import os
import warnings
import pickled_hdf5.pickled_hdf5 as pickled_hdf5
import time
from tqdm import tqdm
import torchvision.transforms as transforms

import torch
import kornia as K
from kornia_moons.feature import opencv_kpts_from_laf, laf_from_opencv_kpts
import cv2
import numpy as np
from PIL import Image
import poselib
import gdown
import zipfile
import tarfile
import csv
import shutil
import bz2
import _pickle as cPickle
import argparse
import math
import copy
import wget
import pycolmap
import scipy
import miho.src.miho as mop_miho
import miho.src.miho_other as mop
import miho.src.ncc as ncc

import matplotlib.pyplot as plt
from matplotlib import colormaps
import plot.viz2d as viz
import plot.utils as viz_utils
import sys
from pathlib import Path

from core import device, pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args
from image_pairs import image_pairs


conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'quadtreeattention'))
sys.path.append(os.path.join(conf_path, 'quadtreeattention/QuadTreeAttention'))
from FeatureMatching.src.config.default import get_cfg_defaults as qta_get_cfg_defaults
from FeatureMatching.src.utils.misc import lower_config as qta_lower_config
from FeatureMatching.src.loftr import LoFTR as qta_LoFTR




def download_quadtreeattention(weight_path='../weights/quadtreeattention'):    
    file_list = [
        'indoor.ckpt',
        'outdoor.ckpt',
    ]
    
    url_list = [
        'https://drive.google.com/file/d/1pSK_8GP1WkqKL5m7J4aHvhFixdLP6Yfa/view?usp=sharing',
        'https://drive.google.com/file/d/1UOYdzbrXHU9kvVy9tscCCO7BB3G4rWK4/view?usp=sharing',
    ]

    os.makedirs(weight_path, exist_ok=True)   

    for file, url in zip(file_list, url_list):

        file_to_download = os.path.join(weight_path, file)    
        if not os.path.isfile(file_to_download):    
            gdown.download(url, file_to_download, fuzzy=True)


    
class quadtreeattention_module:
    """
A dense matching module using QuadTree Attention (hierarchical LoFTR).

QuadTree Attention reduces the quadratic complexity of standard 
Transformers ($O(N^2)$) by selectively attending to image regions. 
It builds a pyramid of features and only performs fine-grained 
matching in regions that show high probability of containing matches 
at a coarser level.

This makes it significantly more memory-efficient than standard LoFTR, 
allowing for higher-resolution processing on the same hardware.

Attributes:
    outdoor (bool): Switch between outdoor (MegaDepth) and indoor (ScanNet) weights.
    resize (list, optional): Processing resolution.
    patch_radius (int): Used to construct local homography metadata.
"""
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
                                    
        self.args = {
            'id_more': '',
            'outdoor': True,
            'resize': None,                      # self.resize = [800, 600]
            'patch_radius': 16,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
            
        self.id_string, self.args = set_args('quadtreeattention', args, self.args)        

        download_quadtreeattention()

        if self.args['outdoor'] == True:
            self.weights = '../weights/quadtreeattention/outdoor.ckpt'
            self.config_path = 'quadtreeattention/FeatureMatching/configs/loftr/outdoor/loftr_ds_quadtree.py'
        else:
            self.weights = '../weights/quadtreeattention/indoor.ckpt'
            self.config_path = 'quadtreeattention/FeatureMatching/configs/loftr/indoor/loftr_ds_quadtree.py'

        parser = argparse.ArgumentParser(description='QuadTreeAttention online demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--weight', type=str, default=self.weights, help="Path to the checkpoint.")
        parser.add_argument('--config_path', type=str, default=self.config_path, help="Path to the config.")

        opt = parser.parse_args()
    
        # init default-cfg and merge it with the main- and data-cfg
        config = qta_get_cfg_defaults()
        config.merge_from_file(opt.config_path)
        _config = qta_lower_config(config)
    
        # Matcher: LoFTR
        self.matcher = qta_LoFTR(config=_config['loftr'])
        state_dict = torch.load(opt.weight, map_location='cpu', weights_only=False)['state_dict']
        self.matcher.load_state_dict(state_dict, strict=True)

        self.matcher.eval()

        if device.type == 'cuda':        
            self.matcher.to('cuda')

    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):
        image0 = K.io.load_image(args['img'][0], K.io.ImageLoadType.GRAY32, device=device)
        image1 = K.io.load_image(args['img'][1], K.io.ImageLoadType.GRAY32, device=device)

        hw1 = image0.shape[1:]
        hw2 = image1.shape[1:]

        if not (self.args['resize'] is None):        
            ms = min(self.args['resize'])
            Ms = max(self.args['resize'])

            if hw1[0] > hw1[1]:
                sz = [Ms, ms]                
                ratio_ori = float(hw1[0]) / hw1[1]                 
            else:
                sz = [ms, Ms]
                ratio_ori = float(hw1[1]) / hw1[0]

            ratio_new = float(Ms) / ms
            if np.abs(ratio_ori - ratio_new) > np.abs(1 - ratio_new):
                sz = [ms, ms]

            image0 = K.geometry.resize(image0, (sz[0], sz[1]), antialias=True)

            if hw2[0] > hw2[1]:
                sz = [Ms, ms]                
                ratio_ori = float(hw2[0]) / hw2[1]                 
            else:
                sz = [ms, Ms]
                ratio_ori = float(hw2[1]) / hw2[0]

            ratio_new = float(Ms) / ms
            if np.abs(ratio_ori - ratio_new) > np.abs(1 - ratio_new):
                sz = [ms, ms]

            image1 = K.geometry.resize(image1, (sz[0], sz[1]), antialias=True)
                    
        hw1_ = image0.shape[1:]
        hw2_ = image1.shape[1:]

        batch = {
            "image0": image0.unsqueeze(0),    # LofTR works on grayscale images
            "image1": image1.unsqueeze(0),
        }

        self.matcher(batch)
        kps1 = batch['mkpts0_f'].detach().to(device).squeeze()
        kps2 = batch['mkpts1_f'].detach().to(device).squeeze()
        m_val = batch['mconf'].detach().to(device)
        m_mask = m_val > 0

        kps1[:, 0] = kps1[:, 0] * (hw1[1] / float(hw1_[1]))
        kps1[:, 1] = kps1[:, 1] * (hw1[0] / float(hw1_[0]))
    
        kps2[:, 0] = kps2[:, 0] * (hw2[1] / float(hw2_[1]))
        kps2[:, 1] = kps2[:, 1] * (hw2[0] / float(hw2_[0]))
        
        kp = [kps1, kps2]
        kH = [
            torch.zeros((kp[0].shape[0], 3, 3), device=device),
            torch.zeros((kp[0].shape[0], 3, 3), device=device),
            ]
        
        kH[0][:, [0, 1], 2] = -kp[0] / self.args['patch_radius']
        kH[0][:, 0, 0] = 1 / self.args['patch_radius']
        kH[0][:, 1, 1] = 1 / self.args['patch_radius']
        kH[0][:, 2, 2] = 1

        kH[1][:, [0, 1], 2] = -kp[1] / self.args['patch_radius']
        kH[1][:, 0, 0] = 1 / self.args['patch_radius']
        kH[1][:, 1, 1] = 1 / self.args['patch_radius']
        kH[1][:, 2, 2] = 1

        kr = [torch.full((kp[0].shape[0],), torch.nan, device=device), torch.full((kp[0].shape[0],), torch.nan, device=device)]        

        m_idx = torch.zeros((kp[0].shape[0], 2), device=device, dtype=torch.int)
        m_idx[:, 0] = torch.arange(kp[0].shape[0])
        m_idx[:, 1] = torch.arange(kp[0].shape[0])

        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}
