
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
sys.path.append(os.path.join(conf_path, 'aspanformer'))

from aspanformer.src.ASpanFormer.aspanformer import ASpanFormer 
from aspanformer.src.config.default import get_cfg_defaults as as_get_cfg_defaults
from aspanformer.src.utils.misc import lower_config as as_lower_config
  



def download_aspanformer(weight_path='../weights/aspanformer'):    
    file = 'weights_aspanformer.tar'    
    url = 'https://drive.google.com/file/d/1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k/view?usp=share_link'

    os.makedirs(os.path.join(weight_path, 'download'), exist_ok=True)   

    file_to_download = os.path.join(weight_path, 'download', file)    
    if not os.path.isfile(file_to_download):    
        gdown.download(url, file_to_download, fuzzy=True)
        
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extractall(weight_path)
            
        shutil.move(os.path.join(weight_path, 'weights', 'indoor.ckpt'), os.path.join(weight_path, 'indoor.ckpt'))
        shutil.move(os.path.join(weight_path, 'weights', 'outdoor.ckpt'), os.path.join(weight_path, 'outdoor.ckpt'))
        os.rmdir(os.path.join(weight_path, 'weights'))
         

class aspanformer_module:
    """
    A detector-free matching module using Asymmetric-Sampling Transformers.

    ASpanFormer improves upon models like LofTR by using an 'asymmetric 
    sampling' strategy. It adaptively adjusts the sampling span (the area 
    it looks at) based on the image content. This allows it to handle 
    large scale differences and extreme camera motions more effectively 
    than fixed-grid transformers.

    Attributes:
        outdoor (bool): Switch between models trained on outdoor (MegaDepth) 
            or indoor (ScanNet) datasets.
        resize (list, optional): Target resolution for processing (e.g., [1024, 1024]).
        patch_radius (int): Defines the local area size for the homography 
            metadata associated with each match.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
                                
        self.args = {
            'id_more': '',
            'outdoor': True,
            'resize': None,                              # default [1024, 1024]
            'patch_radius': 16,
            }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('aspanformer', args, self.args)        

        download_aspanformer()

        if self.args['outdoor']:
            self.weights = os.path.join('../weights/aspanformer/outdoor.ckpt')
            self.config_path = os.path.join('aspanformer/configs/aspan/outdoor/aspan_test.py')
        else:
            self.weights = os.path.join('../weights/aspanformer/indoor.ckpt')
            self.config_path = os.path.join('aspanformer/configs/aspan/indoor/aspan_test.py')
            

        parser = argparse.ArgumentParser(description='AspanFormer online demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--weights_path', type=str, default=self.weights, help="Path to the checkpoint.")
        parser.add_argument('--config_path', type=str, default=self.config_path, help="Path to the config.")

        as_args = parser.parse_args()

        config = as_get_cfg_defaults()
        config.merge_from_file(as_args.config_path)
        _config = as_lower_config(config)
        self.matcher = ASpanFormer(config=_config['aspan'])
        state_dict = torch.load(as_args.weights_path, map_location='cpu', weights_only=False)['state_dict']
        self.matcher.load_state_dict(state_dict,strict=False)

        if device.type == 'cuda':        
            self.matcher.cuda()

        self.matcher.eval()
        
        self.first_warning = True

    
    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args): 
        if not self.first_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.run_actually(**args)
        else:
            return self.run_actually(**args)

    def run_actually(self, **args): 
        if self.first_warning:
            self.first_warning = False

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

        data = {
            "image0": image0.unsqueeze(0),    # LofTR works on grayscale images
            "image1": image1.unsqueeze(0),
        }
                  
        self.matcher(data)

        kps1 = data['mkpts0_f'].detach().to(device).squeeze()
        kps2 = data['mkpts1_f'].detach().to(device).squeeze()
        m_val = data['mconf'].detach().to(device)
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
