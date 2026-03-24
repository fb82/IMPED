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

from core import device, pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, image_pairs, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args


class loftr_module:
    """
    A dense matching module using LoFTR (Local Feature Transformer).

    LoFTR bypasses the traditional 'detect-then-describe' pipeline. Instead, 
    it establishes dense correspondences by processing image pairs through 
    a coarse-to-fine transformer architecture. This allows it to find 
    matches in low-texture areas where traditional detectors fail.

    Attributes:
        outdoor (bool): Switch between 'outdoor' (MegaDepth) and 'indoor' 
            (ScanNet) pretrained weights.
        resize (list, optional): Image resolution for processing.
        patch_radius (int): Normalization factor for local homography metadata.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
                                
        self.args = {
            'id_more': '',
            'outdoor': True,
            'resize': None,                          # self.resize = [800, 600]
            'patch_radius': 16,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('loftr', args, self.args)        

        if self.args['outdoor'] == True:
            pretrained = 'outdoor'
        else:
            pretrained = 'indoor_new'

        self.matcher = K.feature.LoFTR(pretrained=pretrained).to(device).eval()


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

        input_dict = {
            "image0": image0.unsqueeze(0),    # LofTR works on grayscale images
            "image1": image1.unsqueeze(0),
        }

        correspondences = self.matcher(input_dict)

        kps1 = correspondences["keypoints0"]
        kps2 = correspondences["keypoints1"]
        m_val = correspondences['confidence']
                        
        kps1 = kps1.detach().to(device).squeeze()
        kps2 = kps2.detach().to(device).squeeze()

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

        m_mask = m_val > 0

        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}
