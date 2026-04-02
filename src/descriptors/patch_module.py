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

class patch_module:
    """
    A module for refining the orientation and affine shape of keypoints.

    This module takes initial keypoints and 'upgrades' them by estimating 
    their dominant orientation (rotation) and affine shape (tilt/shear). 
    By normalizing these geometric properties, descriptors extracted later 
    (like SIFT or HardNet) become much more robust to viewpoint changes.

    Attributes:
        orinet (bool): Uses a deep neural network (OriNet) to predict 
            the best orientation for the keypoint.
        affnet (bool): Uses a deep neural network (AffNet) to estimate 
            the local affine shape, effectively 'un-tilting' the image patch.
        sift_orientation (bool): Uses traditional gradient-based methods 
            to find the dominant rotation.
    """
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False
        self.add_to_cache = True
                
        self.args = {
            'id_more': '',
            'sift_orientation': False,
            'sift_orientation_params': {},
            'general_orientation_params': {},
            'orinet': True,
            'orinet_params': {
                'pretrained': True,
                },
            'affnet': True,
            'affnet_params': {
                'pretrained': True,
                },
            }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('', args, self.args)

        base_string = ''
        self.ori_module = K.feature.PassLAF()
        if self.args['sift_orientation']:
            base_string = 'sift_orientation'
            self.ori_module = K.feature.LAFOrienter(angle_detector=K.feature.PatchDominantGradientOrientation(**self.args['sift_orientation_params']), **self.args['general_orientation_params'])
        if self.args['orinet']:
            base_string = 'orinet'
            self.ori_module = K.feature.LAFOrienter(angle_detector=K.feature.OriNet(**self.args['orinet_params']).to(device), **self.args['general_orientation_params'])

        if self.args['affnet']:
            if len(base_string): base_string = base_string  + '_' + 'affnet'
            else: base_string = 'affnet'
            self.aff_module =  K.feature.LAFAffNetShapeEstimator(**self.args['affnet_params']).to(device)
        else:
            self.aff_module = K.feature.PassLAF()

        if not len(base_string): base_string = 'pass_laf'
        self.id_string = base_string + self.id_string


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):    
        im = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)

        lafs = homo2laf(args['kp'][args['idx']], args['kH'][args['idx']])

        lafs = self.aff_module(lafs, im)
        lafs = self.ori_module(lafs, im)

        kp, kH = laf2homo(lafs.squeeze(0))
    
        return {'kp': kp, 'kH': kH}
