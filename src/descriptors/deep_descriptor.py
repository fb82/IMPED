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

class deep_descriptor_module:
    """
    A module for extracting learned local descriptors using deep neural networks.

    This module takes existing keypoints (represented as Local Affine Frames) 
    and passes them through a specialized CNN (HardNet, SOSNet, or HyNet). 
    These networks are trained on millions of patches to be invariant to 
    extreme changes in lighting, perspective, and seasonal appearance.

    Attributes:
        descriptor (str): The CNN architecture to use ('hardnet', 'sosnet', 'hynet').
        desc_params (dict): Parameters passed to the specific network (e.g., pretrained=True).
        patch_params (dict): Parameters for patch extraction (e.g., patch_size).
    """
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False
        self.add_to_cache = True
                
        self.args = {
            'id_more': '',
            'descriptor': 'hardnet',
            'desc_params': {
                'pretrained': True,
                },
            'patch_params': {},
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('', args, self.args)        
        
        if self.args['descriptor'] == 'hardnet':
            base_string = 'hardnet'
            desc = K.feature.HardNet(**self.args['desc_params']).to(device)
        if self.args['descriptor'] == 'sosnet':
            desc = K.feature.SOSNet(**self.args['desc_params']).to(device)
            base_string = 'sosnet'
        if self.args['descriptor'] == 'hynet':
            desc = K.feature.HyNet(**self.args['desc_params']).to(device)
            base_string = 'hynet'

        self.ddesc = K.feature.LAFDescriptor(patch_descriptor_module=desc, **self.args['patch_params'])
        self.id_string = base_string + self.id_string


    def get_id(self): 
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):    
        im = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)

        lafs = homo2laf(args['kp'][args['idx']], args['kH'][args['idx']])
        desc = self.ddesc(im, lafs).squeeze(0)
    
        return {'desc': desc}
