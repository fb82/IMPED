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


class sift_module:
    """
    A descriptor extraction module using the SIFT algorithm.

    This module takes pre-existing keypoints and extracts SIFT descriptors 
    from the grayscale version of the image. It supports standard SIFT 
    and the 'RootSIFT' variant, which often provides better matching 
    performance by using Hellinger distance instead of Euclidean distance.

    Note: This module does not 'detect' points; it only 'describes' them.

    Attributes:
        rootsift (bool): If True, applies L1 normalization and a 
            square root to the descriptors to improve matching robustness.
    """
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False
        self.add_to_cache = True
                
        self.args = {
            'id_more': '',
            'rootsift': True,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('', args, self.args)        
        self.descriptor = cv2.SIFT_create()

        if self.args['rootsift']:
            base_string = 'rootsift'
        else:
            base_string = 'sift'
            
        self.id_string = base_string + self.id_string

    def get_id(self): 
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        im = cv2.imread(args['img'][args['idx']], cv2.IMREAD_GRAYSCALE)        
        lafs = homo2laf(args['kp'][args['idx']], args['kH'][args['idx']])                
        kp = opencv_kpts_from_laf(lafs)
        
        _, desc = self.descriptor.compute(im, kp)

        if self.args['rootsift']:
            desc /= desc.sum(axis=1, keepdims=True) + 1e-8
            desc = np.sqrt(desc)
            
        desc = torch.tensor(desc, device=device, dtype=torch.float)
                    
        return {'desc': desc}
