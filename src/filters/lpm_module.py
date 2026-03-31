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

from core import device, pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, image_pairs, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args, enable_quadtree



import lpm.LPM as lpm

class lpm_module:
    """
    An outlier rejection module using Locality Preserving Matching.

    LPM is a non-parametric approach to feature matching. Instead of 
    fitting a global geometric model (like a Homography), it enforces 
    local neighborhood consistency. It is highly effective for:
    1. Non-rigid deformations (e.g., matching a flag waving in the wind).
    2. Scenes with heavy perspective distortion.
    3. High-speed filtering without the need for GPU-heavy deep learning.

    The algorithm determines inliers by checking if the neighbors of a 
    point in Image 1 map to the neighbors of the corresponding point 
    in Image 2.
    """
    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
        self.add_to_cache = True
                        
        self.args = {
            'id_more': '',
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('lpm', args, self.args)        


    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args):  
        pt1_ = args['kp'][0]
        pt2_ = args['kp'][1]
                
        mi = args['m_idx']
        mm = args['m_mask']
        
        pt1 = pt1_[mi[mm][:, 0]]
        pt2 = pt2_[mi[mm][:, 1]]
        
        mask = lpm.LPM_filter(pt1.to('cpu').numpy(), pt2.to('cpu').numpy())        
        mask = torch.tensor(mask, device=device, dtype=torch.bool)
 
        aux = mm.clone()
        mm[aux] = mask
        
        return {'m_mask': mm}

