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

from filters import dtm_module as dtm
import dtm.src.dtm as dtm

class blob_matching_module:
    """
    A pipeline module for performing blob-based feature matching between two sets of descriptors.

    This module wraps the `dtm.blob_matching` function, facilitating the comparison 
    of keypoints and descriptors from two different images. It handles configuration 
    management, device placement, and ensures the output is compatible with 
    downstream filtering or geometry estimation modules.

    Attributes:
        single_image (bool): Flag indicating if the module operates on a single image.
        pipeliner (bool): Flag for integration within a sequential processing pipeline.
        pass_through (bool): If True, allows data to pass without modification.
        add_to_cache (bool): If True, enables result caching for the current configuration.
        args (dict): Dictionary of matching parameters including:
            - pf (int): Threshold/parameter for union matching logic (default: -10).
            - pn (int): Neighbor/proximity parameter (default: 5).
            - ps (int): Search space or offset parameter (default: 16).
            - use_stats (bool): Whether to utilize statistical data during matching.
            - distance (str): Metric for descriptor distance (e.g., 'L2').
            - split_sz (int): Block size for memory-efficient processing (chunking).
            - device (str): Computational device ('cpu' or 'cuda').
        id_string (str): Unique identifier based on the specific parameter configuration.

    Args:
        **args: Arbitrary keyword arguments used to override default matching settings.
    """    
    def __init__(self, **args):
        self.single_image = False    
        self.pipeliner = False      
        self.pass_through = False
        self.add_to_cache = True
                                
        self.args = {
            'id_more': '',
            'pf': -10, # f = 10 with union
            'pn': 5,   # f' = 5
            'ps': 16,  # to = 10
            'use_stats': True,
            'out_idx': 10,
            'distance': 'L2',
            'split_sz': 1024,
            'same_order': True,
            'device': 'cpu',
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('blob_matching', args, self.args)        


    def get_id(self): 
        return self.id_string
    

    def finalize(self):
        return


    def run(self, **args):
        pt1 = args['kp'][0]
        pt2 = args['kp'][1]
        
        desc1 = args['desc'][0].to(torch.float32)
        desc2 = args['desc'][1].to(torch.float32)

        ss = self.args['split_sz']

        midx, val = dtm.blob_matching(pt1, pt2, desc1, desc2,
                  pf=self.args['pf'],
                  pn=self.args['pn'],
                  ps=self.args['ps'],
                  use_stats=self.args['use_stats'],
                  out_idx=self.args['out_idx'],
                  distance=self.args['distance'],
                  ss=ss, # split size
                  same_order=self.args['same_order'],    
                  device=self.args['device'],
        )
        
        midx = midx.to(device)
        val = val.to(device)
    
        return {'m_idx': midx, 'm_val': val, 'm_mask': torch.ones(val.shape[0], device=device, dtype=torch.bool)}
