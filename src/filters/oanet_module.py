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



def download_oanet(weight_path='../weights/oanet'):
    """
    Downloads and extracts the pre-trained weights for the OANet matcher.

    This function automates the environment setup for OANet by:
    1. Downloading a 'tar.gz' archive containing models trained on the 
       GL3D dataset (a large-scale 3D reconstruction dataset).
    2. Extracting the specific 'model_best.pth' file from a nested 
       directory structure within the archive.
    3. Flattening the file structure by moving the weights to the root 
       'oanet' weight directory for easier access by the module.
    4. Cleaning up temporary extraction folders to save disk space.

    Args:
        weight_path (str): The local directory where OANet weights 
            should be stored.

    Returns:
        None: Operates via side-effects on the file system.
    """
    url = "https://drive.google.com/file/d/1Yuk_ZBlY_xgUUGXCNQX-eh8BO2ni_qhm/view?usp=sharing"

    os.makedirs(os.path.join(weight_path, 'download'), exist_ok=True)   

    file_to_download = os.path.join(weight_path, 'download', 'sift-gl3d.tar.gz')    
    if not os.path.isfile(file_to_download):    
        gdown.download(url, file_to_download, fuzzy=True)

    model_file = os.path.join(weight_path, 'model_best.pth')
    if not os.path.isfile(model_file):
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extract('gl3d/sift-4000/model_best.pth', path=weight_path)
        
        shutil.copy(os.path.join(weight_path, 'gl3d/sift-4000/model_best.pth'), model_file)
        shutil.rmtree(os.path.join(weight_path, 'gl3d'))


import oanet.learnedmatcher_custom as oanet

class oanet_module:
    """
    A deep-learning outlier rejection module using Order-Aware Networks.

    OANet is designed to capture the local and global context of image matches. 
    It focuses on the 'order' and spatial distribution of keypoints to 
    distinguish between true inliers (correct matches) and outliers. 
    Unlike standard RANSAC, it is fully differentiable and can be 
    trained to be much more robust in scenes with repetitive patterns.

    Attributes:
        weights (str): Path to the pre-trained PyTorch model (.pth).
        inlier_threshold (int/float): The geometric threshold used during 
            inference to determine if a match is valid.
        lm (oanet.LearnedMatcher): The core OANet inference engine.
    """
    def __init__(self, **args):  

        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
        self.add_to_cache = True
                        
        self.args = {
            'id_more': '',
            'weights': '../weights/oanet/model_best.pth',
            'inlier_threshold': 1,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        download_oanet()
        
        self.id_string, self.args = set_args('oanet', args, self.args)                     
        self.lm = oanet.LearnedMatcher(self.args['weights'], inlier_threshold=self.args['inlier_threshold'], use_ratio=0, use_mutual=0, corr_file=-1)        
        
               
    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args): 
        mi = args['m_idx']
        mm = args['m_mask']

        m12 = mi[mm]

        k1 = args['kp'][0]
        k2 = args['kp'][1]
        
        k1 = k1[m12[:, 0]]
        k2 = k2[m12[:, 1]]        
        
        pt1 = np.ascontiguousarray(k1.detach().cpu())
        pt2 = np.ascontiguousarray(k2.detach().cpu())
                
        l = pt1.shape[0]
        
        if l > 1:
            _, _, _, _, mask = self.lm.infer(pt1, pt2)
            
            mask_aux = torch.tensor(mask, device=device)         
            aux = mm.clone()
            mm[aux] = mask_aux
        
            return {'m_mask': mm}
        else:
            return {'m_mask': args['m_mask']}
