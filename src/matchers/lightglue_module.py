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


from lightglue import LightGlue as lg_lightglue, SuperPoint as lg_superpoint, DISK as lg_disk, SIFT as lg_sift, ALIKED as lg_aliked, DoGHardNet as lg_doghardnet
from lightglue.utils import load_image as lg_load_image, rbd as lg_rbd



class lightglue_module:
    """
    A high-performance feature matcher based on the LightGlue architecture.

    LightGlue is a 'deep matcher' that uses a lightweight Transformer to 
    iteratively update the descriptors of keypoints based on their 
    spatial context and their similarity to points in the other image. 

    Unlike traditional matchers that use a simple distance ratio test, 
    LightGlue is adaptive: it can stop early if a match is easy, or 
    perform more iterations for difficult cases, making it both 
    faster and more accurate than its predecessors.

    Attributes:
        what (str): The type of feature being matched (e.g., 'superpoint', 
            'disk', 'aliked', 'sift').
        desc_cf (float): A scaling factor for descriptors, used to normalize 
            different feature types (e.g., set to 255 for SIFT).
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True
                
        self.what = 'superpoint'
        self.args = {
            'id_more': '',
            'num_features': 8000,
            'resize': 1024,           # this is default, set to None to disable
            'desc_cf': 1,                    # 255 to use R2S2 with what='sift'
            'aliked_model': "aliked-n16rot",          # default is "aliked-n16"
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        if 'what' in args:
            self.what = args['what']
            del args['what']

        self.id_string, self.args = set_args('lightglue', args, self.args)        

        if self.what == 'disk':            
            self.matcher = lg_lightglue(features='disk').eval().to(device)            
        elif self.what == 'aliked':            
            self.matcher = lg_lightglue(features='aliked').eval().to(device)            
        elif self.what == 'sift':            
            self.matcher = lg_lightglue(features='sift').eval().to(device)                            
        elif self.what == 'doghardnet':            
            self.matcher = lg_lightglue(features='doghardnet').eval().to(device)            
        else:   
            self.what = 'superpoint'
            self.matcher = lg_lightglue(features='superpoint').eval().to(device)            


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return
    
    
    def run(self, **args):           
        # dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
        # dict_keys(['matches0', 'matches1', 'matching_scores0', 'matching_scores1', 'stop', 'matches', 'scores', 'prune0', 'prune1'])

        width, height = Image.open(args['img'][0]).size
        sz1 = torch.tensor([width / 2, height / 2], device=device)

        width, height = Image.open(args['img'][1]).size
        sz2 = torch.tensor([width / 2, height / 2], device=device)

        feats1 = {'keypoints': args['kp'][0].unsqueeze(0), 'descriptors': args['desc'][0].unsqueeze(0) * self.args['desc_cf'], 'image_size': sz1.unsqueeze(0)} 
        feats2 = {'keypoints': args['kp'][1].unsqueeze(0), 'descriptors': args['desc'][1].unsqueeze(0) * self.args['desc_cf'], 'image_size': sz2.unsqueeze(0)} 
        
        if (self.what == 'sift') or (self.what == 'doghardnet'):
            lafs1 = homo2laf(args['kp'][0], args['kH'][0])
            lafs2 = homo2laf(args['kp'][1], args['kH'][1])
            
            kp1 = opencv_kpts_from_laf(lafs1)
            kp2 = opencv_kpts_from_laf(lafs2)

            feats1['oris'] = torch.tensor([kp.angle for kp in kp1], device=device).unsqueeze(0)
            feats2['oris'] = torch.tensor([kp.angle for kp in kp2], device=device).unsqueeze(0)

            feats1['scales'] = torch.tensor([kp.size for kp in kp1], device=device).unsqueeze(0)
            feats2['scales'] = torch.tensor([kp.size for kp in kp2], device=device).unsqueeze(0)
            
            
        matches12 = self.matcher({'image0': feats1, 'image1': feats2})
        feats1_, feats2_, matches12 = [lg_rbd(x) for x in [feats1, feats2, matches12]]

        idxs = matches12['matches'].squeeze(0)
        m_val = matches12['scores'].squeeze(0)

        if torch.numel(idxs) == 2:
            idxs = idxs.reshape(1, -1)
            m_val = m_val.reshape(1)
        
        m_mask = torch.ones(idxs.shape[0], device=device, dtype=torch.bool)
                    
        return {'m_idx': idxs, 'm_val': m_val, 'm_mask': m_mask}
    