
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



class show_matches_module:
    """
    A diagnostic module for generating and saving match visualization plots.

    This module creates 'side-by-side' image plots with lines connecting 
    corresponding keypoints. It is highly configurable, allowing for 
    color-coded distinction between inliers (correct matches) and 
    outliers (incorrect matches) based on the geometric mask.

    Attributes:
        cache_path (str): The directory where visualization images are saved.
        mask_idx (list): Controls what to show. [1] shows inliers only, 
            [0, 1] shows both, and -1 shows all raw matches.
        params (list): A list of visual styles (colors) for each mask category.
        fig_max_size (int): Constrains the resolution of the output image 
            to save disk space while maintaining clarity.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'img_prefix': '',
            'img_suffix': '',
            'cache_path': 'show_imgs',
            'prepend_pair': True,
            'ext': '.jpg',
            'force': False,
            'mask_idx': [1], # -1: all, [1]: inliers, [0]: outliers, [0, 1]: outlier and inliers with differen colors
            'fig_min_size': 960,
            'fig_max_size': 1280, 
            'params': [{'color': [1, 0, 0]}, {'color': [0, 1, 0]}],
        }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('show_matches' , args, self.args)

                
    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

    
    def run(self, **args):         
        im0 = os.path.splitext(os.path.split(args['img'][0])[1])[0]
        im1 = os.path.splitext(os.path.split(args['img'][1])[1])[0]

        if self.args['prepend_pair']:            
            cache_path = os.path.join(self.args['cache_path'], im0 + '_' + im1)
        else:
            cache_path = self.args['cache_path']
                
        new_img = os.path.join(cache_path, self.args['img_prefix'] + im0 + '_' + im1 + self.args['img_suffix'] + self.args['ext'])
    
        if not os.path.isfile(new_img) or self.args['force']:
            os.makedirs(cache_path, exist_ok=True)

            fig = plt.figure()    
            img0 = viz_utils.load_image(args['img'][0])
            img1 = viz_utils.load_image(args['img'][1])
            fig, axes = viz.plot_images([img0, img1], fig_num=fig.number)              

            if 'm_idx' in args:
                if self.args['mask_idx'] == -1:
                    mask_idx = -1
                    params = self.args['params'][-1]

                    m_idx = args['m_idx']
                    pt1 = args['kp'][0][m_idx[:, 0]]
                    pt2 = args['kp'][1][m_idx[:, 1]]

                    if pt1.shape[0] > 0:
                        viz.plot_matches(pt1, pt2, color=self.args['params'][0]['color'], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
                else:
                    if not isinstance(self.args['mask_idx'], list): self.args['mask_idx'] = [self.args['mask_idx']]                    
                    mask_idx = self.args['mask_idx']
                    params = self.args['params']

                    m_mask = args['m_mask']
                    m_sum = torch.tensor([(m_mask == i).sum().item() for i in mask_idx], device=device)
                    idx = torch.argsort(m_sum, descending=True)

                    mask_idx = [mask_idx[i] for i in idx]
                    params = [params[i] for i in idx]
                    
                    for i in mask_idx:
                        
                        m_idx = args['m_idx'][args['m_mask'] == i]
                        if m_idx.shape[0] < 1: continue                        

                        pt1 = args['kp'][0][m_idx[:, 0]]
                        pt2 = args['kp'][1][m_idx[:, 1]]

                        viz.plot_matches(pt1, pt2, color=self.args['params'][i]['color'], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
            
            fig_dpi = fig.get_dpi()
            fig_sz = [fig.get_figwidth() * fig_dpi, fig.get_figheight() * fig_dpi]
        
            fig_min_size = self.args['fig_min_size']
            fig_max_size = self.args['fig_max_size']
        
            fig_cz = min(fig_sz)
            if fig_cz < fig_min_size:
                fig_sz[0] = fig_sz[0] / fig_cz * fig_min_size
                fig_sz[1] = fig_sz[1] / fig_cz * fig_min_size
        
            fig_cz = max(fig_sz)
            if fig_cz > fig_min_size:
                fig_sz[0] = fig_sz[0] / fig_cz * fig_max_size
                fig_sz[1] = fig_sz[1] / fig_cz * fig_max_size
                
            fig.set_size_inches(fig_sz[0] / fig_dpi, fig_sz[1]  / fig_dpi)
        
            viz.save_plot(new_img, fig)
            viz.clear_plot(fig)
                        
            plt.close(fig)
        
        return {}


