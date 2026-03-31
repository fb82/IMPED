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

from .sampling import pipe_union

def pair_rot4(pair, cache_path='tmp_imgs', force=False, **dummy_args):
    """
    A generator that yields four rotated versions of an image pair.

    It keeps the first image fixed and rotates the second image by 0, 90, 
    180, and 270 degrees. For each rotation, it calculates the precise 
    3x3 'warp matrix' needed to map coordinates from the rotated image 
    back to the original upright orientation.

    Args:
        pair (list): Paths to the [Reference Image, Target Image].
        cache_path (str): Folder to store the physically rotated .jpg/.png files.
        force (bool): If True, re-rotates and overwrites existing cached images.

    Yields:
        tuple: ((img0, img_rotated), [Identity_Matrix, Warp_Matrix], extra_data)
    """
    yield pair, [torch.eye(3, device=device, dtype=torch.float), torch.eye(3, device=device, dtype=torch.float)], {}

    rot_mat = np.eye(2)
    
    os.makedirs(cache_path, exist_ok=True)
    
    rot_to_do = [
        ['_rot90', cv2.ROTATE_90_CLOCKWISE],
        ['_rot_180', cv2.ROTATE_180],
        ['_rot_270', cv2.ROTATE_90_COUNTERCLOCKWISE],
        ]

    width, height = Image.open(pair[1]).size
    c = [width / 2, height / 2]

    for r in range(len(rot_to_do)):
        img = os.path.split(pair[1])[1]
        img_name, img_ext = os.path.splitext(img)
        new_img = os.path.join(cache_path, img_name + rot_to_do[r][0] + img_ext)

        if not os.path.isfile(new_img) or force:
            im = cv2.imread(pair[1], cv2.IMREAD_UNCHANGED)
            im = cv2.rotate(im, rot_to_do[r][1])
            cv2.imwrite(new_img, im)
                                            
        m0 = [[1, 0, -c[(0 + r + 1) % 2]],
              [0, 1, -c[(1 + r + 1) % 2]],
              [0, 0,          1        ]]

        rot_mat = np.asarray([[0, 1], [-1, 0]]) @ rot_mat
        m1 = np.eye(3)
        m1[:2, :2] = rot_mat

        m2 = [[1, 0, c[0]],
              [0, 1, c[1]],
              [0, 0,   1 ]]

        # from warped to original
        warp_matrix = torch.tensor(m2 @ m1 @ m0, device=device, dtype=torch.float)
            
        yield (pair[0], new_img), [torch.eye(3, device=device, dtype=torch.float), warp_matrix], {}


def pipe_max_matches(pipe_block):
    """
    Selects the single best result from a collection of matching attempts.

    This function evaluates each block in the 'pipe_block' list based on 
    the total count of geometric inliers (m_mask). It identifies which 
    transformation or model produced the most matches and returns only 
    that result, discarding the others.

    Args:
        pipe_block (list): A list of dictionary objects containing 
                           matching results (kp, m_mask, etc.).

    Returns:
        dict: The single dictionary from the input list with the 
              highest number of confirmed inliers.
    """
    n_matches = torch.zeros(len(pipe_block), device=device)
    for i in range(len(pipe_block)):
        if 'm_mask' in pipe_block[i]:
            n_matches[i] = pipe_block[i]['m_mask'].sum()
    
    best = n_matches.max(0)[1]
    
    return pipe_block[best]
        


class image_muxer_module:
    """
    A module that executes a pipeline across multiple image transformations.

    By using a 'pair_generator', this module can create variations of the 
    input image pair (such as 90-degree rotations). It runs the matching 
    pipeline on each variation and then 'unwarps' the results back to the 
    original coordinate system.

    This is highly effective for matchers that are not naturally 
    rotation-invariant or to increase the total number of matches 
    through TTA (Test-Time Augmentation).

    Attributes:
        pair_generator (function): Logic to create image variations (e.g., pair_rot4).
        pipe_gather (function): Method to combine results (e.g., pipe_max_matches).
        check_border (bool): If True, discards keypoints that fall outside 
            the image boundaries after transformation.
    """
    def __init__(self, id_more='', cache_path='tmp_imgs', pair_generator=pair_rot4, pipe_gather=pipe_max_matches, pipeline=None, add_to_cache=True, check_border=True):
        self.single_image = False
        self.pipeliner = True
        self.pass_through = False
                        
        self.id_more = id_more
        self.cache_path = cache_path
        self.pair_generator = pair_generator
        self.pipe_gather = pipe_gather
        self.add_to_cache = add_to_cache
        self.check_border = check_border

        if pipeline is None: pipeline = []
        self.pipeline = pipeline

        self.id_string = 'image_muxer'
        if len(self.id_more): self.id_string = self.id_string + '_' + str(self.id_more)        


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        finalize_pipeline(self.pipeline)
        
        return


    def run(self, db=None, force=False, pipe_data=None, pipe_name='/'):        
        if pipe_data is None: pipe_data = {}
        pair = pipe_data['img']
        warp = pipe_data['warp']
        pipe_data_block = []
        
        for pair_, warp_, aux_data in self.pair_generator(pair, cache_path=self.cache_path, force=force, pipe_data=pipe_data):            
            pipe_data_in = pipe_data.copy()

            for k in aux_data.keys():
                pipe_data_in[k] = aux_data[k]

            pipe_data_in['img'] = [pair_[0], pair_[1]]
            pipe_data_in['warp'] = [warp_[0], warp_[1]]
            
            if 'kp' in pipe_data_in:
                pipe_data_in['kp'] = [    
                    apply_homo(pipe_data_in['kp'][0], warp_[0].inverse()),
                    apply_homo(pipe_data_in['kp'][1], warp_[1].inverse())
                    ]
                
                if self.check_border:
                    sz0 = Image.open(pair_[0]).size
                    sz1 = Image.open(pair_[1]).size     
                    bmask0 = (pipe_data_in['kp'][0][:, 0] >= 0) & (pipe_data_in['kp'][0][:, 0] <= sz0[0] - 1) & (pipe_data_in['kp'][0][:, 1] >= 0) & (pipe_data_in['kp'][0][:, 1] <= sz0[1] - 1)
                    bmask1 = (pipe_data_in['kp'][1][:, 0] >= 0) & (pipe_data_in['kp'][1][:, 0] <= sz1[0] - 1) & (pipe_data_in['kp'][1][:, 1] >= 0) & (pipe_data_in['kp'][1][:, 1] <= sz1[1] - 1)
                       
            if 'kH' in pipe_data_in:
                pipe_data_in['kH'] = [    
                    change_patch_homo(pipe_data_in['kH'][0], warp_[0]),
                    change_patch_homo(pipe_data_in['kH'][1], warp_[1]),
                    ]

            if self.check_border:
                if 'kp' in pipe_data_in:
                    pipe_data_in['kp'] = [pipe_data_in['kp'][0][bmask0], pipe_data_in['kp'][1][bmask1]]
                
                if 'kH' in pipe_data_in:                
                    pipe_data_in['kH'] = [pipe_data_in['kH'][0][bmask0], pipe_data_in['kH'][1][bmask1]]

                if 'kr' in pipe_data_in:                
                    pipe_data_in['kr'] = [pipe_data_in['kr'][0][bmask0], pipe_data_in['kr'][1][bmask1]]

                if 'desc' in pipe_data_in:
                    pipe_data_in['desc'] = [pipe_data_in['desc'][0][bmask0], pipe_data_in['desc'][1][bmask1]]

                if 'm_idx' in pipe_data_in:
                    bmask01 = bmask0[pipe_data_in['m_idx'][:, 0]] & bmask1[pipe_data_in['m_idx'][:, 2]]
                    pipe_data_in['m_idx'] = pipe_data_in['m_idx'][bmask01]
                    pipe_data_in['m_val'] = pipe_data_in['m_val'][bmask01]
                    pipe_data_in['m_mask'] = pipe_data_in['m_mask'][bmask01]
                                
            if ('H' in pipe_data_in) and (not pipe_data_in['H'] is None):
                pipe_data_in['H'] = warp_[1].to(torch.double) @ pipe_data_in['H'] @ warp_[0].to(torch.double)

            if ('F' in pipe_data_in) and (not pipe_data_in['F'] is None):
                pipe_data_in['F'] = warp_[1].permute((1, 0)).to(torch.double) @ pipe_data_in['F'] @ warp_[0].to(torch.double)

            pipe_data_out, pipe_name_out = run_pipeline(pair_, self.pipeline, db, force=force, pipe_data=pipe_data_in, pipe_name=pipe_name)

            pipe_data_out['img'] = pair
            pipe_data_out['warp'] = warp

            if 'kp' in pipe_data_out:
                pipe_data_out['kp'] = [    
                    apply_homo(pipe_data_out['kp'][0], warp_[0]),
                    apply_homo(pipe_data_out['kp'][1], warp_[1])
                    ]

            if 'kH' in pipe_data_out:
                pipe_data_out['kH'] = [    
                    change_patch_homo(pipe_data_out['kH'][0], warp_[0].inverse()),
                    change_patch_homo(pipe_data_out['kH'][1], warp_[1].inverse()),
                    ]
                
            if ('H' in pipe_data_out) and (not pipe_data_out['H'] is None):
                pipe_data_out['H'] = warp_[1].to(torch.double).inverse() @ pipe_data_out['H'] @ warp_[0].to(torch.double).inverse()

            if ('F' in pipe_data_out) and (not pipe_data_out['F'] is None):
                pipe_data_out['F'] = warp_[1].to(torch.double).inverse().permute((1, 0)) @ pipe_data_out['F'] @ warp_[0].to(torch.double).inverse()
                        
            pipe_data_block.append(pipe_data_out)
                    
        return self.pipe_gather(pipe_data_block)
        




class pipeline_muxer_module:
    """
    A meta-module that executes multiple independent pipelines in parallel.

    The Muxer (Multiplexer) allows for complex 'ensemble' strategies. For example, 
    you can run a Transformer-based matcher (LoFTR) and a Sparse matcher (SuperPoint) 
    at the same time. The results from all sub-pipelines are then merged using 
    a gathering function (like 'pipe_union').

    This is particularly useful for achieving high robustness across 
    diverse datasets where a single model might fail.

    Attributes:
        pipeline (list): A list of pipeline definitions to be executed.
        pipe_gather (function): The method used to combine results from 
            different pipelines (defaults to pipe_union).
        id_more (str): An optional suffix to differentiate multiple muxers.
    """
    def __init__(self, id_more='', pipe_gather=pipe_union, pipeline=None, add_to_cache=True):
        self.single_image = False
        self.pipeliner = True
        self.pass_through = False
                
        self.id_more = id_more                
        self.pipe_gather = pipe_gather
        self.add_to_cache = add_to_cache
        
        if pipeline is None: pipeline = []
        self.pipeline = pipeline        

        self.id_string = 'pipeline_muxer'
        if len(self.id_more): self.id_string = self.id_string + '_' + str(self.id_more)        


    def get_id(self): 
        return self.id_string


    def finalize(self):        
        for pipeline in self.pipeline:
            finalize_pipeline(pipeline)

        return


    def run(self, db=None, force=False, pipe_data=None, pipe_name='/'):
        if pipe_data is None: pipe_data = {}

        pipe_data_block = []
        
        for pipeline in self.pipeline:
            pipe_data_in = pipe_data.copy()
            pair = pipe_data['img']
                                       
            pipe_data_out, pipe_name_out = run_pipeline(pair, pipeline, db, force=force, pipe_data=pipe_data_in, pipe_name=pipe_name)        
            pipe_data_block.append(pipe_data_out)
        
        return self.pipe_gather(pipe_data_block)

