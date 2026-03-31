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



def to_pyramid(in_image, cache_path='pyramid_cache', split_max=3, block_sz_max=256, shared=1/3, interpolation=cv2.INTER_AREA, force=False):
    """
    Decomposes an image into a multi-scale pyramid of overlapping tiles.

    This function divides an image into grids of increasing density (from 1xN to 
    split_max x M). It handles padding, calculates overlap (shared area) between 
    adjacent tiles, and resizes tiles to fit a maximum block size. Each tile is 
    saved to disk, and its corresponding coordinate transformation matrix is computed.

    Args:
        in_image (str): Path to the input image file.
        cache_path (str): Directory where the generated tile images will be stored.
        split_max (int): Number of pyramid levels (granularity levels).
        block_sz_max (int): Maximum allowable dimension (width or height) for a tile.
        shared (float): Fraction of the tile size that overlaps with its neighbor.
        interpolation (int): OpenCV interpolation flag for resizing.
        force (bool): If True, overwrites existing cached tiles.

    Returns:
        tuple: 
            - im_list (list of str): Paths to the generated tile images.
            - im_warp (list of torch.Tensor): 3x3 transformation matrices (Affine) 
              mapping original image coordinates to the tile's local coordinate system.
    """
    img = cv2.imread(in_image)
    sz = img.shape[:2]
    sz_min = min(sz)
    sz_max = max(sz)

    os.makedirs(cache_path, exist_ok=True)

    im_list = []
    im_warp = []
    for split in range(1, split_max +1):
        unshared = 1 - (2 * shared)
        total_len = (unshared * split) + (shared * (split - 1)) + (shared * 2)  
        im_block = sz_min / total_len
        
        i = 0
        m_min = []
        while i + im_block - 1 < sz_min:
            m_min.append([round(i), min(sz_min, round(i + im_block))])
            i += (im_block * (shared + unshared))
    
        max_blocks = np.ceil((sz_max - (im_block * shared)) / (im_block - (im_block * shared)))
        padded_len = max_blocks * (im_block - (im_block * shared)) + (im_block * shared)
        padding = (padded_len - sz_max) / 2
    
        i = 0
        m_max = []
        while i + im_block - 1 < padded_len:
            m_max.append([max(0, round(i - padding)), min(sz_max , round(min(padded_len, i + im_block) - padding))])
            i += (im_block * (shared + unshared))    
    
        block_max = max(max([r-l for l, r in m_max]), max([r-l for l, r in m_min]))
        if block_max > block_sz_max:
            scale = block_sz_max / block_max
        else:
            scale = 1.0
    
        if sz_min == sz[0]:
            row = m_min
            col = m_max
        else:
            row = m_max
            col = m_min
            
        for i, r in enumerate(row):
            for j, c in enumerate(col):
                new_img = img[r[0]:r[1], c[0]:c[1]]
                T = torch.eye(3, device=device, dtype=torch.float)
                T[0, 2] = -c[0]
                T[1, 2] = -r[0]

                S = torch.eye(3, device=device, dtype=torch.float)
                if scale != 1.0:
                    sc = round(scale * (c[1] - c[0]))
                    sr = round(scale * (r[1] - r[0]))
                    new_img = cv2.resize(new_img, (sc, sr), 0, 0, interpolation)
                    
                    S[0, 0] = sc / (c[1] - c[0])
                    S[1, 1] = sr / (r[1] - r[0])

                im_name, im_ext = os.path.splitext(os.path.split(in_image)[-1])                
                im_name = os.path.join(cache_path, im_name + '_' + str(split - 1) + '_' + str(i) + '_' + str(j) + '.png')                 
                
                if not os.path.isfile(im_name) or force:                
                    cv2.imwrite(im_name, new_img)
                
                im_list.append(im_name)
                im_warp.append(S @ T)
                
    return im_list, im_warp


def pair_pyramid(pair, cache_path='tmp_imgs', force=False, split_max=3, block_sz_max=256, shared=1/3, interpolation=cv2.INTER_AREA, **dummy_args):
    """
    A generator that yields all possible combinations of tiles from a pair of images.

    This is used for "coarse-to-fine" matching. By tiling both images in a pair, 
    it allows a matcher to compare every sub-region of Image A with every 
    sub-region of Image B at multiple scales.

    Args:
        pair (tuple): A tuple of two image paths (path1, path2).
        ... (Other args same as to_pyramid)

    Yields:
        tuple: 
            - (im1, im2) (tuple): Paths to a pair of tiles (one from each image).
            - [warp1_inv, warp2_inv] (list): Inverse transformation matrices 
              to map coordinates found in the tiles back to the original image space.
            - {} (dict): Empty dictionary for pipeline compatibility.
    """
    im_list1, im_warp1 = to_pyramid(pair[0], cache_path=cache_path, split_max=split_max, block_sz_max=block_sz_max, shared=shared, interpolation=interpolation, force=force)
    im_list2, im_warp2 = to_pyramid(pair[1], cache_path=cache_path, split_max=split_max, block_sz_max=block_sz_max, shared=shared, interpolation=interpolation, force=force)

    for im1, warp1 in zip(im_list1, im_warp1):
        for im2, warp2 in zip(im_list2, im_warp2):                        
            yield (im1, im2), [warp1.inverse(), warp2.inverse()], {}


