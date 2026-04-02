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

from core import device, pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args
from image_pairs import image_pairs

class dog_module:
    """
    A feature detection module using the Difference of Gaussians (DoG) method.

    This module identifies keypoints by subtracting two blurred versions of 
    the same image. This highlights 'blobs' and corners at various scales. 
    It is the standard detector used in the SIFT framework and is highly 
    stable for 3D reconstruction and image alignment.

    Attributes:
        nfeatures (int): The number of top-ranked keypoints to retain 
            (default: 8000).
        upright (bool): If True, forces all keypoints to have an orientation 
            of 0 degrees, disabling rotation invariance.
        contrastThreshold: Filters out weak features in low-contrast regions.
    """
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False                
        self.pass_through = False
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'upright': False,
            'params': {'nfeatures': 8000, 'contrastThreshold': -10000, 'edgeThreshold': 10000},
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('dog', args, self.args)
        self.detector = cv2.SIFT_create(**self.args['params'])


    def get_id(self): 
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):    
        
        im = cv2.imread(args['img'][args['idx']], cv2.IMREAD_GRAYSCALE)
        kp = self.detector.detect(im, None)


        if self.args['upright']:
            idx = np.unique(np.asarray([[k.pt[0], k.pt[1]] for k in kp]), axis=0, return_index=True)[1]
            kp = [kp[ii] for ii in idx]
            for ii in range(len(kp)):
                kp[ii].angle = 0       

        kr = []
        for i in range(len(kp)): kr.append(kp[i].response)
        kr = torch.tensor(kr, device=device, dtype=torch.float)
                
        kp = laf_from_opencv_kpts(kp, device=device)
        kp, kH = laf2homo(kp.detach().to(device).squeeze(0))
    
        return {'kp': kp, 'kH': kH, 'kr': kr}