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

class keynet_module:
    """
    A deep-learning feature detection module using the KeyNet architecture.

    KeyNet is designed to combine the strengths of traditional geometric 
    detectors (like Hessian or Harris) with the power of CNNs. It uses 
    learned filters to find stable, repeatable points that are 
    optimized for matching across different viewpoints.

    Attributes:
        num_features (int): The number of top-scoring keypoints to extract 
            (default: 8000).
        pretrained (bool): Uses weights trained on the HPatches dataset 
            for state-of-the-art repeatability.
    """
    def __init__(self, **args):
        self.single_image = True        
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
                
        self.args = {
            'id_more': '',
            'params': {
                'pretrained': True,
                'num_features': 8000,
                },
        }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('keynet', args, self.args)
        self.detector = K.feature.KeyNetDetector(**self.args['params']).to(device)


    def get_id(self):
        return self.id_string
        

    def finalize(self):
        return

    
    def run(self, **args):
        img = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)
        kp, kr = self.detector(img)
        kp, kH = laf2homo(kp.detach().to(device).squeeze(0))

        return {'kp': kp, 'kH': kH, 'kr': kr.detach().to(device).squeeze(0)}
