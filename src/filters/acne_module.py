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



import uuid
from acne.config import get_config as acne_get_config
import acne.acne_custom as acne

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def download_acne(weight_path='../weights/acne'):
    """
    Downloads and extracts the pre-trained neural network weights for ACNe.

    This utility handles the setup of the ACNe inference environment by:
    1. Creating a directory structure to house weights and temporary downloads.
    2. Downloading a compressed archive from a Google Drive share link using 'gdown'.
    3. Extracting the 'logs' folder, which contains the .ckpt (checkpoint) 
       and .meta files necessary for restoring the TensorFlow session.
    4. Ensuring that the multi-hundred megabyte download only occurs if the 
       files are missing locally.

    Args:
        weight_path (str): The base directory for storing model weights. 
            Defaults to '../weights/acne'.

    Returns:
        None: The function performs file system operations and status checks.
    """
    os.makedirs(os.path.join(weight_path, 'download'), exist_ok=True)   

    file_to_download = os.path.join(weight_path, 'download', 'acne_weights.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1yluw3u3F8qH3oTB3dxVw1re4HI6a0TuQ/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)        

    file_to_unzip = file_to_download
    model_dir = os.path.join(weight_path, 'logs')    
    if not os.path.isdir(model_dir):    
        with zipfile.ZipFile(file_to_unzip,"r") as zip_ref:
            zip_ref.extractall(path=weight_path)


class acne_module:
    """
    A deep-learning outlier rejection module using Attentitive Context Networks.

    This module acts as a 'filter' for keypoint matches. It takes 2D-2D 
    correspondences and estimates a Fundamental (F) or Essential (E) matrix 
    while simultaneously predicting a weight for each match. Matches with weights 
    below a threshold are marked as outliers.

    Attributes:
        outdoor (bool): Whether to use weights trained on outdoor (e.g., Yahoo/YFCC) 
            or indoor (e.g., SUN3D) datasets.
        what (str): The specific model variant (ACNe_F for Fundamental matrix 
            estimation is the default).
    """
    current_net = None
    current_obj_id = None
    
    def __init__(self, **args):
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
        self.add_to_cache = True
                        
        self.args = {
            'id_more': '',
            'outdoor': True,
            'what': 'ACNe_F',
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        download_acne()
        
        self.id_string, self.args = set_args('acne', args, self.args)                     
        self.acne_id = uuid.uuid4()

        if self.args['outdoor']:
            # Model of ACNe_F trained with outdoor dataset.              
            model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_outdoor/models-best"
        else:
            # Model of ACNe_F trained with indoor dataset.                      
             model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_indoor/models-best"        

        self.model_path = os.path.join('../weights/acne', model_path)
        self.acne_id = uuid.uuid4()  
        
        self.outdoor = self.args['outdoor']
        self.prev_outdoor = self.args['outdoor']      
        
        
    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return


    def run(self, **args):
        force_reload = False
        if (self.outdoor != self.prev_outdoor):
            force_reload = True
            warnings.warn("acne modules with both indoor and outdoor model detected, computation will be very slow...")
            self.prev_outdoor = self.outdoor
            
            if self.outdoor:
                # Model of ACNe_F trained with outdoor dataset.              
                model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_outdoor/models-best"
            else:
                # Model of ACNe_F trained with indoor dataset.                      
                model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_indoor/models-best"
            self.model_path = os.path.join('../weights/acne', model_path)

        if (acne_module.current_obj_id != self.acne_id) or force_reload:
            if not (acne_module.current_obj_id is None):
                acne_module.current_net.sess.close()
                tf.reset_default_graph()

            config, unparsed = acne_get_config()
        
            paras = {
                "CNe_E":{
                    "bn_opt":"bn"},
                "ACNe_E":{
                    "gcn_opt":"reweight_vanilla_sigmoid_softmax",  "bn_opt":"gn",
                    "weight_opt":"sigmoid_softmax"},
                "CNe_F":{
                    "bn_opt":"bn", "use_fundamental":2},
                "ACNe_F":{
                    "gcn_opt":"reweight_vanilla_sigmoid_softmax",  "bn_opt":"gn",
                    "weight_opt":"sigmoid_softmax", "use_fundamental":2},
            }
        
            para = paras[self.args['what']]
    
            for ki, vi in para.items():
               setattr(config, ki, vi)
               
            self.use_fundamental = config.use_fundamental # E:0, F:2.
        
            # Instantiate wrapper class
            self.net = acne.NetworkTest(config, self.model_path)
            acne_module.current_net = self.net
            acne_module.current_obj_id = self.acne_id
            
        sz1 = Image.open(args['img'][0]).size
        sz2 = Image.open(args['img'][1]).size

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
        
        if l > 0:    
            corrs = np.hstack((pt1, pt2)).astype(np.float32)
        
            K1 = np.array(
                [[1, 0, sz1[0] / 2.0],
                 [0, 1, sz1[1] / 2.0],
                 [0, 0 ,1]])
        
            K2 = np.array(
                [[1, 0, sz2[0] / 2.0],
                 [0, 1, sz2[1] / 2.0],
                 [0, 0, 1]])
        
            # Prepare input. 
            xs, T1, T2 = acne.prepare_xs(corrs, K1, K2, self.use_fundamental)
            xs = np.array(xs).reshape(1, 1, -1, 4) # reconstruct a batch. Bx1xNx4
        
            # Compute Essential/Fundamental matrix
            E, w_com, score_local = self.net.compute_E(xs)
            E = E[0]
            score_local = score_local[0]
            w_com = w_com[0]
        
            mask = w_com > 1e-5
            mask_aux = torch.tensor(mask, device=device)         
            aux = mm.clone()
            mm[aux] = mask_aux

            return {'m_mask': mm}
        else:
            return {'m_mask': args['m_mask']}

