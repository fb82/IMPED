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

from core import device
from .colmap_ext import coldb_ext, SIMPLE_RADIAL
from ensemble import pipe_union

class to_colmap_module:
    """
    A data-export module that saves pipeline results into a COLMAP database.

    This module handles the heavy lifting of:
    1. Registering new cameras and images in the database.
    2. Converting PyTorch keypoints to COLMAP's binary blob format.
    3. Merging new matches with existing ones using various 'sampling_modes'.
    4. Injecting verified geometric models (Homography, Essential, Fundamental).

    Attributes:
        db (str): Path to the target COLMAP '.db' file.
        focal_cf (float): A multiplier for focal length if camera intrinsics 
            are unknown (defaults to 1.2 * max(width, height)).
        sampling_mode (str): Determines how to handle overlapping keypoints 
            (e.g., 'raw', 'avg_all_matches').
        include_two_view_geometry (bool): If True, saves the 'Verified' 
            matches into the two_view_geometry table.
    """
    def __init__(self, **args):
        from core import set_args
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True
        self.add_to_cache = True
        
        self.args = {
            'id_more': '',
            'db': 'colmap.db',
            'aux_hdf5': 'colmap_aux.hdf5',
            'focal_cf': 1.2,
            'only_keypoints': False,            
            'unique': True,
            'only_matched': False,
            'no_unmatched': True,
            'include_two_view_geometry': True,
            'sampling_mode': 'raw',
            'overlapping_cells' : False,
            'sampling_scale': 1,
            'sampling_offset': 0,
        }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('to_colmap' , args, self.args)

        self.db = coldb_ext(self.args['db'])
        self.db.create_tables()
        self.aux_hdf5 = None
        if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):         
            self.aux_hdf5 = pickled_hdf5.pickled_hdf5(self.args['aux_hdf5'], mode='a', label_prefix='pickled/' + self.id_string)
                

    def finalize(self):
        self.db.commit()
        self.db.close()
        if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):
            self.aux_hdf5.close()
            if os.path.isfile(self.args['aux_hdf5']):
                os.remove(self.args['aux_hdf5'])

                
    def get_id(self): 
        return self.id_string

    
    def run(self, **args):   
        im_ids = []
        imgs = []
        
        for idx in [0, 1]:
            im = args['img'][idx]            
            _, img = os.path.split(im)
            
            im_id = self.db.get_image_id(img)
            if  im_id is None:
                w, h = Image.open(im).size
                cam_id = self.db.add_camera(SIMPLE_RADIAL, w, h, np.array([self.args['focal_cf'] * max(w, h), w / 2, h / 2, 0]))
                im_id = self.db.add_image(img, cam_id)
                self.db.commit()

            imgs.append(img)
            im_ids.append(im_id)
                
        pipe_old = {}
        
        kp_old0 = self.db.get_keypoints(im_ids[0])
        if kp_old0 is None:
            w_old0 = torch.zeros((0, 6), device=device)
            kp_old0 = torch.zeros((0, 2), device=device)
            if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):            
                k_count_old0 = torch.zeros(0, device=device)
        else:
            w_old0 = torch.tensor(kp_old0, device=device)
            kp_old0 = torch.tensor(kp_old0[:, :2], device=device)
            if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):            
                k_count_old0, _ = self.aux_hdf5.get(imgs[0])
            
        kH_old0 = torch.zeros((kp_old0.shape[0], 3, 3), device=device)
        kr_old0 = torch.full((kp_old0.shape[0], ), torch.inf, device=device)
        
        kp_old1 = self.db.get_keypoints(im_ids[1])
        if kp_old1 is None:
            w_old1 = torch.zeros((0, 6), device=device)
            kp_old1 = torch.zeros((0, 2), device=device)
            if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):            
                k_count_old1 = torch.zeros(0, device=device)
        else:
            w_old1 = torch.tensor(kp_old1, device=device)
            kp_old1 = torch.tensor(kp_old1[:, :2], device=device)
            if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):            
                k_count_old1, _ = self.aux_hdf5.get(imgs[1])
            
        kH_old1 = torch.zeros((kp_old1.shape[0], 3, 3), device=device)
        kr_old1 = torch.full((kp_old1.shape[0], ), torch.inf, device=device)

        m_idx_old = torch.zeros((0, 2), device=device, dtype=torch.int)        
        m_val_old = torch.full((m_idx_old.shape[0], ), torch.inf, device=device)
        m_mask_old = torch.full((m_idx_old.shape[0], ), 1, device=device, dtype=torch.bool)
            
        pipe_old['kp'] = [kp_old0, kp_old1]
        pipe_old['kH'] = [kH_old0, kH_old1]
        pipe_old['kr'] = [kr_old0, kr_old1]
        pipe_old['w'] = [w_old0, w_old1]
        
        if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):        
            pipe_old['k_counter'] = [k_count_old0, k_count_old1]

        pipe_old['m_idx'] = m_idx_old
        pipe_old['m_val'] = m_val_old
        pipe_old['m_mask'] = m_mask_old
                
        w0 = kpts_as_colmap(0, **args)
        w1 = kpts_as_colmap(1, **args)
        args['w'] = [w0, w1]

        if (self.args['sampling_mode'] == 'avg_all_matches'):
            k_count0 = torch.full((args['kp'][0].shape[0], ), 1, device=device)
            k_count1 = torch.full((args['kp'][1].shape[0], ), 1, device=device)
            args['k_counter'] = [k_count0, k_count1]
        
        if (self.args['sampling_mode'] == 'avg_inlier_matches'):            
            if 'm_idx' in args:            
                k_count0 = torch.full((args['kp'][0].shape[0], ), 0, device=device)
                for i in torch.arange(args['m_mask'].shape[0]):
                    if args['m_mask'][i]:
                        k_count0[args['m_idx'][i, 0]] = 1

                k_count1 = torch.full((args['kp'][1].shape[0], ), 0, device=device)
                for i in torch.arange(args['m_mask'].shape[0]):
                    if args['m_mask'][i]:
                        k_count1[args['m_idx'][i, 1]] = 1
            else:
                k_count0 = torch.full((args['kp'][0].shape[0], ), 1, device=device)
                k_count1 = torch.full((args['kp'][1].shape[0], ), 1, device=device)
            
            args['k_counter'] = [k_count0, k_count1]

        counter = (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches')
        pipe_out = pipe_union([pipe_old, args], unique=self.args['unique'], no_unmatched=self.args['no_unmatched'], only_matched=self.args['only_matched'], sampling_mode=self.args['sampling_mode'], sampling_scale=self.args['sampling_scale'], sampling_offset=self.args['sampling_offset'], overlapping_cells=self.args['overlapping_cells'], preserve_order=True, counter=counter)

        pts0 = pipe_out['w'][0].to('cpu').numpy()
        pts1 = pipe_out['w'][1].to('cpu').numpy()
        
        if counter:
            self.aux_hdf5.add(imgs[0], pipe_out['k_counter'][0])
            self.aux_hdf5.add(imgs[1], pipe_out['k_counter'][1])
        
        self.db.update_keypoints(im_ids[0], pts0)
        self.db.update_keypoints(im_ids[1], pts1)

        if not self.args['only_keypoints']:
            m_idx = pipe_out['m_idx'].to('cpu').numpy()
            self.db.update_matches(im_ids[0], im_ids[1], m_idx)

            if self.args['include_two_view_geometry']:

                m_idx = pipe_out['m_idx'][pipe_out['m_mask']].to('cpu').numpy()
                models = {}
                for m in ['H', 'E', 'F']:
                    if (m in args):
                        if not (args[m] is None):
                            models[m] = args[m].to('cpu').numpy()
                                
                self.db.update_two_view_geometry(im_ids[0], im_ids[1], m_idx, model=models)

        self.db.commit()
        
        return {}



def kpts_as_colmap(idx, **args): 
    """
    Decomposes a local homography (kH) into COLMAP's affine components.

    COLMAP expects keypoints in the format: (x, y, a11, a12, a21, a22). 
    This function extracts the (x, y) coordinates and solves for the 
    2x2 affine matrix by removing the translation component from the 
    full 3x3 local homography matrix.

    Args:
        idx (int): The image index (0 or 1) in the current processing pair.
        **args: The pipeline data dictionary containing 'kp' and 'kH'.

    Returns:
        torch.Tensor: A tensor of shape (N, 6) formatted for COLMAP storage.
    """
    kp = args['kp'][idx]
    kH = args['kH'][idx]
     
    t = torch.zeros((kp.shape[0], 3, 3), device=device)        
    t[:, [0, 1], 2] = -kH[:, [0, 1], 2]
    t[:, 0, 0] = 1
    t[:, 1, 1] = 1
    t[:, 2, 2] = 1           
     
    h = t.bmm(kH.inverse())
     
    v = torch.zeros((kp.shape[0], 3, 3), device=device)        
    v[:, 2, :] = h[:, 2, :]
    v[:, 0, 0] = 1
    v[:, 1, 1] = 1
     
    w = h.bmm(v.inverse())
    w = w[:, :2, :2].reshape(-1, 4)
         
    return torch.cat((kp[:, :2], w), dim=1)
