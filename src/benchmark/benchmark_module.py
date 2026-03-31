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
from .metrics import relative_pose_error_angular, relative_pose_error_metric, estimate_pose, error_auc, invalid_matches, homography_error_heat_map, epipolar_error_heat_map, register_by_Horn, evaluate_rec
from visualization import show_kpts_module, visualize_LAF, show_matches_module, show_homography_module, show_patches_module, colorize_plane




class pairwise_benchmark_module:
    """
    A comprehensive evaluation module for benchmarking pairwise 3D reconstruction.

    This module computes various error metrics (AUC, Accuracy, Precision) by 
    comparing estimated matrices and keypoints against ground truth data. 
    It supports multiple geometric modes and logs results into a persistent 
    HDF5 cache for large-dataset analysis.

    Attributes:
        args (dict): Configuration including:
            - mode (str): 'fundamental', 'essential', 'epipolar', or 'homography'.
            - metric (bool): If True, computes errors in physical units (meters) 
              rather than angular degrees.
            - err_th_list (list): Thresholds (pixels) for inlier counting.
            - angular_thresholds (list): Thresholds (degrees) for AUC/Acc calculation.
            - aux_hdf5 (str): Path to the storage file for stats.
        gt (dict): The ground truth data structure containing K, R, T, and H.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True
        self.add_to_cache = True
        
        self.args = { 
            'id_more': '',
            'gt': None,
            'to_add_path': '',
            'aux_hdf5': 'stats.hdf5',
            'err_th_list': list(range(1,16)),
            'essential_th': 0.5,
            'mode': 'fundamental',
            'metric': False,
            'angular_thresholds': [5, 10, 20],
            'metric_thresholds': [0.5, 1, 2],
            'planar_thresholds': [5, 10, 15],
            'homography_mask_rad': 15,
            'am_scaling' : 10, # current metric error requires that angular_thresholds[i] / metric_thresholds[i] = am_scaling
            'save_to': None,
            # to save homography heat map
            'img_prefix': '',
            'img_suffix': '',
            'cache_path': 'show_imgs',
            'prepend_pair': False,
            'ext': '.png',            
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                                        
        self.id_string, self.args = set_args('pairwise_benchmark', args, self.args)    
        
        if self.args['gt'] is None:
            warnings.warn("no gt data given!")

        self.args['to_add_path_size'] = len(self.args['to_add_path'])    

        if self.args['err_th_list'] is None:
            self.args['err_th_list'] = list(range(1,16))            
                        
        self.aux_hdf5 = pickled_hdf5.pickled_hdf5(self.args['aux_hdf5'], mode='a', label_prefix='pickled/' + self.id_string)
                

    def finalize(self, **args):
        """
        Aggregates stats from all processed pairs and prints/saves final metrics:
        - AUC (Area Under the Curve): Measures overall robustness.
        - Acc@X: Percentage of pairs with error below X degrees/meters.
        - Precision: Ratio of valid matches (inliers) to total matches.
        """
        if self.args['mode'] == 'homography':
            return self.finalize_planar(**args)
        elif self.args['mode'] == 'epipolar':
            return self.finalize_epipolar(**args)
        else:
            return self.finalize_non_planar(**args)


    def finalize_epipolar(self):
        keys = self.aux_hdf5.get_keys()

        fe = 'F*'
            
        if not (self.args['save_to'] is None):
            f = open(self.args['save_to'], 'w')

        F_error_1 = []
        F_error_2 = []
        n = 0
        inliers = torch.zeros(len(self.args['err_th_list']), device=device, dtype=torch.int)
        auc = []
        acc = []

        for key in keys:      
            val, is_found = self.aux_hdf5.get(key)

            F_error_1.append(val['F_error_1'])
            F_error_2.append(val['F_error_2'])
            n = n + val['n']

            inliers = inliers + val['inliers']
                        
            aux = np.asarray([F_error_1, F_error_2]).T            
            max_F_err = np.max(aux, axis=1)        
            tmp = np.concatenate((aux, np.expand_dims(max_F_err, axis=1)), axis=1)

        for a in self.args['angular_thresholds']:       
            auc_F1 = error_auc(F_error_1, a).item()
            auc_F2 = error_auc(F_error_2, a).item()
            auc_max_F = error_auc(max_F_err, a).item()
            acc_ = np.sum(tmp < a, axis=0) / np.shape(tmp)[0]

            auc.append([a, auc_F1, auc_F2, auc_max_F])
            acc.append([a, acc_[0].item(), acc_[1].item(), acc_[2].item()])

        avg_inliers = inliers.type(torch.float).mean().to('cpu').numpy().item()
        avg_precision = (inliers / n).type(torch.float).mean().to('cpu').numpy().item()

        if self.args['save_to'] is None:
            print("            F12      F21  max(F12,F21)")
            for i, a in enumerate(self.args['angular_thresholds']):       
                print(f"AUC@{str(a).ljust(2,' ')} ({fe}) : {auc[i][1]*100: >6.2f}%, {auc[i][2]*100: >6.2f}%, {auc[i][3]*100: >6.2f}%")        
            for i, a in enumerate(self.args['angular_thresholds']):       
                print(f"Acc@{str(a).ljust(2,' ')} ({fe}) : {acc[i][1]*100: >6.2f}%, {acc[i][2]*100: >6.2f}%, {acc[i][3]*100: >6.2f}%")        
            print(f"Prec  ({fe}) : {avg_inliers: .0f} / {n} = {avg_precision*100: >6.2f}%")
        else:
            print("what; angular th; metric th; mode; F12; F21; max(F12,F21); inliers; matches; prec", file=f)
            for i, am in enumerate(zip(self.args['planar_thresholds'], self.args['metric_thresholds'])):       
                a = am[0]
                print(f"AUC; {str(a)}; nan; {fe}; {auc[i][1]}; {auc[i][2]}; {auc[i][3]}; nan; nan; nan", file=f)    
            for i, am in enumerate(zip(self.args['planar_thresholds'], self.args['metric_thresholds'])):       
                a = am[0]
                print(f"Acc; {str(a)}; nan; {fe}; {acc[i][1]}; {acc[i][2]}; {acc[i][3]}; nan; nan; nan", file=f)    
            print(f"Prec; nan; nan; {fe}; nan; nan; nan; {avg_inliers}; {n}; {avg_precision}", file=f)

        self.aux_hdf5.close()

        if not (self.args['save_to'] is None):
            f.close()


    def finalize_planar(self):
        keys = self.aux_hdf5.get_keys()

        fe = 'H'
            
        if not (self.args['save_to'] is None):
            f = open(self.args['save_to'], 'w')

        H_error_1 = []
        H_error_2 = []
        n = 0
        v = 0
        inliers = torch.zeros(len(self.args['err_th_list']), device=device, dtype=torch.int)
        auc = []
        acc = []

        for key in keys:      
            val, is_found = self.aux_hdf5.get(key)

            H_error_1.append(val['H_error_1'])
            H_error_2.append(val['H_error_2'])
            n = n + val['n']
            v = v + val['valid']

            inliers = inliers + val['inliers']
                        
            aux = np.asarray([H_error_1, H_error_2]).T            
            max_H_err = np.max(aux, axis=1)        
            tmp = np.concatenate((aux, np.expand_dims(max_H_err, axis=1)), axis=1)

        for a in self.args['planar_thresholds']:       
            auc_H1 = error_auc(H_error_1, a).item()
            auc_H2 = error_auc(H_error_2, a).item()
            auc_max_H = error_auc(max_H_err, a).item()
            acc_ = np.sum(tmp < a, axis=0) / np.shape(tmp)[0]

            auc.append([a, auc_H1, auc_H2, auc_max_H])
            acc.append([a, acc_[0].item(), acc_[1].item(), acc_[2].item()])

        avg_inliers = inliers.type(torch.float).mean().to('cpu').numpy().item()
        avg_precision = (inliers / n).type(torch.float).mean().to('cpu').numpy().item()
        avg_precision_valid = (inliers / v).type(torch.float).mean().to('cpu').numpy().item()

        if self.args['save_to'] is None:
            print("            H12      H21  max(H12,H21)")
            for i, a in enumerate(self.args['planar_thresholds']):       
                print(f"AUC@{str(a).ljust(2,' ')} ({fe}) : {auc[i][1]*100: >6.2f}%, {auc[i][2]*100: >6.2f}%, {auc[i][3]*100: >6.2f}%")        
            for i, a in enumerate(self.args['planar_thresholds']):       
                print(f"Acc@{str(a).ljust(2,' ')} ({fe}) : {acc[i][1]*100: >6.2f}%, {acc[i][2]*100: >6.2f}%, {acc[i][3]*100: >6.2f}%")        
            print(f"Prec  ({fe}) : {avg_inliers: .0f} / {n} = {avg_precision*100: >6.2f}%")
            print(f"Prec* ({fe}) : {avg_inliers: .0f} / {v} = {avg_precision_valid*100: >6.2f}%")
        else:
            print("what; planar th; metric th; mode; H12; H21; max(H12,H21); inliers; matches; prec", file=f)
            for i, am in enumerate(zip(self.args['planar_thresholds'], self.args['metric_thresholds'])):       
                a = am[0]
                print(f"AUC; {str(a)}; nan; {fe}; {auc[i][1]}; {auc[i][2]}; {auc[i][3]}; nan; nan; nan", file=f)    
            for i, am in enumerate(zip(self.args['planar_thresholds'], self.args['metric_thresholds'])):       
                a = am[0]
                print(f"Acc; {str(a)}; nan; {fe}; {acc[i][1]}; {acc[i][2]}; {acc[i][3]}; nan; nan; nan", file=f)    
            print(f"Prec; nan; nan; {fe}; nan; nan; nan; {avg_inliers}; {n}; {avg_precision}", file=f)
            print(f"Prec*; nan; nan; {fe}; nan; nan; nan; {avg_inliers}; {v}; {avg_precision_valid}", file=f)

        self.aux_hdf5.close()

        if not (self.args['save_to'] is None):
            f.close()


    def finalize_non_planar(self):
        keys = self.aux_hdf5.get_keys()

        if self.args['mode'] == 'fundamental':
            fe = 'F'
        else:
            fe = 'E'
            
        if not (self.args['save_to'] is None):
            f = open(self.args['save_to'], 'w')

        R_error = []
        t_error = []
        n = 0
        inliers = torch.zeros(len(self.args['err_th_list']), device=device, dtype=torch.int)
        auc = []
        acc = []

        for key in keys:      
            val, is_found = self.aux_hdf5.get(key)

            R_error.append(val['R_error'])
            t_error.append(val['t_error'])
            n = n + val['n']
            inliers = inliers + val['inliers']
                        
            aux = np.asarray([R_error, t_error]).T
            if self.args['metric']:
                aux[:, 1] = aux[:, 1] * self.args['am_scaling']
            
            max_Rt_err = np.max(aux, axis=1)        
            tmp = np.concatenate((aux, np.expand_dims(max_Rt_err, axis=1)), axis=1)

        if not self.args['metric']:    
            for a in self.args['angular_thresholds']:       
                auc_R = error_auc(R_error, a).item()
                auc_t = error_auc(t_error, a).item()
                auc_max_Rt = error_auc(max_Rt_err, a).item()
                acc_ = np.sum(tmp < a, axis=0) / np.shape(tmp)[0]
    
                auc.append([a, auc_R, auc_t, auc_max_Rt])
                acc.append([a, acc_[0].item(), acc_[1].item(), acc_[2].item()])

            avg_inliers = inliers.type(torch.float).mean().to('cpu').numpy().item()
            avg_precision = (inliers / n).type(torch.float).mean().to('cpu').numpy().item()

            if self.args['save_to'] is None:
                print("             R        t        max(R,t)")
                for i, a in enumerate(self.args['angular_thresholds']):       
                    print(f"AUC@{str(a).ljust(2,' ')} ({fe}) : {auc[i][1]*100: >6.2f}%, {auc[i][2]*100: >6.2f}%, {auc[i][3]*100: >6.2f}%")        
                for i, a in enumerate(self.args['angular_thresholds']):       
                    print(f"Acc@{str(a).ljust(2,' ')} ({fe}) : {acc[i][1]*100: >6.2f}%, {acc[i][2]*100: >6.2f}%, {acc[i][3]*100: >6.2f}%")        
                print(f"Prec ({fe}) : {avg_inliers: .0f} / {n} = {avg_precision*100: >6.2f}%")
            else:
                print("what; angular th; metric th; mode; R; t; max(R,t); inliers; matches; prec", file=f)
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"AUC; {str(a)}; nan; {fe}; {auc[i][1]}; {auc[i][2]}; {auc[i][3]}; nan; nan; nan", file=f)    
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"Acc; {str(a)}; nan; {fe}; {acc[i][1]}; {acc[i][2]}; {acc[i][3]}; nan; nan; nan", file=f)    
                print(f"Prec; nan; nan; {fe}; nan; nan; nan; {avg_inliers}; {n}; {avg_precision}", file=f)
        else:
            for a, m in zip(self.args['angular_thresholds'], self.args['metric_thresholds']):       
                auc_R = error_auc(R_error, a).item()
                auc_t = error_auc(t_error, m).item()
                auc_max_Rt = error_auc(max_Rt_err, a).item()
                                
                aa = (aux[:, 0] < a)[:, np.newaxis]
                mm = (aux[:, 1] < m)[:, np.newaxis]
                tmp = np.concatenate((aa, mm, aa & mm), axis=1)
                acc_ = np.sum(tmp, axis=0) / np.shape(tmp)[0]

                auc.append([a, auc_R, auc_t, auc_max_Rt])
                acc.append([a, acc_[0].item(), acc_[1].item(), acc_[2].item()])

            avg_inliers = inliers.type(torch.float).mean().to('cpu').numpy().item()
            avg_precision = (inliers / n).type(torch.float).mean().to('cpu').numpy().item()

            if self.args['save_to'] is None:    
                print("                 R        t        max(R,t)")
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"@AUC{str(a).ljust(2,' ')},{str(m).ljust(3,' ')} ({fe}) : {auc[i][1]*100: >6.2f}%, {auc[i][2]*100: >6.2f}%, {auc[i][3]*100: >6.2f}%")    
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"@Acc{str(a).ljust(2,' ')},{str(m).ljust(3,' ')} : {acc[i][1]*100: >6.2f}%, {acc[i][2]*100: >6.2f}%, {acc[i][3]*100: >6.2f}%")    
                print(f"Prec ({fe}) : {avg_inliers: .0f} / {n} = {avg_precision*100: >6.2f}%")
            else:
                print("what; angular th; metric th; mode; R; t; max(R,t); inliers; matches; prec", file=f)
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"AUC; {str(a)}; {str(m)}; {fe}; {auc[i][1]}; {auc[i][2]}; {auc[i][3]}; nan; nan; nan", file=f)    
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"Acc; {str(a)}; {str(m)}; {fe}; {acc[i][1]}; {acc[i][2]}; {acc[i][3]}; nan; nan; nan", file=f)    
                print(f"Prec; nan; nan; {fe}; nan; nan; nan; {avg_inliers}; {n}; {avg_precision}", file=f)

        self.aux_hdf5.close()

        if not (self.args['save_to'] is None):
            f.close()


    def get_id(self): 
        return self.id_string


    def run(self, **args):
        """
        Executes the benchmark for a single image pair.
        Calculates:
        - Reprojection errors for keypoints.
        - Matrix errors (Angular or Metric) for R and t.
        - Heat maps for spatial error visualization (via 'epipolar' or 'homography' modes).
        """
        if self.args['mode'] == 'fundamental':
            return self.run_fundamental(**args)
        elif self.args['mode'] == 'essential':
            return self.run_essential(**args)
        elif self.args['mode'] == 'epipolar':
            return self.run_epipolar(**args)
        else:
            return self.run_homography(**args)


    def run_epipolar(self, **args):
        err_th_list = self.args['err_th_list']
        
        img1 = args['img'][0]
        img2 = args['img'][1]
                
        data_key = '/' + os.path.split(img1)[-1] + '/' + os.path.split(img2)[-1] + '/epipolar'
        
        out_data, is_found = self.aux_hdf5.get(data_key)
        if is_found:
            return {}

        cannot_do = False

        img1_key = img1[self.args['to_add_path_size'] + 1:]
        img2_key = img2[self.args['to_add_path_size'] + 1:]
        
        if cannot_do or (not (img1_key in self.args['gt'])):
            cannot_do = True
                        
        if cannot_do or (not (img2_key in self.args['gt'][img1_key])):
            cannot_do = None
        
        use_scale = self.args['gt']['use_scale']
        
        if not cannot_do:
            gt = self.args['gt'][img1_key][img2_key]
        else:
            gt = None

        if not (gt is None):
            K1 = gt['K1']
            K2 = gt['K2']    
            R_gt = gt['R']
            t_gt = gt['T']
                
            mm = args['m_idx'][args['m_mask']]
        
            pts1 = args['kp'][0][mm[:, 0]]
            pts2 = args['kp'][1][mm[:, 1]]
                       
            if torch.is_tensor(pts1):
                pts1 = pts1.detach().cpu().numpy()
                pts2 = pts2.detach().cpu().numpy()
        
            scales = gt['image_pair_scale'] if use_scale else np.asarray([[1.0, 1.0], [1.0, 1.0]])    
        
            pts1 = pts1 * scales[0]
            pts2 = pts2 * scales[1]
        
            nn = pts1.shape[0]

            inl_sum = torch.zeros(len(err_th_list), device=device, dtype=torch.int)
        
            if nn < 8:
                F = None
            else:
                if 'F' in args:
                    s1 = torch.eye(3, device=device)
                    s2 = torch.eye(3, device=device)

                    s1[0, 0] = 1 / scales[0, 0]
                    s1[1, 1] = 1 / scales[0, 1]

                    s2[0, 0] = 1 / scales[1, 0]
                    s2[1, 1] = 1 / scales[1, 1]

                    F = s2 @ args['F'].type(torch.float) @ s1
                    F = F / F[2, 2]
                else:
                    F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
                    if not (F is None): F = torch.tensor(F, device=device)
        
            if nn > 0:
                F_gt = torch.tensor(K2.T, device=device, dtype=torch.float64).inverse() @ \
                       torch.tensor([[0, -t_gt[2], t_gt[1]],
                                    [t_gt[2], 0, -t_gt[0]],
                                    [-t_gt[1], t_gt[0], 0]], device=device) @ \
                       torch.tensor(R_gt, device=device) @ \
                       torch.tensor(K1, device=device, dtype=torch.float64).inverse()
                F_gt = F_gt / F_gt.sum()
        
                pt1_ = torch.vstack((torch.tensor(pts1.T, device=device), torch.ones((1, nn), device=device)))
                pt2_ = torch.vstack((torch.tensor(pts2.T, device=device), torch.ones((1, nn), device=device)))
        
                l1_ = F_gt @ pt1_
                d1 = pt2_.permute(1,0).unsqueeze(-2).bmm(l1_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l1_[:2]**2).sum(0).sqrt()
        
                l2_ = F_gt.T @ pt2_
                d2 = pt1_.permute(1,0).unsqueeze(-2).bmm(l2_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l2_[:2]**2).sum(0).sqrt()
        
                epi_max_err = torch.maximum(d1, d2)
                inl_sum = (epi_max_err.unsqueeze(-1) < torch.tensor(err_th_list, device=device).unsqueeze(0)).sum(dim=0).type(torch.int)        
            
            if F is None:
                F_error_1 = np.inf
                F_error_2 = np.inf
            else:
                sz1 = np.asarray(Image.open(img1).size)[-1::-1]
                sz2 = np.asarray(Image.open(img2).size)[-1::-1]
                
                heat1 = epipolar_error_heat_map(F_gt, F, sz1)
                heat2 = epipolar_error_heat_map(F_gt.T, F.T, sz2)

                F_error_1 = heat1.mean().detach().cpu().numpy() 
                F_error_2 = heat2.mean().detach().cpu().numpy()                  
        
                if not (self.args['cache_path'] is None):
                    im1 = os.path.splitext(os.path.split(img1)[1])[0]
                    im2 = os.path.splitext(os.path.split(img2)[1])[0]                
                    
                    if self.args['prepend_pair']:            
                        cache_path = os.path.join(self.args['cache_path'], im1 + '_' + im2)
                    else:
                        cache_path = self.args['cache_path']
                            
                    heat_img1 = os.path.join(cache_path, self.args['img_prefix'] + im1 + self.args['img_suffix'] + self.args['ext'])
                    heat_img2 = os.path.join(cache_path, self.args['img_prefix'] + im2 + self.args['img_suffix'] + self.args['ext'])
    
                    os.makedirs(cache_path, exist_ok=True)
    
                    colorize_plane(img1, heat1, cmap_name='viridis', max_val=45, cf=0.7, save_to=heat_img1)            
                    colorize_plane(img2, heat2, cmap_name='viridis', max_val=45, cf=0.7, save_to=heat_img2)                
                
            out_data = {'F_error_1': F_error_1, 'F_error_2': F_error_2, 'n': nn, 'inliers': inl_sum}
        else:
            warnings.warn("image pair not in gt data!")            
            out_data = None

        self.aux_hdf5.add(data_key, out_data)
        return {}
            
    
    def run_fundamental(self, **args):
        err_th_list = self.args['err_th_list']
        
        img1 = args['img'][0]
        img2 = args['img'][1]
        
        if self.args['metric']:
            key_metric = '_metric'
        else:
            key_metric = ''
        
        data_key = '/' + os.path.split(img1)[-1] + '/' + os.path.split(img2)[-1] + '/fundamental' + key_metric
        
        out_data, is_found = self.aux_hdf5.get(data_key)
        if is_found:
            return {}

        cannot_do = False

        img1_key = img1[self.args['to_add_path_size'] + 1:]
        img2_key = img2[self.args['to_add_path_size'] + 1:]
        
        if cannot_do or (not (img1_key in self.args['gt'])):
            cannot_do = True
                        
        if cannot_do or (not (img2_key in self.args['gt'][img1_key])):
            cannot_do = None

        use_scale = self.args['gt']['use_scale']
        
        if not cannot_do:
            gt = self.args['gt'][img1_key][img2_key]
        else:
            gt = None

        if not (gt is None):
            K1 = gt['K1']
            K2 = gt['K2']    
            R_gt = gt['R']
            t_gt = gt['T']
            
            if self.args['metric']:
                scene_scale = gt['scene_scale']
    
            mm = args['m_idx'][args['m_mask']]
        
            pts1 = args['kp'][0][mm[:, 0]]
            pts2 = args['kp'][1][mm[:, 1]]
                       
            if torch.is_tensor(pts1):
                pts1 = pts1.detach().cpu().numpy()
                pts2 = pts2.detach().cpu().numpy()
        
            scales = gt['image_pair_scale'] if use_scale else np.asarray([[1.0, 1.0], [1.0, 1.0]])  
        
            pts1 = pts1 * scales[0]
            pts2 = pts2 * scales[1]
        
            nn = pts1.shape[0]

            inl_sum = torch.zeros(len(err_th_list), device=device, dtype=torch.int)
        
            if nn < 8:
                Rt_ = None
            else:
                if 'F' in args:
                    s1 = torch.eye(3, device=device)
                    s2 = torch.eye(3, device=device)

                    s1[0, 0] = 1 / scales[0, 0]
                    s1[1, 1] = 1 / scales[0, 1]

                    s2[0, 0] = 1 / scales[1, 0]
                    s2[1, 1] = 1 / scales[1, 1]

                    F = s2 @ args['F'].type(torch.float) @ s1
                    F = F / F[2, 2]
                    F = F.to('cpu').numpy()
                else:
                    F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]

                if F is None:
                    Rt_ = None
                else:
                    E = K2.T @ F @ K1
                    Rt_ = cv2.decomposeEssentialMat(E)
        
            if nn > 0:
                F_gt = torch.tensor(K2.T, device=device, dtype=torch.float64).inverse() @ \
                       torch.tensor([[0, -t_gt[2], t_gt[1]],
                                    [t_gt[2], 0, -t_gt[0]],
                                    [-t_gt[1], t_gt[0], 0]], device=device) @ \
                       torch.tensor(R_gt, device=device) @ \
                       torch.tensor(K1, device=device, dtype=torch.float64).inverse()
                F_gt = F_gt / F_gt.sum()
        
                pt1_ = torch.vstack((torch.tensor(pts1.T, device=device), torch.ones((1, nn), device=device)))
                pt2_ = torch.vstack((torch.tensor(pts2.T, device=device), torch.ones((1, nn), device=device)))
        
                l1_ = F_gt @ pt1_
                d1 = pt2_.permute(1,0).unsqueeze(-2).bmm(l1_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l1_[:2]**2).sum(0).sqrt()
        
                l2_ = F_gt.T @ pt2_
                d2 = pt1_.permute(1,0).unsqueeze(-2).bmm(l2_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l2_[:2]**2).sum(0).sqrt()
        
                epi_max_err = torch.maximum(d1, d2)
                inl_sum = (epi_max_err.unsqueeze(-1) < torch.tensor(err_th_list, device=device).unsqueeze(0)).sum(dim=0).type(torch.int)        
            
            if Rt_ is None:
                R_error = np.inf
                t_error = np.inf
            else:
                R_a, t_a, = Rt_[0], Rt_[2].squeeze()
                R_b, t_b, = Rt_[1], Rt_[2].squeeze()

                if not self.args['metric']:
                    t_err_a, R_err_a = relative_pose_error_angular(R_gt, t_gt, R_a, t_a)
                    t_err_b, R_err_b = relative_pose_error_angular(R_gt, t_gt, R_b, t_b)
            
                    if max(R_err_a, t_err_a) < max(R_err_b, t_err_b):
                        R_err, t_err = R_err_a, t_err_b
                    else:
                        R_err, t_err = R_err_b, t_err_b
                else:
                    t_err, R_err = relative_pose_error_metric(R_gt, t_gt, [Rt_[0], Rt_[1]], Rt_[2].squeeze(), scale_cf=scene_scale)
        
                R_error = R_err
                t_error = t_err
                
            out_data = {'R_error': R_error.item(), 't_error': t_error.item(), 'n': nn, 'inliers': inl_sum}
        else:
            warnings.warn("image pair not in gt data!")            
            out_data = None

        self.aux_hdf5.add(data_key, out_data)
        return {}


    def run_essential(self, **args):        
        img1 = args['img'][0]
        img2 = args['img'][1]
        
        if self.args['metric']:
            key_metric = '_metric'
        else:
            key_metric = ''
                
        data_key = '/' + os.path.split(img1)[-1] + '/' + os.path.split(img2)[-1] + '/essential' + key_metric
        
        out_data, is_found = self.aux_hdf5.get(data_key)
        if is_found:
            return {}

        cannot_do = False

        img1_key = img1[self.args['to_add_path_size'] + 1:]
        img2_key = img2[self.args['to_add_path_size'] + 1:]
        
        if cannot_do or (not (img1_key in self.args['gt'])):
            cannot_do = True
                        
        if cannot_do or (not (img2_key in self.args['gt'][img1_key])):
            cannot_do = None
        
        if not cannot_do:
            gt = self.args['gt'][img1_key][img2_key]
        else:
            gt = None
            
        use_scale = self.args['gt']['use_scale']
            
        if not (gt is None):
            K1 = gt['K1']
            K2 = gt['K2']    
            R_gt = gt['R']
            t_gt = gt['T']

            if self.args['metric']:
                scene_scale = gt['scene_scale']
    
            mm = args['m_idx'][args['m_mask']]
        
            pts1 = args['kp'][0][mm[:, 0]]
            pts2 = args['kp'][1][mm[:, 1]]
                       
            if torch.is_tensor(pts1):
                pts1 = pts1.detach().cpu().numpy()
                pts2 = pts2.detach().cpu().numpy()
        
            scales = gt['image_pair_scale'] if use_scale else np.asarray([[1.0, 1.0], [1.0, 1.0]])  
        
            pts1 = pts1 * scales[0]
            pts2 = pts2 * scales[1]
        
            nn = pts1.shape[0]

            inl_sum = 0
        
            if nn < 5:
                Rt = None
            else:
                Rt = estimate_pose(pts1, pts2, K1, K2, self.args['essential_th'])                                                        

            if Rt is None:
                R_error = np.inf
                t_error = np.inf                          
            else:
                R, t, inliers = Rt

                if not self.args['metric']:
                    t_err, R_err = relative_pose_error_angular(R_gt, t_gt, R, t)
                else:
                    t_err, R_err = relative_pose_error_metric(R_gt, t_gt, R, t, scale_cf=scene_scale)
        
                R_error = R_err
                t_error = t_err
                inl_sum = inliers.sum()
                
            out_data = {'R_error': R_error.item(), 't_error': t_error.item(), 'n': nn, 'inliers': inl_sum}
        else:
            warnings.warn("image pair not in gt data!")            
            out_data = None

        self.aux_hdf5.add(data_key, out_data)
        return {}


    def run_homography(self, **args):
        img1 = args['img'][0]
        img2 = args['img'][1]
                        
        data_key = '/' + os.path.split(img1)[-1] + '/' + os.path.split(img2)[-1] + '/homography'
        
        out_data, is_found = self.aux_hdf5.get(data_key)
        if is_found: return {}

        rad = self.args['homography_mask_rad']
        
        cannot_do = False
        
        img1_key = img1[self.args['to_add_path_size'] + 1:]
        img2_key = img2[self.args['to_add_path_size'] + 1:]
        
        if cannot_do or (not (img1_key in self.args['gt'])):
            cannot_do = True
                        
        if cannot_do or (not (img2_key in self.args['gt'][img1_key])):
            cannot_do = None
        
        if not cannot_do:
            gt = self.args['gt'][img1_key][img2_key]
        else:
            gt = None

        use_scale = self.args['gt']['use_scale']
        
        if not (gt is None):
            H_gt = torch.tensor(gt['H'], device=device)
              
            mm = args['m_idx'][args['m_mask']]
           
            pts1 = args['kp'][0][mm[:, 0]]
            pts2 = args['kp'][1][mm[:, 1]]
                       
            if torch.is_tensor(pts1):
                pts1 = pts1.detach().cpu().numpy()
                pts2 = pts2.detach().cpu().numpy()        
            
            scales = gt['image_pair_scale'] if use_scale else np.asarray([[1.0, 1.0], [1.0, 1.0]])  
        
            pts1 = pts1 * scales[0]
            pts2 = pts2 * scales[1]
            
            nn = pts1.shape[0]
                                                    
            if (nn < 4):
                H = None
            else:
                if not ('H' in args):                
                    H = torch.tensor(cv2.findHomography(pts1, pts2, 0)[0], device=device)
                else:
                    H = args['H']
        
            if nn > 0:
                H_gt_inv = H_gt.inverse()
                
                pts1 = torch.tensor(pts1, device=device)
                pts2 = torch.tensor(pts2, device=device)
                 
                pts1_reproj = apply_homo(pts1, H_gt.to(torch.float))
                d1 = ((pts2 - pts1_reproj)**2).sum(1).sqrt()
                 
                pts2_reproj = apply_homo(pts2, H_gt_inv.to(torch.float))
                d2 = ((pts1 - pts2_reproj)**2).sum(1).sqrt()
                 
                valid_matches = torch.ones(nn, device=device, dtype=torch.bool)                            
                valid_matches = valid_matches & ~invalid_matches(gt['mask1'], gt['full_mask2'], pts1, pts2, rad)          
                valid_matches = valid_matches & ~invalid_matches(gt['mask2'], gt['full_mask1'], pts2, pts1, rad)
                                                     
                reproj_max_err_ = torch.maximum(d1, d2)                                
                reproj_max_err = reproj_max_err_[valid_matches]
                inl_sum = (reproj_max_err.unsqueeze(-1) < torch.tensor(self.args['err_th_list'], device=device).unsqueeze(0)).sum(dim=0).type(torch.int)
                valid_sum = valid_matches.sum()
            else:                                                    
                H = None
                inl_sum = torch.zeros(len(self.args['err_th_list']), device=device, dtype=torch.int)
                valid_sum = 0

            if not (H is None):
                heat1 = homography_error_heat_map(H_gt, H, torch.tensor(gt['full_mask1'], device=device))
                heat2 = homography_error_heat_map(H_gt_inv, H.inverse(), torch.tensor(gt['full_mask2'], device=device))

                H_error_1 = heat1[heat1 != -1].mean().detach().cpu().numpy() 
                H_error_2 = heat2[heat2 != -1].mean().detach().cpu().numpy()                  
        
                if not (self.args['cache_path'] is None):
                    im1 = os.path.splitext(os.path.split(img1)[1])[0]
                    im2 = os.path.splitext(os.path.split(img2)[1])[0]                
                    
                    if self.args['prepend_pair']:            
                        cache_path = os.path.join(self.args['cache_path'], im1 + '_' + im2)
                    else:
                        cache_path = self.args['cache_path']
                            
                    heat_img1 = os.path.join(cache_path, self.args['img_prefix'] + im1 + self.args['img_suffix'] + self.args['ext'])
                    heat_img2 = os.path.join(cache_path, self.args['img_prefix'] + im2 + self.args['img_suffix'] + self.args['ext'])
    
                    os.makedirs(cache_path, exist_ok=True)
    
                    colorize_plane(img1, heat1, cmap_name='viridis', max_val=45, cf=0.7, save_to=heat_img1)            
                    colorize_plane(img2, heat2, cmap_name='viridis', max_val=45, cf=0.7, save_to=heat_img2)                    
            else:
                H_error_1 = np.inf
                H_error_2 = np.inf
   
            out_data = {'H_error_1': H_error_1, 'H_error_2': H_error_2, 'n': nn, 'inliers': inl_sum, 'valid': valid_sum}
        else:
            warnings.warn("image pair not in gt data!")            
            out_data = None

        self.aux_hdf5.add(data_key, out_data)
        return {}
