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






def relative_pose_error_angular(R_gt, t_gt, R, t, ignore_gt_t_thr=0.0):
    """
    Computes the angular error for both rotation and translation direction.

    This metric is standard for Monocular Structure-from-Motion (SfM) because 
    it ignores absolute scale and focuses purely on whether the camera 
    moved in the correct direction. 

    Args:
        R_gt, t_gt: Ground truth rotation (3x3) and translation vector (3x1).
        R, t: Predicted rotation and translation.
        ignore_gt_t_thr (float): Threshold for 'pure rotation'. If the ground 
            truth movement is smaller than this, the translation error is 
            ignored (set to 0) to avoid noisy metrics in static scenes.

    Returns:
        tuple: (translation_angular_error, rotation_angular_error) in degrees.
    """
    # angle error between 2 vectors
    # t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    # R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, max_iters=10000):
    """
    Estimates the relative rotation (R) and translation (t) between two cameras.

    This function performs the following steps:
    1. Normalizes 2D keypoints using intrinsic camera matrices (K).
    2. Robustly estimates the Essential Matrix using RANSAC to handle outliers.
    3. Decomposes the Essential Matrix into physically plausible R and t.
    4. Performs 'cheirality check' to ensure points are in front of both cameras.

    Args:
        kpts0, kpts1 (np.array): Matched 2D coordinates in Image 0 and Image 1.
        K0, K1 (3x3 array): Internal camera calibration (focal length, principal point).
        thresh (float): RANSAC inlier threshold in pixels.
        conf (float): Desired confidence level for the RANSAC result.
        max_iters (int): Maximum number of RANSAC iterations.

    Returns:
        tuple: (R, t, inlier_mask) or None if estimation fails.
    """
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC, maxIters=max_iters)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def relative_pose_error_metric(R_gt, t_gt, R, t, scale_cf=1.0, use_gt_norm=True, t_ambiguity=True):
    """
    Computes the angular rotation error and translation distance between two poses.

    In Monocular Vision, we can estimate direction but not absolute scale 
    (the 'size' of the world). This function handles that limitation by 
    optionally normalizing translations or accounting for 180-degree 
    ambiguities in the vector direction.

    Args:
        R_gt, t_gt: Ground truth rotation matrix (3x3) and translation vector (3x1).
        R, t: Predicted rotation matrix and translation vector.
        scale_cf (float): Scale correction factor to align units (e.g., mm to meters).
        use_gt_norm (bool): If True, scales the predicted translation to match 
            the length of the ground truth (evaluating direction only).
        t_ambiguity (bool): If True, accounts for the fact that a translation 
            vector 'forward' might be predicted as 'backward' (cheating/baselines).

    Returns:
        tuple: (translation_error, rotation_error_in_degrees)
    """
    t_gt = t_gt * scale_cf
    t = t * scale_cf
    if use_gt_norm: 
        n_gt = np.linalg.norm(t_gt)
        n = np.linalg.norm(t)
        t = t / n * n_gt

    if t_ambiguity:
        t_err = np.minimum(np.linalg.norm(t_gt - t), np.linalg.norm(t_gt + t))
    else:
        t_err = np.linalg.norm(t_gt - t)

    if not isinstance(R, list):
        R = [R]
        
    R_err = []
    for R_ in R:        
        cos = (np.trace(np.dot(R_.T, R_gt)) - 1) / 2
        cos = np.clip(cos, -1., 1.)  # handle numercial errors
        R_err.append(np.rad2deg(np.abs(np.arccos(cos))))
    
    R_err = np.min(R_err)

    return t_err, R_err


def error_auc(errors, thr):
    """
    Calculates the Area Under the Curve (AUC) for a cumulative error distribution.

    This function measures the 'recall' of a model under a specific error 
    threshold (thr). It represents the probability that the estimated 
    geometric model (Homography, Essential Matrix, etc.) has an error 
    lower than the threshold.

    A higher AUC indicates that the pipeline consistently produces 
    accurate results across many image pairs.

    Args:
        errors (list/np.array): A list of scalar error values (e.g., degrees or pixels).
        thr (float): The maximum error threshold to consider for the AUC 
            (e.g., 5°, 10°, or 20°).

    Returns:
        float: The normalized AUC value between 0 and 1.
    """
    errors = [0] + sorted(errors)
    recall = list(np.linspace(0, 1, len(errors)))

    last_index = np.searchsorted(errors, thr)
    y = recall[:last_index] + [recall[last_index-1]]
    x = errors[:last_index] + [thr]
    return np.trapezoid(y, x) / thr    





# This is the IMC 3D error metric code
def register_by_Horn(ev_coord, gt_coord, ransac_threshold, inl_cf, strict_cf):
    '''Return the best similarity transforms T that registers 3D points pt_ev in <ev_coord> to
    the corresponding ones pt_gt in <gt_coord> according to a RANSAC-like approach for each
    threshold value th in <ransac_threshold>.
    
    Given th, each triplet of 3D correspondences is examined if not already present as strict inlier,
    a correspondence is a strict inlier if <strict_cf> * err_best < th, where err_best is the registration
    error for the best model so far.
    The minimal model given by the triplet is then refined using also its inliers if their total is greater
    than <inl_cf> * ninl_best, where ninl_best is th number of inliers for the best model so far. Inliers
    are 3D correspondences (pt_ev, pt_gt) for which the Euclidean distance |pt_gt-T*pt_ev| is less than th.'''
    
    # remove invalid cameras, the index is returned
    idx_cams = np.all(np.isfinite(ev_coord), axis=0)
    ev_coord = ev_coord[:, idx_cams]
    gt_coord = gt_coord[:, idx_cams]

    # initialization
    n = ev_coord.shape[1]
    r = ransac_threshold.shape[0]
    ransac_threshold = np.expand_dims(ransac_threshold, axis=0)
    ransac_threshold2 = ransac_threshold**2
    ev_coord_1 = np.vstack((ev_coord, np.ones(n)))

    max_no_inl = np.zeros((1, r))
    best_inl_err = np.full(r, np.Inf)
    best_transf_matrix = np.zeros((r, 4, 4))
    best_err = np.full((n, r), np.Inf)
    strict_inl = np.full((n, r), False)
    triplets_used = np.zeros((3, r))

    # run on camera triplets
    for ii in range(n-2):
        for jj in range(ii+1, n-1):
            for kk in range(jj+1, n):
                i = [ii, jj, kk]
                triplets_used_now = np.full((n), False)
                triplets_used_now[i] = True
                # if both ii, jj, kk are strict inliers for the best current model just skip
                if np.all(strict_inl[i]):
                    continue
                # get transformation T by Horn on the triplet camera center correspondences
                transf_matrix = affine_matrix_from_points(ev_coord[:, i], gt_coord[:, i], usesvd=False)
                # apply transformation T to test camera centres
                rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                # compute error and inliers
                err = np.sum((rotranslated - gt_coord)**2, axis=0)
                inl = np.expand_dims(err, axis=1) < ransac_threshold2
                no_inl = np.sum(inl, axis=0)
                # if the number of inliers is close to that of the best model so far, go for refinement
                to_ref = np.squeeze(((no_inl > 2) & (no_inl > max_no_inl * inl_cf)), axis=0)
                for q in np.argwhere(to_ref):                        
                    qq = q[0]
                    if np.any(np.all((np.expand_dims(inl[:, qq], axis=1) == inl[:, :qq]), axis=0)):
                        # already done for this set of inliers
                        continue
                    # get transformation T by Horn on the inlier camera center correspondences
                    transf_matrix = affine_matrix_from_points(ev_coord[:, inl[:, qq]], gt_coord[:, inl[:, qq]])
                    # apply transformation T to test camera centres
                    rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                    # compute error and inliers
                    err_ref = np.sum((rotranslated - gt_coord)**2, axis=0)
                    err_ref_sum = np.sum(err_ref, axis=0)
                    err_ref = np.expand_dims(err_ref, axis=1)
                    inl_ref = err_ref < ransac_threshold2
                    no_inl_ref = np.sum(inl_ref, axis=0)
                    # update the model if better for each threshold
                    to_update = np.squeeze((no_inl_ref > max_no_inl) | ((no_inl_ref == max_no_inl) & (err_ref_sum < best_inl_err)), axis=0)
                    if np.any(to_update):
                        triplets_used[0, to_update] = ii
                        triplets_used[1, to_update] = jj
                        triplets_used[2, to_update] = kk
                        max_no_inl[:, to_update] = no_inl_ref[to_update]
                        best_err[:, to_update] = np.sqrt(err_ref)
                        best_inl_err[to_update] = err_ref_sum
                        strict_inl[:, to_update] = (best_err[:, to_update] < strict_cf * ransac_threshold[:, to_update])
                        best_transf_matrix[to_update] = transf_matrix

    # print("\n")
    # for i in range(r):
    #    print(f'Registered cameras {max_no_inl[0, i]} of {n} for threshold {ransac_threshold[0, i]}')

    best_model = {
        "valid_cams": idx_cams,        
        "no_inl": max_no_inl,
        "err": best_err,
        "triplets_used": triplets_used,
        "transf_matrix": best_transf_matrix}
    return best_model




def homography_error_heat_map(H12_gt, H12, mask1):
    """
    Generates a 2D heat map of the reprojection error between two homographies.

    For every pixel marked as True in 'mask1', the function projects the 
    coordinate into the second image plane using both the ground truth ($H12_{gt}$) 
    and the estimated ($H12$) homography. The error is the Euclidean distance 
    between these two projected points.

    Args:
        H12_gt (torch.Tensor): The $3 \times 3$ ground truth Homography matrix.
        H12 (torch.Tensor): The $3 \times 3$ estimated Homography matrix.
        mask1 (torch.Tensor): A boolean mask of shape (H, W) defining the 
            region of interest (e.g., a specific plane or the whole image).

    Returns:
        torch.Tensor: A 2D tensor of shape (H, W). Valid pixels contain the 
            pixel distance error; background/masked pixels are set to -1.
    """
    pt1 = mask1.argwhere()
    
    pt1 = torch.cat((pt1, torch.ones(pt1.shape[0], 1, device=device)), dim=1).permute(1,0)   

    pt2_gt_ = H12_gt.type(torch.float) @ pt1
    pt2_gt_ = pt2_gt_[:2] / pt2_gt_[2].unsqueeze(0)

    pt2_ = H12.type(torch.float) @ pt1
    pt2_ = pt2_[:2] / pt2_[2].unsqueeze(0)

    d1 = ((pt2_gt_ - pt2_)**2).sum(dim=0).sqrt()
    d1[~d1.isfinite()] = np.inf

    heat_map = torch.full(mask1.shape, -1, device=device, dtype=torch.float)
    heat_map[mask1] = d1
    
    return heat_map


def epipolar_error_heat_map(F_gt, F, sz):
    """
    Computes a 2D heat map of the angular error between two Fundamental matrices.

    For every pixel in an image of size 'sz', the function calculates the 
    corresponding epipolar line in the second view using both the ground truth 
    matrix ($F_{gt}$) and the estimated matrix ($F$). The error is defined as 
    the angle (in degrees) between these two lines.

    Args:
        F_gt (torch.Tensor): The ground truth $3 \times 3$ Fundamental matrix.
        F (torch.Tensor): The estimated $3 \times 3$ Fundamental matrix.
        sz (tuple): The (height, width) of the image grid to evaluate.

    Returns:
        torch.Tensor: A 2D tensor of shape (height, width) containing the 
            angular error in degrees for each pixel. Values range from 0 
            to 180 (or 360 for invalid results).
    """
    y, x = torch.meshgrid(torch.arange(sz[0], device=device), torch.arange(sz[1], device=device))
    pt = torch.stack((y.flatten(), x.flatten(), torch.ones(sz[0] * sz[1], device=device))).type(torch.float)

    l_gt = F_gt.type(torch.float) @ pt
    l = F.type(torch.float) @ pt
    
    l_gt_n = l_gt / l_gt.norm(dim=0)
    l_n = l / l.norm(dim=0)

    sim = torch.linalg.vecdot(l_gt_n.T, l_n.T).abs()
    sim[sim > 1] = 1    

    sim = sim.acos().rad2deg()
    sim[~sim.isfinite()] = 360

    return sim.reshape((sz[0], sz[1]))



   
def invalid_matches(mask1, mask2, pts1, pts2, rad):
    """
    Identifies matches that are outside valid image boundaries or mask regions.

    This function checks two conditions for every match:
    1. Boundary Check: Are the keypoints located within the actual pixel 
       dimensions of the images?
    2. Mask Check: Do the keypoints fall on 'False' regions of the provided masks? 
       To be robust, the second mask is dilated by a specified radius to allow 
       a small margin of error near the edges.

    Args:
        mask1 (np.ndarray): Boolean mask for the first image (H, W).
        mask2 (np.ndarray): Boolean mask for the second image (H, W).
        pts1 (torch.Tensor): Keypoints in the first image (N, 2).
        pts2 (torch.Tensor): Keypoints in the second image (N, 2).
        rad (int): The radius for dilation of the second mask.

    Returns:
        torch.Tensor: A boolean tensor of shape (N,) where 'True' indicates 
            an invalid match that should be discarded.
    """
    dmask2 = cv2.dilate(mask2.astype(np.ubyte), np.ones((rad*2 + 1, rad*2 + 1)))
    
    pt1 = pts1.round().permute(1, 0)
    pt2 = pts2.round().permute(1, 0)

    invalid_ = torch.zeros(pt1.shape[1], device=device, dtype=torch.bool)

    to_exclude = (pt1[0] < 0) | (pt2[0] < 0) | (pt1[0] >= mask1.shape[1]) | (pt2[0] >= mask2.shape[1]) | (pt1[1] < 0) | (pt2[1] < 0) | (pt1[1] >= mask1.shape[0]) | (pt2[1] >= mask2.shape[0])

    pt1 = pt1[:, ~to_exclude]
    pt2 = pt2[:, ~to_exclude]
    
    l1 = (pt1[1, :] * mask1.shape[1] + pt1[0,:]).type(torch.long)
    l2 = (pt2[1, :] * mask2.shape[1] + pt2[0,:]).type(torch.long)

    invalid_check = ~(torch.tensor(mask1, device=device).flatten()[l1]) & ~(torch.tensor(dmask2, device=device, dtype=torch.bool).flatten()[l2])
    invalid_[~to_exclude] = invalid_check 

    return invalid_



def evaluate_rec(gt_df, user_df, inl_cf = 0.8, strict_cf=0.5, thresholds=[0.05]):
    ''' Register the <user_df> camera centers to the ground-truth <gt_df> camera centers and
    return the corresponding mAA as the average percentage of registered camera threshold.
    
    For each threshold value in <thresholds>, the best similarity transformation found which
    maximizes the number of registered cameras is employed. A camera is marked as registered
    if after the transformation its Euclidean distance to the corresponding ground-truth camera
    center is less than the mentioned threshold. Current measurements are in meter.
    
    Registration parameters:
    <inl_cf> coefficient to activate registration refinement, set to 1 to refine a new model
    only when it gives more inliers, to 0 to refine a new model always; high values increase
    speed but decrease precision.
    <strict_cf> threshold coefficient to define strict inliers for the best registration so far,
    new minimal models made up of strict inliers are skipped. It can vary from 0 (slower) to
    1 (faster); set to -1 to check exhaustively all the minimal model triplets.'''
    
    # get camera centers
    ucameras = user_df
    gcameras = gt_df    
        
    # get the image list to use
    good_cams = []
    for image_path in gcameras.keys():
        if image_path in ucameras.keys():
            good_cams.append(image_path)
        
    # put corresponding camera centers into matrices
    n = len(good_cams)
    u_cameras = np.zeros((3, n))
    g_cameras = np.zeros((3, n))
    
    ii = 0
    for i in good_cams:
        u_cameras[:, ii] = ucameras[i]
        g_cameras[:, ii] = gcameras[i]
        ii += 1
        
    # Horn camera centers registration, a different best model for each camera threshold
    model = register_by_Horn(u_cameras, g_cameras, np.asarray(thresholds), inl_cf, strict_cf)
    
    # transformation matrix
    # print("Transformation matrix for maximum threshold")
    # T = np.squeeze(model['transf_matrix'][-1])
    # print(T)
    
    return model
    
 