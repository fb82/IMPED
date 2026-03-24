from .device import device
import torch

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


def laf2homo(kps, with_scale=False):
    c = kps[:, :, 2].type(torch.float)
    
    Hi = torch.zeros((kps.shape[0], 3, 3), device=device)
    Hi[:, :2, :] = kps    
    Hi[:, 2, 2] = 1 

    if with_scale:
        s = torch.sqrt(torch.abs(kps[:, 0, 0] * kps[:, 1, 1] - kps[:, 0, 1] * kps[:, 1, 0]))   
        Hi[:, :2, :] = Hi[:, :2, :] / s.reshape(-1, 1, 1)
        s = s.type(torch.float)
        
    H = torch.linalg.inv(Hi).type(torch.float)

    if with_scale:    
        return c, H, s
    
    return c, H


def homo2laf(c, H, s=None):
    
    aux = torch.zeros((H.shape[0], 3 , 3), device=device)
    aux[:, 0, 0] = 1
    aux[:, 1, 1] = 1
    aux[:, 2] = 1
    pt3 = H.inverse().bmm(aux)
    pt2 = pt3 / pt3[:, 2, :].unsqueeze(1)
    kp = torch.stack((pt2[:, :2, 0] - pt2[:, :2, 2], pt2[:, :2, 1] - pt2[:, :2, 2], pt2[:, :2, 2]), dim=-1)
     
#   Hi = torch.linalg.inv(H)
#   kp = Hi[:, :2, :]
    
    if not (s is None):
        kp = kp * s.reshape(-1, 1, 1)

    return kp.unsqueeze(0)

def apply_homo(p, H):
    """
    Transforms 2D points from one coordinate system to another using a Homography.

    This function converts 2D points (x, y) into 'Homogeneous Coordinates' (x, y, 1), 
    multiplies them by the transformation matrix H, and then projects them 
    back into 2D space by dividing by the third (w) coordinate.

    Args:
        p (torch.Tensor): A tensor of shape (N, 2) representing N points.
        H (torch.Tensor): A 3x3 Homography transformation matrix.

    Returns:
        torch.Tensor: The transformed (N, 2) points.
    """
    
    pt = torch.zeros((p.shape[0], 3), device=device)
    pt[:, :2] = p
    pt[:, 2] = 1
    pt_ = (H @ pt.permute((1, 0))).permute((1, 0))
    return pt_[:, :2] / pt_[:, 2].unsqueeze(-1)    


def change_patch_homo(kH, warp):       
    return kH @ warp.unsqueeze(0)


def decompose_H(H, ret_err=False):
    """
    Decomposes a batch of homography matrices into elementary geometric components.

    The function factors the homography matrix $H$ such that:
    $H = H_t \cdot H_s \cdot H_m \cdot H_r \cdot H_a \cdot H_p$
    
    Where:
    - $H_t$: Translation
    - $H_s$: Uniform Scaling
    - $H_m$: Reflection (Mirroring)
    - $H_r$: Pure Rotation
    - $H_a$: Affine (Shear/Stretch)
    - $H_p$: Projective (Perspective) components

    The decomposition uses QR factorization on the upper-left $2 \times 2$ block 
    after removing projective and translative effects. It handles determinant 
    signs to ensure valid rotation and affine matrices.

    Args:
        H (torch.Tensor): A batch of homography matrices of shape (N, 3, 3).
        ret_err (bool, optional): If True, calculates and returns the 
            reconstruction error. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - H_t (torch.Tensor): Translation matrices (N, 3, 3).
            - H_s (torch.Tensor): Scale matrices (N, 3, 3).
            - H_m (torch.Tensor): Reflection matrices (N, 3, 3).
            - H_r (torch.Tensor): Rotation matrices (N, 3, 3).
            - H_a (torch.Tensor): Affine/Shear matrices (N, 3, 3).
            - H_p (torch.Tensor): Projective matrices (N, 3, 3).
            - err (torch.Tensor or None): The L1 reconstruction error per 
              matrix if ret_err is True, else None.

    Note:
        The decomposition assumes the input matrices are defined in a 
        coordinate system where the transformation is applied as $H \cdot x$.
    """
    # H = torch.rand((5, 3, 3), device=device)
    
    v = H[:, -1, -1]
    V = H[:, -1, :2]
    T = H[:, :2, -1] / v.unsqueeze(-1)
    W = H[:, :2, :2] - T.unsqueeze(-1).bmm(V.unsqueeze(1))
    [R_, K_] = torch.linalg.qr(W)
    M_ = torch.eye(2, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    
    # the determinant inversion can be obtained by multiplying a row or a column by -1
    # this is done for instance by the orthogonal matrix [1 0; 0 -1] (reflection matrix)
    # notice that [1 0; 0 -1]*[1 0; 0 -1]=eye(2);
    
    # det sign
    t = K_.det().sign() < 0
    # this change the sign of det
    K_[t] = torch.tensor([[1., 0.], [0., -1.]], device=device) @ K_[t] 
    R_[t] = R_[t] @ torch.tensor([[1., 0.], [0., -1.]], device=device)
    
    # det sign    
    t = R_.det().sign() < 0    
    # this change the sign of det
    R_[t] = torch.tensor([[1., 0.], [0., -1.]], device=device) @ R_[t]
    # this is the inverse to nullify the total effect
    M_[t] = M_[t] @ torch.tensor([[1., 0.], [0., -1.]], device=device)
    
    s = R_.bmm(K_).det().abs() ** 0.5
    K = K_ / (K_.det().abs() ** 0.5).unsqueeze(-1).unsqueeze(-1)
    R = R_ / (R_.det().abs() ** 0.5).unsqueeze(-1).unsqueeze(-1)
    M = M_
    
    # projective
    H_p = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_p[:, -1, :2] = V
    H_p[:, -1, -1] = v
     
    # affine
    H_a = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_a[:, :2, :2] = K
    
    # rotation
    H_r = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_r[:, :2, :2] = R
    
    # reflection
    H_m = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_m[:, :2, :2] = M
    
    # scale
    H_s = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_s[:, 0, 0] = s
    H_s[:, 1, 1] = s
    
    # translation
    H_t = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_t[:, :2, -1] = T
    
    if ret_err:    
        err = (H - H_t.bmm(H_s.bmm(H_m.bmm(H_r.bmm(H_a.bmm(H_p)))))).abs().sum(dim=(1, 2))
    else:
        err = None
        
    return H_t, H_s, H_m, H_r, H_a, H_p, err


def decompose_H_other(H, ret_err=False):
    """
    Alternative decomposition of homography matrices using RQ-based factorization.

    This function decomposes a batch of $3 \times 3$ homography matrices $H$ into 
    geometric components using a permutation-based RQ decomposition on the 
    affine block. This approach typically isolates intrinsic and extrinsic 
    parameters differently than a standard QR decomposition.

    The decomposition follows the reconstruction order:
    $H = H_p \cdot H_a \cdot H_t \cdot H_s \cdot H_m \cdot H_r$

    Args:
        H (torch.Tensor): Batch of homography matrices of shape (N, 3, 3).
        ret_err (bool, optional): If True, calculates the L1 reconstruction 
            error between the original $H$ and the product of its components. 
            Defaults to False.

    Returns:
        tuple: A tuple containing:
            - H_p (torch.Tensor): Projective (perspective) matrices.
            - H_a (torch.Tensor): Affine (shear) matrices.
            - H_t (torch.Tensor): Translation matrices.
            - H_s (torch.Tensor): Uniform scale matrices.
            - H_m (torch.Tensor): Reflection (mirroring) matrices.
            - H_r (torch.Tensor): Pure rotation matrices.
            - err (torch.Tensor or None): The reconstruction error if 
              ret_err is True, otherwise None.

    Note:
        Unlike `decompose_H`, this function performs an RQ decomposition by 
        applying a permutation matrix $P$ before and after a standard QR 
        factorization. It also handles coordinate shifts for the translation 
        and projective vectors to maintain consistency with the new chain order.
    """
    # H = torch.rand((5, 3, 3), device=device)
    
    A_ = H[:, :2, :2]
    a_ = H[:, -1, :2]
    t_ = H[:, :2, -1]
    a = H[:, -1, -1]
    
    # RQ decomposition
    P = torch.tensor([[0., 1.], [1., 0.]], device=device)
    R_, K_ = torch.linalg.qr(A_.permute(0, 2, 1) @ P)
    K_ = P @ K_.permute(0, 2, 1) @ P
    R_ = P @ R_.permute(0, 2, 1)
        
    M_ = torch.eye(2, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)

    # the determinant inversion can be obtained by multiplying a row or a column by -1
    # this is done for instance by the orthogonal matrix [1 0; 0 -1] (reflection matrix)
    # notice that [1 0; 0 -1]*[1 0; 0 -1]=eye(2);

    # det sign
    t = K_.det().sign() < 0
    # this change the sign of det
    K_[t] =  K_[t] @ torch.tensor([[1., 0.], [0., -1.]], device=device)
    R_[t] = torch.tensor([[1., 0.], [0., -1.]], device=device) @ R_[t]

    # det sign    
    t = R_.det().sign() < 0    
    # this change the sign of det
    R_[t] = torch.tensor([[1., 0.], [0., -1.]], device=device) @ R_[t]
    # this is the inverse to nullify the total effect
    M_[t] = M_[t] @ torch.tensor([[1., 0.], [0., -1.]], device=device)

    s = R_.bmm(K_).det().abs() ** 0.5
    K = K_ / (K_.det().abs() ** 0.5).unsqueeze(-1).unsqueeze(-1)
    R = R_ / (R_.det().abs() ** 0.5).unsqueeze(-1).unsqueeze(-1)
    M = M_

    V = torch.linalg.inv(A_.permute(0, 2, 1)).bmm(a_.unsqueeze(-1))
    T = torch.linalg.inv(K).bmm(t_.unsqueeze(-1))
    v = a - V.permute(0, 2, 1).bmm(t_.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    # projective
    H_p = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_p[:, -1, :2] = V.squeeze(-1)
    H_p[:, -1, -1] = v
     
    # affine
    H_a = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_a[:, :2, :2] = K
    
    # rotation
    H_r = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_r[:, :2, :2] = R
    
    # reflection
    H_m = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_m[:, :2, :2] = M
    
    # scale
    H_s = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_s[:, 0, 0] = s
    H_s[:, 1, 1] = s
    
    # translation
    H_t = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_t[:, :2, -1] = T.squeeze(-1)
    
    if ret_err:    
        err = (H - H_p.bmm(H_a.bmm(H_t.bmm(H_s.bmm(H_m.bmm(H_r)))))).abs().sum(dim=(1, 2))
    else:
        err = None
        
    return H_p, H_a, H_t, H_s, H_m, H_r, err
