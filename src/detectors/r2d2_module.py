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

from core import device, pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, image_pairs, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args

conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'r2d2'))

from r2d2.tools import common as r2d2_common
from r2d2.tools.dataloader import norm_RGB as r2d2_norm_RGB
import r2d2.nets.patchnet as r2d2_patchnet 

class r2d2_module:
    """
    A unified keypoint detector and descriptor module.

    R2D2 learns to jointly estimate three outputs for an image:
    1. Repeatability Map: High values indicate locations likely to be 
       detected under different viewing conditions.
    2. Reliability Map: High values indicate locations where the 
       resulting descriptor is unique and trustworthy for matching.
    3. Dense Descriptors: High-dimensional vectors used for point-to-point 
       correspondence.

    This module supports multi-scale extraction, allowing it to find 
    features that are robust to large changes in distance/zoom.

    Attributes:
        reliability-thr (float): Minimum confidence for a point to be 
            considered reliable.
        repeatability-thr (float): Minimum confidence for a point to be 
            considered repeatable.
        top-k (int): Maximum number of keypoints to keep per image.
    """
    def load_network(model_fn): 
        checkpoint = torch.load(model_fn, weights_only=False)
        # print("\n>> Creating net = " + checkpoint['net']) 
        net = eval('r2d2_patchnet.' + checkpoint['net'])
        # nb_of_weights = r2d2_common.model_size(net)
        # print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")
    
        # initialization
        weights = checkpoint['state_dict']
        net.load_state_dict({k.replace('module.',''): v for k,v in weights.items()})
        return net.eval()
    
    
    class NonMaxSuppression(torch.nn.Module):
        def __init__(self, rel_thr=0.7, rep_thr=0.7):
            r2d2_patchnet.nn.Module.__init__(self)
            self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.rel_thr = rel_thr
            self.rep_thr = rep_thr

        
        def forward(self, reliability, repeatability, **kw):
            assert len(reliability) == len(repeatability) == 1
            reliability, repeatability = reliability[0], repeatability[0]
    
            # local maxima
            maxima = (repeatability == self.max_filter(repeatability))
    
            # remove low peaks
            maxima *= (repeatability >= self.rep_thr)
            maxima *= (reliability   >= self.rel_thr)
    
            return maxima.nonzero().t()[2:4]
    
    
    def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                            min_scale=0.0, max_scale=1, 
                            min_size=256, max_size=1024, 
                            verbose=False):
        old_bm = torch.backends.cudnn.benchmark 
        torch.backends.cudnn.benchmark = False # speedup
        
        # extract keypoints at multiple scales
        B, three, H, W = img.shape
        assert B == 1 and three == 3, "should be a batch with a single RGB image"
        
        assert max_scale <= 1
        s = 1.0 # current scale factor
        
        X,Y,S,C,Q,D = [],[],[],[],[],[]
        while  s+0.001 >= max(min_scale, min_size / max(H,W)):
            if s-0.001 <= min(max_scale, max_size / max(H,W)):
                nh, nw = img.shape[2:]
                if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
                # extract descriptors
                with torch.no_grad():
                    res = net(imgs=[img])
                    
                # get output and reliability map
                descriptors = res['descriptors'][0]
                reliability = res['reliability'][0]
                repeatability = res['repeatability'][0]
    
                # normalize the reliability for nms
                # extract maxima and descs
                y,x = detector(**res) # nms
                c = reliability[0,0,y,x]
                q = repeatability[0,0,y,x]
                d = descriptors[0,:,y,x].t()
                n = d.shape[0]
    
                # accumulate multiple scales
                X.append(x.float() * W/nw)
                Y.append(y.float() * H/nh)
                S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
                C.append(c)
                Q.append(q)
                D.append(d)
            s /= scale_f
    
            # down-scale the image for next iteration
            nh, nw = round(H*s), round(W*s)
            img = r2d2_patchnet.F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)
    
        # restore value
        torch.backends.cudnn.benchmark = old_bm
    
        Y = torch.cat(Y)
        X = torch.cat(X)
        S = torch.cat(S) # scale
        scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
        XYS = torch.stack([X,Y,S], dim=-1)
        D = torch.cat(D)
        return XYS, D, scores


    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True
                                
        self.args = { 
            'id_more': '',
            'patch_radius': 16,            
            'top-k': 5000,
            'scale_f': 2**0.25,
            'min_size': 256,
            'max_size': 1024,
            'min_scale': 0,
            'max_scale': 1,
            'reliability-thr': 0.7,
            'repeatability-thr': 0.7,
            'model': 'r2d2/models/r2d2_WAF_N16.pt',
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                        
        self.id_string, self.args = set_args('r2d2', args, self.args)        

        if device.type == 'cuda':
            cuda = 0
        else:
            cuda = -1

        self.iscuda = r2d2_common.torch_set_gpu(cuda)
    
        # load the network...
        self.net = r2d2_module.load_network(self.args['model'])
        if self.iscuda: self.net = self.net.cuda()
    
        # create the non-maxima detector
        self.detector = r2d2_module.NonMaxSuppression(
            rel_thr = self.args['reliability-thr'], 
            rep_thr = self.args['repeatability-thr'])


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):
        img = Image.open(args['img'][args['idx']]).convert('RGB')
        W, H = img.size
        img = r2d2_norm_RGB(img)[None] 
        if self.iscuda: img = img.cuda()
        
        # extract keypoints/descriptors for a single image
        xys, desc, scores = r2d2_module.extract_multiscale(self.net, img, self.detector,
            scale_f   = self.args['scale_f'], 
            min_scale = self.args['min_scale'], 
            max_scale = self.args['max_scale'],
            min_size  = self.args['min_size'], 
            max_size  = self.args['max_size'], 
            verbose = False)

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[-self.args['top-k'] or None:]
        
        keypoints = torch.tensor(xys[idxs], device=device) 
        descriptors = torch.tensor(desc[idxs], device=device) 
        scores = torch.tensor(scores[idxs], device=device)

        kp = keypoints[:, :2]       
        desc = descriptors
        scales = keypoints[:, 2] / 2

        kH = torch.zeros((kp.shape[0], 3, 3), device=device)        
        kH[:, [0, 1], 2] = -kp / self.args['patch_radius']
        kH[:, 0, 0] = 1 / scales
        kH[:, 1, 1] = 1 / scales
        kH[:, 2, 2] = 1

        kr = scores        
        
        # todo: add feats['keypoint_scores'] as kr        
        return {'kp': kp, 'kH': kH, 'kr': kr, 'desc': desc}
