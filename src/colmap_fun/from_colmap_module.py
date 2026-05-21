import os

import numpy as np
import torch

from core import device as global_device

#pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, image_pairs, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args, enable_quadtree
from .colmap_ext import coldb_ext


def kpts_from_colmap(kp): 
    """
    Converts COLMAP-style keypoints into the pipeline's standard LAF format.

    COLMAP stores keypoints as (x, y, a11, a12, a21, a22), where the last 
    four elements describe the affine shape (scale and orientation) of 
    the feature. This function reconstructs the full homography matrix 
    representing the local patch around the keypoint.

    Args:
        kp (torch.Tensor): A tensor of shape (N, 4) or (N, 6). 
            Columns 0-1 are (x, y). 
            Columns 2-5 are the affine shape matrix components.

    Returns:
        tuple: (keypoint_coords, local_homographies, reliability_scores)
    """
    w_ = kp[:, 2:]
    kp = kp[:, :2]
    w = torch.zeros((kp.shape[0], 3, 3), device=global_device)
    w[:, 2, 2] = 1
    w[:, :2, :2] = w_.reshape(-1, 2, 2)
         
    t = torch.zeros((kp.shape[0], 3, 3), device=global_device)        
    t[:, [0, 1], 2] = kp
    t[:, 0, 0] = 1
    t[:, 1, 1] = 1
    t[:, 2, 2] = 1           
     
    kH = t.bmm(w).inverse()
     
    kr = torch.full((kp.shape[0], ), np.inf, device=global_device)    
             
    return kp, kH, kr


class from_colmap_module:
    """
    A data-loading module that retrieves keypoints and geometry from a COLMAP database.

    This module allows you to use COLMAP's pre-computed results (like SIFT 
    features or geometric verification masks) within the pipeline. It can 
    be used for 'evaluation only' tasks or to feed COLMAP-detected points 
    into a deep-learning refiner like FCGNN.

    Attributes:
        db (str): Path to the COLMAP '.db' file.
        only_keypoints (bool): If True, only extracts points for a single 
            image (useful for feature-only evaluation).
        include_two_view_geometry (bool): If True, retrieves the verified 
            inliers and geometric models (H, E, F) computed by COLMAP.
    """
    def __init__(self, **args):
        from core import set_args
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True
        self.add_to_cache = True

        self.device = torch.device(self.args.get('device', str(global_device)))
        
        self.args = {
            'id_more': '',
            'db': 'colmap.db',
            'only_keypoints': False,            
            'include_two_view_geometry': True,
        }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('from_colmap' , args, self.args)

        self.db = coldb_ext(self.args['db'])
        if self.args['only_keypoints']:
            self.single_image = True
                

    def finalize(self):
        self.db.close()

                
    def get_id(self): 
        return self.id_string

    
    def run(self, **args):   
        if self.single_image:
            im = args['img'][args['idx']]
            _, img = os.path.split(im)           
            im_id = self.db.get_image_id(img)

            if im_id is None:
                kp = torch.zeros((0, 2), device=self.device)
                kr = torch.zeros((0, ), device=self.device)
                kH = torch.zeros((0, 3, 3), device=self.device)
            else:                
                kp_ = self.db.get_keypoints(im_id)
                kp, kH, kr = kpts_from_colmap(torch.tensor(kp_, device=self.device))

            return {'kp': kp, 'kH': kH, 'kr': kr}
        
        else:
            out_data = {}
            
            im0 = args['img'][0]            
            _, img0 = os.path.split(im0)           
            im0_id = self.db.get_image_id(img0)

            if im0_id is None:
                kp0 = torch.zeros((0, 2), device=self.device)
                kr0 = torch.zeros((0, ), device=self.device)
                kH0 = torch.zeros((0, 3, 3), device=self.device)
            else:                
                kp0_ = self.db.get_keypoints(im0_id)
                kp0, kH0, kr0 = kpts_from_colmap(torch.tensor(kp0_, device=self.device))

            im1 = args['img'][1]            
            _, img1 = os.path.split(im1)           
            im1_id = self.db.get_image_id(img1)

            if im1_id is None:
                kp1 = torch.zeros((0, 2), device=self.device)
                kr1 = torch.zeros((0, ), device=self.device)
                kH1 = torch.zeros((0, 3, 3), device=self.device)
            else:                
                kp1_ = self.db.get_keypoints(im1_id)
                kp1, kH1, kr1 = kpts_from_colmap(torch.tensor(kp1_, device=self.device))

            kp = [kp0, kp1]
            kH = [kH0, kH1]
            kr = [kr0, kr1]
            
            out_data['kp'] = kp
            out_data['kH'] = kH
            out_data['kr'] = kr

            if (im0_id is not None) and (im1_id is not None):
                m_idx = self.db.get_matches(im0_id, im1_id)
                
                if m_idx is not None:
                    m_idx = torch.tensor(np.copy(m_idx), device=self.device, dtype=torch.int)
                    
                    if not self.args['include_two_view_geometry']:
                        m_mask = torch.full((m_idx.shape[0],), 1, device=self.device, dtype=torch.bool)
                        m_val = torch.full((m_idx.shape[0],), np.inf, device=self.device)
                    
                    else:
                        s_idx, models = self.db.get_two_view_geometry(im0_id, im1_id)
                    
                        if s_idx is None:
                            m_mask = torch.full((m_idx.shape[0],), 1, device=self.device, dtype=torch.bool)
                        else:
                            s_idx = torch.tensor(np.copy(s_idx), device=self.device, dtype=torch.int)
                            
                            if len(models.keys()) == 1:
                                for model in ['H', 'F', 'E']:
                                    if model in models: out_data[model] = torch.tensor(models[model], device=self.device)
                            
                            m_mask = torch.zeros(m_idx.shape[0], device=self.device, dtype=torch.bool)
                            
                            idx = torch.argsort(m_idx[:, 1].type(torch.int), stable=True)
                            m_idx = m_idx[idx]
                            idx = torch.argsort(m_idx[:, 0].type(torch.int), stable=True)
                            m_idx = m_idx[idx]

                            idx = torch.argsort(s_idx[:, 1].type(torch.int), stable=True)
                            s_idx = s_idx[idx]
                            idx = torch.argsort(s_idx[:, 0].type(torch.int), stable=True)
                            s_idx = s_idx[idx]

                            q0 = 0
                            q1 = 0
                            while (q0 < s_idx.shape[0]) and (q1 < m_idx.shape[0]):                       
                                if (s_idx[q0, 0] < m_idx[q1, 0]) or ((s_idx[q0, 0] == m_idx[q1, 0]) and (s_idx[q0, 1] < m_idx[q1, 1])):
                                    q0 = q0 + 1
                                elif (s_idx[q0, 0] == m_idx[q1, 0]) and (s_idx[q0, 1] == m_idx[q1, 1]):
                                    m_mask[q1] = 1
                                    q0 = q0 + 1
                                    q1 = q1 + 1
                                else:
                                    q1 = q1 + 1

                        m_val = torch.full((m_idx.shape[0],), np.inf, device=self.device)
                                                    
                    out_data['m_idx'] = m_idx
                    out_data['m_val'] = m_val
                    out_data['m_mask'] = m_mask
        
        return out_data

