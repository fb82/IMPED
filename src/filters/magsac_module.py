import warnings

import cv2
import numpy as np
import torch

from core import device as global_device
from core import set_args


class magsac_module:
    """
    A geometric verification module using the MAGSAC++ robust estimator.

    MAGSAC++ is an improvement over traditional RANSAC. It is less sensitive 
    to the 'px_th' (pixel threshold) parameter because it considers 
    multiple noise scales simultaneously. This makes it exceptionally 
    good at filtering out outliers in noisy deep-learning matches.

    Attributes:
        mode (str): 'fundamental_matrix' for 3D scenes or 'homography' 
            for planes/rotations.
        px_th (float): The maximum expected noise scale (in pixels).
        conf (float): Confidence level (0 to 1).
        max_try (int): Number of retry attempts if the solver fails 
            due to numerical instability.
    """
    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False  
        self.pass_through = False
        self.add_to_cache = True
                                
        self.args = {
            'id_more': '',
            'mode': 'fundamental_matrix',
            'conf': 0.9999,
            'max_iters': 100000,
            'px_th': 3,
            'max_try': 3
            }
        self.device = torch.device(self.args.get('device', str(global_device)))
        
                
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('magsac', args, self.args)      


    def get_id(self): 
        return self.id_string


    def finalize(self):
        return

        
    def run(self, **args):  
        pt1_ = args['kp'][0]
        pt2_ = args['kp'][1]
        mi = args['m_idx']
        mm = args['m_mask']
        
        pt1 = pt1_[mi[mm][:, 0]]
        pt2 = pt2_[mi[mm][:, 1]]
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
            
        if self.args['mode'] == 'fundamental_matrix':
            sac_to_run = cv2.findFundamentalMat
            sac_min = 8
        else:
            sac_to_run = cv2.findHomography
            sac_min = 4
            
        if (pt1.shape)[0] >= sac_min:  
            try:                     
                F, mask = sac_to_run(pt1, pt2, cv2.USAC_MAGSAC, self.args['px_th'], self.args['conf'], self.args['max_iters'])
            except:
                for i in range(self.args['max_try'] - 1):
                    try:
                        idx = np.random.permutation(pt1.shape[0])
                        jdx = np.argsort(idx)
                        F, mask = sac_to_run(pt1[idx], pt2[idx], cv2.USAC_MAGSAC, self.args['px_th'], self.args['conf'], self.args['max_iters'])
                        mask = mask[jdx]
                    except:
                        warnings.warn("MAGSAC failed, tentative " + str(i + 1) + ' of ' + str(self.args['max_try']))
                        continue
                    
        if not isinstance(mask, np.ndarray):
            mask = torch.zeros(pt1.shape[0], device=self.device, dtype=torch.bool)
        else:
            if len(mask.shape) > 1: mask = mask.squeeze(1) > 0
            mask = torch.tensor(mask, device=self.device, dtype=torch.bool)
 
        aux = mm.clone()
        mm[aux] = mask
        
        if F is not None:
            F = torch.tensor(F, device=self.device)
        
        if self.args['mode'] == 'fundamental_matrix':
            return {'m_mask': mm, 'F': F}
        else:
            return {'m_mask': mm, 'H': F}
