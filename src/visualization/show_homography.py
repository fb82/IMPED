import os

import cv2
import numpy as np
import torch

from core import apply_homo, device, set_args


class show_homography_module:
    """
    A visualization module for verifying homography-based image alignment.

    This module takes two images and a homography matrix, warps them into a 
    common coordinate system, and generates diagnostic images. It specifically 
    creates a 'checkerboard' overlay where alternating squares come from 
    different images, making it easy to see if features (like edges or lines) 
    align perfectly across the boundary.

    Attributes:
        args (dict): Configuration including:
            - chessboard_size (int): The width/height of the checkerboard tiles.
            - alpha (float): Blending factor between the two images.
            - reference_image (int): Which image (0 or 1) defines the base canvas.
            - img_max_size (int): Rescales the output to fit this maximum dimension.
            - show_merged (bool): If True, generates the checkerboard blend file.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'img_prefix': '',
            'img_suffix': '',
            'cache_path': 'show_imgs',
            'prepend_pair': True,
            'ext': '.png',
            'force': False,
            'img_max_size': 1280,
            'img_exp_length': 0.33,
            'reference_image': 0,
            'show_separated': True,
            'show_merged': True,
            'alpha': 1.0,
            'chessboard_size': 100,
            'interpolation': cv2.INTER_LANCZOS4,
        }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('show_hompgraphy' , args, self.args)

                
    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

    
    def run(self, **args): 
        """
        Processes the images and saves the visualizations.

        Logic:
        1. Coordinate Transformation: Determines the shared bounding box 
           required to fit both the reference and the warped secondary image.
        2. Scaling: Applies a global scale factor to prevent massive output 
           files if the homography has a large translation component.
        3. Warping: Uses `cv2.warpPerspective` with high-quality Lanczos 
           interpolation to transform the images.
        4. Checkerboard Masking: Generates an alternating 2D grid to assign 
           pixels from either Image A or Image B.
        5. Saving: Writes the aligned individual images and the combined 
           checkerboard blend to the cache path.
        """                
        H = args['H']
        
        if H is None: return {}
        
        alpha = 1.0 if self.args['alpha'] is None else self.args['alpha']      
        max_sz = np.inf if self.args['img_max_size'] is None else self.args['img_max_size']
        exp_len = np.inf if self.args['img_exp_length'] is None else self.args['img_exp_length']
        chess_sz = np.inf if self.args['chessboard_size'] is None else self.args['chessboard_size']

        if self.args['reference_image'] == 0:
            img0 = args['img'][0]
            img1 = args['img'][1]
        else:
            img0 = args['img'][1]
            img1 = args['img'][0]
            H = H.inverse()

        im0 = os.path.splitext(os.path.split(img0)[1])[0]
        im1 = os.path.splitext(os.path.split(img1)[1])[0]

        if self.args['prepend_pair']:            
            cache_path = os.path.join(self.args['cache_path'], im0 + '_' + im1)
        else:
            cache_path = self.args['cache_path']
                
        new_img0 = os.path.join(cache_path, self.args['img_prefix'] + im0 + self.args['img_suffix'] + self.args['ext'])
        new_img1 = os.path.join(cache_path, self.args['img_prefix'] + im1 + self.args['img_suffix'] + self.args['ext'])
        new_img01 = os.path.join(cache_path, self.args['img_prefix'] + im0 + '_' + im1 + self.args['img_suffix'] + self.args['ext'])
    
        can_return_separated = False
        if self.args['show_separated']:    
            if os.path.isfile(new_img0) and os.path.isfile(new_img1) and (not self.args['force']): can_return_separated = True

        can_return_merged = False
        if self.args['show_merged']:    
            if os.path.isfile(new_img01) and (not self.args['force']): can_return_merged = True

        if can_return_separated and can_return_merged: return {}

        os.makedirs(cache_path, exist_ok=True)
        
        ima0 = cv2.imread(img0, cv2.IMREAD_UNCHANGED)
        ima1 = cv2.imread(img1, cv2.IMREAD_UNCHANGED)

        bts0 = torch.tensor([[0.0, 0], [0, ima0.shape[0]], [ima0.shape[1], 0],  [ima0.shape[1], ima0.shape[0]]], device=device, dtype=torch.float)
        bts0_offset = (torch.tensor([ima0.shape[1], ima0.shape[0]], device=device) * exp_len / 2).round()
        bts0_proj = bts0 + bts0_offset * torch.tensor([[-1.0, -1], [-1, 1], [1, -1], [1, 1]], device=device)

        bts1 = torch.tensor([[0.0, 0], [0, ima1.shape[0]], [ima1.shape[1], 0],  [ima1.shape[1], ima1.shape[0]]], device=device, dtype=torch.float)
        bts1_proj = apply_homo(bts1, H.inverse().to(torch.float)).round()

        bts_all = torch.cat((bts0, bts1_proj), axis=0)
        bts_small = [max(min(bts_all[:, 0]), min(bts0_proj[:, 0])).item(), max(min(bts_all[:, 1]), min(bts0_proj[:, 1])).item()]
        bts_big = [min(max(bts_all[:, 0]), max(bts0_proj[:, 0])).item(), min(max(bts_all[:, 1]), max(bts0_proj[:, 1])).item()]

        bts_orig = torch.tensor(bts_small, device=device)
        bts_size = torch.tensor(bts_big, device=device) - bts_orig 

        s_rev = max(bts_size / max_sz)
        s = 1.0 if s_rev <= 1 else 1/s_rev
        
        T = torch.eye(3, device=device, dtype=H.dtype)
        T[:2, 2] = -s * bts_orig
        T[0, 0] = s
        T[1, 1] = s

        interp = self.args['interpolation']

        ima0 = np.concatenate((ima0, np.full((ima0.shape[0], ima0.shape[1], 1), 255, dtype=np.uint8)), axis=-1)
        ima1 = np.concatenate((ima1, np.full((ima1.shape[0], ima0.shape[1], 1), 255, dtype=np.uint8)), axis=-1)

        ima0_warp = cv2.warpPerspective(ima0, T.to('cpu').numpy(), (bts_size * s).to(torch.int).to('cpu').numpy(), flags=interp)
        ima1_warp = cv2.warpPerspective(ima1, (T @ H.inverse()).to('cpu').numpy(), (bts_size * s).to(torch.int).to('cpu').numpy(), flags=interp)

        if self.args['show_separated']:
            cv2.imwrite(new_img0, ima0_warp)
            cv2.imwrite(new_img1, ima1_warp)        

        if not self.args['show_merged']: return {}
                
        ima0_warp = ima0_warp.astype(float)
        ima1_warp = ima1_warp.astype(float)


        mask0 = (ima0_warp[:, :, 3] == 255).astype(float)       
        mask1 = (ima1_warp[:, :, 3] == 255).astype(float)     
        mask01 = mask0 + mask1       

        for k1, i in enumerate(np.arange(0, ima0_warp.shape[0], chess_sz)):
            for k2, j in enumerate(np.arange(0, ima0_warp.shape[1], chess_sz)):
                ii = i.astype(int)
                jj = j.astype(int)

                alpha0 = alpha if (k1 + k2) % 2 else 1 - alpha
                alpha1 = 1 - alpha if (k1 + k2) % 2 else alpha

                b0 = mask0[ii:min(ii + chess_sz, ima0_warp.shape[0]), jj:min(jj + chess_sz, ima0_warp.shape[1])]
                b1 = mask1[ii:min(ii + chess_sz, ima1_warp.shape[0]), jj:min(jj + chess_sz, ima1_warp.shape[1])]
                b01 = mask01[ii:min(ii + chess_sz, ima0_warp.shape[0]), jj:min(jj + chess_sz, ima1_warp.shape[1])]

                b0[b01 == 2] *= alpha0
                b1[b01 == 2] *= alpha1

                mask0[ii:min(ii + chess_sz, ima0_warp.shape[0]), jj:min(jj + chess_sz, ima0_warp.shape[1])] = b0
                mask1[ii:min(ii + chess_sz, ima1_warp.shape[0]), jj:min(jj + chess_sz, ima1_warp.shape[1])] = b1

        ima01_warp = (ima0_warp * np.expand_dims(mask0, -1) +
                      ima1_warp * np.expand_dims(mask1, -1)).astype(np.uint8)
        ima01_warp[:, :, 3] = (mask01 > 0) * 255

        cv2.imwrite(new_img01, ima01_warp)

        return {}
