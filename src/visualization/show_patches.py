import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import miho.src.ncc as ncc
from core import device as global_device
from core import set_args


class show_patches_module:
    """
    A visualization module for extracting and saving matched image patches.

    This module takes matched keypoints and their associated local homographies 
    to extract 'rectified' patches from both images. These patches are normalized 
    to account for lighting and perspective differences, then saved as grids 
    to help users visually assess if the matches are correct.

    Attributes:
        args (dict): Configuration parameters including:
            - patch_radius (int): Scaling factor for the patch extraction.
            - w (int): Half-width of the resulting patch (total size = 2w + 1).
            - show_mode (set): Determines visualization style ('overlay', 'separated', or 'both').
            - cache_path (str): Directory where patch grid images are saved.
            - only_valid (bool): If True, only extracts patches for matches marked as valid.
            - affine_laf_miho (bool): If True, applies an affine normalization 
              based on Local Affine Frames.

    Methods:
        go_save_diff_patches: Static method that creates a two-channel 'diff' 
            image (Red/Green) to show the overlay/alignment of two patches.
    """
    @staticmethod
    def go_save_diff_patches(im1, im2, pt1, pt2, Hs, w, save_prefix='patch_diff_', stretch=False, grid=[40, 50], save_suffix='.png'):        
        # warning image must be grayscale and not rgb!
    
        pt1_, pt2_, _, Hi1, Hi2 = ncc.get_inverse(pt1, pt2, Hs) 
                
        patch1 = ncc.patchify(im1, pt1_, Hi1, w)
        patch2 = ncc.patchify(im2, pt2_, Hi2, w)
        
        for k in range(pt1.shape[0]):
            pp = patch1[k]
            pm = torch.isfinite(pp)
            if pp[pm].numel() == 0:
                continue  # or handle the empty case however makes sense (e.g., m_ = 0)
            m_ = pp[pm].min()
            M_ = pp[pm].max()
            pp[pm] = (pp[pm] - m_) / (M_ - m_)            
            patch1[k] = pp * 255        
        
            pp = patch2[k]
            pm = torch.isfinite(pp)
            if pp[pm].numel() == 0:
                continue  # or handle the empty case however makes sense (e.g., m_ = 0)
            m_ = pp[pm].min()
            M_ = pp[pm].max()
            pp[pm] = (pp[pm] - m_) / (M_ - m_)            
            patch2[k] = pp * 255       
        
        mask1 = torch.isfinite(patch1) & (~torch.isfinite(patch2))
        patch2[mask1] = 0
    
        mask2 = torch.isfinite(patch2) & (~torch.isfinite(patch1))
        patch1[mask2] = 0
    
        both_patches = torch.zeros((3, patch1.shape[0], patch1.shape[1], patch1.shape[2]), dtype=torch.float32, device=global_device)
        both_patches[0] = patch1
        both_patches[1] = patch2
    
        ncc.save_patch(both_patches, save_prefix=save_prefix, save_suffix=save_suffix, stretch=stretch, grid=grid)


    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True
        self.add_to_cache = True
        self.device = torch.device(self.args.get('device', str(global_device)))

        self.args = {
            'id_more': '',
            'img_prefix': '',
            'img_suffix': '',
            'cache_path': 'show_imgs',
            'prepend_pair': True,
            'ext': '.png',
            'force': False,
            'grid': [40, 50],
            'stretch': True,
            'max_patches': np.inf,
            'only_valid': True,
            'show_mode': {'overlay', 'separated'},
            'patch_radius': 16,
            'w': 10,
            'affine_laf_miho': False,
        }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                        
        self.id_string, self.args = set_args('show_patches' , args, self.args)

        self.transform_gray = transforms.Compose([
            transforms.Grayscale(),
            transforms.PILToTensor() 
            ]) 

        self.transform = transforms.PILToTensor() 


    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

    
    def run(self, **args):   
        """
        Executes the patch extraction and saving pipeline.

        The process follows these steps:
        1. Filters matches based on validity and confidence scores.
        2. Computes the final warping matrices (Hs) by combining keypoint 
           locations with local patch homographies (kH).
        3. Optional: Normalizes the Local Affine Frames (LAF) to ensure 
           patches from both images have a consistent scale.
        4. Warps the images using Normalized Cross Correlation (ncc) utilities 
           to produce square patches.
        5. Saves 'separated' grids (Image 0 patches and Image 1 patches separately) 
           or 'overlay' grids (Image 0 and 1 combined in color channels).

        Args:
            **args: Dictionary containing input data:
                - img (list[str]): Paths to the source images.
                - kp (list[Tensor]): Keypoint coordinates.
                - kH (list[Tensor]): Local homography matrices.
                - m_idx (Tensor): Match indices.
                - m_mask (Tensor): Validity mask for matches.
                - m_val (Tensor): Confidence values used for sorting/ranking.

        Returns:
            dict: An empty dictionary (this module is used for side-effects/saving files).
        """  
        img0 = args['img'][0]
        img1 = args['img'][1]
            
        im0 = os.path.splitext(os.path.split(img0)[1])[0]
        im1 = os.path.splitext(os.path.split(img1)[1])[0]

        if self.args['prepend_pair']:            
            cache_path = os.path.join(self.args['cache_path'], im0 + '_' + im1)
        else:
            cache_path = self.args['cache_path']
                
        new_img0_prefix = os.path.join(cache_path, self.args['img_prefix'] + im0 + '_')
        new_img1_prefix = os.path.join(cache_path, self.args['img_prefix'] + im1 + '_')
        new_img01_prefix = os.path.join(cache_path, self.args['img_prefix'] + im0 + '_' + im1 + '_')
        new_img_suffix = self.args['img_suffix'] + self.args['ext']
        
        if 'm_idx' not in args: return {}
                
        mi = args['m_idx']                     
        mm = args['m_mask']

        lidx = torch.arange(mm.shape[0], device=self.device)
        if self.args['only_valid']: lidx = lidx[mm]
                
        pt1 = args['kp'][0][mi[lidx, 0]]
        pt2 = args['kp'][1][mi[lidx, 1]]

        H1 = args['kH'][0][mi[lidx, 0]]
        H2 = args['kH'][1][mi[lidx, 1]]

        v = args['m_val'][lidx]
        
        if len(v) == 0: return {}
        
        zidx = v.argsort(descending=True)
        zidx = zidx[:min(len(zidx), self.args['max_patches'])]

        pt1 = pt1[zidx]
        pt2 = pt2[zidx]

        kH1 = H1[zidx]
        kH2 = H2[zidx]

        run_separated = True if ('separated' in self.args['show_mode']) or ('both' in self.args['show_mode']) else False
        run_overlay = True if ('overlay' in self.args['show_mode']) or ('both' in self.args['show_mode']) else False

        if run_separated or run_overlay: os.makedirs(cache_path, exist_ok=True)
        
        l = len(zidx)       

        r = self.args['patch_radius']
        S = torch.tensor([[r, 0, 0],[0, r, 0],[0, 0, 1.]], device=self.device).unsqueeze(0).repeat(l, 1, 1)

        p1_ = kH1.bmm(torch.cat((pt1, torch.ones((pt1.shape[0], 1), device=self.device)), dim=1).unsqueeze(-1))
        p1_ = p1_ / p1_[:, 2].unsqueeze(-1)

        p2_ = kH2.bmm(torch.cat((pt2, torch.ones((pt2.shape[0], 1), device=self.device)), dim=1).unsqueeze(-1))
        p2_ = p2_ / p2_[:, 2].unsqueeze(-1)

        T1 = torch.eye(3, device=self.device).unsqueeze(0).repeat(p1_.shape[0], 1, 1)
        T1[:, :2, 2] = p1_[:, :2].squeeze(-1)

        T2 = torch.eye(3, device=self.device).unsqueeze(0).repeat(p2_.shape[0], 1, 1)
        T2[:, :2, 2] = p2_[:, :2].squeeze(-1)

        Z1 = T1.bmm(S).bmm(kH1)
        Z2 = T2.bmm(S).bmm(kH2)
        
        if self.args['affine_laf_miho']:                
            N1 = Z1 / Z1[:, 2, 2].unsqueeze(1).unsqueeze(2)
            N2 = Z2 / Z2[:, 2, 2].unsqueeze(1).unsqueeze(2)

            is_affine = (N1[:, 2, :2].abs().sum(dim=1) < 1.0e-8) & (N2[:, 2, :2].abs().sum(dim=1) < 1.0e-8)

            s1 = (N1[:, 0, 0] * N1[:, 1, 1] - N1[:, 0, 1] * N1[:, 1, 0]) ** 0.5 
            s2 = (N2[:, 0, 0] * N2[:, 1, 1] - N2[:, 0, 1] * N2[:, 1, 0]) ** 0.5 

            s1[~is_affine] = 1 
            s2[~is_affine] = 1
            
            s12 = (s1 * s2) ** 0.5

            Z1[:, :2, :] = Z1[:, :2, :] / s12.unsqueeze(1).unsqueeze(2)
            Z2[:, :2, :] = Z2[:, :2, :] / s12.unsqueeze(1).unsqueeze(2) 

        Hs = torch.stack((Z1, Z2), dim=1)

        if run_separated:
            pt1_, pt2_, _, Hi1, Hi2 = ncc.get_inverse(pt1, pt2, Hs) 
                    
            ima0 = self.transform(Image.open(img0)).type(torch.float16).to(self.device)
            ima1 = self.transform(Image.open(img1)).type(torch.float16).to(self.device)

            patch1 = ncc.patchify(ima0, pt1_, Hi1, self.args['w'])
            patch2 = ncc.patchify(ima1, pt2_, Hi2, self.args['w'])
        
            ncc.save_patch(patch1, save_prefix=new_img0_prefix, save_suffix=new_img_suffix, grid=self.args['grid'], stretch=self.args['stretch'])
            ncc.save_patch(patch2, save_prefix=new_img1_prefix, save_suffix=new_img_suffix, grid=self.args['grid'], stretch=self.args['stretch'])

        if run_overlay:
            ima0 = self.transform_gray(Image.open(img0)).type(torch.float16).to(self.device)
            ima1 = self.transform_gray(Image.open(img1)).type(torch.float16).to(self.device)

            self.go_save_diff_patches(ima0, ima1, pt1, pt2, Hs, self.args['w'], save_prefix=new_img01_prefix, stretch=self.args['stretch'], grid=self.args['grid'], save_suffix=new_img_suffix)

        return {}

