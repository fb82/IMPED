import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import miho.src.miho as mop_miho
import miho.src.miho_other as mop
import miho.src.ncc as ncc
from core import device as global_device
from core import set_args


class mop_miho_ncc_module:
    """
    A high-level refinement module combining planar clustering and radiometric optimization.

    This module performs three primary tasks:
    1. Planar Clustering (MOP/MIHO): Grouping matches into spatially coherent planes 
       using homography-based motion partitioning.
    2. Local Affine/Projective Adjustment: Handling Local Affine Frames (LAF) to 
       account for perspective distortion.
    3. NCC Refinement: Fine-tuning keypoint positions and homographies by maximizing 
       the Normalized Cross Correlation between local image patches.

    Attributes:
        args (dict): Configuration dictionary including:
            - mop (bool): Enable Motion Partitioning (planar clustering).
            - miho (bool): Enable Multiple Image Homography Optimization.
            - ncc (bool): Enable Normalized Cross Correlation refinement.
            - ncc_todo (set): Types of NCC to perform ('eye', 'laf', 'mop_miho').
            - patch_radius (int): Scale used for patch extraction and kH construction.
            - affine_laf_miho (bool): Apply specific affine normalization to frames.
        mop (mop_miho.miho or mop.miho): The underlying clustering engine.

    Args:
        **args: Keyword arguments to override default configuration.
    """
    def __init__(self, device=None, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
        self.add_to_cache = True
        self.device = device if device is not None else global_device
                        
        self.args = {
            'id_more': '',
            'patch_radius': 16,
            'mop': True,
            'miho': True,
            'mop_miho_patches': True,
            'mop_miho_cfg': None,
            'ncc': True,
            'ncc_todo': None,          # ncc_to_do = {'eye', 'laf', 'mop_miho'}
            'ncc_cfg': None,
            'affine_laf_miho': False,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('', args, self.args)        

        id_prefix = ''
        if self.args['mop']: id_prefix = id_prefix + '_mop' 
        if self.args['miho']: id_prefix = id_prefix + '_miho' 
        if self.args['ncc']: id_prefix = id_prefix + '_ncc' 
        if id_prefix == '': id_prefix = 'no_mop_miho_ncc'
        self.id_string = id_prefix + self.id_string

        self.mop = None
        if self.args['mop']:
            if self.args['miho']: self.mop = mop_miho.miho()  
            else: self.mop = mop.miho()
        
            mop_miho_cfg = self.mop.get_current()
        
            if self.args['mop_miho_cfg'] is not None and isinstance(self.args['mop_miho_cfg'], dict):
                for k in self.args['mop_miho_cfg']:
                    mop_miho_cfg[k] = self.args['mop_miho_cfg'][k]

            self.mop.update_params(mop_miho_cfg)  
            
        if self.args['ncc_todo'] is None: self.args['ncc_todo'] = {'eye', 'laf', 'mop_miho'}
        if not self.args['ncc']: self.args['ncc_todo'] = {}

        if self.args['ncc_cfg'] is None:
            self.args['ncc_cfg'] = {
                'w': 10,
                'w_big': None,
                'angle': [-30, -15, 0, 15, 30],
                'scale': [[10/14, 1], [10/12, 1], [1, 1], [1, 12/10], [1, 14/10]],
                'subpix': True,
                'ref_image': 'both',
                'use_covariance': True,
                'centered_derivative': True,                
                }

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.PILToTensor() 
            ]) 

    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args):   
        """
        Executes the clustering and refinement pipeline.

        Processing Flow:
        1. Planar Clustering: If MOP is enabled, matches are clustered into planes. 
           Matches not belonging to a valid plane are masked out.
        2. Initial Warping: If NCC is disabled but 'mop_miho_patches' is enabled, 
           local homographies (kH) are updated based on the detected planes.
        3. NCC Refinement: 
           - 'eye': Refines based on a simple identity translation.
           - 'laf': Refines using the Local Affine Frames from keypoint detectors.
           - 'mop_miho': Refines using the homographies derived from planar clustering.
        4. Selection: For each match, the refinement method yielding the highest 
           NCC score (val_) is selected as the winner.
        5. Output Reconstruction: Merges refined matches with original unchanged 
           data into a unified pipeline format.

        Args:
            **args: Pipeline dictionary containing 'kp', 'kH', 'm_idx', 'm_mask', 
                    'kr', and 'img' paths.

        Returns:
            dict: Updated pipeline dictionary containing refined keypoints, 
                  homographies, and updated validity masks.
        """     
        from ensemble import pipe_union
        if self.mop is not None:
            mi = args['m_idx']                     
            mm = args['m_mask']

            # pad mm to match mi if sizes differ
            if mm.shape[0] < mi.shape[0]:
                mm_padded = torch.zeros(mi.shape[0], device=mm.device, dtype=torch.bool)
                mm_padded[:mm.shape[0]] = mm
                mm = mm_padded
        
            pt1 = args['kp'][0][mi[mm][:, 0]]
            pt2 = args['kp'][1][mi[mm][:, 1]]

            # params = self.mop.all_params()
            # params['go_assign']['method'] = mop_miho.cluster_assign_base
            # params['get_avg_hom']['min_plane_pts'] = 24
            # params['get_avg_hom']['min_pt_gap'] = 12
            # self.mop = mop_miho.miho(params)
            
            # self.mop.attach_images(Image.open(args['img'][0]),Image.open(args['img'][1]))

            lidx = torch.arange(mm.shape[0], device=self.device)[mm]
            Hs_mop_, Hidx = self.mop.planar_clustering(pt1, pt2)
            
            # self.mop.show_clustering()

            mask = Hidx > -1
            mm[lidx] = mask   
            
            if not len(self.args['ncc_todo']):
                if (not self.args['mop_miho_patches']) or (not len(Hs_mop_)):
                    return {'m_mask': mm}
                            
                kH0 = args['kH'][0]
                kH1 = args['kH'][1]
                
                lidx = lidx[Hidx > -1]                
                
                p1 = args['kp'][0][mi[lidx, 0]]
                p2 = args['kp'][1][mi[lidx, 1]]

                r = self.args['patch_radius']
                S = torch.tensor([[1/r, 0, 0],[0, 1/r, 0],[0, 0, 1]], device=self.device).unsqueeze(0).repeat(p1.shape[0], 1, 1)
                                
                if self.args['miho']:                
                    Ha = torch.stack([Hs_mop_[i][0] for i in range(len(Hs_mop_))], dim=0)
                    Hb = torch.stack([Hs_mop_[i][1] for i in range(len(Hs_mop_))], dim=0)
                else:
                    Ha = torch.stack([Hs_mop_[i][0][0] for i in range(len(Hs_mop_))], dim=0)
                    Hb = Ha.inverse()
                    
                H1 = Ha[Hidx[Hidx > -1]]
                H2 = Hb[Hidx[Hidx > -1]]
                
                p1_ = H1.bmm(torch.cat((p1, torch.ones((p1.shape[0], 1), device=self.device)), dim=1).unsqueeze(-1))
                p1_ = p1_ / p1_[:, 2].unsqueeze(-1)

                p2_ = H2.bmm(torch.cat((p2, torch.ones((p2.shape[0], 1), device=self.device)), dim=1).unsqueeze(-1))
                p2_ = p2_ / p2_[:, 2].unsqueeze(-1)

                T1 = torch.eye(3, device=self.device).unsqueeze(0).repeat(p1_.shape[0], 1, 1)
                T1[:, :2, 2] = -p1_[:, :2].squeeze(-1)

                T2 = torch.eye(3, device=self.device).unsqueeze(0).repeat(p2_.shape[0], 1, 1)
                T2[:, :2, 2] = -p2_[:, :2].squeeze(-1)

                kH0[mi[lidx, 0]] = S.bmm(T1).bmm(H1)
                kH1[mi[lidx, 1]] = S.bmm(T2).bmm(H2)
                
                return {'m_mask': mm, 'kH': [kH0, kH1]}
            
        if len(self.args['ncc_todo']):
            pt1 = args['kp'][0]
            pt2 = args['kp'][1]
    
            H1 = args['kH'][0]
            H2 = args['kH'][1]

            im1 = Image.open(args['img'][0])
            im2 = Image.open(args['img'][1])
    
            im1 = self.transform(im1).type(torch.float16).to(self.device)
            im2 = self.transform(im2).type(torch.float16).to(self.device)               
                    
            mi = args['m_idx']                     
            if self.mop is None: mm = args['m_mask']

            lidx = torch.arange(mm.shape[0], device=self.device)[mm]
            l = lidx.shape[0]
                    
            pt1_base = args['kp'][0][mi[lidx, 0]]
            pt2_base = args['kp'][1][mi[lidx, 1]]

            kr1 = args['kr'][0][mi[lidx, 0]]
            kr2 = args['kr'][1][mi[lidx, 1]]

            pt1_ = pt1_base
            pt2_ = pt2_base
            Hs_ = torch.eye(3, device=self.device).repeat(l * 2, 1).reshape(l, 2, 3, 3)
            T_ = torch.eye(3, device=self.device).repeat(l * 2, 1).reshape(l, 2, 3, 3)
            val_ = torch.full((l, ), -np.inf, device=self.device)
                        
        if ('eye' in self.args['ncc_todo']) and mm.sum():
            Hs_in = torch.eye(3, device=self.device).repeat(l * 2, 1).reshape(l, 2, 3, 3)
            
            pt1_eye, pt2_eye, Hs_eye, val_eye, T_eye = ncc.refinement_norm_corr_alternate(im1, im2, pt1_base, pt2_base, Hs_in, **self.args['ncc_cfg'], img_patches=False)   
            replace_idx = torch.argwhere((torch.cat((val_.unsqueeze(0),val_eye.unsqueeze(0)), dim=0)).max(dim=0)[1] == 1)
            pt1_[replace_idx] = pt1_eye[replace_idx]
            pt2_[replace_idx] = pt2_eye[replace_idx]
            Hs_[replace_idx] = Hs_eye[replace_idx]
            val_[replace_idx] = val_eye[replace_idx]
            T_[replace_idx] = T_eye.reshape(T_eye.shape[0] // 2, 2, 3, 3)[replace_idx]
                        
        if ('laf' in self.args['ncc_todo']) and mm.sum():
            r = self.args['patch_radius']
            S = torch.tensor([[r, 0, 0],[0, r, 0],[0, 0, 1.]], device=self.device).unsqueeze(0).repeat(l, 1, 1)

            kH1 = args['kH'][0][mi[lidx, 0]]
            kH2 = args['kH'][1][mi[lidx, 1]]

            p1_ = kH1.bmm(torch.cat((pt1_base, torch.ones((pt1_base.shape[0], 1), device=self.device)), dim=1).unsqueeze(-1))
            p1_ = p1_ / p1_[:, 2].unsqueeze(-1)

            p2_ = kH2.bmm(torch.cat((pt2_base, torch.ones((pt2_base.shape[0], 1), device=self.device)), dim=1).unsqueeze(-1))
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
    
            Hs_in = torch.stack((Z1, Z2), dim=1)

            pt1_laf, pt2_laf, Hs_laf, val_laf, T_laf = ncc.refinement_norm_corr_alternate(im1, im2, pt1_base, pt2_base, Hs_in, **self.args['ncc_cfg'])   
            replace_idx = torch.argwhere((torch.cat((val_.unsqueeze(0),val_laf.unsqueeze(0)), dim=0)).max(dim=0)[1] == 1)
            pt1_[replace_idx] = pt1_laf[replace_idx]
            pt2_[replace_idx] = pt2_laf[replace_idx]
            Hs_[replace_idx] = Hs_laf[replace_idx]
            val_[replace_idx] = val_laf[replace_idx]
            T_[replace_idx] = T_laf.reshape(T_laf.shape[0] // 2, 2, 3, 3)[replace_idx]
            
        if (self.mop is not None) and ('mop_miho' in self.args['ncc_todo']) and mm.sum():                        
            Hs_in = torch.zeros((l, 2, 3, 3), device=self.device)
                        
            Hidx_ = Hidx[Hidx > -1]
            for i in torch.arange(l):                           
                 Hs_in[i, 0] = Hs_mop_[Hidx_[i]][0]
                 Hs_in[i, 1] = Hs_mop_[Hidx_[i]][1]
                 
            pt1_mop, pt2_mop, Hs_mop, val_mop, T_mop = ncc.refinement_norm_corr_alternate(im1, im2, pt1_base, pt2_base, Hs_in, **self.args['ncc_cfg'], img_patches=False)   
            replace_idx = torch.argwhere((torch.cat((val_.unsqueeze(0),val_mop.unsqueeze(0)), dim=0)).max(dim=0)[1] == 1)
            pt1_[replace_idx] = pt1_mop[replace_idx]
            pt2_[replace_idx] = pt2_mop[replace_idx]
            Hs_[replace_idx] = Hs_mop[replace_idx]
            val_[replace_idx] = val_mop[replace_idx]
            T_[replace_idx] = T_mop.reshape(T_mop.shape[0] // 2, 2, 3, 3)[replace_idx]

        if len(self.args['ncc_todo']) and mm.sum():            
            pipe_unchanged = {
                'kp': args['kp'],
                'kr': args['kr'],
                'kH': args['kH'],
                'm_idx': args['m_idx'][~mm],
                'm_val': torch.full(((~mm).sum(),), np.nan, device=self.device, dtype=torch.bool),
                'm_mask': mm[~mm],
                }
    
            r = self.args['patch_radius']
            S = torch.tensor([[1/r, 0, 0],[0, 1/r, 0],[0, 0, 1]], device=self.device).unsqueeze(0).repeat(mm.sum(), 1, 1)
                            
            H1 = Hs_[:, 0]
            H2 = Hs_[:, 1]
                                    
            p1_ = H1.bmm(torch.cat((pt1_, torch.ones((pt1_.shape[0], 1), device=self.device)), dim=1).unsqueeze(-1))
            p1_ = p1_ / p1_[:, 2].unsqueeze(-1)
    
            p2_ = H2.bmm(torch.cat((pt2_, torch.ones((pt2_.shape[0], 1), device=self.device)), dim=1).unsqueeze(-1))
            p2_ = p2_ / p2_[:, 2].unsqueeze(-1)
    
            T1 = torch.eye(3, device=self.device).unsqueeze(0).repeat(p1_.shape[0], 1, 1)
            T1[:, :2, 2] = -p1_[:, :2].squeeze(-1)
    
            T2 = torch.eye(3, device=self.device).unsqueeze(0).repeat(p2_.shape[0], 1, 1)
            T2[:, :2, 2] = -p2_[:, :2].squeeze(-1)
    
            Hs1 = S.bmm(T1).bmm(H1)
            Hs2 = S.bmm(T2).bmm(H2)
    
            pipe_mod = {
                'kp': [pt1_, pt2_],
                'kr': [kr1, kr2],
                'kH': [Hs1, Hs2],
                'm_idx': torch.arange(pt1_.shape[0], device=self.device).unsqueeze(1).repeat(1, 2),
                'm_val': val_,
                'm_mask': mm[mm],
                }
    
            pipe_out = pipe_union([pipe_unchanged, pipe_mod], unique=True, no_unmatched=False, only_matched=False, sampling_mode=None, preserve_order=True, patch_matters=True)
            return pipe_out
        
        return {}

