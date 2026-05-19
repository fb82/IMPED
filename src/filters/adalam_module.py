

import torch
from kornia_moons.feature import opencv_kpts_from_laf
from PIL import Image

import adalam.adalam.adalam as adalam
from core import device as global_device
from core import homo2laf, set_args


class adalam_module:
    """
    A geometric outlier rejection module using Adaptive Local Affine Matching.

    AdaLAM filters matches by ensuring they are locally consistent with a 
    smooth affine transformation. It specifically checks:
    1. Orientation Consistency: Do the keypoints rotate by a similar amount?
    2. Scale Consistency: Is the zooming factor between images consistent?
    3. Spatial Coherence: Are neighboring points moving in a similar pattern?

    This module is highly effective for 'hand-crafted' features (like SIFT) 
    where local affine information (LAF) is available.

    Attributes:
        adalam_params (dict): Configuration for the filter, including:
            - orientation_difference_threshold: Max allowed rotation variance.
            - scale_rate_threshold: Max allowed scale change variance.
            - min_inliers: Minimum points required to validate a local region.
    """
    class adalamfilter_custom(adalam.AdalamFilter):
        def __init__(self, custom_config=None):         
            super().__init__(custom_config=custom_config)
            

        def match_and_filter(self, k1, k2, im1shape=None, im2shape=None, o1=None, o2=None, s1=None, s2=None, putative_matches=None, scores=None, mnn=None):    
            if s1 is None or s2 is None:
                if self.config['scale_rate_threshold'] is not None:
                    raise AttributeError("Current configuration considers keypoint scales for filtering, but scales have not been provided.\n"
                                         "Please either provide scales or set 'scale_rate_threshold' to None to disable scale filtering")
            if o1 is None or o2 is None:
                if self.config['orientation_difference_threshold'] is not None:
                    raise AttributeError(
                        "Current configuration considers keypoint orientations for filtering, but orientations have not been provided.\n"
                        "Please either provide orientations or set 'orientation_difference_threshold' to None to disable orientations filtering")
       
            if not self.config['force_seed_mnn']:
                mnn = None
    
            return self.filter_matches(k1, k2, putative_matches, scores, mnn, im1shape, im2shape, o1, o2, s1, s2)
        

    def __init__(self, device=None, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
        self.add_to_cache = True
        self.device = device if device is not None else global_device
                        
        self.args = {
            'id_more': '',
            'adalam_params': {
                    'area_ratio': 100,
                    'search_expansion': 4,
                    'ransac_iters': 128,
                    'min_inliers': 6,
                    'min_confidence': 200,
                    'orientation_difference_threshold': 30, 
                    'scale_rate_threshold': 1.5, 
                    'detected_scale_rate_threshold': 5, 
                    'refit': True, 
                    'force_seed_mnn': False,
                    'device': device,
                    'th': 0.8 **2,
                },   
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('adalam', args, self.args)        
        if self.args['adalam_params']['device'] is None:
            self.args['adalam_params']['device'] = self.device
        
        self.matcher = self.adalamfilter_custom(self.args['adalam_params'])


    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args):  
        sz1 = Image.open(args['img'][0]).size
        sz2 = Image.open(args['img'][1]).size

        k1 = args['kp'][0]
        k2 = args['kp'][1]
        
        lafs1 = homo2laf(args['kp'][0], args['kH'][0])
        lafs2 = homo2laf(args['kp'][1], args['kH'][1])
        
        kp1 = opencv_kpts_from_laf(lafs1)
        kp2 = opencv_kpts_from_laf(lafs2)

        o1 = torch.tensor([kp.angle for kp in kp1], device=self.device)
        o2 = torch.tensor([kp.angle for kp in kp2], device=self.device)

        s1 = torch.tensor([kp.size for kp in kp1], device=self.device)
        s2 = torch.tensor([kp.size for kp in kp2], device=self.device)

        mi = args['m_idx']
        mm = args['m_mask']
        mv = args['m_val']

        m12 = mi[mm]
        scores = mv[mm]
        
        k1 = k1[m12[:, 0]]
        k2 = k2[m12[:, 1]]

        s1 = s1[m12[:, 0]]
        s2 = s2[m12[:, 1]]

        o1 = o1[m12[:, 0]]
        o2 = o2[m12[:, 1]]
        
        puta_match = torch.arange(0, m12.shape[0], device=self.device)
        mask = self.matcher.match_and_filter(k1, k2, im1shape=sz1, im2shape=sz2, o1=o1, o2=o2, s1=s1, s2=s2, putative_matches=puta_match, scores=scores, mnn=None)
        
        mask_aux = torch.zeros(m12.shape[0], device=self.device, dtype=torch.bool)
        mask_aux[mask[:, 0]] = True
         
        aux = mm.clone()
        mm[aux] = mask_aux
        
        return {'m_mask': mm}

