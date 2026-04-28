
import kornia as K
import torch

import hz.hz as hz
from core import laf2homo, set_args


class hz_module:
    """
    A high-precision feature detection module based on the Hessian-Affine algorithm.

    This module identifies stable 'blobs' in an image by analyzing the 
    Hessian matrix of the intensity values. It specifically targets 
    affine-invariant regions, meaning it can detect the same physical 
    surface even if the camera's viewpoint causes significant 
    geometric stretching or tilting.

    Attributes:
        plus (bool): If True, uses the 'hz_plus' variant (color/multichannel). 
            If False, uses standard grayscale Hessian detection.
        max_max_pts (int): The maximum number of keypoints to extract 
            (default: 8000).
        block_mem (int): Memory management parameter for GPU processing.
    """
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'plus': True,
            'params': {'max_max_pts': 8000, 'block_mem': 16*10**6},
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('' , args, self.args)
        if self.args['plus']:
            self.id_string = 'hz_plus' + self.id_string                
            self.hz_to_run = hz.hz_plus
        else:
            self.id_string = 'hz' + self.id_string                
            self.hz_to_run = hz.hz
        
    def get_id(self): 
        return self.id_string


    def finalize(self):
        return

    
    def run(self, **args):  
        if self.args['plus']:        
            img = hz.load_to_tensor(args['img'][args['idx']]).to(torch.float)
        else:
            img = hz.load_to_tensor(args['img'][args['idx']], grayscale=True).to(torch.float)

        kp, kr = self.hz_to_run(img, output_format='laf', **self.args['params'])
        kp, kH = laf2homo(K.feature.ellipse_to_laf(kp[None]).squeeze(0))

        return {'kp': kp, 'kH': kH, 'kr': kr.type(torch.float)}
