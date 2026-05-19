
import kornia as K

from core import device as global_device
from core import homo2laf, laf2homo, set_args
import torch


class patch_module:
    """
    A module for refining the orientation and affine shape of keypoints.

    This module takes initial keypoints and 'upgrades' them by estimating 
    their dominant orientation (rotation) and affine shape (tilt/shear). 
    By normalizing these geometric properties, descriptors extracted later 
    (like SIFT or HardNet) become much more robust to viewpoint changes.

    Attributes:
        orinet (bool): Uses a deep neural network (OriNet) to predict 
            the best orientation for the keypoint.
        affnet (bool): Uses a deep neural network (AffNet) to estimate 
            the local affine shape, effectively 'un-tilting' the image patch.
        sift_orientation (bool): Uses traditional gradient-based methods 
            to find the dominant rotation.
    """
    def __init__(self, device=None, **args):
        self.device = device if device is not None else global_device
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False
        self.add_to_cache = True
                
        self.args = {
            'id_more': '',
            'sift_orientation': False,
            'sift_orientation_params': {},
            'general_orientation_params': {},
            'orinet': True,
            'orinet_params': {
                'pretrained': True,
                },
            'affnet': True,
            'affnet_params': {
                'pretrained': True,
                },
            }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('', args, self.args)

        base_string = ''
        self.ori_module = K.feature.PassLAF()
        if self.args['sift_orientation']:
            base_string = 'sift_orientation'
            self.ori_module = K.feature.LAFOrienter(angle_detector=K.feature.PatchDominantGradientOrientation(**self.args['sift_orientation_params']), **self.args['general_orientation_params'])
        if self.args['orinet']:
            base_string = 'orinet'
            self.ori_module = K.feature.LAFOrienter(angle_detector=K.feature.OriNet(**self.args['orinet_params']).to(self.device), **self.args['general_orientation_params'])

        if self.args['affnet']:
            if len(base_string): base_string = base_string  + '_' + 'affnet'
            else: base_string = 'affnet'
            self.aff_module =  K.feature.LAFAffNetShapeEstimator(**self.args['affnet_params']).to(device)
        else:
            self.aff_module = K.feature.PassLAF()

        if not len(base_string): base_string = 'pass_laf'
        self.id_string = base_string + self.id_string


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):    
        import cv2
        import numpy as np

        try:
            im = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=self.device).unsqueeze(0)
        except FileExistsError as e:
            print(f"Error loading image {args['img'][args['idx']]}: {e}")
            # Fallback: try loading with OpenCV
            img_cv = cv2.imread(str(args['img'][args['idx']]), cv2.IMREAD_GRAYSCALE)
            if img_cv is not None:
                im = torch.from_numpy(img_cv).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.0
            else:
                raise RuntimeError(f"Failed to load image: {args['img'][args['idx']]}")


        lafs = homo2laf(args['kp'][args['idx']], args['kH'][args['idx']])

        lafs = self.aff_module(lafs, im)
        lafs = self.ori_module(lafs, im)

        kp, kH = laf2homo(lafs.squeeze(0))
    
        return {'kp': kp, 'kH': kH}
