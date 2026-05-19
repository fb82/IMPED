
import cv2
import numpy as np
import torch
from kornia_moons.feature import laf_from_opencv_kpts

from core import device as global_device
from core import laf2homo, set_args


class dog_module:
    """
    A feature detection module using the Difference of Gaussians (DoG) method.

    This module identifies keypoints by subtracting two blurred versions of 
    the same image. This highlights 'blobs' and corners at various scales. 
    It is the standard detector used in the SIFT framework and is highly 
    stable for 3D reconstruction and image alignment.

    Attributes:
        nfeatures (int): The number of top-ranked keypoints to retain 
            (default: 8000).
        upright (bool): If True, forces all keypoints to have an orientation 
            of 0 degrees, disabling rotation invariance.
        contrastThreshold: Filters out weak features in low-contrast regions.
    """
    def __init__(self, device=None, **args):
        self.device = device if device is not None else global_device
        self.single_image = True
        self.pipeliner = False                
        self.pass_through = False
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'upright': False,
            'params': {'nfeatures': 8000, 'contrastThreshold': -10000, 'edgeThreshold': 10000},
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('dog', args, self.args)
        self.detector = cv2.SIFT_create(**self.args['params'])


    def get_id(self): 
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):    
        
        im = cv2.imread(args['img'][args['idx']], cv2.IMREAD_GRAYSCALE)
        kp = self.detector.detect(im, None)


        if self.args['upright']:
            idx = np.unique(np.asarray([[k.pt[0], k.pt[1]] for k in kp]), axis=0, return_index=True)[1]
            kp = [kp[ii] for ii in idx]
            for ii in range(len(kp)):
                kp[ii].angle = 0       

        kr = []
        for i in range(len(kp)): kr.append(kp[i].response)
        kr = torch.tensor(kr, device=self.device, dtype=torch.float)
                
        kp = laf_from_opencv_kpts(kp, device=self.device)
        kp, kH = laf2homo(kp.detach().to(self.device).squeeze(0))
    
        return {'kp': kp, 'kH': kH, 'kr': kr}