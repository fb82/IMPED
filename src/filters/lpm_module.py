
import torch

import lpm.LPM as lpm
from core import device as global_device
from core import set_args


class lpm_module:
    """
    An outlier rejection module using Locality Preserving Matching.

    LPM is a non-parametric approach to feature matching. Instead of 
    fitting a global geometric model (like a Homography), it enforces 
    local neighborhood consistency. It is highly effective for:
    1. Non-rigid deformations (e.g., matching a flag waving in the wind).
    2. Scenes with heavy perspective distortion.
    3. High-speed filtering without the need for GPU-heavy deep learning.

    The algorithm determines inliers by checking if the neighbors of a 
    point in Image 1 map to the neighbors of the corresponding point 
    in Image 2.
    """
    def __init__(self, device=None, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
        self.add_to_cache = True
        self.device = device if device is not None else global_device
                        
        self.args = {
            'id_more': '',
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('lpm', args, self.args)        


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
        
        mask = lpm.LPM_filter(pt1.to('cpu').numpy(), pt2.to('cpu').numpy())        
        mask = torch.tensor(mask, device=self.device, dtype=torch.bool)
 
        aux = mm.clone()
        mm[aux] = mask
        
        return {'m_mask': mm}

