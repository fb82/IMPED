
import kornia as K
import torch

from core import device as global_device
from core import set_args


class smnn_module:
    """
    A feature matching module using Symmetric Mutual Nearest Neighbors (SMNN).

    SMNN is a strict filtering strategy. For a pair of points (A, B) to be 
    considered a match, two conditions must be met:
    1. B must be the closest neighbor to A in the second image.
    2. A must be the closest neighbor to B in the first image.

    This 'double-check' significantly reduces the number of false 
    positives compared to a simple one-way nearest neighbor search.

    Attributes:
        th (float): The distance threshold (ratio test). Only matches 
            with a distance ratio better than this value are kept.
    """
    def __init__(self, **args):
        self.single_image = False    
        self.pipeliner = False      
        self.pass_through = False
        self.add_to_cache = True
        self.device = torch.device(self.args.get('device', str(global_device)))
                                
        self.args = {
            'id_more': '',
            'th': 0.95,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('smnn', args, self.args)        


    def get_id(self): 
        return self.id_string
    

    def finalize(self):
        return


    def run(self, **args):
        val, idxs = K.feature.match_smnn(args['desc'][0], args['desc'][1], self.args['th'])

        return {'m_idx': idxs, 'm_val': val.squeeze(1), 'm_mask': torch.ones(idxs.shape[0], device=self.device, dtype=torch.bool)}

