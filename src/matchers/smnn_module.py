
import kornia as K
import torch

from core import device as global_device
from core import set_args, check_data, DONTCARE, NKEYPOINTS, NMATCHES


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
                                
        self.args = {
            'id_more': '',
            'th': 0.95,
            'use_keypt_mask': False,
            }
        self.device =  torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('smnn', args, self.args)
        if self.args['use_keypt_mask']:
            self.id_string += '_keypt_mask'

        self.required_input = {'desc': 2}
        self.required_output = {
            'm_idx':  [NMATCHES, 2],
            'm_val':  [NMATCHES],
            'm_mask': [NMATCHES],
        }


    def get_id(self): 
        return self.id_string
    

    def finalize(self):
        return


    def run(self, **args):
        check_data(args, self.required_input)

        desc0, desc1 = args['desc'][0], args['desc'][1]
        use_mask = self.args['use_keypt_mask'] and 'keypt_mask' in args
        if use_mask:
            idx0 = torch.nonzero(args['keypt_mask'][0].to(desc0.device)).squeeze(1)
            idx1 = torch.nonzero(args['keypt_mask'][1].to(desc1.device)).squeeze(1)
            desc0, desc1 = desc0[idx0], desc1[idx1]

        val, idxs = K.feature.match_smnn(desc0, desc1, self.args['th'])
        if use_mask:
            idxs = torch.stack([idx0[idxs[:, 0]], idx1[idxs[:, 1]]], dim=1)

        result = {'m_idx': idxs, 'm_val': val.squeeze(1), 'm_mask': torch.ones(idxs.shape[0], device=self.device, dtype=torch.bool)}
        check_data(result, self.required_output)
        return result

