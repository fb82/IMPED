
import torch

import dtm.src.dtm as dtm
from core import device as global_device
from core import set_args


class dtm_module:
    """
    Interface module for the Dynamic Token Matching (DTM) algorithm.
    
    This class acts as a wrapper to manage the filtering and validation of 
    feature matches between image pairs. It utilizes the DTM algorithm to 
    identify spatially consistent matches and filter out geometric outliers.

    Attributes:
        single_image (bool): Flag indicating if the module operates on a single image context.
        pipeliner (bool): Flag for integration within a sequential processing pipeline.
        pass_through (bool): If True, the module allows data to pass without transformation.
        add_to_cache (bool): If True, enables caching of the computed results.
        args (dict): Configuration dictionary for DTM parameters (e.g., 'full_dtm', 'st').
        id_string (str): A unique identifier generated based on the specific configuration.

    Args:
        **args: Arbitrary keyword arguments used to configure the module. 
            Expected keys include 'full_dtm' (bool), 'show_progress' (bool), 
            'st' (list/tuple), and 'prepare_data' (callable).
    """    
    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
        self.add_to_cache = True
        self.device = torch.device(self.args.get('device', str(global_device)))
                        
        self.args = {
            'id_more': '',
            'full_dtm': True,
            'show_progress': False,
            'st': [1., 0.],
            'prepare_data': dtm.prepare_data_shaped,
            'only_spatial': False,
            'guided_matching': False,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('dtm', args, self.args)        


    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args):                
        match_data = {
            'img': args['img'],
            'kp': args['kp'],
            'm_idx': args['m_idx'],
            'm_val': args['m_val'].clone(),
            'm_mask': args['m_mask'].clone(),
            }

        if self.args['only_spatial']: match_data['m_val'][:] = 1.

        if self.args['guided_matching']: 
            match_data['m_val'][match_data['m_mask']] = 0.
            match_data['m_mask'][:] = True

        dtm_mask = dtm.dtm(match_data, show_in_progress=self.args['show_progress'], full_dtm=self.args['full_dtm'], st=self.args['st'], prepare_data=self.args['prepare_data'])
   
        return {'m_mask': torch.tensor(dtm_mask <= 0, dtype=torch.bool, device=self.device)}        
