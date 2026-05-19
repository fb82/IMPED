
import kornia as K

from core import device as global_device
from core import homo2laf, set_args


class deep_descriptor_module:
    """
    A module for extracting learned local descriptors using deep neural networks.

    This module takes existing keypoints (represented as Local Affine Frames) 
    and passes them through a specialized CNN (HardNet, SOSNet, or HyNet). 
    These networks are trained on millions of patches to be invariant to 
    extreme changes in lighting, perspective, and seasonal appearance.

    Attributes:
        descriptor (str): The CNN architecture to use ('hardnet', 'sosnet', 'hynet').
        desc_params (dict): Parameters passed to the specific network (e.g., pretrained=True).
        patch_params (dict): Parameters for patch extraction (e.g., patch_size).
    """
    def __init__(self, device=None, **args):
        self.device = device if device is not None else global_device
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False
        self.add_to_cache = True
                
        self.args = {
            'id_more': '',
            'descriptor': 'hardnet',
            'desc_params': {
                'pretrained': True,
                },
            'patch_params': {},
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('', args, self.args)        
        
        if self.args['descriptor'] == 'hardnet':
            base_string = 'hardnet'
            desc = K.feature.HardNet(**self.args['desc_params']).to(device)
        if self.args['descriptor'] == 'sosnet':
            desc = K.feature.SOSNet(**self.args['desc_params']).to(device)
            base_string = 'sosnet'
        if self.args['descriptor'] == 'hynet':
            desc = K.feature.HyNet(**self.args['desc_params']).to(device)
            base_string = 'hynet'

        self.ddesc = K.feature.LAFDescriptor(patch_descriptor_module=desc, **self.args['patch_params'])
        self.id_string = base_string + self.id_string


    def get_id(self): 
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):    
        im = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=self.device).unsqueeze(0)

        lafs = homo2laf(args['kp'][args['idx']], args['kH'][args['idx']])
        desc = self.ddesc(im, lafs).squeeze(0)
    
        return {'desc': desc}
