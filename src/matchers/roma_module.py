
import torch
from PIL import Image
from romatch import roma_indoor, roma_outdoor, tiny_roma_v1_outdoor

from core import device as global_device
from core import set_args

torch.set_float32_matmul_precision('highest')


class roma_module:
    """
    A robust dense matching module using a warping-based transformer.

    RoMa is designed to provide high-quality correspondences by combining 
    a coarse global matching stage with a fine-grained refinement stage. 
    It estimates a dense 'warp' field and a 'certainty' map, allowing it 
    to find matches even in challenging scenarios like large viewpoint 
    changes or extreme lighting conditions.

    Attributes:
        use_tiny (bool): If True, uses the lightweight 'TinyRoMa' variant 
            for faster processing on lower-end hardware.
        coarse_resolution (int): Resolution for the initial global match.
        upsample_resolution (int): Resolution for the final pixel-accurate refinement.
        max_keypoints (int): The number of points to sample from the dense warp field.
    """
    def __init__(self, **args):
        torch.set_float32_matmul_precision('highest')
        self.device = torch.device(self.args.get('device', str(global_device)))

        self.roma_model = roma_outdoor(device=self.device, **args)
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
                                
        self.args = {
            'id_more': '',
            'outdoor': True,
            'use_tiny': False,
            'coarse_resolution': 280,
            'upsample_resolution': 432,
            'max_keypoints': 2000,
            'patch_radius': 16,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('roma', args, self.args)        

        roma_args = {}
        roma_args['use_custom_corr'] = False
        if self.args['coarse_resolution'] is not None:
            roma_args['coarse_res'] = self.args['coarse_resolution']
        if self.args['upsample_resolution'] is not None:
            roma_args['upsample_res'] = self.args['upsample_resolution']

        if self.args['use_tiny']:
            self.roma_model = tiny_roma_v1_outdoor(device=self.device)            
        else:
            if self.args['outdoor'] == True:
                self.roma_model = roma_outdoor(device=self.device, **roma_args)
            else:
                self.roma_model = roma_indoor(device=self.device, **roma_args)


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):
        H, W = self.roma_model.get_output_resolution()

        im1 = Image.open(args['img'][0])
        im1 = im1.convert("RGB")

        im2 = Image.open(args['img'][1])
        im2 = im2.convert("RGB")

        W_A, H_A = im1.size
        W_B, H_B = im2.size

        im1 = im1.resize((W, H))
        im2 = im2.resize((W, H))
    
        # Match
        if self.args['use_tiny']:
            warp, certainty = self.roma_model.match(args['img'][0], args['img'][1])
        else:
            warp, certainty = self.roma_model.match(args['img'][0], args['img'][1])
        # Sample matches for estimation
        
        sampling_args = {}
        if self.args['max_keypoints'] is not None:
            sampling_args['num'] = self.args['max_keypoints']
        
        matches, certainty = self.roma_model.sample(warp, certainty, **sampling_args)
        kpts1, kpts2 = self.roma_model.to_pixel_coordinates(matches, H, W, H, W)    

        kps1 = kpts1.detach().to(self.device)
        kps2 = kpts2.detach().to(self.device)

        kps1 = kps1 / torch.tensor([W/float(W_A), H/float(H_A)], device=self.device).unsqueeze(0)
        kps2 = kps2 / torch.tensor([W/float(W_B), H/float(H_B)], device=self.device).unsqueeze(0)
        
        kp = [kps1, kps2]
        kH = [
            torch.zeros((kp[0].shape[0], 3, 3), device=self.device),
            torch.zeros((kp[0].shape[0], 3, 3), device=self.device),
            ]
        
        kH[0][:, [0, 1], 2] = -kp[0] / self.args['patch_radius']
        kH[0][:, 0, 0] = 1 / self.args['patch_radius']
        kH[0][:, 1, 1] = 1 / self.args['patch_radius']
        kH[0][:, 2, 2] = 1

        kH[1][:, [0, 1], 2] = -kp[1] / self.args['patch_radius']
        kH[1][:, 0, 0] = 1 / self.args['patch_radius']
        kH[1][:, 1, 1] = 1 / self.args['patch_radius']
        kH[1][:, 2, 2] = 1

        kr = [torch.full((kp[0].shape[0],), torch.nan, device=self.device), torch.full((kp[0].shape[0],), torch.nan, device=self.device)]        

        m_idx = torch.zeros((kp[0].shape[0], 2), device=self.device, dtype=torch.int)
        m_idx[:, 0] = torch.arange(kp[0].shape[0])
        m_idx[:, 1] = torch.arange(kp[0].shape[0])

        m_mask = torch.ones(m_idx.shape[0], device=self.device, dtype=torch.bool)

        m_val = certainty.detach().to(self.device)        

        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}
