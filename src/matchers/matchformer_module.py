import argparse
import os
import shutil
import sys

import gdown
import kornia as K
import numpy as np
import torch

from core import device as global_device
from core import set_args

conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'matchformer'))

from matchformer.config.defaultmf import get_cfg_defaults as mf_get_cfg_defaults
from matchformer.model.matchformer import Matchformer
from matchformer.model.utils.misc import lower_config as mf_lower_config


def download_matchformer(weight_path='../weights/matchformer'):    
    file_list = [
        'indoor-large-SEA.ckpt',
        'indoor-lite-LA.ckpt',
        'outdoor-large-LA.ckpt',
        'outdoor-lite-SEA.ckpt',
    ]
    
    url_list = [
        'https://drive.google.com/file/d/1EjeSvU3ARZg5mn2PlqNDWMu9iwS7Zf_m/view?usp=drive_link',
        'https://drive.google.com/file/d/11ClOQ_VrlsT7PxK6jQr5AW1Fd0YMbB3R/view?usp=drive_link',
        'https://drive.google.com/file/d/1Ii-z3dwNwGaxoeFVSE44DqHdMhubYbQf/view?usp=drive_link',
        'https://drive.google.com/file/d/1etaU9mM8bGT2AKT56ph6iqUdpV1daFBz/view?usp=drive_link',
    ]

    os.makedirs(weight_path, exist_ok=True)   

    for file, url in zip(file_list, url_list):

        file_to_download = os.path.join(weight_path, file)    
        if not os.path.isfile(file_to_download):    
            gdown.download(url, file_to_download, fuzzy=True)

class matchformer_module:
    """
    A dense matching module using MatchFormer (Interleaving Attention).

    MatchFormer treats the image matching task as a dual-softmax problem 
    using a hierarchical transformer. It interleaves self-attention 
    (understanding an image's own context) and cross-attention 
    (finding similarities between the two images) at multiple scales.

    Attributes:
        model (str): The specific architecture variant (e.g., 'outdoor-large-LA').
        resize (list, optional): Target resolution for processing.
        patch_radius (int): Scaling factor for the local homography metadata.
    """
    def __init__(self, device=None, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
        self.device = device if device is not None else global_device
                                
        self.args = {
            'id_more': '',
            'model': 'outdoor-large-LA',
            'resize': None,                          # self.resize = [800, 600]
            'patch_radius': 16,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('matchformer', args, self.args)        

        download_matchformer()
        cfgs = os.listdir('../data/matchfomer_cfgs')
        for fname in cfgs:
            dst = os.path.join('matchformer/config', fname)
            src = os.path.join('../data/matchfomer_cfgs', fname)
            if not os.path.isfile(dst):
                shutil.copy(src, dst)

        self.weights = os.path.join('../weights/matchformer/', self.args['model'] + '.ckpt')
        self.config_path = os.path.join('matchformer/config', self.args['model'] + '.py')
        self.base_cfg = 'matchformer/config/defaultmf.py'


        parser = argparse.ArgumentParser(description='MatchFormer online demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--weight', type=str, default=self.weights, help="Path to the checkpoint.")
        parser.add_argument('--config_path', type=str, default=self.base_cfg, help="Path to the config.")
        parser.add_argument('--config_path_other', type=str, default=self.config_path, help="Path to the config.")

        opt = parser.parse_args()
    
        # init default-cfg and merge it with the main- and data-cfg
        config = mf_get_cfg_defaults()
        config.merge_from_file(opt.config_path_other)
        _config = mf_lower_config(config)
    
        # Matcher: LoFTR
        self.matcher = Matchformer(config=_config['matchformer'])        
        self.matcher.load_state_dict({k.replace('matcher.',''):v  for k,v in torch.load(opt.weight, map_location='cpu', weights_only=False).items()})

        self.matcher.eval()
        
        if self.device.type == 'cuda':        
            self.matcher.to('cuda')

    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):
        image0 = K.io.load_image(args['img'][0], K.io.ImageLoadType.GRAY32, device=self.device)
        image1 = K.io.load_image(args['img'][1], K.io.ImageLoadType.GRAY32, device=self.device)

        hw1 = image0.shape[1:]
        hw2 = image1.shape[1:]

        if self.args['resize'] is not None:        
            ms = min(self.args['resize'])
            Ms = max(self.args['resize'])

            if hw1[0] > hw1[1]:
                sz = [Ms, ms]                
                ratio_ori = float(hw1[0]) / hw1[1]                 
            else:
                sz = [ms, Ms]
                ratio_ori = float(hw1[1]) / hw1[0]

            ratio_new = float(Ms) / ms
            if np.abs(ratio_ori - ratio_new) > np.abs(1 - ratio_new):
                sz = [ms, ms]

            image0 = K.geometry.resize(image0, (sz[0], sz[1]), antialias=True)

            if hw2[0] > hw2[1]:
                sz = [Ms, ms]                
                ratio_ori = float(hw2[0]) / hw2[1]                 
            else:
                sz = [ms, Ms]
                ratio_ori = float(hw2[1]) / hw2[0]

            ratio_new = float(Ms) / ms
            if np.abs(ratio_ori - ratio_new) > np.abs(1 - ratio_new):
                sz = [ms, ms]

            image1 = K.geometry.resize(image1, (sz[0], sz[1]), antialias=True)
                    
        hw1_ = image0.shape[1:]
        hw2_ = image1.shape[1:]

        batch = {
            "image0": image0.unsqueeze(0),    # LofTR works on grayscale images
            "image1": image1.unsqueeze(0),
        }

        self.matcher(batch)
        kps1 = batch['mkpts0_f'].detach().to(self.device).squeeze()
        kps2 = batch['mkpts1_f'].detach().to(self.device).squeeze()
        m_val = batch['mconf'].detach().to(self.device)
        m_mask = m_val > 0

        kps1[:, 0] = kps1[:, 0] * (hw1[1] / float(hw1_[1]))
        kps1[:, 1] = kps1[:, 1] * (hw1[0] / float(hw1_[0]))
    
        kps2[:, 0] = kps2[:, 0] * (hw2[1] / float(hw2_[1]))
        kps2[:, 1] = kps2[:, 1] * (hw2[0] / float(hw2_[0]))
        
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

        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}

