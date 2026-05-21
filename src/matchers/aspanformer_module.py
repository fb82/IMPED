

import argparse
import os
import shutil
import sys
import tarfile
import warnings

import gdown
import kornia as K
import numpy as np
import torch

from core import device as global_device
from core import set_args

conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'aspanformer'))

from aspanformer.src.ASpanFormer.aspanformer import ASpanFormer
from aspanformer.src.config.default import get_cfg_defaults as as_get_cfg_defaults
from aspanformer.src.utils.misc import lower_config as as_lower_config


def download_aspanformer(weight_path='../weights/aspanformer'):    
    file = 'weights_aspanformer.tar'    
    url = 'https://drive.google.com/file/d/1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k/view?usp=share_link'

    os.makedirs(os.path.join(weight_path, 'download'), exist_ok=True)   

    file_to_download = os.path.join(weight_path, 'download', file)    
    if not os.path.isfile(file_to_download):    
        gdown.download(url, file_to_download, fuzzy=True)
        
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extractall(weight_path)
            
        shutil.move(os.path.join(weight_path, 'weights', 'indoor.ckpt'), os.path.join(weight_path, 'indoor.ckpt'))
        shutil.move(os.path.join(weight_path, 'weights', 'outdoor.ckpt'), os.path.join(weight_path, 'outdoor.ckpt'))
        os.rmdir(os.path.join(weight_path, 'weights'))
         

class aspanformer_module:
    """
    A detector-free matching module using Asymmetric-Sampling Transformers.

    ASpanFormer improves upon models like LofTR by using an 'asymmetric 
    sampling' strategy. It adaptively adjusts the sampling span (the area 
    it looks at) based on the image content. This allows it to handle 
    large scale differences and extreme camera motions more effectively 
    than fixed-grid transformers.

    Attributes:
        outdoor (bool): Switch between models trained on outdoor (MegaDepth) 
            or indoor (ScanNet) datasets.
        resize (list, optional): Target resolution for processing (e.g., [1024, 1024]).
        patch_radius (int): Defines the local area size for the homography 
            metadata associated with each match.
    """
    def __init__(self, **args):

        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
        self.device = torch.device(self.args.get('device', str(global_device)))
                                
        self.args = {
            'id_more': '',
            'outdoor': True,
            'resize': None,                              # default [1024, 1024]
            'patch_radius': 16,
            }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('aspanformer', args, self.args)        

        download_aspanformer()

        if self.args['outdoor']:
            self.weights = os.path.join('../weights/aspanformer/outdoor.ckpt')
            self.config_path = os.path.join('aspanformer/configs/aspan/outdoor/aspan_test.py')
        else:
            self.weights = os.path.join('../weights/aspanformer/indoor.ckpt')
            self.config_path = os.path.join('aspanformer/configs/aspan/indoor/aspan_test.py')
            

        parser = argparse.ArgumentParser(description='AspanFormer online demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--weights_path', type=str, default=self.weights, help="Path to the checkpoint.")
        parser.add_argument('--config_path', type=str, default=self.config_path, help="Path to the config.")

        as_args = parser.parse_args()

        import glob
        # Find the aspanformer directory
        matches = glob.glob(os.path.join(conf_path, '**', 'aspanformer*'), recursive=True)
        print("aspanformer:", matches)

        # Also print conf_path so we know the base
        print("conf_path:", conf_path)
        print("contents:", os.listdir(conf_path))
        print(100*'-')

        
        import importlib.util
        import types

        aspanformer_dir = os.path.join(conf_path, '..', 'aspanformer')

        def register_aspanformer_src():
            src_path = os.path.join(aspanformer_dir, 'src')
            config_path = os.path.join(src_path, 'config')

            # Register 'src' namespace package
            src_module = types.ModuleType('src')
            src_module.__path__ = [src_path]
            src_module.__package__ = 'src'
            sys.modules['src'] = src_module

            # Register 'src.config' namespace package
            config_module = types.ModuleType('src.config')
            config_module.__path__ = [config_path]
            config_module.__package__ = 'src.config'
            sys.modules['src.config'] = config_module

            # Load and register 'src.config.default' from actual file
            default_file = os.path.join(config_path, 'default.py')
            spec = importlib.util.spec_from_file_location('src.config.default', default_file)
            default_module = importlib.util.module_from_spec(spec)
            sys.modules['src.config.default'] = default_module
            spec.loader.exec_module(default_module)

        if 'src.config.default' not in sys.modules:
            register_aspanformer_src()

        config = as_get_cfg_defaults()
        config.merge_from_file(as_args.config_path)
        _config = as_lower_config(config)


        self.matcher = ASpanFormer(config=_config['aspan'])
        state_dict = torch.load(as_args.weights_path, map_location='cpu', weights_only=False)['state_dict']
        self.matcher.load_state_dict(state_dict,strict=False)

        if self.device.type == 'cuda':        
            self.matcher.cuda()

        self.matcher.eval()
        
        self.first_warning = True

    
    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args): 
        if not self.first_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.run_actually(**args)
        else:
            return self.run_actually(**args)

    def run_actually(self, **args): 
        if self.first_warning:
            self.first_warning = False

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

        data = {
            "image0": image0.unsqueeze(0),    # LofTR works on grayscale images
            "image1": image1.unsqueeze(0),
        }
                  
        self.matcher(data)

        kps1 = data['mkpts0_f'].detach().to(self.device).squeeze()
        kps2 = data['mkpts1_f'].detach().to(self.device).squeeze()
        m_val = data['mconf'].detach().to(self.device)
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
