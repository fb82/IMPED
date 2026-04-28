
import torch
from kornia_moons.feature import opencv_kpts_from_laf
from lightglue import ALIKED as lg_aliked
from lightglue import DISK as lg_disk
from lightglue import SIFT as lg_sift
from lightglue import DoGHardNet as lg_doghardnet
from lightglue import LightGlue as lg_lightglue
from lightglue import SuperPoint as lg_superpoint
from lightglue.utils import load_image as lg_load_image
from lightglue.utils import rbd as lg_rbd
from PIL import Image

from core import device, homo2laf, set_args


class lightglue_module:
    """
    A high-performance feature matcher based on the LightGlue architecture.

    LightGlue is a 'deep matcher' that uses a lightweight Transformer to 
    iteratively update the descriptors of keypoints based on their 
    spatial context and their similarity to points in the other image. 

    Unlike traditional matchers that use a simple distance ratio test, 
    LightGlue is adaptive: it can stop early if a match is easy, or 
    perform more iterations for difficult cases, making it both 
    faster and more accurate than its predecessors.

    Attributes:
        what (str): The type of feature being matched (e.g., 'superpoint', 
            'disk', 'aliked', 'sift').
        desc_cf (float): A scaling factor for descriptors, used to normalize 
            different feature types (e.g., set to 255 for SIFT).
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True
                
        self.what = 'superpoint'
        self.args = {
            'id_more': '',
            'num_features': 8000,
            'resize': 1024,           # this is default, set to None to disable
            'desc_cf': 1,                    # 255 to use R2S2 with what='sift'
            'aliked_model': "aliked-n16rot",          # default is "aliked-n16"
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        if 'what' in args:
            self.what = args['what']
            del args['what']

        self.id_string, self.args = set_args('lightglue', args, self.args)        

        if self.what == 'disk':            
            self.matcher = lg_lightglue(features='disk').eval().to(device)            
        elif self.what == 'aliked':            
            self.matcher = lg_lightglue(features='aliked').eval().to(device)            
        elif self.what == 'sift':            
            self.matcher = lg_lightglue(features='sift').eval().to(device)                            
        elif self.what == 'doghardnet':            
            self.matcher = lg_lightglue(features='doghardnet').eval().to(device)            
        else:   
            self.what = 'superpoint'
            self.matcher = lg_lightglue(features='superpoint').eval().to(device)            


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return
    
    
    def run(self, **args):           
        # dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
        # dict_keys(['matches0', 'matches1', 'matching_scores0', 'matching_scores1', 'stop', 'matches', 'scores', 'prune0', 'prune1'])

        width, height = Image.open(args['img'][0]).size
        sz1 = torch.tensor([width / 2, height / 2], device=device)

        width, height = Image.open(args['img'][1]).size
        sz2 = torch.tensor([width / 2, height / 2], device=device)

        feats1 = {'keypoints': args['kp'][0].unsqueeze(0), 'descriptors': args['desc'][0].unsqueeze(0) * self.args['desc_cf'], 'image_size': sz1.unsqueeze(0)} 
        feats2 = {'keypoints': args['kp'][1].unsqueeze(0), 'descriptors': args['desc'][1].unsqueeze(0) * self.args['desc_cf'], 'image_size': sz2.unsqueeze(0)} 
        
        if (self.what == 'sift') or (self.what == 'doghardnet'):
            lafs1 = homo2laf(args['kp'][0], args['kH'][0])
            lafs2 = homo2laf(args['kp'][1], args['kH'][1])
            
            kp1 = opencv_kpts_from_laf(lafs1)
            kp2 = opencv_kpts_from_laf(lafs2)

            feats1['oris'] = torch.tensor([kp.angle for kp in kp1], device=device).unsqueeze(0)
            feats2['oris'] = torch.tensor([kp.angle for kp in kp2], device=device).unsqueeze(0)

            feats1['scales'] = torch.tensor([kp.size for kp in kp1], device=device).unsqueeze(0)
            feats2['scales'] = torch.tensor([kp.size for kp in kp2], device=device).unsqueeze(0)
            
            
        matches12 = self.matcher({'image0': feats1, 'image1': feats2})
        feats1_, feats2_, matches12 = [lg_rbd(x) for x in [feats1, feats2, matches12]]

        idxs = matches12['matches'].squeeze(0)
        m_val = matches12['scores'].squeeze(0)

        if torch.numel(idxs) == 2:
            idxs = idxs.reshape(1, -1)
            m_val = m_val.reshape(1)
        
        m_mask = torch.ones(idxs.shape[0], device=device, dtype=torch.bool)
                    
        return {'m_idx': idxs, 'm_val': m_val, 'm_mask': m_mask}
    






class deep_joined_module:
    """
    A unified extractor for deep-learning-based keypoints and descriptors.

    This module acts as a factory, allowing you to switch between different 
    neural feature extractors (SuperPoint, DISK, ALIKED, etc.) using a 
    single interface. It handles image loading, resizing, and the 
    transformation of raw network outputs into the pipeline's standard 
    Local Affine Frame (LAF) format.

    Attributes:
        what (str): The specific extractor to use. Options include 
            'superpoint', 'disk', 'aliked', 'sift', or 'doghardnet'.
        num_features (int): The maximum number of keypoints to extract.
        resize (int): The maximum dimension for image scaling before 
            extraction (helps maintain VRAM and speed).
    """
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True
                                
        self.what = 'superpoint'
        self.args = { 
            'id_more': '',
            'patch_radius': 16,            
            'num_features': 8000,
            'resize': 1024,           # this is default, set to None to disable
            'aliked_model': "aliked-n16rot",          # default is "aliked-n16"
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        if 'what' in args:
            self.what = args['what']
            del args['what']
        
        self.id_string, self.args = set_args(self.what, args, self.args)        

        if self.what == 'disk':            
            self.extractor = lg_disk(max_num_keypoints=self.args['num_features']).eval().to(device)
        elif self.what == 'aliked':            
            self.extractor = lg_aliked(max_num_keypoints=self.args['num_features'], model_name=self.args['aliked_model']).eval().to(device)
        elif self.what == 'sift':            
            self.extractor = lg_sift(max_num_keypoints=self.args['num_features']).eval().to(device)
        elif self.what == 'doghardnet':            
            self.extractor = lg_doghardnet(max_num_keypoints=self.args['num_features']).eval().to(device)
        else:   
            self.what = 'superpoint'
            self.extractor = lg_superpoint(max_num_keypoints=self.args['num_features']).eval().to(device)


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):
        # dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])         
        img = lg_load_image(args['img'][args['idx']]).to(device)
        
        feats = self.extractor.extract(img, resize=self.args['resize'])
        kp = feats['keypoints'].squeeze(0)       
        desc = feats['descriptors'].squeeze(0)       

        kH = torch.zeros((kp.shape[0], 3, 3), device=device)        
        kH[:, [0, 1], 2] = -kp / self.args['patch_radius']
        kH[:, 0, 0] = 1 / self.args['patch_radius']
        kH[:, 1, 1] = 1 / self.args['patch_radius']
        kH[:, 2, 2] = 1

        kr = torch.full((kp.shape[0], ), torch.nan, device=device)        
        
        # todo: add feats['keypoint_scores'] as kr        
        return {'kp': kp, 'kH': kH, 'kr': kr, 'desc': desc}

