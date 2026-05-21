import copy
import os

import cv2
import torch
import wget

from core import device as global_device
from core import set_args


def download_fcgnn(weight_path='../weights/fcgnn'):
    """
    Downloads the pre-trained FCGNN model weights for sub-pixel refinement.

    This function facilitates the setup of the refinement stage by:
    1. Ensuring the local directory structure exists for weight storage.
    2. Downloading the 'fcgnn.model' binary file directly from the 
       official GitHub release assets.
    3. Using 'wget' for a reliable stream-based download.
    4. Skipping the process if the model file is already detected locally 
       to minimize network overhead.

    Args:
        weight_path (str): The destination directory for the model file.
            Defaults to '../weights/fcgnn'.

    Returns:
        None: Performs a network download and file-system write.
    """
    file = 'fcgnn.model'    
    url = "https://github.com/xuy123456/fcgnn/releases/download/v0/fcgnn.model"

    os.makedirs(weight_path, exist_ok=True)   

    file_to_download = os.path.join(weight_path, file)    
    if not os.path.isfile(file_to_download):    
        wget.download(url, file_to_download)


import fcgnn.fcgnn as fcgnn


class fcgnn_module:
    """
    A refinement module using Graph Neural Networks for sub-pixel accuracy.

    FCGNN treats a set of matches as a graph and uses local image patches 
    around each keypoint to predict an 'offset' (dx, dy). This corrects 
    slight misalignments caused by the initial detector or noise. It 
    simultaneously outputs a confidence score to perform final outlier 
    filtering.

    Attributes:
        thd (float): Confidence threshold (0.0 to 1.0) for keeping a match. 
            Default is very strict (0.999).
        min_matches (int): Minimum number of matches to return; if scores 
            are low, it will force-pick the top-K best matches.
        fcgnn_refiner (fcgnn_custom): The internal GNN architecture with 
            attention layers.
    """
    class fcgnn_custom(fcgnn.GNN):
        def __init__(self, depth=9):
            torch.nn.Module.__init__(self)
    
            in_dim, r, self.n = 256, 20, 8
            AttnModule = fcgnn.Attention(in_dim=in_dim, num_heads=8)
            self.layers = torch.nn.ModuleList([copy.deepcopy(AttnModule) for _ in range(depth)])
    
            self.embd_p = torch.nn.Sequential(fcgnn.BasicBlock(self.n*4, in_dim, torch.nn.Tanh()))
            self.embd_f = torch.nn.Sequential(fcgnn.BasicBlock((2*r+1)**2*2, 3*in_dim), torch.nn.LayerNorm(3*in_dim), fcgnn.BasicBlock(3*in_dim, in_dim))
    
            self.extract = fcgnn.ExtractPatch(r)
    
            self.mlp_s = torch.nn.Sequential(fcgnn.OutBlock(in_dim, 1), torch.nn.Sigmoid())
            self.mlp_o = torch.nn.Sequential(fcgnn.OutBlock(in_dim, 2))
                
            download_fcgnn()
            local_path = '../weights/fcgnn/fcgnn.model'    
            self.load_state_dict(torch.load(local_path, map_location='cpu', weights_only=False)) 

                
        def optimize_matches_custom(self, img1, img2, matches, thd=0.999, min_matches=10):
    
            if len(matches.shape) == 2:
                matches = matches.unsqueeze(0)
    
            matches = matches.round()
            offsets, scores = self.forward(img1, img2, matches)
            matches[:, :, 2:] = matches[:, :, 2:] + offsets[:, :, [1, 0]]
            mask = scores[0] > thd
            
            if mask.sum() < min_matches:
                mask_i = scores[0].topk(k=min(matches.shape[1], min_matches))
                mask[mask_i[1]] = True
                    
            return matches[0].detach(), mask                
                

    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
        self.add_to_cache = True
        self.device = torch.device(self.args.get('device', str(global_device)))
                        
        self.args = {
            'id_more': '',
            'thd': 0.999,
            'min_matches': 10,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('fcgnn', args, self.args)     
        self.fcgnn_refiner = self.fcgnn_custom().to(self.device)        


    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args): 
        img1 = cv2.imread(args['img'][0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(args['img'][1], cv2.IMREAD_GRAYSCALE)
        
        img1_ = torch.tensor(img1.astype('float32') / 255.)[None, None].to(self.device)
        img2_ = torch.tensor(img2.astype('float32') / 255.)[None, None].to(self.device)        
        
        mi = args['m_idx']
        mm = args['m_mask']

        m12 = mi[mm]

        k1 = args['kp'][0]
        k2 = args['kp'][1]
        
        k1 = k1[m12[:, 0]]
        k2 = k2[m12[:, 1]]
        
        matches = torch.hstack((k1, k2))

        matches_refined, mask = self.fcgnn_refiner.optimize_matches_custom(img1_, img2_, matches, thd=self.args['thd'], min_matches=self.args['min_matches']) 

        aux = mm.clone()
        mm[aux] = mask
        
        k1 = matches_refined[mask, :2]
        k2 = matches_refined[mask, 2:]

        kp1 = args['kp'][0]
        kp2 = args['kp'][1]

        m12 = mi[mm]

        kp1[m12[:, 0]] = k1        
        kp2[m12[:, 1]] = k2        
        
        kp = [kp1, kp2]
        
        # masked keypoints are refined too but the patch shape remain the same!
        return {'kp': kp, 'm_mask': mm}
