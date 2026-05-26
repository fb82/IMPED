import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image

from core import device as global_device
from core import set_args

conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'mast3r'))

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference as mast3r_inference
from dust3r.utils.image import load_images as mast3r_load_images
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R


class mast3r_module:
    """
    A pipeline module using the MASt3R model for fast, 3D-aware feature matching.

    MASt3R extends DUSt3R by outputting dense local descriptors for each pixel. 
    This module performs inference to obtain these descriptors and uses a 
    Fast Reciprocal Nearest Neighbor search to establish matches, skipping 
    the expensive global point cloud alignment used in standard DUSt3R.

    Attributes:
        args (dict): Configuration parameters including:
            - model (str): HuggingFace model path (default: MASt3R ViT Large).
            - max_matches (int): Maximum number of matches to subsample.
            - resize (int): Image resolution for inference (default: 512).
            - patch_radius (int): Scale factor for the local patch homographies (kH).
        model (AsymmetricMASt3R): The loaded MASt3R neural network.

    Args:**args: Keyword arguments to override default settings.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
                                
        self.args = {
            'id_more': '',
            'model': 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
            'max_matches': 2048,
            'schedule': 'cosine',
            'lr': 0.01,
            'niter': 300,
            'resize': 512, 
            'patch_radius': 16,
            }
        self.device =  torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('mast3r', args, self.args)        

        # you can put the path to a local checkpoint in model_name if needed
        self.model = AsymmetricMASt3R.from_pretrained(self.args['model']).to(self.device)
        
        
    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        warnings.simplefilter(action='always', category=FutureWarning)        
        
        return


    def run(self, **args):   
        """
        Executes the MASt3R matching pipeline on a pair of images.

        The process follows these steps:
        1. Load and resize the image pair to the model's native resolution.
        2. Perform inference to extract view-specific descriptors.
        3. Use `fast_reciprocal_NNs` to find matches based on descriptor dot-product similarity.
        4. Apply a border-rejection filter to remove matches too close to image edges.
        5. Subsample matches if they exceed 'max_matches'.
        6. Rescale 2D keypoints back to the original image dimensions.
        7. Compute local patch homographies (kH) centered at each keypoint.

        Args:
            **args: Dictionary containing the input data:
                - img (list[str]): Paths to the two images to be matched.

        Returns:
            dict: A dictionary containing:
                - kp (list[Tensor]): Keypoint coordinates for both images.
                - kH (list[Tensor]): Patch-based homography matrices ($3 \times 3$).
                - kr (list[Tensor]): Placeholders for rotation values (NaN).
                - m_idx (Tensor): Identity mapping (0->0, 1->1) for established matches.
                - m_val (Tensor): Confidence values (all True).
                - m_mask (Tensor): Boolean validity mask (all True).
        """     
        warnings.simplefilter(action='ignore', category=FutureWarning)        
        
        image0 = args['img'][0]
        image1 = args['img'][1]
        
        images = mast3r_load_images([image0, image1], size=self.args['resize'], verbose=False)
        output = mast3r_inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)
        
        # at this stage, you have the raw dust3r predictions        
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
    
        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    
        # find 2D-2D matches between the two images    
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                       device=self.device, dist='dot', block_size=2**13)
    
        # ignore small border around the edge
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
    
        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
    
        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
    
        # # visualize a few matches
        # import numpy as np
        # import torch
        # import torchvision.transforms.functional
        # from matplotlib import pyplot as pl
    
        # n_viz = 10
        # num_matches = matches_im0.shape[0]
        # match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        # viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    
        # image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        # image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    
        # viz_imgs = []
        # for i, view in enumerate([view1, view2]):
        #     rgb_tensor = view['img'] * image_std + image_mean
        #     viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    
        # H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        # img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        # img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        # img = np.concatenate((img0, img1), axis=1)
        # pl.figure()
        # pl.imshow(img)
        # cmap = pl.get_cmap('jet')
        # for i in range(n_viz):
        #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        #     pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        # pl.show(block=True)

        kps1 = matches_im0
        kps2 = matches_im1
        
        max_m = self.args['max_matches']
        n_m = kps1.shape[0]
        if np.isfinite(max_m) and (n_m > max_m):
            idx = np.linspace(0, n_m - 1, max_m).astype(int)
            kps1 = kps1[idx]
            kps2 = kps2[idx]

        s1 = max(Image.open(image0).size)
        s2 = max(Image.open(image1).size)
        
        kps1 = torch.tensor(kps1 * s1 / self.args['resize'], device=self.device, dtype=torch.float)
        kps2 = torch.tensor(kps2 * s2 / self.args['resize'], device=self.device, dtype=torch.float)
        
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

        m_mask = torch.full((kps1.shape[0], ), 1, device=self.device, dtype=torch.bool)
        m_val = torch.full((kps1.shape[0], ), 1, device=self.device, dtype=torch.bool)

        m_idx = torch.zeros((kp[0].shape[0], 2), device=self.device, dtype=torch.int)
        m_idx[:, 0] = torch.arange(kp[0].shape[0])
        m_idx[:, 1] = torch.arange(kp[0].shape[0])

        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}
