import torch
from PIL import Image
from romav2 import RoMaV2

from core import device as global_device
from core import set_args

torch.set_float32_matmul_precision('highest')


class romav2_module:
    """
    A robust dense matching module using RoMaV2.

    RoMaV2 is an improved version of RoMa that provides high-quality 
    correspondences through dense matching. It estimates dense matches 
    with overlaps and precision scores, allowing it to find reliable 
    matches even in challenging scenarios like large viewpoint changes 
    or extreme lighting conditions.

    Attributes:
        max_keypoints (int): The number of points to sample from the dense matches.
        patch_radius (int): Radius for patch-based feature extraction.
    """
    def __init__(self, **args):
        torch.set_float32_matmul_precision('highest')
        from romav2.romav2 import RoMaV2
        self.device = torch.device(self.args.get('device', str(global_device)))

        self.romav2_model = None
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
                                
        self.args = {
            'id_more': '',
            'max_keypoints': 2000,
            'patch_radius': 16,
            }
        
        if 'add_to_cache' in args.keys(): 
            self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('romav2', args, self.args)        

        # Load pretrained RoMaV2 model
        self.romav2_model = RoMaV2()
        self.romav2_model = self.romav2_model.to(self.device)


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):
        """
        Run RoMaV2 matching on a pair of images.
        
        Args:
            args: Dictionary containing 'img' key with list of two image paths
            
        Returns:
            Dictionary with keypoints, homographies, rotations, match indices, 
            match values, and match mask
        """
        # Load images to get original dimensions
        im1 = Image.open(args['img'][0])
        im1 = im1.convert("RGB")

        im2 = Image.open(args['img'][1])
        im2 = im2.convert("RGB")

        W_A, H_A = im1.size
        W_B, H_B = im2.size

        # Match densely using RoMaV2
        preds = self.romav2_model.match(args['img'][0], args['img'][1])
        
        # Sample matches
        num_samples = self.args['max_keypoints'] if self.args['max_keypoints'] is not None else 5000
        matches, overlaps, precision_AB, precision_BA = self.romav2_model.sample(preds, num_samples)
        
        # Convert to pixel coordinates (RoMaV2 produces matches in [-1,1]x[-1,1])
        kpts1, kpts2 = self.romav2_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        
        # Move to device
        kps1 = kpts1.detach().to(self.device)
        kps2 = kpts2.detach().to(self.device)
        
        # Create keypoint list
        kp = [kps1, kps2]
        
        # Create homography matrices for patch extraction
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

        # Create rotation arrays (NaN for no rotation info)
        kr = [
            torch.full((kp[0].shape[0],), torch.nan, device=self.device), 
            torch.full((kp[0].shape[0],), torch.nan, device=self.device)
        ]

        # Create match indices
        m_idx = torch.zeros((kp[0].shape[0], 2), device=self.device, dtype=torch.int)
        m_idx[:, 0] = torch.arange(kp[0].shape[0])
        m_idx[:, 1] = torch.arange(kp[0].shape[0])

        # Create match mask (all valid)
        m_mask = torch.ones(m_idx.shape[0], device=self.device, dtype=torch.bool)

        # Use overlaps as match confidence/certainty scores
        m_val = overlaps.detach().to(self.device)

        return {
            'kp': kp, 
            'kH': kH, 
            'kr': kr, 
            'm_idx': m_idx, 
            'm_val': m_val, 
            'm_mask': m_mask
        }