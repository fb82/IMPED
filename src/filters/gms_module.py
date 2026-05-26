
import math

import cv2
import numpy as np
import torch
from PIL import Image

import gms.python.gms_matcher as gms
from core import device as global_device
from core import set_args


class gms_module:
    """
    An outlier rejection module using Grid-based Motion Statistics.

    GMS converts the problem of identifying correct matches into a 
    statistical counting problem. It divides the images into grids 
    and assumes that for a true match, the neighborhood (grid cells) 
    in Image 1 should map to the corresponding neighborhood in Image 2.

    The algorithm is particularly effective for high-recall scenarios 
    (where you have thousands of initial matches) and provides a 
    massive speedup over traditional iterative geometric filters.

    Attributes:
        gms_matcher_custom: An internal extension of the GMS library 
            optimized for the pipeline's keypoint format.
    """
    class gms_matcher_custom(gms.GmsMatcher):
        def __init__(self, kp1, kp2, m12):
            self.kp1 = kp1
            self.kp2 = kp2
            self.m12 = m12
    
            self.scale_ratios = [1.0, 1.0 / 2, 1.0 / math.sqrt(2.0), math.sqrt(2.0), 2.0]
            
            # Normalized vectors of 2D points
            self.normalized_points1 = []
            self.normalized_points2 = []
            # Matches - list of pairs representing numbers
            self.matches = []
            self.matches_number = 0
            # Grid Size
            self.grid_size_right = gms.Size(0, 0)
            self.grid_number_right = 0
            # x      : left grid idx
            # y      :  right grid idx
            # value  : how many matches from idx_left to idx_right
            self.motion_statistics = []
    
            self.number_of_points_per_cell_left = []
            # Inldex  : grid_idx_left
            # Value   : grid_idx_right
            self.cell_pairs = []
    
            # Every Matches has a cell-pair
            # first  : grid_idx_left
            # second : grid_idx_right
            self.match_pairs = []
    
            # Inlier Mask for output
            self.inlier_mask = []
            self.grid_neighbor_right = []
    
            # Grid initialize
            self.grid_size_left = gms.Size(20, 20)
            self.grid_number_left = self.grid_size_left.width * self.grid_size_left.height
    
            # Initialize the neihbor of left grid
            self.grid_neighbor_left = np.zeros((self.grid_number_left, 9))
    
            self.gms_matches = []
            self.keypoints_image1 = []
            self.keypoints_image2 = []
    
    
        def compute_matches(self, sz1r, sz1c, sz2r, sz2c):
            self.keypoints_image1=self.kp1
            self.keypoints_image2=self.kp2
        
            size1 = gms.Size(sz1c, sz1r)
            size2 = gms.Size(sz2c, sz2r)
    
            if self.gms_matches:
                self.empty_matches()
    
            all_matches=self.m12
                    
            self.normalize_points(self.keypoints_image1, size1, self.normalized_points1)
            self.normalize_points(self.keypoints_image2, size2, self.normalized_points2)
            self.matches_number = len(all_matches)
            self.convert_matches(all_matches, self.matches)
                    
            self.initialize_neighbours(self.grid_neighbor_left, self.grid_size_left)
            
            mask, num_inliers = self.get_inlier_mask(False, False)
    
            for i in range(len(mask)):
                if mask[i]:
                    self.gms_matches.append(all_matches[i])
            return self.gms_matches, mask


    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
        self.add_to_cache = True
                        
        self.args = {
            'id_more': '',
            }
        self.device =  torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('gms', args, self.args)     
        

    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args):  
        kp1 = args['kp'][0]
        kp2 = args['kp'][1]
        
        kp1_=[cv2.KeyPoint(float(kp1[i, 0]), float(kp1[i, 1]), 1) for i in range(kp1.shape[0])]
        kp2_=[cv2.KeyPoint(float(kp2[i, 0]), float(kp2[i, 1]), 1) for i in range(kp2.shape[0])]

        mi = args['m_idx']
        mm = args['m_mask']
        mv = args['m_val']

        m12 = mi[mm]
        v12 = mv[mm]

        sz1c, sz1r = Image.open(args['img'][0]).size
        sz2c, sz2r = Image.open(args['img'][1]).size

        m12_ = [cv2.DMatch(int(m12[i, 0]), int(m12[i, 1]), float(v12[i])) for i in range(m12.shape[0])]    

        gms = self.gms_matcher_custom(kp1_, kp2_, m12_)

        _, mask = gms.compute_matches(sz1r, sz1c, sz2r, sz2c)

        mask = torch.tensor(mask, device=self.device, dtype=torch.bool)
 
        aux = mm.clone()
        mm[aux] = mask
        
        return {'m_mask': mm}

