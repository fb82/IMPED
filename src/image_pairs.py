import os
import warnings

from PIL import Image

from colmap_fun import coldb_ext


class image_pairs:
    """
    An intelligent iterator for generating image pairs from a dataset.

    This class supports two primary modes:
    1. Exhaustive (Base): Generates every possible pair from a folder of images 
       (N images result in N*(N-1)/2 pairs).
    2. Explicit: Processes a specific list of image-to-image matches provided by the user.

    The class features a robust filtering system to 'include' or 'exclude' 
    specific images or pairs based on:
    - Direct filename lists.
    - Pre-existing COLMAP databases (checking for prior matches or keypoints).
    - Image validity (verifying that files are not corrupt).

    Attributes:
        imgs (list): List of image paths or pair tuples.
        mode (str): Filtering behavior, either 'include' (only process listed) 
            or 'exclude' (process everything except listed).
    """
    def init_additional_image_pair_check(self, colmap_db_or_list, mode, colmap_req, colmap_min_matches):

        self.additional_colmap_db = None     
        self.additional_img_list = None
        self.additional_pair_list = None       

        if isinstance(colmap_db_or_list, str) and os.path.isdir(colmap_db_or_list):
            file_list = os.listdir(colmap_db_or_list)                    

            self.additional_img_list = {}

            for i in file_list:
                self.additional_img_list[os.path.split(i)[-1]] = True
        
        elif isinstance(colmap_db_or_list, list) or isinstance(colmap_db_or_list, tuple):
            for k in colmap_db_or_list:
                if isinstance(k, str):
                    
                    if self.additional_img_list is None: self.additional_img_list = {}
                    
                    self.additional_img_list[os.path.split(k)[-1]] = True

                elif (isinstance(k, list) or isinstance(k, tuple)) and (len(k) == 2) and isinstance(k[0], str) and isinstance(k[1], str):

                    if self.additional_pair_list is None: self.additional_pair_list = {}

                    i = os.path.split(k[0])[-1]
                    j = os.path.split(k[1])[-1]

                    if i not in self.additional_pair_list.keys():
                        self.additional_pair_list[i] = {}

                    self.additional_pair_list[i][j] = True
                    
        elif isinstance(colmap_db_or_list, str) and (os.path.isfile(colmap_db_or_list)):
            self.additional_colmap_db = coldb_ext(colmap_db_or_list)            
        
        self.mode = mode
        self.colmap_req = colmap_req
        self.colmap_min_matches = colmap_min_matches 
        
    
    def must_skip_after_additional_image_pair_check(self, ii, jj):
        must_skip = False

        i = os.path.split(ii)[-1]
        j = os.path.split(jj)[-1]        

        if (not must_skip) and (self.additional_img_list is not None):
            in_img_list = False
            
            if (i in self.additional_img_list.keys()) or (j in self.additional_img_list.keys()):
                in_img_list = True
            
            must_skip = (in_img_list and self.mode == 'exclude') or ((not in_img_list) and self.mode == 'include') 

        if (not must_skip) and (self.additional_pair_list is not None):
            in_pair_list = False
            
            if (i in self.additional_pair_list.keys() and j in self.additional_pair_list[i].keys()) or (j in self.additional_pair_list.keys() and i in self.additional_pair_list[j].keys()):
                in_pair_list = True
            
            must_skip = (in_pair_list and self.mode == 'exclude') or ((not in_pair_list) and self.mode == 'include') 

        if (not must_skip) and (self.additional_colmap_db is not None):
            in_colmap_db = True
            
            if self.additional_colmap_db is not None:
                im0_id = self.additional_colmap_db.get_image_id(i)
                im1_id = self.additional_colmap_db.get_image_id(j)
                
                if (im0_id is None) or (im1_id is None): in_colmap_db = False
    
                if in_colmap_db and (self.colmap_req != 'keypoints'):
                    if self.colmap_req == 'matches':                            
                        m_idx = self.additional_colmap_db.get_matches(im0_id, im1_id)
                        if (m_idx is None) or (m_idx.shape[0] < self.colmap_min_matches): in_colmap_db = False                                
                    else:
                        m_idx = self.additional_colmap_db.get_matches(im0_id, im1_id)
                        if (m_idx is None) or (m_idx.shape[0] < self.colmap_min_matches): in_colmap_db = False                                
                                            
            must_skip = (in_colmap_db and self.mode == 'exclude') or ((not in_colmap_db) and self.mode == 'include') 
            
        return must_skip
    

    def __init__(self, to_list, add_path='', check_img=True, colmap_db_or_list=None, mode='exclude', colmap_req='geometry', colmap_min_matches=0):
        imgs = []        

        if isinstance(to_list, str):
            warnings.warn("retrieving image list from the image folder")
    
            add_path = os.path.join(add_path, to_list)
    
            if os.path.isdir(add_path):
                file_list = os.listdir(add_path)
            else:
                warnings.warn("image folder does not exist!")
                file_list = []
                
            is_match_list = False
            
            if not is_match_list:                
                for i in file_list:
                    ii = os.path.join(add_path, i)
                    
                    if check_img:
                        try:
                            Image.open(ii).verify()
                        except:
                            continue
    
                    imgs.append(ii)
            
                imgs.sort()
                iter_base = True
            
        if isinstance(to_list, list):
            is_match_list = True
            
            for i in to_list:
                if ((not isinstance(i, tuple)) and (not isinstance(i, list))) or not (len(i) == 2):
                    is_match_list = False
                    break
            
            file_list = to_list
    
            # to_list is a list of images
            if not is_match_list:    
                warnings.warn("reading image list")
                
                for i in file_list:
                    ii = os.path.join(add_path, i)
                    
                    if check_img:                
                        try:
                            Image.open(ii).verify()
                        except:
                            continue
    
                    imgs.append(ii)
            
                imgs.sort()
                iter_base = True

            # dir_name is a list of image pairs
            else:
                warnings.warn("reading image pairs")
                iter_base = False

        self.iter_base = iter_base  
        
        if iter_base:
            self.imgs = imgs    
            self.i = 0
            self.j = 1
        else:
            self.imgs = file_list
            self.add_path = add_path
            self.k = 0
            self.check_img = check_img            
    
        self.init_additional_image_pair_check(colmap_db_or_list, mode, colmap_req, colmap_min_matches)

        if self.iter_base:
            self.len = (len(self.imgs) * (len(self.imgs) - 1)) // 2
        else:
            self.len = len(self.imgs)


    def __iter__(self):
        return self
    

    def __len__(self):
        return self.len

    
    def __next__(self):
        if self.iter_base:
            in_loop = True
            while in_loop:
                if (self.i < len(self.imgs)) and (self.j < len(self.imgs)):                    
                        ii, jj = self.imgs[self.i], self.imgs[self.j]
                    
                        self.j = self.j + 1
    
                        if self.j >= len(self.imgs):                    
                            self.i = self.i + 1
                            self.j = self.i + 1
    
                        if self.must_skip_after_additional_image_pair_check(ii, jj):
                            self.len = max(0, self.len - 1)
                            continue
                                                       
                        return ii, jj
                else:
                    if self.additional_colmap_db is not None: self.additional_colmap_db.close()
                    raise StopIteration

        else:
            while self.k < len(self.imgs):            
                i, j = self.imgs[self.k]
                self.k = self.k + 1

                ii = os.path.join(self.add_path, i)
                jj = os.path.join(self.add_path, j)
        
                if self.check_img:
                    try:
                        Image.open(ii).verify()
                        Image.open(jj).verify()
                    except:
                        self.len = max(0, self.len - 1)
                        continue
                        
                if self.must_skip_after_additional_image_pair_check(ii, jj):
                    self.len = max(0, self.len - 1)
                    continue
    
                return ii, jj            

            if self.additional_colmap_db is not None: self.additional_colmap_db.close()
            raise StopIteration
