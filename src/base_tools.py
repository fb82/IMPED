import os
import warnings
import pickled_hdf5.pickled_hdf5 as pickled_hdf5
import time
from tqdm import tqdm

import torch
import kornia as K
from kornia_moons.feature import opencv_kpts_from_laf, laf_from_opencv_kpts
from lightglue import LightGlue as lg_lightglue, SuperPoint as lg_superpoint, DISK as lg_disk, SIFT as lg_sift, ALIKED as lg_aliked, DoGHardNet as lg_doghardnet
from lightglue.utils import load_image as lg_load_image, rbd as lg_rbd
import cv2
import numpy as np
import hz.hz as hz
from PIL import Image
import poselib

import colmap_db.database as coldb
import matplotlib.pyplot as plt
import plot.viz2d as viz
import plot.utils as viz_utils
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
pipe_color = ['red', 'blue', 'lime', 'fuchsia', 'yellow']
show_progress = True


def go_iter(to_iter, msg='', active=True, params=None):
    if params is None: params = {}
    
    if show_progress and active:
        return tqdm(to_iter, desc=msg, **params)
    else:
        return to_iter 


def visualize_LAF(img, LAF, img_idx = 0, color='r', linewidth=1, draw_ori = True, fig=None, ax = None, return_fig_ax = False, **kwargs):
    from kornia_moons.feature import to_numpy_image

    x, y = K.feature.laf.get_laf_pts_to_draw(K.feature.laf.scale_laf(LAF, 0.5), img_idx)

    if not draw_ori:
        x= x[1:]
        y= y[1:]

    if (fig is None and ax is None):
        fig, ax = plt.subplots(1,1, **kwargs)

    if (fig is not None and ax is None):
        ax = fig.add_axes([0, 0, 1, 1])
    
    if not (img is None):
        ax.imshow(to_numpy_image(img[img_idx]))

    ax.plot(x, y, color, linewidth=linewidth)
    if return_fig_ax : return fig, ax

    return


class image_pairs:
    def __init__(self, to_list, add_path='', check_img=True):
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
    

    def __iter__(self):
        return self
    

    def __len__(self):
        if self.iter_base:
            return (len(self.imgs) * (len(self.imgs) - 1)) // 2
        else:
            return len(self.imgs)

    
    def __next__(self):
        if self.iter_base:            
            if (self.i < len(self.imgs)) and (self.j < len(self.imgs)):                    
                    ii, jj = self.imgs[self.i], self.imgs[self.j]
                
                    self.j = self.j + 1

                    if self.j >= len(self.imgs):                    
                        self.i = self.i + 1
                        self.j = self.i + 1

                    return ii, jj
            else:
                raise StopIteration

        else:
            while self.k < len(self.file_list):            
                i, j = self.file_list[self.k]
                self.k = self.k + 1

                ii = os.path.join(self.add_path, i)
                jj = os.path.join(self.add_path, j)
        
                if self.check_img:
                    try:
                        Image.open(ii).verify()
                        Image.open(jj).verify()
                    except:
                        continue
    
                return ii, jj            

            raise StopIteration

            
# def image_pairs(to_list, add_path='', check_img=True):
#     imgs = []

#     # to_list is effectively an image folder
#     if isinstance(to_list, str):
#         warnings.warn("retrieving image list from the image folder")

#         add_path = os.path.join(add_path, to_list)

#         if os.path.isdir(add_path):
#             file_list = os.listdir(add_path)
#         else:
#             warnings.warn("image folder does not exist!")
#             file_list = []
            
#         is_match_list = False
        
#         if not is_match_list:                
#             for i in file_list:
#                 ii = os.path.join(add_path, i)
                
#                 if check_img:
#                     try:
#                         Image.open(ii).verify()
#                     except:
#                         continue

#                 imgs.append(ii)
        
#             imgs.sort()
#             for i in range(len(imgs)):
#                 for j in range(i + 1, len(imgs)):
#                     yield imgs[i], imgs[j]        
        
#     if isinstance(to_list, list):
#         is_match_list = True
        
#         for i in to_list:
#             if ((not isinstance(i, tuple)) and (not isinstance(i, list))) or not (len(i) == 2):
#                 is_match_list = False
#                 break
        
#         file_list = to_list

#         # to_list is a list of images
#         if not is_match_list:    
#             warnings.warn("reading image list")
            
#             for i in file_list:
#                 ii = os.path.join(add_path, i)
                
#                 if check_img:                
#                     try:
#                         Image.open(ii).verify()
#                     except:
#                         continue

#                 imgs.append(ii)

    
#             imgs.sort()
#             for i in range(len(imgs)):
#                 for j in range(i + 1, len(imgs)):
#                     yield imgs[i], imgs[j]

#         # dir_name is a list of image pairs
#         else:
#             warnings.warn("reading image pairs")

#             for i, j in file_list:
#                 ii = os.path.join(add_path, i)
#                 jj = os.path.join(add_path, j)

#                 if check_img:
#                     try:
#                         Image.open(ii).verify()
#                         Image.open(jj).verify()
#                     except:
#                         continue

#                 yield ii, jj


def run_pairs(pipeline, imgs, db_name='database.hdf5', db_mode='a', force=False):    
    db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)

    for pair in go_iter(image_pairs(imgs), msg='          processed pairs'):
        run_pipeline(pair, pipeline, db, force=force, show_progress=True)

                
def run_pipeline(pair, pipeline, db, force=False, pipe_data=None, pipe_name='/', show_progress=False):  
    if pipe_data is None: pipe_data = {}

    if not pipe_data:
        pipe_data['img'] = [pair[0], pair[1]]
        pipe_data['warp'] = [torch.eye(3, device=device, dtype=torch.float), torch.eye(3, device=device, dtype=torch.float)]
        
    for pipe_module in go_iter(pipeline, msg='current pipeline progress', active=show_progress, params={'leave': False}):
        if hasattr(pipe_module, 'pass_through') and pipe_module.pass_through:  
            pipe_id = '/'
            key_data = '/' + pipe_module.get_id()
        else:
            pipe_id = '/' + pipe_module.get_id()
            key_data = '/data'
            
        if pipe_name == '': pipe_name = '/'
        pipe_name_prev = pipe_name            
        pipe_name = pipe_name + pipe_id
        
        if hasattr(pipe_module, 'single_image') and pipe_module.single_image:            
            for n in range(len(pipe_data['img'])):
                im = os.path.split(pipe_data['img'][n])[-1]
                data_key = '/' + im + pipe_name + key_data                    

                out_data, is_found = db.get(data_key)                    
                if (not is_found) or force:
                    start_time = time.time()
                    out_data = pipe_module.run(idx=n, **pipe_data)
                    stop_time = time.time()
                    out_data['running_time'] = stop_time - start_time
                    db.add(data_key, out_data)
                del out_data['running_time']

                for k, v in out_data.items():
                    if k in pipe_data:
                        if len(pipe_data[k]) == len(pipe_data['img']):
                            pipe_data[k][n] = v
                        else:
                            pipe_data[k].append(v)
                    else:
                        pipe_data[k] = [v]
                        
        else:            
            im0 = os.path.split(pipe_data['img'][0])[-1]
            im1 = os.path.split(pipe_data['img'][1])[-1]
            data_key = '/' + im0 + '/' + im1 + pipe_name + key_data 

            out_data, is_found = db.get(data_key)                    
            if (not is_found) or force:
                start_time = time.time()

                if hasattr(pipe_module, 'pipeliner') and pipe_module.pipeliner:
                    out_data = pipe_module.run(pipe_data=pipe_data, pipe_name=pipe_name_prev, db=db, force=force)
                else:
                    out_data = pipe_module.run(**pipe_data)

                stop_time = time.time()
                out_data['running_time'] = stop_time - start_time
                db.add(data_key, out_data)
            out_data['running_time']
                

            for k, v in out_data.items(): pipe_data[k] = v
                
    return pipe_data, pipe_name


def laf2homo(kps, with_scale=False):
    c = kps[:, :, 2].type(torch.float)
    
    Hi = torch.zeros((kps.shape[0], 3, 3), device=device)
    Hi[:, :2, :] = kps    
    Hi[:, 2, 2] = 1 

    if with_scale:
        s = torch.sqrt(torch.abs(kps[:, 0, 0] * kps[:, 1, 1] - kps[:, 0, 1] * kps[:, 1, 0]))   
        Hi[:, :2, :] = Hi[:, :2, :] / s.reshape(-1, 1, 1)
        s = s.type(torch.float)
        
    H = torch.linalg.inv(Hi).type(torch.float)

    if with_scale:    
        return c, H, s
    
    return c, H


def homo2laf(c, H, s=None):
    Hi = torch.linalg.inv(H)
    kp = Hi[:, :2, :]
    
    if not (s is None):
        kp = kp * s.reshape(-1, 1, 1)

    return kp.unsqueeze(0)


def set_args(id_string, args, args_):
        
    if args:
        for k, v in args.items():
            args_[k] = v
            if k == 'id_more':
                id_string = id_string + '_' + str(v)

    id_string = id_string.lower()
    
    return id_string, args_    


class dog_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False                
        self.pass_through = False

        self.args = {
            'id_more': '',
            'upright': False,
            'params': {'nfeatures': 8000, 'contrastThreshold': -10000, 'edgeThreshold': 10000},
        }

        self.id_string, self.args = set_args('dog', args, self.args)
        self.detector = cv2.SIFT_create(**self.args['params'])


    def get_id(self): 
        return self.id_string


    def run(self, **args):    
        
        im = cv2.imread(args['img'][args['idx']], cv2.IMREAD_GRAYSCALE)
        kp = self.detector.detect(im, None)


        if self.args['upright']:
            idx = np.unique(np.asarray([[k.pt[0], k.pt[1]] for k in kp]), axis=0, return_index=True)[1]
            kp = [kp[ii] for ii in idx]
            for ii in range(len(kp)):
                kp[ii].angle = 0       

        kr = []
        for i in range(len(kp)): kr.append(kp[i].response)
        kr = torch.tensor(kr, device=device, dtype=torch.float)
                
        kp = laf_from_opencv_kpts(kp, device=device)
        kp, kH = laf2homo(kp.detach().to(device).squeeze(0))
    
        return {'kp': kp, 'kH': kH, 'kr': kr}


class keynet_module:
    def __init__(self, **args):
        self.single_image = True        
        self.pipeliner = False   
        self.pass_through = False
        
        self.args = {
            'id_more': '',
            'params': {'num_features': 8000},
        }
        
        self.id_string, self.args = set_args('keynet', args, self.args)
        self.detector = K.feature.KeyNetDetector(**self.args['params']).to(device)


    def get_id(self):
        return self.id_string
        
    
    def run(self, **args):
        img = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)
        kp, kr = self.detector(img)
        kp, kH = laf2homo(kp.detach().to(device).squeeze(0))

        return {'kp': kp, 'kH': kH, 'kr': kr.detach().to(device).squeeze(0)}


class hz_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False

        self.args = {
            'id_more': '',
            'plus': True,
            'params': {'max_max_pts': 8000, 'block_mem': 16*10**6},
        }

        self.id_string, self.args = set_args('' , args, self.args)
        if self.args['plus']:
            self.id_string = 'hz_plus' + self.id_string                
            self.hz_to_run = hz.hz_plus
        else:
            self.id_string = 'hz' + self.id_string                
            self.hz_to_run = hz.hz
        
    def get_id(self): 
        return self.id_string

    
    def run(self, **args):  
        if self.args['plus']:        
            img = hz.load_to_tensor(args['img'][args['idx']]).to(torch.float)
        else:
            img = hz.load_to_tensor(args['img'][args['idx']], grayscale=True).to(torch.float)

        kp, kr = self.hz_to_run(img, output_format='laf', **self.args['params'])
        kp, kH = laf2homo(K.feature.ellipse_to_laf(kp[None]).squeeze(0))

        return {'kp': kp, 'kH': kH, 'kr': kr.type(torch.float)}


class show_kpts_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = True

        self.args = {
            'id_more': '',
            'img_prefix': '',
            'img_suffix': '',
            'cache_path': 'show_imgs',
            'prepend_pair': False,
            'ext': '.jpg',
            'force': False,
            'mask_idx': None, # None: all single image, -1: all both images, list: filtered both images
            'params': [{'color': 'r', 'linewidth': 1, 'draw_ori': True}, {'color': 'g', 'linewidth': 1, 'draw_ori': True}],
        }
        
        self.id_string, self.args = set_args('show_kpts' , args, self.args)
        if not (self.args['mask_idx'] is None): self.single_image = False

                
    def get_id(self): 
        return self.id_string

    
    def run(self, **args): 
        if not self.single_image:
            idxs = [0, 1]
        else:
            idxs = [args['idx']]

        for idx in idxs:
            im = args['img'][idx]
            cache_path = self.args['cache_path']
            img = os.path.split(im)[1]
            img_name, _ = os.path.splitext(img)
            if self.args['prepend_pair']:
                img0 = os.path.splitext(os.path.split(args['img'][0])[1])[0]
                img1 = os.path.splitext(os.path.split(args['img'][1])[1])[0]
                cache_path = os.path.join(cache_path, img0 + '_' + img1)
                
            new_img = os.path.join(cache_path, self.args['img_prefix'] + img_name + self.args['img_suffix'] + self.args['ext'])
    
            if not os.path.isfile(new_img) or self.args['force']:
                os.makedirs(cache_path, exist_ok=True)
                img = cv2.cvtColor(cv2.imread(args['img'][idx]), cv2.COLOR_BGR2RGB)    
                lafs = homo2laf(args['kp'][idx], args['kH'][idx])
    
                if (self.args['mask_idx'] is None) or (self.args['mask_idx'] == -1) or (not 'm_idx' in args):
                    mask_idx = -1
                    params = self.args['params'][-1]
                else:
                    if not isinstance(self.args['mask_idx'], list): self.args['mask_idx'] = [self.args['mask_idx']]
                    mask_idx = self.args['mask_idx']
                    params = self.args['params']
                                    
                fig = plt.figure()
                ax = None
                img = K.image_to_tensor(img, False)
    
                if mask_idx == -1: 
                    fig, ax = visualize_LAF(img, lafs, 0, fig=fig, ax=ax, return_fig_ax=True, **params)
    
                else:
                    for i in mask_idx:                
                        m_idx = args['m_idx'][:, idx]
                        m_mask = args['m_mask']
                        m_idx = m_idx[m_mask == i]
                        if m_idx.shape[0] < 1: continue
                        lafs_ = lafs[:, m_idx]
                        
                        fig, ax = visualize_LAF(img, lafs_, 0, fig=fig, ax=ax, return_fig_ax=True, **params[i])
                        img = None
    
                plt.axis('off')
                plt.savefig(new_img, dpi=150, bbox_inches='tight')
                plt.close(fig)

        return {}


class show_matches_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True

        self.args = {
            'id_more': '',
            'img_prefix': '',
            'img_suffix': '',
            'cache_path': 'show_imgs',
            'prepend_pair': True,
            'ext': '.jpg',
            'force': False,
            'mask_idx': [1], # -1: all, [1]: inliers, [0]: outliers, [0, 1]: outlier and inliers with differen colors
            'fig_min_size': 960,
            'fig_max_size': 1280, 
            'params': [{'color': [1, 0, 0]}, {'color': [0, 1, 0]}],
        }
        
        self.id_string, self.args = set_args('show_matches' , args, self.args)

                
    def get_id(self): 
        return self.id_string

    
    def run(self, **args):         
        im0 = os.path.splitext(os.path.split(args['img'][0])[1])[0]
        im1 = os.path.splitext(os.path.split(args['img'][1])[1])[0]

        if self.args['prepend_pair']:            
            cache_path = os.path.join(self.args['cache_path'], im0 + '_' + im1)
        else:
            cache_path = self.args['cache_path']
                
        new_img = os.path.join(cache_path, self.args['img_prefix'] + im0 + '_' + im1 + self.args['img_suffix'] + self.args['ext'])
    
        if not os.path.isfile(new_img) or self.args['force']:
            os.makedirs(cache_path, exist_ok=True)

            fig = plt.figure()    
            img0 = viz_utils.load_image(args['img'][0])
            img1 = viz_utils.load_image(args['img'][1])
            fig, axes = viz.plot_images([img0, img1], fig_num=fig.number)              

            if 'm_idx' in args:
                if self.args['mask_idx'] == -1:
                    mask_idx = -1
                    params = self.args['params'][-1]

                    m_idx = args['m_idx']
                    pt1 = args['kp'][0][m_idx[:, 0]]
                    pt2 = args['kp'][1][m_idx[:, 1]]

                    if pt1.shape[0] > 0:
                        viz.plot_matches(pt1, pt2, color=self.args['params'][0]['color'], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
                else:
                    if not isinstance(self.args['mask_idx'], list): self.args['mask_idx'] = [self.args['mask_idx']]                    
                    mask_idx = self.args['mask_idx']
                    params = self.args['params']

                    m_mask = args['m_mask']
                    m_sum = torch.tensor([(m_mask == i).sum().item() for i in mask_idx], device=device)
                    idx = torch.argsort(m_sum, descending=True)

                    mask_idx = [mask_idx[i] for i in idx]
                    params = [params[i] for i in idx]
                    
                    for i in mask_idx:
                        
                        m_idx = args['m_idx'][args['m_mask'] == i]
                        if m_idx.shape[0] < 1: continue                        

                        pt1 = args['kp'][0][m_idx[:, 0]]
                        pt2 = args['kp'][1][m_idx[:, 1]]

                        viz.plot_matches(pt1, pt2, color=self.args['params'][i]['color'], lw=0.2, ps=6, a=0.3, axes=axes, fig_num=fig.number)
            
            fig_dpi = fig.get_dpi()
            fig_sz = [fig.get_figwidth() * fig_dpi, fig.get_figheight() * fig_dpi]
        
            fig_min_size = self.args['fig_min_size']
            fig_max_size = self.args['fig_max_size']
        
            fig_cz = min(fig_sz)
            if fig_cz < fig_min_size:
                fig_sz[0] = fig_sz[0] / fig_cz * fig_min_size
                fig_sz[1] = fig_sz[1] / fig_cz * fig_min_size
        
            fig_cz = max(fig_sz)
            if fig_cz > fig_min_size:
                fig_sz[0] = fig_sz[0] / fig_cz * fig_max_size
                fig_sz[1] = fig_sz[1] / fig_cz * fig_max_size
                
            fig.set_size_inches(fig_sz[0] / fig_dpi, fig_sz[1]  / fig_dpi)
        
            viz.save_plot(new_img, fig)
            viz.clear_plot(fig)
                        
            plt.close(fig)
        
        return {}


class patch_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False
        
        self.args = {
            'id_more': '',
            'sift_orientation': False,
            'sift_orientation_params': {},
            'orinet': True,
            'orinet_params': {},
            'affnet': True,
            'affnet_params': {},
            }

        self.id_string, self.args = set_args('', args, self.args)

        base_string = ''
        self.ori_module = K.feature.PassLAF()
        if self.args['sift_orientation']:
            base_string = 'sift_orientation'
            self.ori_module = K.feature.LAFOrienter(angle_detector=K.feature.PatchDominantGradientOrientation(), **self.args['orinet_params'])
        if self.args['orinet']:
            base_string = 'orinet'
            self.ori_module = K.feature.LAFOrienter(angle_detector=K.feature.OriNet().to(device), **self.args['orinet_params'])

        if self.args['affnet']:
            if len(base_string): base_string = base_string  + '_' + 'affnet'
            else: base_string = 'affnet'
            self.aff_module = K.feature.LAFAffineShapeEstimator(**self.args['affnet_params'])
        else:
            self.aff_module = K.feature.PassLAF()

        if not len(base_string): base_string = 'pass_laf'
        self.id_string = base_string + self.id_string


    def get_id(self): 
        return self.id_string

    def run(self, **args):    
        im = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)

        lafs = homo2laf(args['kp'][args['idx']], args['kH'][args['idx']])

        lafs = self.ori_module(lafs, im)
        lafs = self.aff_module(lafs, im)

        kp, kH = laf2homo(lafs.squeeze(0))
    
        return {'kp': kp, 'kH': kH}


class deep_descriptor_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False
        
        self.args = {
            'id_more': '',
            'descriptor': 'hardnet',
            'desc_params': {},
            'patch_params': {},
            }

        self.id_string, self.args = set_args('', args, self.args)        
        
        if self.args['descriptor'] == 'hardnet':
            base_string = 'hardnet'
            desc = K.feature.HardNet().to(device)
        if self.args['descriptor'] == 'sosnet':
            desc = K.feature.SOSNet().to(device)
            base_string = 'sosnet'
        if self.args['descriptor'] == 'hynet':
            desc = K.feature.HyNet(**self.args['desc_params']).to(device)
            base_string = 'hynet'

        self.ddesc = K.feature.LAFDescriptor(patch_descriptor_module=desc, **self.args['patch_params'])
        self.id_string = base_string + self.id_string

    def get_id(self): 
        return self.id_string


    def run(self, **args):    
        im = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)

        lafs = homo2laf(args['kp'][args['idx']], args['kH'][args['idx']])
        desc = self.ddesc(im, lafs).squeeze(0)
    
        return {'desc': desc}


class sift_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = False
        
        self.args = {
            'id_more': '',
            'rootsift': True,
            }
        
        self.id_string, self.args = set_args('', args, self.args)        
        self.descriptor = cv2.SIFT_create()

        if self.args['rootsift']:
            base_string = 'rootsift'
        else:
            base_string = 'sift'
            
        self.id_string = base_string + self.id_string

    def get_id(self): 
        return self.id_string


    def run(self, **args):
        im = cv2.imread(args['img'][args['idx']], cv2.IMREAD_GRAYSCALE)        
        lafs = homo2laf(args['kp'][args['idx']], args['kH'][args['idx']])                
        kp = opencv_kpts_from_laf(lafs)
        
        _, desc = self.descriptor.compute(im, kp)

        if self.args['rootsift']:
            desc /= desc.sum(axis=1, keepdims=True) + 1e-8
            desc = np.sqrt(desc)
            
        desc = torch.tensor(desc, device=device, dtype=torch.float)
                    
        return {'desc': desc}


class smnn_module:
    def __init__(self, **args):
        self.single_image = False    
        self.pipeliner = False      
        self.pass_through = False
                        
        self.args = {
            'id_more': '',
            'th': 0.95,
            }
        
        self.id_string, self.args = set_args('smnn', args, self.args)        


    def get_id(self): 
        return self.id_string


    def run(self, **args):
        val, idxs = K.feature.match_smnn(args['desc'][0], args['desc'][1], self.args['th'])

        return {'m_idx': idxs, 'm_val': val.squeeze(1), 'm_mask': torch.ones(idxs.shape[0], device=device, dtype=torch.bool)}


def pair_rot4(pair, cache_path='tmp_imgs', force=False, **dummy_args):
    yield pair, [torch.eye(3, device=device, dtype=torch.float), torch.eye(3, device=device, dtype=torch.float)], {}

    rot_mat = np.eye(2)
    
    os.makedirs(cache_path, exist_ok=True)
    
    rot_to_do = [
        ['_rot90', cv2.ROTATE_90_CLOCKWISE],
        ['_rot_180', cv2.ROTATE_180],
        ['_rot_270', cv2.ROTATE_90_COUNTERCLOCKWISE],
        ]

    width, height = Image.open(pair[1]).size
    c = [width / 2, height / 2]

    for r in range(len(rot_to_do)):
        img = os.path.split(pair[1])[1]
        img_name, img_ext = os.path.splitext(img)
        new_img = os.path.join(cache_path, img_name + rot_to_do[r][0] + img_ext)

        if not os.path.isfile(new_img) or force:
            im = cv2.imread(pair[1], cv2.IMREAD_UNCHANGED)
            im = cv2.rotate(im, rot_to_do[r][1])
            cv2.imwrite(new_img, im)
                                            
        m0 = [[1, 0, -c[(0 + r + 1) % 2]],
              [0, 1, -c[(1 + r + 1) % 2]],
              [0, 0,          1        ]]

        rot_mat = np.asarray([[0, 1], [-1, 0]]) @ rot_mat
        m1 = np.eye(3)
        m1[:2, :2] = rot_mat

        m2 = [[1, 0, c[0]],
              [0, 1, c[1]],
              [0, 0,   1 ]]

        # from warped to original
        warp_matrix = torch.tensor(m2 @ m1 @ m0, device=device, dtype=torch.float)
            
        yield (pair[0], new_img), [torch.eye(3, device=device, dtype=torch.float), warp_matrix], {}


def pipe_max_matches(pipe_block):
    n_matches = torch.zeros(len(pipe_block), device=device)
    for i in range(len(pipe_block)):
        if 'm_mask' in pipe_block[i]:
            n_matches[i] = pipe_block[i]['m_mask'].sum()
    
    best = n_matches.max(0)[1]
    
    return pipe_block[best]
        

class image_muxer_module:
    def __init__(self, id_more='', cache_path='tmp_imgs', pair_generator=pair_rot4, pipe_gather=pipe_max_matches, pipeline=None):
        self.single_image = False
        self.pipeliner = True
        self.pass_through = False
                
        self.id_more = id_more
        self.cache_path = cache_path
        self.pair_generator = pair_generator
        self.pipe_gather = pipe_gather

        if pipeline is None: pipeline = []
        self.pipeline = pipeline
        
        self.id_string = 'image_muxer'
        if len(self.id_more): self.id_string = self.id_string + '_' + str(self.id_more)        


    def get_id(self): 
        return self.id_string


    def run(self, db=None, force=False, pipe_data=None, pipe_name='/'):        
        if pipe_data is None: pipe_data = {}
        pair = pipe_data['img']
        warp = pipe_data['warp']
        pipe_data_block = []
        
        for pair_, warp_, aux_data in self.pair_generator(pair, cache_path=self.cache_path, force=force, pipe_data=pipe_data):            
            pipe_data_in = pipe_data.copy()

            for k in aux_data.keys():
                pipe_data_in[k] = aux_data[k]

            pipe_data_in['img'] = [pair_[0], pair_[1]]
            pipe_data_in['warp'] = [warp_[0], warp_[1]]
            
            if 'kp' in pipe_data:
                pipe_data_in['kp'] = [    
                    apply_homo(pipe_data['kp'][0], warp_[0].inverse()),
                    apply_homo(pipe_data['kp'][1], warp_[1].inverse())
                    ]

            if 'kH' in pipe_data:
                pipe_data_in['kH'] = [    
                    change_patch_homo(pipe_data['kH'][0], warp_[0]),
                    change_patch_homo(pipe_data['kH'][1], warp_[1]),
                    ]
                                       
            pipe_data_out, pipe_name_out = run_pipeline(pair_, self.pipeline, db, force=force, pipe_data=pipe_data_in, pipe_name=pipe_name)

            pipe_data_out['img'] = pair
            pipe_data_out['warp'] = warp

            if 'kp' in pipe_data_out:
                pipe_data_out['kp'] = [    
                    apply_homo(pipe_data_out['kp'][0], warp_[0]),
                    apply_homo(pipe_data_out['kp'][1], warp_[1])
                    ]

            if 'kH' in pipe_data_out:
                pipe_data_in['kH'] = [    
                    change_patch_homo(pipe_data_out['kH'][0], warp_[0].inverse()),
                    change_patch_homo(pipe_data_out['kH'][1], warp_[1].inverse()),
                    ]
        
            pipe_data_block.append(pipe_data_out)
        
        return self.pipe_gather(pipe_data_block)
        

def change_patch_homo(kH, warp):       
    return kH @ warp.unsqueeze(0)


def apply_homo(p, H):
    
    pt = torch.zeros((p.shape[0], 3), device=device)
    pt[:, :2] = p
    pt[:, 2] = 1
    pt_ = (H @ pt.permute((1, 0))).permute((1, 0))
    return pt_[:, :2] / pt_[:, 2].unsqueeze(-1)    


class magsac_module:
    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False  
        self.pass_through = False
                        
        self.args = {
            'id_more': '',
            'mode': 'fundamental_matrix',
            'conf': 0.9999,
            'max_iters': 100000,
            'px_th': 3,
            'max_try': 3
            }
        
        self.id_string, self.args = set_args('magsac', args, self.args)        


    def get_id(self): 
        return self.id_string

        
    def run(self, **args):  
        pt1_ = args['kp'][0]
        pt2_ = args['kp'][1]
        mi = args['m_idx']
        mm = args['m_mask']
        
        pt1 = pt1_[mi[mm][:, 0]]
        pt2 = pt2_[mi[mm][:, 1]]
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
            
        if self.args['mode'] == 'fundamental_matrix':
            sac_to_run = cv2.findFundamentalMat
            sac_min = 8
        else:
            sac_to_run = cv2.findHomography
            sac_min = 4
            
        if (pt1.shape)[0] >= sac_min:  
            try:                     
                F, mask = sac_to_run(pt1, pt2, cv2.USAC_MAGSAC, self.args['px_th'], self.args['conf'], self.args['max_iters'])
            except:
                for i in range(self.args['max_try'] - 1):
                    try:
                        idx = np.random.permutation(pt1.shape[0])
                        jdx = np.argsort(idx)
                        F, mask = sac_to_run(pt1[idx], pt2[idx], cv2.USAC_MAGSAC, self.args['px_th'], self.args['conf'], self.args['max_iters'])
                        mask = mask[jdx]
                    except:
                        warnings.warn("MAGSAC failed, tentative " + str(i + 1) + ' of ' + str(self.args['max_try']))
                        continue
                    
        if not isinstance(mask, np.ndarray):
            mask = torch.zeros(pt1.shape[0], device=device, dtype=torch.bool)
        else:
            if len(mask.shape) > 1: mask = mask.squeeze(1) > 0
            mask = torch.tensor(mask, device=device, dtype=torch.bool)
 
        aux = mm.clone()
        mm[aux] = mask
        
        if not (F is None):
            F = torch.tensor(F, device=device)
        
        if self.args['mode'] == 'fundamental_matrix':
            return {'m_mask': mm, 'F': F}
        else:
            return {'m_mask': mm, 'H': F}


class poselib_module:
    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
                
        self.args = {
            'id_more': '',
            'mode': 'fundamental_matrix',
            'conf': 0.9999,
            'max_iters': 100000,
            'min_iters': 50,
            'px_th': 3,
            'max_try': 3
            }
        
        self.id_string, self.args = set_args('poselib', args, self.args)        


    def get_id(self): 
        return self.id_string
    
        
    def run(self, **args):  
        pt1_ = args['kp'][0]
        pt2_ = args['kp'][1]
        mi = args['m_idx']
        mm = args['m_mask']
        
        pt1 = pt1_[mi[mm][:, 0]]
        pt2 = pt2_[mi[mm][:, 1]]
        
        if torch.is_tensor(pt1):
            pt1 = np.ascontiguousarray(pt1.detach().cpu())
            pt2 = np.ascontiguousarray(pt2.detach().cpu())

        F = None
        mask = []
            
        if self.args['mode'] == 'fundamental_matrix':
            sac_to_run = poselib.estimate_fundamental
            sac_min = 8
        else:
            sac_to_run = poselib.estimate_homography
            sac_min = 4
            
        params = {         
            'max_iterations': self.args['max_iters'],
            'min_iterations': self.args['min_iters'],
            'success_prob': self.args['conf'],
            'max_epipolar_error': self.args['px_th'],
            }
            
        if (pt1.shape)[0] >= sac_min:  
            F, info = sac_to_run(pt1, pt2, params, {})
            mask = info['inliers']

        if (not isinstance(mask, list)) or (mask == []):
            mask = torch.zeros(pt1.shape[0], device=device, dtype=torch.bool)
        else:
            mask = torch.tensor(mask, device=device, dtype=torch.bool)
 
        aux = mm.clone()
        mm[aux] = mask
        
        if not (F is None):
            F = torch.tensor(F, device=device)
        
        if self.args['mode'] == 'fundamental_matrix':
            return {'m_mask': mm, 'F': F}
        else:
            return {'m_mask': mm, 'H': F}


def pipe_union(pipe_block, unique=True, no_unmatched=False, only_matched=False, sampling_mode=None, sampling_scale=1, sampling_offset=0, preserve_order=False, counter=False):
    if not isinstance(pipe_block, list): pipe_block = [pipe_block]

    kp0 = []
    kH0 = []
    kr0 = []

    kp1 = []
    kH1 = []
    kr1 = []
    
    use_w = False
    for pipe_data in pipe_block:
        if 'w' in pipe_data:
            use_w = True
            break
    
    if use_w:
        w0 = []
        w1 = []
        
    counter0 = None
    counter1 = None    
        
    if counter:
        counter0 = []
        counter1 = []
        
    if preserve_order:
        rank0 = []
        rank1 = []
    
    m_idx = []
    m_val = []
    m_mask = []
    
    m0_offset = 0
    m1_offset = 0
    
    idx0 = None
    idx1 = None
        
    c_rank0 = 0
    c_rank1 = 0
    for i, pipe_data in enumerate(pipe_block):
        if 'kp' in pipe_data:
        
            kp0.append(pipe_data['kp'][0])
            kp1.append(pipe_data['kp'][1])
    
            kH0.append(pipe_data['kH'][0])
            kH1.append(pipe_data['kH'][1])

            kr0.append(pipe_data['kr'][0])
            kr1.append(pipe_data['kr'][1])
            
            if use_w:
                w0.append(pipe_data['w'][0])
                w1.append(pipe_data['w'][1])

            if preserve_order:
                rank0.append(torch.arange(c_rank0, c_rank0 + pipe_data['kp'][0].shape[0], device=device))
                rank1.append(torch.arange(c_rank1, c_rank1 + pipe_data['kp'][1].shape[0], device=device))
                c_rank0 = c_rank0 + pipe_data['kp'][0].shape[0]
                c_rank1 = c_rank1 + pipe_data['kp'][1].shape[0]
                
                if i==0:
                    q_rank0 = pipe_data['kp'][0].shape[0]
                    q_rank1 = pipe_data['kp'][1].shape[0]

            if counter:
                counter0.append(pipe_data['k_counter'][0])
                counter1.append(pipe_data['k_counter'][1])
            
            if 'm_idx' in pipe_data:

                if only_matched:
                    to_retain = pipe_data['m_mask'].clone()
                else:
                    to_retain = torch.full((pipe_data['m_mask'].shape[0], ), 1, device=device, dtype=torch.bool)
                                
                m_idx.append(pipe_data['m_idx'][to_retain] + torch.tensor([m0_offset, m1_offset], device=device).unsqueeze(0))
                m_val.append(pipe_data['m_val'][to_retain])
                m_mask.append(pipe_data['m_mask'][to_retain])
                    
                m0_offset = m0_offset + pipe_data['kp'][0].shape[0]
                m1_offset = m1_offset + pipe_data['kp'][1].shape[0]

    if 'kp' in pipe_data:
        kp0 = torch.cat(kp0)
        kp1 = torch.cat(kp1)

        kH0 = torch.cat(kH0)
        kH1 = torch.cat(kH1)

        kr0 = torch.cat(kr0)
        kr1 = torch.cat(kr1)
        
        if use_w:
            w0 = torch.cat(w0)
            w1 = torch.cat(w1)
            
        if preserve_order:
            rank0 = torch.cat(rank0)
            rank1 = torch.cat(rank1)
            
        if counter:
            counter0 = torch.cat(counter0)
            counter1 = torch.cat(counter1)
          
        if 'm_idx' in pipe_data:
            m_idx = torch.cat(m_idx)
            m_val = torch.cat(m_val)
            m_mask = torch.cat(m_mask)
            
    if not (sampling_mode is None):
        kp0_unsampled = kp0.clone()
        kp1_unsampled = kp1.clone()
        
        kp0 = ((kp0 + sampling_offset) / sampling_scale).round() * sampling_scale - sampling_offset
        kp1 = ((kp1 + sampling_offset) / sampling_scale).round() * sampling_scale - sampling_offset
            
        if 'm_idx' in pipe_data:
            m0_idx = m_idx[:, 0]
            m1_idx = m_idx[:, 1]
            ms_val = m_val
            ms_mask = m_mask
        else:
            m0_idx = None
            m1_idx = None
            ms_val = None
            ms_mask = None
        
        kp0 = sampling(sampling_mode, kp0, kp0_unsampled, kr0, m0_idx, ms_val, ms_mask, counter=counter0)            
        kp1 = sampling(sampling_mode, kp1, kp1_unsampled, kr1, m1_idx, ms_val, ms_mask, counter=counter1)            
            
    if unique:
        if 'm_idx' in pipe_data:
            idx = torch.argsort(m_val, descending=True, stable=True)

            m_idx = m_idx[idx]
            m_val = m_val[idx]
            m_mask = m_mask[idx]

            idx = torch.argsort(m_mask.type(torch.float), descending=True, stable=True)

            m_idx = m_idx[idx]
            m_val = m_val[idx]
            m_mask = m_mask[idx]

            idx0 = torch.full((kp0.shape[0], ), m_idx.shape[0], device=device, dtype=torch.int)
            for i in range(m_idx.shape[0] - 1, -1, -1):
                idx0[m_idx[i, 0]] = i            
            idx0 = torch.argsort(idx0, stable=True)
            
            idx1 = torch.full((kp1.shape[0], ), m_idx.shape[0], device=device, dtype=torch.int)
            idx1[:] = m_idx.shape[0] + 1
            for i in range(m_idx.shape[0] - 1, -1, -1):
                idx1[m_idx[i, 1]] = i            
            idx1 = torch.argsort(idx1, stable=True)
            
        if 'kp' in pipe_data:
            if not preserve_order:
                rank0 = None
                rank1 = None
            
            idx0u, idx0r = sortrows(kp0.clone(), idx0, rank0)
            
            if counter:
                counter_new = torch.zeros(idx0u.shape[0], device=device)
                for i in range(idx0r.shape[0]):
                    counter_new[idx0r[i]] = counter_new[idx0r[i]] + counter0[i] 
                counter0 = counter_new
            
            kp0 = kp0[idx0u]
            kH0 = kH0[idx0u]
            kr0 = kr0[idx0u]

            idx1u, idx1r = sortrows(kp1.clone(), idx1, rank1)
 
            if counter:
                counter_new = torch.zeros(idx1u.shape[0], device=device)
                for i in range(idx1r.shape[0]):
                    counter_new[idx1r[i]] = counter_new[idx1r[i]] + counter1[i] 
                counter1 = counter_new
  
            kp1 = kp1[idx1u]
            kH1 = kH1[idx1u]
            kr1 = kr1[idx1u]
            
            if use_w:
                w0 = w0[idx0u]
                w1 = w1[idx1u]
                
            if preserve_order:
                rank0 = rank0[idx0u]
                rank1 = rank1[idx1u]
                            
            if 'm_idx' in pipe_data:
                m_idx_new = torch.cat((idx0r[m_idx[:, 0]].unsqueeze(1), idx1r[m_idx[:, 1]].unsqueeze(1)), dim=1)
                idxmu, _ = sortrows(m_idx_new.clone())
                m_idx = m_idx_new[idxmu]
                m_val = m_val[idxmu]
                m_mask = m_mask[idxmu]
    
    if no_unmatched and ('m_idx' in pipe_data):
        t0 = torch.zeros(kp0.shape[0], device=device, dtype=torch.bool)
        t1 = torch.zeros(kp1.shape[0], device=device, dtype=torch.bool)
        
        if preserve_order:
            t0[rank0 < q_rank0] = True
            t1[rank1 < q_rank1] = True

        t0[m_idx[:, 0]] = True
        t1[m_idx[:, 1]] = True
        
        idx0 = t0.cumsum(dim=0) - 1
        idx1 = t1.cumsum(dim=0) - 1

        m_idx = torch.cat((idx0[m_idx[:, 0]].unsqueeze(1), idx1[m_idx[:, 1]].unsqueeze(1)), dim=1)

        kp0 = kp0[t0]
        kH0 = kH0[t0]
        kr0 = kr0[t0]

        kp1 = kp1[t1]
        kH1 = kH1[t1]
        kr1 = kr1[t1]
                
        if use_w:
            w0 = w0[t0]            
            w1 = w1[t1]            
        
        if preserve_order:
            rank0 = rank0[t0]
            rank1 = rank1[t1]
            
        if counter:
            counter0 = counter0[t0]
            counter1 = counter1[t1]
            
    if preserve_order:
        if 'kp' in pipe_data:        
            idx0 = torch.argsort(rank0)
            idr0 = torch.argsort(idx0)

            idx1 = torch.argsort(rank1)
            idr1 = torch.argsort(idx1)

            kp0 = kp0[idx0]
            kH0 = kH0[idx0]
            kr0 = kr0[idx0]
    
            kp1 = kp1[idx1]
            kH1 = kH1[idx1]
            kr1 = kr1[idx1]
                    
            if use_w:
                w0 = w0[idx0]            
                w1 = w1[idx1] 
                
            if counter:
                counter0 = counter0[idx0]
                counter1 = counter1[idx1]
                
            if 'm_idx' in pipe_data:
                m_idx = torch.cat((idr0[m_idx[:, 0]].unsqueeze(1), idr1[m_idx[:, 1]].unsqueeze(1)), dim=1)
        
    pipe_data_out = {}
                
    if 'kp' in pipe_data:
        pipe_data_out['kp'] = [kp0, kp1]
        pipe_data_out['kH'] = [kH0, kH1]
        pipe_data_out['kr'] = [kr0, kr1]
        
        if use_w:
            w0[:, :2] = kp0
            w1[:, :2] = kp1
            pipe_data_out['w'] = [w0, w1]
            
        if counter:
            pipe_data_out['k_counter'] = [counter0, counter1]
            
        if 'm_idx' in pipe_data:
            pipe_data_out['m_idx'] = m_idx
            pipe_data_out['m_val'] = m_val
            pipe_data_out['m_mask'] = m_mask
                
    return pipe_data_out


def sampling(sampling_mode, kp, kp_unsampled, kr, ms_idx, ms_val, ms_mask, counter=None):
    if (sampling_mode == 'raw') or (kp.shape[0] == 0): return kp
            
    if (sampling_mode == 'avg_all_matches'):                    
        if counter is None:
            counter = torch.full((kp.shape[0], ), 1, device=device, dtype=torch.bool)
        
    if (sampling_mode == 'avg_inlier_matches'):                
        if ms_idx is None:
            mask = torch.full((kp.shape[0], ), 1, device=device, dtype=torch.bool)
        else:            
            mask = torch.zeros(kp.shape[0], device=device, dtype=torch.bool)
            for i in torch.arange(ms_idx.shape[0]):
                mask[ms_idx[i]] = ms_mask[i]
                
        if counter is None:
            counter = torch.zeros(kp.shape[0], device=device)
            for i in torch.arange(ms_idx.shape[0]):
                if ms_mask[i]: counter[ms_idx[i]] = 1            
            
    if (sampling_mode == 'best'):
        if ms_idx is None:
            mask = torch.full((kp.shape[0], ), 1, device=device, dtype=torch.bool)  
            val = torch.full((kp.shape[0], ), 1, device=device)  
        else:
            mask = torch.zeros(kp.shape[0], device=device, dtype=torch.bool)
            for i in torch.arange(ms_idx.shape[0]):
                mask[ms_idx[i]] = ms_mask[i]
    
            val = torch.zeros(kp.shape[0], device=device)
            for i in torch.arange(ms_idx.shape[0]):
                val[ms_idx[i]] = ms_val[i]
    
    aux = kp.clone()
    idx = torch.arange(len(aux), device=device)
    for i in range(aux.shape[1] - 1, -1, -1):            
        sidx = torch.argsort(aux[:, i], stable=True)
        idx = idx[sidx]
        aux = aux[sidx]            
    
    i = 0
    j = 1
    while j < aux.shape[0]:
        if torch.all(aux[i] == aux[j]):
            j = j + 1
            continue

        if (sampling_mode == 'avg_all_matches'): 
            c_sum = counter[idx[i:j]].sum()
            kp[idx[i:j]] = (kp_unsampled[idx[i:j]] * counter[idx[i:j]].unsqueeze(-1)).sum(dim=0) / c_sum
            
        if (sampling_mode == 'avg_inlier_matches'):
            tmp = kp_unsampled[idx[i:j]]
            tmp_c = counter[idx[i:j]]
            tmp_mask = mask[idx[i:j]]
            if torch.any(tmp_mask):
                c_sum = tmp_c[tmp_mask].sum()                
                kp[idx[i:j]] = (tmp[tmp_mask] * tmp_c[tmp_mask].unsqueeze(-1)).sum(dim=0) / c_sum

        if (sampling_mode == 'best'):
            tmp_mask = torch.stack((mask[idx[i:j]], val[idx[i:j]], kr[idx[i:j]]), dim=1)
            
            max_idx = 0
            max_val = tmp_mask[0]
            for q in torch.arange(1, tmp_mask.shape[0]):
                if (max_val[0] < tmp_mask[q][0]) or ((max_val[0] == tmp_mask[q][0]) and (max_val[1] < tmp_mask[q][1])) or ((max_val[0] == tmp_mask[q][0]) and (max_val[1] == tmp_mask[q][1]) and (max_val[2] < tmp_mask[q][2])):
                    max_val = tmp_mask[q]
                    max_idx = q
            
            best = idx[i:j][max_idx]
            kp[idx[i:j]] = kp_unsampled[best]            
                        
        i = j     
        j = j + 1

    if (sampling_mode == 'avg_all_matches'):
        c_sum = counter[idx[i:j]].sum()
        kp[idx[i:j]] = (kp_unsampled[idx[i:j]] * counter[idx[i:j]].unsqueeze(-1)).sum(dim=0) / c_sum
        
    if (sampling_mode == 'avg_inlier_matches'):
        tmp = kp_unsampled[idx[i:j]]
        tmp_c = counter[idx[i:j]]
        tmp_mask = mask[idx[i:j]]
        if torch.any(tmp_mask):
            c_sum = tmp_c[tmp_mask].sum()                
            kp[idx[i:j]] = (tmp[tmp_mask] * tmp_c[tmp_mask].unsqueeze(-1)).sum(dim=0) / c_sum
    
    if (sampling_mode == 'best'):
        tmp_mask = torch.stack((mask[idx[i:j]], val[idx[i:j]], kr[idx[i:j]]), dim=1)
        
        max_idx = 0
        max_val = tmp_mask[0]
        for q in torch.arange(1, tmp_mask.shape[0]):
            if (max_val[0] < tmp_mask[q][0]) or ((max_val[0] == tmp_mask[q][0]) and (max_val[1] < tmp_mask[q][1])) or ((max_val[0] == tmp_mask[q][0]) and (max_val[1] == tmp_mask[q][1]) and (max_val[2] < tmp_mask[q][2])):
                max_val = tmp_mask[q]
                max_idx = q
        
        best = idx[i:j][max_idx]
        kp[idx[i:j]] = kp_unsampled[best]  
       
    return kp


def sortrows(kp, idx_prev=None, rank=None):    
    idx = torch.arange(kp.shape[0], device=device)

    if not (idx_prev is None):
        idx = idx[idx_prev]
        kp = kp[idx_prev]
        
        if not (rank is None):
            rank = rank[idx_prev]
        
    for i in range(kp.shape[1] - 1, -1, -1):            
        sidx = torch.argsort(kp[:, i], stable=True)
        idx = idx[sidx]
        kp = kp[sidx]

        if not (rank is None):
            rank = rank[sidx]

    idxa = torch.zeros(kp.shape[0], device=device, dtype=torch.int)
    idxb = torch.zeros(kp.shape[0], device=device, dtype=torch.int)

    k = 0
    cur = torch.zeros((0, 2), device=device)
    for i in range(kp.shape[0]):
        if (cur.shape[0] == 0) or (not torch.all(kp[i] == cur)):
            cur = kp[i]
            idxa[k] = idx[i]                                        
            k = k + 1
            
            if not (rank is None):
                cur_rank = rank[i]

        if not (rank is None):
            if cur_rank > rank[i]:
                cur_rank = rank[i]
                idxa[k - 1] = idx[i]
            
        idxb[idx[i]] = k - 1
            
    idxa = idxa[:k]

    return idxa, idxb

class pipeline_muxer_module:
    def __init__(self, id_more='', pipe_gather=pipe_union, pipeline=None):
        self.single_image = False
        self.pipeliner = True
        self.pass_through = False
        
        self.id_more = id_more                
        self.pipe_gather = pipe_gather
        
        if pipeline is None: pipeline = []
        self.pipeline = pipeline        

        self.id_string = 'pipeline_muxer'
        if len(self.id_more): self.id_string = self.id_string + '_' + str(self.id_more)        


    def get_id(self): 
        return self.id_string


    def run(self, db=None, force=False, pipe_data=None, pipe_name='/'):
        if pipe_data is None: pipe_data = {}

        pipe_data_block = []
        
        for pipeline in self.pipeline:
            pipe_data_in = pipe_data.copy()
            pair = pipe_data['img']
                                       
            pipe_data_out, pipe_name_out = run_pipeline(pair, pipeline, db, force=force, pipe_data=pipe_data_in, pipe_name=pipe_name)        
            pipe_data_block.append(pipe_data_out)
        
        return self.pipe_gather(pipe_data_block)


class deep_joined_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False
        self.pass_through = False
                        
        self.what = 'superpoint'
        self.args = { 
            'id_more': '',
            'patch_radius': 16,            
            'num_features': 8000,
            'resize': 1024,           # this is default, set to None to disable
            'aliked_model': "aliked-n16rot",          # default is "aliked-n16"
            }
                        
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


class lightglue_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        
        self.what = 'superpoint'
        self.args = {
            'id_more': '',
            'num_features': 8000,
            'resize': 1024,           # this is default, set to None to disable
            'aliked_model': "aliked-n16rot",          # default is "aliked-n16"
            }

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
    

    def run(self, **args):           
        # dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
        # dict_keys(['matches0', 'matches1', 'matching_scores0', 'matching_scores1', 'stop', 'matches', 'scores', 'prune0', 'prune1'])

        width, height = Image.open(args['img'][0]).size
        sz1 = torch.tensor([width / 2, height / 2], device=device)

        width, height = Image.open(args['img'][1]).size
        sz2 = torch.tensor([width / 2, height / 2], device=device)

        feats1 = {'keypoints': args['kp'][0].unsqueeze(0), 'descriptors': args['desc'][0].unsqueeze(0), 'image_size': sz1.unsqueeze(0)} 
        feats2 = {'keypoints': args['kp'][1].unsqueeze(0), 'descriptors': args['desc'][1].unsqueeze(0), 'image_size': sz2.unsqueeze(0)} 
        
        matches12 = self.matcher({'image0': feats1, 'image1': feats2})
        feats1_, feats2_, matches12 = [lg_rbd(x) for x in [feats1, feats2, matches12]]


        idxs = matches12['matches'].squeeze(0)
        m_val = matches12['scores'].squeeze(0)
        m_mask = torch.ones(idxs.shape[0], device=device, dtype=torch.bool)
            
        return {'m_idx': idxs, 'm_val': m_val, 'm_mask': m_mask}
    

class loftr_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
                        
        self.args = {
            'id_more': '',
            'outdoor': True,
            'resize': None,                          # self.resize = [800, 600]
            'patch_radius': 16,
            }

        self.id_string, self.args = set_args('loftr', args, self.args)        

        if self.args['outdoor'] == True:
            pretrained = 'outdoor'
        else:
            pretrained = 'indoor_new'

        self.matcher = K.feature.LoFTR(pretrained=pretrained).to(device).eval()


    def get_id(self): 
        return self.id_string


    def run(self, **args):
        image0 = K.io.load_image(args['img'][0], K.io.ImageLoadType.GRAY32, device=device)
        image1 = K.io.load_image(args['img'][1], K.io.ImageLoadType.GRAY32, device=device)

        hw1 = image0.shape[1:]
        hw2 = image1.shape[1:]

        if not (self.args['resize'] is None):        
            ms = min(self.resize)
            Ms = max(self.resize)

            if hw1[0] > hw1[1]:
                sz = [Ms, ms]                
                ratio_ori = float(hw1[0]) / hw1[1]                 
            else:
                sz = [ms, Ms]
                ratio_ori = float(hw1[1]) / hw1[0]

            ratio_new = float(Ms) / ms
            if np.abs(ratio_ori - ratio_new) > np.abs(1 - ratio_new):
                sz = [ms, ms]

            K.geometry.resize(image0, (sz[0], sz[1]), antialias=True)

            if hw2[0] > hw2[1]:
                sz = [Ms, ms]                
                ratio_ori = float(hw2[0]) / hw2[1]                 
            else:
                sz = [ms, Ms]
                ratio_ori = float(hw2[1]) / hw2[0]

            ratio_new = float(Ms) / ms
            if np.abs(ratio_ori - ratio_new) > np.abs(1 - ratio_new):
                sz = [ms, ms]

            K.geometry.resize(image1, (sz[0], sz[1]), antialias=True)
                    
        hw1_ = image0.shape[1:]
        hw2_ = image1.shape[1:]

        input_dict = {
            "image0": image0.unsqueeze(0),    # LofTR works on grayscale images
            "image1": image1.unsqueeze(0),
        }

        correspondences = self.matcher(input_dict)

        kps1 = correspondences["keypoints0"]
        kps2 = correspondences["keypoints1"]
        m_val = correspondences['confidence']
                        
        kps1 = kps1.detach().to(device).squeeze()
        kps2 = kps2.detach().to(device).squeeze()

        kps1[:, 0] = kps1[:, 0] * (hw1[1] / float(hw1_[1]))
        kps1[:, 1] = kps1[:, 1] * (hw1[0] / float(hw1_[0]))
    
        kps2[:, 0] = kps2[:, 0] * (hw2[1] / float(hw2_[1]))
        kps2[:, 1] = kps2[:, 1] * (hw2[0] / float(hw2_[0]))
        
        kp = [kps1, kps2]
        kH = [
            torch.zeros((kp[0].shape[0], 3, 3), device=device),
            torch.zeros((kp[0].shape[0], 3, 3), device=device),
            ]
        
        kH[0][:, [0, 1], 2] = -kp[0] / self.args['patch_radius']
        kH[0][:, 0, 0] = 1 / self.args['patch_radius']
        kH[0][:, 1, 1] = 1 / self.args['patch_radius']
        kH[0][:, 2, 2] = 1

        kH[1][:, [0, 1], 2] = -kp[1] / self.args['patch_radius']
        kH[1][:, 0, 0] = 1 / self.args['patch_radius']
        kH[1][:, 1, 1] = 1 / self.args['patch_radius']
        kH[1][:, 2, 2] = 1

        kr = [torch.full((kp[0].shape[0],), torch.nan, device=device), torch.full((kp[0].shape[0],), torch.nan, device=device)]        

        m_idx = torch.zeros((kp[0].shape[0], 2), device=device, dtype=torch.int)
        m_idx[:, 0] = torch.arange(kp[0].shape[0])
        m_idx[:, 1] = torch.arange(kp[0].shape[0])

        m_mask = torch.ones(m_idx.shape[0], device=device, dtype=torch.bool)

        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}

class sampling_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        
        self.args = {
            'id_more': '',
            'unique': True,
            'no_unmatched': True,
            'only_matched': True,
            'sampling_mode': 'raw', # None, raw, best, avg_inlier_matches, avg_all_matches
            'sampling_scale': 1,
            'sampling_offset': 0,
            }

        self.id_string, self.args = set_args('sampling', args, self.args)        


    def get_id(self): 
        return self.id_string
    

    def run(self, **args):           
        pipe_data = args

        return pipe_union(pipe_data, unique=self.args['unique'],
                          no_unmatched=self.args['no_unmatched'],
                          only_matched=self.args['only_matched'],
                          sampling_mode=self.args['sampling_mode'],
                          sampling_scale=self.args['sampling_scale'],
                          sampling_offset=self.args['sampling_offset'])    


class coldb_ext(coldb.COLMAPDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def get_image_id(self, image):
        cursor = self.execute(
            "SELECT image_id FROM images WHERE name=?",
            (image, ),
        )
        image_id = cursor.fetchone()
        if not (image_id is None): image_id = image_id[0]
        return image_id

        
    def get_keypoints(self, image_id):
        cursor = self.execute("SELECT data, rows, cols FROM keypoints where image_id=?", (image_id, ))
        kpts = cursor.fetchone()
        if kpts is None:
            return None
        else:
            k, r, c = kpts
            return np.reshape(coldb.blob_to_array(kpts[0], np.float32), (r, c))


    def update_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        
        if self.get_keypoints(image_id) is None:
            self.add_keypoints(image_id, keypoints)
        else: 
            self.execute(
                "UPDATE keypoints SET rows=?, cols=?, data=? WHERE image_id=?",
                keypoints.shape + (coldb.array_to_blob(keypoints), ) + (image_id, ),
                )


    def get_matches(self, image_id1, image_id2):
        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)
        cursor = self.execute("SELECT data, rows, cols FROM matches where pair_id=?", (pair_id, ))
        m = cursor.fetchone()
        if m is None:
            return None
        else:
            m, r, c = m
            m = np.reshape(coldb.blob_to_array(m[0], np.float32), (r, c))
            if image_id1 > image_id2: m = m[:, ::-1]
            return m


    def update_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)

        if self.get_matches(image_id1, image_id2) is None:
            self.add_matches(image_id1, image_id2, matches)
        else:
            matches = np.asarray(matches, np.uint32)

            if image_id1 > image_id2:
                matches = matches[:, ::-1]

            self.execute(
                "UPDATE matches SET rows=?, cols=?, data=? WHERE pair_id=?",
                matches.shape + (coldb.array_to_blob(matches), ) + (pair_id, ),
                )


def kpts_as_colmap(idx, **args): 
    kp = args['kp'][idx]
    kH = args['kH'][idx]
     
    t = torch.zeros((kp.shape[0], 3, 3), device=device)        
    t[:, [0, 1], 2] = -kH[:, [0, 1], 2]
    t[:, 0, 0] = 1
    t[:, 1, 1] = 1
    t[:, 2, 2] = 1           
     
    h = t.bmm(kH.inverse())
     
    v = torch.zeros((kp.shape[0], 3, 3), device=device)        
    v[:, 2, :] = h[:, 2, :]
    v[:, 0, 0] = 1
    v[:, 1, 1] = 1
     
    w = h.bmm(v.inverse())
    w = w[:, :2, :2].reshape(-1, 4)
         
    return torch.cat((kp[:, :2], w), dim=1)


SIMPLE_RADIAL = 2

UNDEFINED = 0  # Not provided
DEGENERATE = 1 # Degenerate configuration (e.g., no overlap or not enough inliers).
CALIBRATED = 2 # Essential matrix.
UNCALIBRATED = 3 # Fundamental matrix.
PLANAR = 4 # Homography, planar scene with baseline.
PANORAMIC = 5 # Homography, pure rotation without baseline.
PLANAR_OR_PANORAMIC = 6 # Homography, planar or panoramic.
WATERMARK = 7 # Watermark, pure 2D translation in image borders.
MULTIPLE = 8 # Multi-model configuration, i.e. the inlier matches result from multiple individual, non-degenerate configurations.

class to_colmap_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True

        self.args = {
            'id_more': '',
            'db': 'colmap.db',
            'aux_hdf5': 'colmap_aux.hdf5',
            'focal_cf': 1.2,
            'only_keypoints': False,            
            'unique': True,
            'only_matched': True,
            'no_unmatched': True,
            'sampling_mode': 'raw',
            'sampling_scale': 1,
            'sampling_offset': 0,
        }
        
        self.id_string, self.args = set_args('colmap' , args, self.args)

        self.db = coldb_ext(self.args['db'])
        self.db.create_tables()
        self.aux_hdf5 = None
        if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):         
            self.aux_hdf5 = pickled_hdf5.pickled_hdf5(self.args['aux_hdf5'], mode='a', label_prefix='pickled/' + self.id_string)
                

    def __del__(self):
        self.db.close()
        if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):
            self.aux_hdf5.close()
            if os.path.isfile(self.args['aux_hdf5']):
                os.remove(self.args['aux_hdf5'])

                
    def get_id(self): 
        return self.id_string

    
    def run(self, **args):   
        im_ids = []
        imgs = []
        
        for idx in [0, 1]:
            im = args['img'][idx]            
            _, img = os.path.split(im)
            
            im_id = self.db.get_image_id(img)
            if  im_id is None:
                w, h = Image.open(im).size
                cam_id = self.db.add_camera(SIMPLE_RADIAL, w, h, np.array([self.args['focal_cf'] * max(w, h), w / 2, h / 2, 0]))
                im_id = self.db.add_image(img, cam_id)
                self.db.commit()

            imgs.append(img)
            im_ids.append(im_id)
                
        pipe_old = {}
        
        kp_old0 = self.db.get_keypoints(im_ids[0])
        if kp_old0 is None:
            w_old0 = torch.zeros((0, 6), device=device)
            kp_old0 = torch.zeros((0, 2), device=device)
            if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):            
                k_count_old0 = torch.zeros(0, device=device)
        else:
            w_old0 = torch.tensor(kp_old0, device=device)
            kp_old0 = torch.tensor(kp_old0[:, :2], device=device)
            if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):            
                k_count_old0, _ = self.aux_hdf5.get(imgs[0])
            
        kH_old0 = torch.zeros((kp_old0.shape[0], 3, 3), device=device)
        kr_old0 = torch.full((kp_old0.shape[0], ), torch.inf, device=device)
        
        kp_old1 = self.db.get_keypoints(im_ids[1])
        if kp_old1 is None:
            w_old1 = torch.zeros((0, 6), device=device)
            kp_old1 = torch.zeros((0, 2), device=device)
            if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):            
                k_count_old1 = torch.zeros(0, device=device)
        else:
            w_old1 = torch.tensor(kp_old1, device=device)
            kp_old1 = torch.tensor(kp_old1[:, :2], device=device)
            if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):            
                k_count_old1, _ = self.aux_hdf5.get(imgs[1])
            
        kH_old1 = torch.zeros((kp_old1.shape[0], 3, 3), device=device)
        kr_old1 = torch.full((kp_old1.shape[0], ), torch.inf, device=device)

        m_idx_old = self.db.get_matches(im_ids[0], im_ids[1])    
        if m_idx_old is None:
            m_idx_old = torch.zeros((0, 2), device=device, dtype=torch.int)
        else:
            m_idx_old = torch.tensor(m_idx_old, device=device, dtype=torch.int)
        
        m_val_old = torch.full((m_idx_old.shape[0], ), torch.inf, device=device)
        m_mask_old = torch.full((m_idx_old.shape[0], ), 1, device=device, dtype=torch.bool)
            
        pipe_old['kp'] = [kp_old0, kp_old1]
        pipe_old['kH'] = [kH_old0, kH_old1]
        pipe_old['kr'] = [kr_old0, kr_old1]
        pipe_old['w'] = [w_old0, w_old1]
        
        if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):        
            pipe_old['k_counter'] = [k_count_old0, k_count_old1]

        pipe_old['m_idx'] = m_idx_old
        pipe_old['m_val'] = m_val_old
        pipe_old['m_mask'] = m_mask_old
                
        w0 = kpts_as_colmap(0, **args)
        w1 = kpts_as_colmap(1, **args)
        args['w'] = [w0, w1]

        if (self.args['sampling_mode'] == 'avg_all_matches'):
            k_count0 = torch.full((args['kp'][0].shape[0], ), 1, device=device)
            k_count1 = torch.full((args['kp'][1].shape[0], ), 1, device=device)
            args['k_counter'] = [k_count0, k_count1]
        
        if (self.args['sampling_mode'] == 'avg_inlier_matches'):            
            if 'm_idx' in args:            
                k_count0 = torch.full((args['kp'][0].shape[0], ), 0, device=device)
                for i in torch.arange(args['m_mask'].shape[0]):
                    if args['m_mask'][i]:
                        k_count0[args['m_idx'][i, 0]] = 1

                k_count1 = torch.full((args['kp'][1].shape[0], ), 0, device=device)
                for i in torch.arange(args['m_mask'].shape[0]):
                    if args['m_mask'][i]:
                        k_count1[args['m_idx'][i, 1]] = 1
            else:
                k_count0 = torch.full((args['kp'][0].shape[0], ), 1, device=device)
                k_count1 = torch.full((args['kp'][1].shape[0], ), 1, device=device)
            
            args['k_counter'] = [k_count0, k_count1]

        counter = (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches')
        pipe_out = pipe_union([pipe_old, args], unique=self.args['unique'], no_unmatched=self.args['no_unmatched'], only_matched=self.args['only_matched'], sampling_mode=self.args['sampling_mode'], sampling_scale=self.args['sampling_scale'], sampling_offset=self.args['sampling_offset'], preserve_order=True, counter=counter)

        pts0 = pipe_out['w'][0].to('cpu').numpy()
        pts1 = pipe_out['w'][1].to('cpu').numpy()
        
        if counter:
            self.aux_hdf5.add(imgs[0], pipe_out['k_counter'][0])
            self.aux_hdf5.add(imgs[1], pipe_out['k_counter'][1])
        
        self.db.update_keypoints(im_ids[0], pts0)
        self.db.update_keypoints(im_ids[1], pts1)

        if not self.args['only_keypoints']:
            m_idx = pipe_out['m_idx'].to('cpu').numpy()
            self.db.update_matches(im_ids[0], im_ids[1], m_idx)

        self.db.commit()
        
        return {}


if __name__ == '__main__':    
    with torch.inference_mode():     
#       pipeline = [
#           dog_module(),
#         # show_kpts_module(id_more='first', prepend_pair=False),
#           patch_module(),
#         # show_kpts_module(id_more='second', img_prefix='orinet_affnet_', prepend_pair=True),
#           deep_descriptor_module(),
#           smnn_module(),
#           magsac_module(),
#         # show_kpts_module(id_more='third', img_prefix='ransac_', prepend_pair=True, mask_idx=[0, 1]),
#         # show_matches_module(id_more='forth', img_prefix='matches_', mask_idx=[1, 0]),
#         # show_matches_module(id_more='fifth', img_prefix='matches_inliers_', mask_idx=[1]),
#         # show_matches_module(id_more='sixth', img_prefix='matches_all_', mask_idx=-1),
#           show_matches_module(id_more='only', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#       ]

#       pipeline = [
#           loftr_module(),
#           show_kpts_module(id_more='first', prepend_pair=False),
#           magsac_module(),
#           show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#       ]

#       pipeline = [
#           deep_joined_module(),
#           show_kpts_module(id_more='first', prepend_pair=False),
#           lightglue_module(),
#           magsac_module(),
#           show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#       ]

#       pipeline = [
#           image_muxer_module(pair_generator=pair_rot4, pipe_gather=pipe_max_matches, pipeline=[
#               hz_module(),
#               patch_module(sift_orientation=True, orinet=False),
#               deep_descriptor_module(),
#               show_kpts_module(id_more='first', prepend_pair=False),
#               smnn_module(),
#               magsac_module(),
#               show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           ]),
#           show_kpts_module(id_more='third', img_prefix='best_rot_', prepend_pair=False),
#           show_matches_module(id_more='fourth', img_prefix='best_rot_matches_', mask_idx=[1, 0], prepend_pair=False),            
#       ]

#       pipeline = [
#           image_muxer_module(pair_generator=pair_rot4, pipe_gather=pipe_max_matches, pipeline=[
#               deep_joined_module(),
#               show_kpts_module(id_more='first', prepend_pair=False),
#               lightglue_module(),
#               magsac_module(),
#               show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           ]),
#           show_kpts_module(id_more='third', img_prefix='best_rot_', prepend_pair=False),
#           show_matches_module(id_more='fourth', img_prefix='best_rot_matches_', mask_idx=[1, 0], prepend_pair=False),            
#       ]
       
#       pipeline = [
#           pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
#               [
#                   loftr_module(),
#                   show_kpts_module(id_more='a_first', img_prefix='a_', prepend_pair=False),
#                   magsac_module(),
#                   show_matches_module(id_more='a_second', img_prefix='a_matches_', mask_idx=[1, 0], prepend_pair=False),
#               ],
#               [
#                   deep_joined_module(),
#                   show_kpts_module(id_more='b_first', img_prefix='b_', prepend_pair=False),
#                   lightglue_module(),
#                   magsac_module(),
#                   show_matches_module(id_more='b_second', img_prefix='b_matches_', mask_idx=[1, 0], prepend_pair=False),                    
#               ],
#           ]),
#           show_kpts_module(id_more='third', img_prefix='union_', prepend_pair=False),
#           show_matches_module(id_more='fourth', img_prefix='union_matches_', mask_idx=[1, 0], prepend_pair=False),            
#       ]
        
        
#       pipeline = [
#           pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
#               [
#                   deep_joined_module(),
#                   show_kpts_module(id_more='a_first', img_prefix='a_', prepend_pair=False),
#                   lightglue_module(),
#                   magsac_module(),
#                   show_matches_module(id_more='a_second', img_prefix='a_matches_', mask_idx=[1, 0], prepend_pair=False),                    
#               ],
#               [
#                   deep_joined_module(),
#                   show_kpts_module(id_more='b_first', img_prefix='b_', prepend_pair=False),
#                   lightglue_module(),
#                   magsac_module(),
#                   show_matches_module(id_more='b_second', img_prefix='b_matches_', mask_idx=[1, 0], prepend_pair=False),                    
#               ],
#           ]),
#           show_kpts_module(id_more='third', img_prefix='union_', prepend_pair=False),
#           show_matches_module(id_more='fourth', img_prefix='union_matches_', mask_idx=[1, 0], prepend_pair=False),            
#       ]        

#       pipeline = [
#           loftr_module(),
#           magsac_module(),
#           show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           sampling_module(sampling_mode='avg_inlier_matches', sampling_scale=20),
#           show_matches_module(id_more='second', img_prefix='matches_sampled_', mask_idx=[1, 0], prepend_pair=False),
#       ]

        pipeline = [
            loftr_module(),
            magsac_module(),
            show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
            to_colmap_module(),
        ]        
                
#       imgs = '../data/ET_random_rotated'
        imgs = '../data/ET'
        run_pairs(pipeline, imgs)
