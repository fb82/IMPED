import os
import warnings
import pickled_hdf5.pickled_hdf5 as pickled_hdf5
import time

import torch
import kornia as K
from kornia_moons.feature import opencv_kpts_from_laf, laf_from_opencv_kpts
import cv2
import numpy as np
import hz.hz as hz
from PIL import Image
import poselib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def image_pairs(to_list, add_path='', check_img=True):
    imgs = []

    # to_list is effectively an image folder
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
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    yield imgs[i], imgs[j]        
        
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
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    yield imgs[i], imgs[j]

        # dir_name is a list of image pairs
        else:
            warnings.warn("reading image pairs")

            for i, j in file_list:
                ii = os.path.join(add_path, i)
                jj = os.path.join(add_path, j)

                if check_img:
                    try:
                        Image.open(ii).verify()
                        Image.open(jj).verify()
                    except:
                        continue

                yield ii, jj


def run_pairs(pipeline, imgs, db_name='database.hdf5', db_mode='a', force=False):    
    db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)

    for pair in image_pairs(imgs):
        with torch.inference_mode():        
            run_pipeline(pair, pipeline, db, force=force)

                
def run_pipeline(pair, pipeline, db, force=False, pipe_data={}, pipe_name=''):        
    if not pipe_data:
        pipe_data['img'] = [pair[0], pair[1]]
        pipe_data['warp'] = [torch.eye(3, device=device, dtype=torch.float), torch.eye(3, device=device, dtype=torch.float)]
        
    for pipe_module in pipeline:
        if pipe_name == '':
            pipe_name = pipe_module.get_id()
        else:
            pipe_name = pipe_name + '/' + pipe_module.get_id()
        
        if hasattr(pipe_module, 'single_image') and pipe_module.single_image:            
            for n in range(len(pipe_data['img'])):
                im = os.path.split(pipe_data['img'][n])[-1]
                data_key = '/' + im + '/' + pipe_name + '/data'                    

                out_data, is_found = db.get(data_key)                    
                if (not is_found) or force:
                    start_time = time.time()
                    out_data = pipe_module.run(idx=n, **pipe_data)
                    stop_time = time.time()
                    out_data['running_time'] = stop_time - start_time
                    db.add(data_key, out_data)

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
            data_key = '/' + im0 + '/' + im1 + '/' + pipe_name + '/data'   

            out_data, is_found = db.get(data_key)                    
            if (not is_found) or force:
                start_time = time.time()

                if hasattr(pipe_module, 'pipeliner') and pipe_module.pipeliner:
                    out_data = pipe_module.run(pipe_data=pipe_data, pipe_name=pipe_name, db=db, force=force)
                else:
                    out_data = pipe_module.run(**pipe_data)

                stop_time = time.time()
                out_data['running_time'] = stop_time - start_time
                db.add(data_key, out_data)

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
            id_string = id_string + '_' + k + '_' + str(v)

    id_string = id_string.lower()
    
    return id_string, args_    


class sift_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False                
        self.args = {
            'upright': False,
            'params': {'nfeatures': 8000, 'contrastThreshold': -10000, 'edgeThreshold': 10000},
        }

        self.id_string, self.args = set_args('sift', args, self.args)
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
                
        kp = laf_from_opencv_kpts(kp, device=device)
        kp, kH = laf2homo(kp.squeeze(0).detach().to(device))
    
        return {'kp': kp, 'kH': kH}


class keynet_module:
    def __init__(self, **args):
        self.single_image = True        
        self.pipeliner = False        
        self.args = {
            'params': {'num_features': 8000},
        }
        
        self.id_string, self.args = set_args('keynet', args, self.args)
        self.detector = K.feature.KeyNetDetector(**self.args['params']).to(device)


    def get_id(self):
        return self.id_string
        
    
    def run(self, **args):  
        img = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)
        kp, _ = self.detector(img)        
        kp, kH = laf2homo(kp.squeeze(0).detach().to(device))
    
        return {'kp': kp, 'kH': kH}


class hz_plus_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.args = {
            'params': {'max_max_pts': 8000, 'block_mem': 16*10**6}
        }
        
        self.id_string, self.args = set_args('hz_plus', args, self.args)

                
    def get_id(self): 
        return self.id_string

    
    def run(self, **args):    
        img = hz.load_to_tensor(args['img'][args['idx']]).to(torch.float)
        kp = hz.hz_plus(img, output_format='laf', **self.args['params'])
        kp, kH = laf2homo(K.feature.ellipse_to_laf(kp[None]).squeeze(0))
    
        return {'kp': kp, 'kH': kH}


class deep_patch_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.args = {
            'orinet': True,
            'orinet_params': {},
            'affnet': True,
            'affnet_params': {},
            }

        self.id_string, self.args = set_args('deep_patch', args, self.args)

        if self.args['orinet']:
            self.ori_module = K.feature.LAFOrienter(angle_detector=K.feature.OriNet().to(device), **self.args['orinet_params'])
        else:
            self.ori_module = K.feature.PassLAF()

        if self.args['affnet']:
            self.aff_module = K.feature.LAFAffineShapeEstimator(**self.args['affnet_params'])
        else:
            self.aff_module = K.feature.PassLAF()


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
        self.args = {
            'descriptor': 'hardnet',
            'desc_params': {},
            'patch_params': {},
            }

        self.id_string, self.args = set_args('deep_descriptor', args, self.args)        
        
        if self.args['descriptor'] == 'hardnet':
            desc = K.feature.HardNet().to(device)
        if self.args['descriptor'] == 'sosnet':
            desc = K.feature.SOSNet().to(device)
        if self.args['descriptor'] == 'hynet':
            desc = K.feature.HyNet(**self.args['desc_params']).to(device)

        self.ddesc = K.feature.LAFDescriptor(patch_descriptor_module=desc, **self.args['patch_params'])


    def get_id(self): 
        return self.id_string


    def run(self, **args):    
        im = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)

        lafs = homo2laf(args['kp'][args['idx']], args['kH'][args['idx']])
        desc = self.ddesc(im, lafs).squeeze(0)
    
        return {'desc': desc}


class sift_descriptor_module:
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.args = {
            'rootsift': True,
            }
        
        self.id_string, self.args = set_args('sift_descriptor', args, self.args)        
        self.descriptor = cv2.SIFT_create()


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
        self.args = {
            'th': 0.95,
            }
        
        self.id_string, self.args = set_args('smnn', args, self.args)        


    def get_id(self): 
        return self.id_string


    def run(self, **args):
        val, idxs = K.feature.match_smnn(args['desc'][0], args['desc'][1], self.args['th'])

        return {'m_idx': idxs, 'm_val': val.squeeze(1), 'm_mask': torch.ones(idxs.shape[0], device=device, dtype=torch.bool)}


def pair_rot4(pair, cache_path='tmp_imgs', force=False):

    yield pair, [torch.eye(3, device=device, dtype=torch.float), torch.eye(3, device=device, dtype=torch.float)]

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
                                            
        m0 = [[1, 0,  c[(0 + r + 1) % 2]],
              [0, 1,  c[(1 + r + 1) % 2]],
              [0, 0,          1        ]]

        rot_mat = np.asarray([[0, -1], [1, 0]]) @ rot_mat
        m1 = np.eye(3)
        m1[:2, :2] = rot_mat

        m2 = [[1, 0, -c[0]],
              [0, 1, -c[1]],
              [0, 0,    1 ]]

        # from warped to original
        warp_matrix = torch.tensor(m2 @ m1 @ m0, device=device, dtype=torch.float)
            
        yield (pair[0], new_img), [torch.eye(3, device=device, dtype=torch.float), warp_matrix]


def pipe_max_matches(pipe_block):
    n_matches = torch.zeros(len(pipe_block), device=device)
    for i in range(len(pipe_block)):
        if 'm_mask' in pipe_block[i]:
            n_matches[i] = pipe_block[i]['m_mask'].sum()
    
    best = n_matches.max(0)[1]
    
    return pipe_block[best]
        

class image_muxer_module:
    def __init__(self, what='default', cache_path='tmp_imgs', pair_generator=pair_rot4, pipe_gather=pipe_max_matches, pipeline=[]):
        self.single_image = False
        self.pipeliner = True
        
        self.cache_path = cache_path
        self.pair_generator = pair_generator
        self.pipe_gather = pipe_gather
        self.pipeline = pipeline
        
        self.id_string = ('image_muxer_' + str(what)).lower()        


    def get_id(self): 
        return self.id_string


    def run(self, db=None, force=False, pipe_data={}, pipe_name=''):
        pair = pipe_data['img']
        warp = pipe_data['warp']
        pipe_data_block = []
        
        for pair_, warp_ in image_pairs(self.pair_generator(pair, cache_path=self.cache_path, force=force)):
            pipe_data_in = pipe_data.copy()
            pipe_data_in['img'] = [pair_[0], pair_[1]]
            pipe_data_in['warp'] = [warp_[0], warp_[1]]
            
            if 'kp' in pipe_data:
                pipe_data_in['kp'] = [    
                    apply_homo(pipe_data['kp'][0], warp_[0].inverse()),
                    apply_homo(pipe_data['kp'][1], warp_[1].inverse())
                    ]

            if 'kH' in pipe_data:
                pipe_data_in['kH'] = [    
                    change_patch_homo(pipe_data['kp'][0], pipe_data['kH'][0], warp_[0]),
                    change_patch_homo(pipe_data['kp'][1], pipe_data['kH'][1], warp_[1]),
                    ]
                                       
            pipe_data_out, pipe_name_out = run_pipeline(pair_, pipeline, db, force=force, pipe_data=pipe_data_in, pipe_name=pipe_name)

            pipe_data_out['img'] = pair
            pipe_data_out['warp'] = warp

            if 'kp' in pipe_data_out:
                pipe_data_out['kp'] = [    
                    apply_homo(pipe_data_out['kp'][0], warp_[0]),
                    apply_homo(pipe_data_out['kp'][1], warp_[1])
                    ]

            if 'kH' in pipe_data_out:
                pipe_data_in['kH'] = [    
                    change_patch_homo(pipe_data_out['kp'][0], pipe_data_out['kH'][0], warp_[0].inverse()),
                    change_patch_homo(pipe_data_out['kp'][1], pipe_data_out['kH'][1], warp_[1].inverse()),
                    ]
        
            pipe_data_block.append(pipe_data_out)
        
        return self.pipe_gather(pipe_data_block)
        

def change_patch_homo(kp, kH, warp):
    
    pt_old = torch.zeros((kp.shape[0], 3), device=device)
    pt_old[:, :2] = kp
    pt_old[:, 2] = 1
    pt_old = pt_old.permute((1,0))

    pt_new = warp.inverse() @ pt_old
    pt_new / pt_new [:, 2]    

    t_old = torch.zeros((kp.shape[0], 3, 3), device=device)
    t_new = torch.zeros((kp.shape[0], 3, 3), device=device)

    t_old [:, :2, 2] =  pt_old.unsqueeze(-1)
    t_new [:, :2, 2] = -pt_new.unsqueeze(-1)    
    kH_ = (kH.bmm(t_new) @ warp.unsqueeze(0)).bmm(t_old)
    
    return kH_


def apply_homo(p, H):
    
    pt = torch.zeros((p.shape[0], 3), device=device)
    pt[:, :2] = p
    pt[:, 2] = 1
    pt_ = (H @ pt.permute((1, 0))).permute((1, 0))
    return pt_ [:, :2] / pt_[:, 2]    


class magsac_module:
    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False        
        self.args = {
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
        
        pt1 = pt1_[mi[mm][0]]
        pt2 = pt2_[mi[mm][1]]
        
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
 
        aux = mm[:]
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
        self.args = {
            'mode': 'fundamental_matrix',
            'conf': 0.9999,
            'max_iters': 100000,
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
        
        pt1 = pt1_[mi[mm][0]]
        pt2 = pt2_[mi[mm][1]]
        
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

        if not isinstance(mask, np.ndarray):
            mask = torch.zeros(pt1.shape[0], device=device, dtype=torch.bool)
        else:
            mask = torch.tensor(mask, device=device, dtype=torch.bool)
 
        aux = mm[:]
        mm[aux] = mask
        
        if not (F is None):
            F = torch.tensor(F, device=device)
        
        if self.args['mode'] == 'fundamental_matrix':
            return {'m_mask': mm, 'F': F}
        else:
            return {'m_mask': mm, 'H': F}


def pipe_union(pipe_block, unique=True):

    kp0 = []
    kH0 = []

    kp1 = []
    kH1 = []
    
    m_idx = []
    m_val = []
    m_mask = []
    
    m0_offset = 0
    m1_offset = 0
    
    for pipe_data in pipe_block:
        if 'kp' in pipe_data:
        
            kp0.append(pipe_data['kp'][0])
            kp1.append(pipe_data['kp'][1])
    
            kH0.append(pipe_data['kH'][0])
            kH1.append(pipe_data['kH'][1])
    
            if 'm_idx' in pipe_data:
                m_idx.append(pipe_data['m_val'] + torch.tensor([m0_offset, m1_offset], device=device).unsqueeze(0))
                m_val.append(pipe_data['m_val'])
                m_mask.append(pipe_data['m_mask'])
    
                m0_offset = m0_offset + pipe_data['kp'][0].shape[0]
                m1_offset = m1_offset + pipe_data['kp'][1].shape[1]

    if 'kp' in pipe_data:
        kp0 = torch.cat(kp0)
        kp1 = torch.cat(kp1)

        kH0 = torch.cat(kH0)
        kH1 = torch.cat(kH1)

        if 'm_idx' in pipe_data:
            m_idx = torch.cat(m_idx)
            m_val = torch.cat(m_val)
            m_mask = torch.cat(m_mask)

    if unique:
        if 'm_idx' in pipe_data:
            idx = torch.argsort(m_val)

            m_idx = m_idx[idx]
            m_val = m_val[idx]
            m_mask = m_mask[idx]

            idx = torch.argsort(m_mask, descending=True, stable=True)

            m_idx = m_idx[idx]
            m_val = m_val[idx]
            m_mask = m_mask[idx]

            idx0 = torch.zeros(kp0.shape[0], device=device, dtype=torch.int)
            idx0[:] = m_idx.shape[0] + 1
            for i in range(m_idx.shape[0] - 1,-1,-1):
                idx0[m_idx[i, 0]] = i
            
            idx0 = torch.argsort(idx0)
            
            idx1 = torch.zeros(kp0.shape[0], device=device, dtype=torch.int)
            idx1[:] = m_idx.shape[0] + 1
            for i in range(m_idx.shape[0] - 1,-1,-1):
                idx1[m_idx[i, 1]] = i
            
            idx1 = torch.argsort(idx1)
            

        if 'kp' in pipe_data:
            kp0, idx0, idx0_rev = sortrows(kp0)
            kp1, idx1, idx1_rev = sortrows(kp1)
            
            if 'm_idx' in pipe_data:
                m_idx_new = torch.zeros_like(m_idx)
                m_idx_new[:, 0] = idx0_rev[m_idx[0], 0]
                m_idx_new[:, 1] = idx1_rev[m_idx[1], 0]
            
    return pipe_block[0]


def sortrows(kp):
    idx = torch.arange(len(kp))

    for i in range(kp.shape[1] - 1,-1,-1):            
        _, sidx = torch.sort(kp, dim=i, stable=True)
        idx = idx[sidx]
        kp = kp[sidx]            

    kp_ = torch.zeros(kp.shape, device=device)
    iidx = torch.zeros((kp.shape[0], 2), device=device)

    k = 0
    uk_cur = kp[0]
    for i in range(kp.shape[0]):
        if torch.all(kp[i] == uk_cur):
            iidx[i, 0] = k
            iidx[i, 1] = idx[i]
        else:
            kp_[k] = uk_cur
            k = k + 1                                        
            uk_cur = kp[i]

    _, aux = torch.sort(iidx, dim=1)
    iidx_rev = iidx[aux,[1, 0]] 
    
    return kp_[:k + 1], iidx, iidx_rev


class pipeline_muxer_module:
    def __init__(self, what='default', pipe_gather=pipe_union, pipeline=[]):
        self.single_image = False
        self.pipeliner = True
        self.pipe_gather = pipe_gather
        self.pipeline = pipeline
        
        self.id_string = ('pipeline_muxer_' + str(what)).lower()        


    def get_id(self): 
        return self.id_string


    def run(self, db=None, force=False, pipe_data={}, pipe_name=''):
        pipe_data_block = []
        
        for pipeline in self.args['pipeline']:
            pipe_data_in = pipe_data.copy()
            pair = pipe_data['img']
                                       
            pipe_data_out, pipe_name_out = run_pipeline(pair, pipeline, db, force=force, pipe_data=pipe_data_in, pipe_name=pipe_name)        
            pipe_data_block.append(pipe_data_out)
        
        return self.pipe_gather(pipe_data_block)


if __name__ == '__main__':
    
    pipeline = [keynet_module(),
                deep_patch_module(),
                deep_descriptor_module(descriptor='hardnet'),
                smnn_module()
                ]
    
    imgs = '/media/bellavista/Dati2/colmap_working/villa_giulia2/imgs'
    run_pairs(pipeline, imgs)
