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
                out_data = pipe_module.run(**pipe_data)
                stop_time = time.time()
                out_data['running_time'] = stop_time - start_time
                db.add(data_key, out_data)

            for k, v in out_data.items(): pipe_data[k] = v
                
    return pipe_data, pipe_name


def laf2homo(kps):
    c = kps[:, :, 2].type(torch.float)
    s = torch.sqrt(torch.abs(kps[:, 0, 0] * kps[:, 1, 1] - kps[:, 0, 1] * kps[:, 1, 0]))   
    
    Hi = torch.zeros((kps.shape[0], 3, 3), device=device)
    Hi[:, :2, :] = kps / s.reshape(-1, 1, 1)
    Hi[:, 2, 2] = 1 

    H = torch.linalg.inv(Hi).type(torch.float)
    s = s.type(torch.float)
    
    return c, H, s


def homo2laf(c, H, s):
    Hi = torch.linalg.inv(H)
    kp = Hi[:, :2, :] * s.reshape(-1, 1, 1)

    return kp.unsqueeze(0)


class sift_module:
    def __init__(self, **args):
        self.upright = False
        self.num_features = 8000
        self.single_image = True
        
        for k, v in args.items():
           setattr(self, k, v)

        self.detector = cv2.SIFT_create(self.num_features, contrastThreshold=-10000, edgeThreshold=10000)


    def get_id(self):
        return ('sift_upright_' + str(self.upright) + '_nfeat_' + str(self.num_features)).lower()


    def run(self, **args):    
        
        im1 = cv2.imread(args['img'][args['idx']], cv2.IMREAD_GRAYSCALE)
        kp = self.detector.detect(im1, None)

        if self.upright:
            idx = np.unique(np.asarray([[k.pt[0], k.pt[1]] for k in kp]), axis=0, return_index=True)[1]
            kp = [kp[ii] for ii in idx]
            for ii in range(len(kp)):
                kp[ii].angle = 0       
                
        kp = laf_from_opencv_kpts(kp, device=device)
        kp, H, s = laf2homo(kp.squeeze(0).detach().to(device))
    
        return {'kp': kp, 'H': H, 's': s}


class keynet_module:
    def __init__(self, **args):
        self.num_features = 8000
        self.single_image = True
        
        for k, v in args.items():
           setattr(self, k, v)

        self.detector = K.feature.KeyNetDetector(num_features=self.num_features).to(device)
        
        
    def get_id(self):
        return ('keynet_nfeat_' + str(self.num_features)).lower()

    
    def run(self, **args):    
        kp, val = self.detector(K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0))
        
        kp, H, s = laf2homo(kp.squeeze(0).detach().to(device))
        val = val.squeeze(0).detach().to(device)
    
        return {'kp': kp, 'H': H, 's': s, 'val': val}


class hz_plus_module:
    def __init__(self, **args):
        self.num_features = 8000
        self.single_image = True
        self.block_memory = 16*10**6 
        
        for k, v in args.items():
           setattr(self, k, v)

                
    def get_id(self):
        return ('hz_plus_nfeat_' + str(self.num_features)).lower()

    
    def run(self, **args):    
        img = hz.load_to_tensor(args['img'][args['idx']]).to(torch.float)
        kp = hz.hz_plus(img, output_format='laf', block_mem=self.block_memory, max_max_pts=self.num_features)
        lafs = K.feature.ellipse_to_laf(kp[None]).squeeze(0)
        kp, H, s = laf2homo(lafs)
    
        return {'kp': kp, 'H': H, 's': s}


class orinet_affnet_module:
    def __init__(self, **args):
        self.orinet = True
        self.orinet_args = {}

        self.affnet = True
        self.affnet_args = {}

        self.single_image = True
        
        for k, v in args.items():
           setattr(self, k, v)

        if self.orinet:
            self.ori_module = K.feature.LAFOrienter(angle_detector=K.feature.OriNet().to(device), **self.orinet_args)
        else:
            self.ori_module = K.feature.PassLAF()

        if self.affnet:
            self.aff_module = K.feature.LAFAffineShapeEstimator(**self.affnet_args)
        else:
            self.aff_module = K.feature.PassLAF()


    def get_id(self):
        return ('orinet_' + str(self.orinet) + '_affnet_' + str(self.affnet)).lower()


    def run(self, **args):    
        im = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)

        lafs = homo2laf(args['kp'][args['idx']], args['H'][args['idx']], args['s'][args['idx']])

        lafs = self.ori_module(lafs, im)
        lafs = self.aff_module(lafs, im)

        kp, H, s = laf2homo(lafs.squeeze(0))
    
        return {'kp': kp, 'H': H, 's': s}


class deep_descriptor_module:
    def __init__(self, **args):
        self.single_image = True
        self.descriptor = 'hardnet'
        self.desc_args = {}
        self.patch_args = {}
        
        for k, v in args.items():
           setattr(self, k, v)

        if self.descriptor == 'hardnet':
            desc = K.feature.HardNet().to(device)
        if self.descriptor == 'sosnet':
            desc = K.feature.SOSNet().to(device)
        if self.descriptor == 'hynet':
            desc = K.feature.HyNet(**self.desc_args).to(device)

        self.ddesc = K.feature.LAFDescriptor(patch_descriptor_module=desc, **self.patch_args)


    def get_id(self):
        return (self.descriptor + '_descriptor').lower()


    def run(self, **args):    
        im = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)

        lafs = homo2laf(args['kp'][args['idx']], args['H'][args['idx']], args['s'][args['idx']])
        desc = self.ddesc(im, lafs).squeeze(0)
    
        return {'desc': desc}


class sift_descriptor_module:
    def __init__(self, **args):
        self.rootsift = True
        self.single_image = True
        
        for k, v in args.items():
           setattr(self, k, v)

        self.descriptor = cv2.SIFT_create()


    def get_id(self):
        if self.rootsift:
            prefix = 'root'
        else:
            prefix = ''
        
        return (prefix + 'sift_descriptor').lower()


    def run(self, **args):
        im = cv2.imread(args['img'][args['idx']], cv2.IMREAD_GRAYSCALE)
        
        lafs = homo2laf(args['kp'][args['idx']], args['H'][args['idx']], args['s'][args['idx']])        
        
        kp = opencv_kpts_from_laf(lafs)
        
        _, desc = self.descriptor.compute(im, kp)

        if self.rootsift:
            desc /= desc.sum(axis=1, keepdims=True) + 1e-8
            desc = np.sqrt(desc)
            
        desc = torch.tensor(desc, device=device, dtype=torch.float)
                    
        return {'desc': desc}


class smnn_module:
    def __init__(self, **args):
        self.th = 0.99
        
        for k, v in args.items():
           setattr(self, k, v)


    def get_id(self):        
        return ('smnn_th_' + str(self.th)).lower()


    def run(self, **args):
        val, idxs = K.feature.match_smnn(args['desc'][0], args['desc'][1], self.th)

        return {'midx': idxs, 'vidx': val.squeeze(1)}


if __name__ == '__main__':
    
    pipeline = [hz_plus_module(),
                orinet_affnet_module(),
                deep_descriptor_module(descriptor='hardnet'),
                smnn_module()
                ]
    
    imgs = '/media/bellavista/Dati2/colmap_working/villa_giulia2/imgs'
    run_pairs(pipeline, imgs)
