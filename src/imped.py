import os
import warnings
import pickled_hdf5.pickled_hdf5 as pickled_hdf5
import time
from tqdm import tqdm

import torch
import kornia as K
from kornia_moons.feature import opencv_kpts_from_laf, laf_from_opencv_kpts
import cv2
import numpy as np
from PIL import Image
import poselib
import gdown
import zipfile
import tarfile
import csv
import shutil
import bz2
import _pickle as cPickle
import argparse
import math
import copy
import wget
import pycolmap
import scipy

import matplotlib.pyplot as plt
import plot.viz2d as viz
import plot.utils as viz_utils
import sys

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
pipe_color = ['red', 'blue', 'lime', 'fuchsia', 'yellow']
show_progress = True

enable_quadtree = False

def megadepth_1500_list(ppath='bench_data/gt_data/megadepth'):
    npz_list = [i for i in os.listdir(ppath) if (os.path.splitext(i)[1] == '.npz')]

    data = {'im1': [], 'im2': [], 'K1': [], 'K2': [], 'T': [], 'R': []}
    # Sort to avoid os.listdir issues 
    for name in sorted(npz_list):
        scene_info = np.load(os.path.join(ppath, name), allow_pickle=True)
    
        # Sort to avoid pickle issues 
        pidx = sorted([[pair_info[0][0], pair_info[0][1]] for pair_info in scene_info['pair_infos']])
    
        # Collect pairs
        for idx in pidx:
            id1, id2 = idx
            im1 = scene_info['image_paths'][id1].replace('Undistorted_SfM/', '')
            im2 = scene_info['image_paths'][id2].replace('Undistorted_SfM/', '')                        
            K1 = scene_info['intrinsics'][id1].astype(np.float32)
            K2 = scene_info['intrinsics'][id2].astype(np.float32)
    
            # Compute relative pose
            T1 = scene_info['poses'][id1]
            T2 = scene_info['poses'][id2]
            T12 = np.matmul(T2, np.linalg.inv(T1))
    
            data['im1'].append(im1)
            data['im2'].append(im2)
            data['K1'].append(K1)
            data['K2'].append(K2)
            data['T'].append(T12[:3, 3])
            data['R'].append(T12[:3, :3])   
    return data


def scannet_1500_list(ppath='bench_data/gt_data/scannet'):
    intrinsic_path = 'intrinsics.npz'
    npz_path = 'test.npz'

    data = np.load(os.path.join(ppath, npz_path))
    data_names = data['name']
    intrinsics = dict(np.load(os.path.join(ppath, intrinsic_path)))
    rel_pose = data['rel_pose']
    
    data = {'im1': [], 'im2': [], 'K1': [], 'K2': [], 'T': [], 'R': []}
    
    for idx in range(data_names.shape[0]):
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_names[idx]
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
    
        # read the grayscale image which will be resized to (1, 480, 640)
        im1 = os.path.join(scene_name, 'color', f'{stem_name_0}.jpg')
        im2 = os.path.join(scene_name, 'color', f'{stem_name_1}.jpg')
        
        # read the intrinsic of depthmap
        K1 = intrinsics[scene_name]
        K2 = intrinsics[scene_name]
    
        # pose    
        T12 = np.concatenate((rel_pose[idx],np.asarray([0, 0, 0, 1.0]))).reshape(4,4)
        
        data['im1'].append(im1)
        data['im2'].append(im2)
        data['K1'].append(K1)
        data['K2'].append(K2)  
        data['T'].append(T12[:3, 3])
        data['R'].append(T12[:3, :3])     
    return data


def resize_megadepth(im, res_path='imgs', bench_path='bench_data', force=False, max_sz=1200):
    aux = im.split('/')
    flat_img = os.path.join('megadepth', aux[0], '_'.join((aux[0], aux[-1])))
    flat_img = os.path.splitext(flat_img)[0] + '.png'
    
    mod_im = os.path.join(bench_path, res_path, flat_img)
    ori_im= os.path.join(bench_path, 'megadepth_test_1500/Undistorted_SfM', im)

    if os.path.isfile(mod_im) and not force:
        # PIL does not load image, so it's faster to get only image size
        return np.asarray(Image.open(ori_im).size) / np.asarray(Image.open(mod_im).size), flat_img 
        # return np.array(cv2.imread(ori_im).shape)[:2][::-1] / np.array(cv2.imread(mod_im).shape)[:2][::-1], flat_img

    img = cv2.imread(ori_im)
    sz_ori = np.array(img.shape)[:2][::-1]
    sz_max = float(max(sz_ori))

    if sz_max > max_sz:
        cf = max_sz / sz_max                    
        sz_new = np.ceil(sz_ori * cf).astype(int) 
        img = cv2.resize(img, tuple(sz_new), interpolation=cv2.INTER_LANCZOS4)
        sc = sz_ori/sz_new
        os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
        cv2.imwrite(mod_im, img)
        return sc, flat_img
    else:
        os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
        cv2.imwrite(mod_im, img)
        return np.array([1., 1.]), flat_img


def resize_scannet(im, res_path='imgs', bench_path='bench_data', force=False):
    aux = im.split('/')
    flat_img = os.path.join('scannet', aux[0], '_'.join((aux[0], aux[-1])))
    flat_img = os.path.splitext(flat_img)[0] + '.png'

    mod_im = os.path.join(bench_path, res_path, flat_img)
    ori_im= os.path.join(bench_path, 'scannet_test_1500', im)

    if os.path.isfile(mod_im) and not force:
        # PIL does not load image, so it's faster to get only image size
        return np.asarray(Image.open(ori_im).size) / np.asarray(Image.open(mod_im).size), flat_img 
        # return np.array(cv2.imread(ori_im).shape)[:2][::-1] / np.array(cv2.imread(mod_im).shape)[:2][::-1], flat_img

    img = cv2.imread(ori_im)
    sz_ori = np.array(img.shape)[:2][::-1]

    sz_new = np.array([640, 480])
    img = cv2.resize(img, tuple(sz_new), interpolation=cv2.INTER_LANCZOS4)
    sc = sz_ori/sz_new
    os.makedirs(os.path.dirname(mod_im), exist_ok=True)                 
    cv2.imwrite(mod_im, img)
    return sc, flat_img


def setup_images_megadepth(megadepth_data, data_file='bench_data/megadepth_scannet.pbz2', bench_path='bench_data', bench_imgs='imgs', max_sz=1200):
    n = len(megadepth_data['im1'])
    im_pair_scale = np.zeros((n, 2, 2))

    new_im1 = [None] * n
    new_im2 = [None] * n
    
    res_path = bench_imgs
    for i in tqdm(range(n), desc='megadepth image setup'):
        im_pair_scale[i, 0], new_im1[i] = resize_megadepth(megadepth_data['im1'][i], res_path, bench_path, max_sz=max_sz)
        im_pair_scale[i, 1], new_im2[i] = resize_megadepth(megadepth_data['im2'][i], res_path, bench_path, max_sz=max_sz)
    megadepth_data['im_pair_scale'] = im_pair_scale
 
    megadepth_data['im1'] = new_im1   
    megadepth_data['im2'] = new_im2   
 
    return megadepth_data


def setup_images_scannet(scannet_data, data_file='bench_data/megadepth_scannet.pbz2', bench_path='bench_data', bench_imgs='imgs', max_sz=None):       
    n = len(scannet_data['im1'])
    im_pair_scale = np.zeros((n, 2, 2))
    
    new_im1 = [None] * n
    new_im2 = [None] * n
    
    res_path = bench_imgs
    for i in tqdm(range(n), desc='scannet image setup'):
        im_pair_scale[i, 0], new_im1[i] = resize_scannet(scannet_data['im1'][i], res_path, bench_path)
        im_pair_scale[i, 1], new_im2[i] = resize_scannet(scannet_data['im2'][i], res_path, bench_path)
    scannet_data['im_pair_scale'] = im_pair_scale
         
    scannet_data['im1'] = new_im1   
    scannet_data['im2'] = new_im2       
    
    return scannet_data


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
        

def download_quadtreeattention(weight_path='../weights/quadtreeattention'):    
    file_list = [
        'indoor.ckpt',
        'outdoor.ckpt',
    ]
    
    url_list = [
        'https://drive.google.com/file/d/1pSK_8GP1WkqKL5m7J4aHvhFixdLP6Yfa/view?usp=sharing',
        'https://drive.google.com/file/d/1UOYdzbrXHU9kvVy9tscCCO7BB3G4rWK4/view?usp=sharing',
    ]

    os.makedirs(weight_path, exist_ok=True)   

    for file, url in zip(file_list, url_list):

        file_to_download = os.path.join(weight_path, file)    
        if not os.path.isfile(file_to_download):    
            gdown.download(url, file_to_download, fuzzy=True)


def download_matchformer(weight_path='../weights/matchformer'):    
    file_list = [
        'indoor-large-SEA.ckpt',
        'indoor-lite-LA.ckpt',
        'outdoor-large-LA.ckpt',
        'outdoor-lite-SEA.ckpt',
    ]
    
    url_list = [
        'https://drive.google.com/file/d/1EjeSvU3ARZg5mn2PlqNDWMu9iwS7Zf_m/view?usp=drive_link',
        'https://drive.google.com/file/d/11ClOQ_VrlsT7PxK6jQr5AW1Fd0YMbB3R/view?usp=drive_link',
        'https://drive.google.com/file/d/1Ii-z3dwNwGaxoeFVSE44DqHdMhubYbQf/view?usp=drive_link',
        'https://drive.google.com/file/d/1etaU9mM8bGT2AKT56ph6iqUdpV1daFBz/view?usp=drive_link',
    ]

    os.makedirs(weight_path, exist_ok=True)   

    for file, url in zip(file_list, url_list):

        file_to_download = os.path.join(weight_path, file)    
        if not os.path.isfile(file_to_download):    
            gdown.download(url, file_to_download, fuzzy=True)
    

def download_megadepth(bench_path ='bench_data'):   
    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)   

    file_to_download = os.path.join(bench_path, 'downloads', 'megadepth_scannet_gt_data.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1GtpHBN6RLcgSW5RPPyqYLyfbn7ex360G/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'gt_data')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile(file_to_download, "r") as zip_ref:
            zip_ref.extractall(bench_path)    

    file_to_download = os.path.join(bench_path, 'downloads', 'megadepth_test_1500.tar.gz')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1Vwk_htrvWmw5AtJRmHw10ldK57ckgZ3r/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)
    
    out_dir = os.path.join(bench_path, 'megadepth_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    return


def download_scannet(bench_path ='bench_data'):   
    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)   

    file_to_download = os.path.join(bench_path, 'downloads', 'megadepth_scannet_gt_data.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1GtpHBN6RLcgSW5RPPyqYLyfbn7ex360G/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'gt_data')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile(file_to_download, "r") as zip_ref:
            zip_ref.extractall(bench_path)    
    
    file_to_download = os.path.join(bench_path, 'downloads', 'scannet_test_1500.tar.gz')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/13KCCdC1k3IIZ4I3e4xJoVMvDA84Wo-AG/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'scannet_test_1500')
    if not os.path.isdir(out_dir):    
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extractall(bench_path)
    
    return


def benchmark_setup(bench_path='bench_data', bench_imgs='imgs', bench_gt='gt_data', dataset='megadepth', debug_pairs=None, force=False, sample_size=800, seed=42, covisibility_range=[0.1, 0.7], new_sample=False, scene_list=None):        
    if (dataset == 'megadepth') or (dataset == 'scannet'):
        return megadepth_scannet_setup(bench_path=bench_path, bench_imgs=bench_imgs, bench_gt=bench_gt, dataset=dataset, debug_pairs=debug_pairs, force=force)
    
    if dataset == 'imc':
        return imc_phototourism_setup(bench_path=bench_path, bench_imgs=bench_imgs, dataset=dataset, sample_size=sample_size, seed=seed, covisibility_range=covisibility_range, new_sample=new_sample, force=force)
        

def megadepth_scannet_setup(bench_path='bench_data', bench_imgs='imgs', bench_gt='gt_data', dataset='megadepth', debug_pairs=None, force=False, max_sz=1200):        
    if dataset == 'megadepth':
        download = download_megadepth
        img_list = megadepth_1500_list
        setup_images = setup_images_megadepth

    if dataset == 'scannet':
        download = download_scannet
        img_list = scannet_1500_list
        setup_images = setup_images_scannet

    os.makedirs(bench_path, exist_ok=True)
    db_file = os.path.join(bench_path, dataset + '.hdf5')    
    db = pickled_hdf5.pickled_hdf5(db_file, mode='a')    

    data_key = '/' + dataset

    data, is_found = db.get(data_key)                    
    if (not is_found) or force:
        download(bench_path)        
        data = img_list(os.path.join(bench_path, bench_gt, dataset))
    
        # for debugging, use only first debug_pairs image pairs
        if not (debug_pairs is None):
            for what in data.keys():
                data[what] = [data[what][i] for i in range(debug_pairs)]
    
        data = setup_images(data, bench_path=bench_path, bench_imgs=bench_imgs, max_sz=max_sz)    
        
        pairs = [(im1, im2) for im1, im2 in zip(data['im1'], data['im2'])]
        gt = {}
        for i in range(len(data['im1'])):
            if not data['im1'][i] in gt:
                gt[data['im1'][i]] = {}
            
            gt[data['im1'][i]][data['im2'][i]] = {
                'K1': data['K1'][i],
                'K2': data['K2'][i],
                'R': data['R'][i],
                'T': data['T'][i],
                'image_pair_scale': data['im_pair_scale'][i],
                }
                
        data = {'image_pairs': pairs, 'gt': gt, 'image_path': os.path.join(bench_path, bench_imgs)}
        db.add(data_key, data)
        db.close()
        
    return data['image_pairs'], data['gt'], data['image_path']


# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data, add_ext=False):
    if add_ext:
        ext = '.pbz2'
    else:
        ext = ''
        
    with bz2.BZ2File(title + ext, 'w') as f: 
        cPickle.dump(data, f)
        

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def imc_phototourism_setup(bench_path='bench_data', bench_imgs='imgs', dataset='imc', sample_size=800, seed=42, covisibility_range=[0.1, 0.7], new_sample=False, force=False):
    
    os.makedirs(bench_path, exist_ok=True)
    db_file = os.path.join(bench_path, dataset + '.hdf5')    
    db = pickled_hdf5.pickled_hdf5(db_file, mode='a')   

    data_key = '/' + dataset

    data, is_found = db.get(data_key)                    
    if is_found and (not force):
        db.close()
        return data['image_pairs'], data['gt'], data['image_path']   

    rng = np.random.default_rng(seed=seed)    
    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)

    file_to_download = os.path.join(bench_path, 'downloads', 'image-matching-challenge-2022.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1RyqsKr_X0Itkf34KUv2C7XP35drKSXht/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'imc_phototourism')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile(file_to_download, "r") as zip_ref:
            zip_ref.extractall(out_dir)        
        
    scenes = sorted([scene for scene in os.listdir(os.path.join(out_dir, 'train')) if os.path.isdir(os.path.join(out_dir, 'train', scene))])

    scale_file = os.path.join(out_dir, 'train', 'scaling_factors.csv')
    scale_dict = {}
    with open(scale_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            scale_dict[row['scene']] = float(row['scaling_factor'])
        
    im1 = []
    im2 = []
    K1 = []
    K2 = []
    R = []
    T = []
    scene_scales = []
    covisibility = []
    
    if new_sample:
        sampled_idx = {}
    else:
        file_to_download = os.path.join(bench_path, 'downloads', 'imc_sampled_idx.pbz2')    
        if not os.path.isfile(file_to_download):    
            url = "https://drive.google.com/file/d/13AE6pbkJ8bNfVYjkxYvpVN6mkok98NuM/view?usp=drive_link"
            gdown.download(url, file_to_download, fuzzy=True)
        
        sampled_idx = decompress_pickle(file_to_download)
    
    for sn in tqdm(range(len(scenes)), desc='imc setup'):    
        scene = scenes[sn]
                        
        work_path = os.path.join(out_dir, 'train', scene)
        pose_file  = os.path.join(work_path, 'calibration.csv')
        covis_file  = os.path.join(work_path, 'pair_covisibility.csv')

        if (not os.path.isfile(pose_file)) or (not os.path.isfile(covis_file)):
            continue
        
        im1_ = []
        im2_ = []
        covis_val = []
        with open(covis_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pp = row['pair'].split('-')
                im1_.append(os.path.join(scene, pp[0]))
                im2_.append(os.path.join(scene, pp[1]))
                covis_val.append(float(row['covisibility']))

        covis_val = np.asarray(covis_val)
        
        if new_sample:
            mask_val = (covis_val >= covisibility_range[0]) & (covis_val <= covisibility_range[1])

            n = covis_val.shape[0]
            
            full_idx = np.arange(n)  
            full_idx = full_idx[mask_val]

            m = full_idx.shape[0]
            
            idx = rng.permutation(m)[:sample_size]
            full_idx = np.sort(full_idx[idx])

            sampled_idx[scene] = full_idx
        else:
            full_idx = sampled_idx[scene]
                    
        covis_val = covis_val[full_idx]
        im1_ = [im1_[i] for i in full_idx]
        im2_ = [im2_[i] for i in full_idx]
        
        img_path = os.path.join(bench_path, bench_imgs, 'imc_phototourism')
        os.makedirs(os.path.join(img_path, scene), exist_ok=True)

        im1_new = []        
        im2_new = []

        for im in im1_:
            im_flat = os.path.split(im)
            im_new = os.path.join('imc_phototourism', im_flat[0], '_'.join(im_flat)) + '.jpg'

            im1_new.append(im_new)    
            shutil.copyfile(os.path.join(bench_path, 'imc_phototourism', 'train', scene, 'images', os.path.split(im)[1] + '.jpg'), os.path.join(bench_path, bench_imgs, im_new))

        for im in im2_:
            im_flat = os.path.split(im)
            im_new = os.path.join('imc_phototourism', im_flat[0], '_'.join(im_flat)) + '.jpg'

            im2_new.append(im_new) 
            shutil.copyfile(os.path.join(bench_path, 'imc_phototourism', 'train', scene, 'images', os.path.split(im)[1] + '.jpg'), os.path.join(bench_path, bench_imgs, im_new))

        Kv = {}
        Tv = {}
        calib_file = os.path.join(work_path, 'calibration.csv')
        with open(calib_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cam = os.path.join(scene, row['image_id'])
                Kv[cam] = np.asarray([float(i) for i in row['camera_intrinsics'].split(' ')]).reshape((3, 3))
                tmp = np.eye(4)
                tmp[:3, :3] = np.asarray([float(i) for i in row['rotation_matrix'].split(' ')]).reshape((3, 3))
                tmp[:3, 3] = np.asarray([float(i) for i in row['translation_vector'].split(' ')])
                Tv[cam] = tmp

        K1_ = []
        K2_ = []
        T_ = []
        R_ = []
        scales_ = []
        for i in range(len(im1_)):
            K1_.append(Kv[im1_[i]])
            K2_.append(Kv[im2_[i]])
            T1 = Tv[im1_[i]]
            T2 = Tv[im2_[i]]
            T12 = np.matmul(T2, np.linalg.inv(T1))
            T_.append(T12[:3, 3])
            R_.append(T12[:3, :3])
            scales_.append(scale_dict[scene])
            
            
        im1 = im1 + im1_new
        im2 = im2 + im2_new
        K1 = K1 + K1_
        K2 = K2 + K2_
        T = T + T_
        R = R + R_
        scene_scales = scene_scales + scales_
        covisibility = covisibility + covis_val.tolist()  
        
    imc_data = {}
    imc_data['im1'] = im1
    imc_data['im2'] = im2
    imc_data['K1'] = np.asarray(K1)
    imc_data['K2'] = np.asarray(K2)
    imc_data['T'] = np.asarray(T)
    imc_data['R'] = np.asarray(R)
    imc_data['scene_scales'] = np.asarray(scene_scales)
    imc_data['covisibility'] = np.asarray(covisibility)
    imc_data['im_pair_scale'] = np.full((len(im1), 2, 2), 1)
    
    
    pairs = [(im1, im2) for im1, im2 in zip(imc_data['im1'], imc_data['im2'])]
    gt = {}
    for i in range(len(imc_data['im1'])):
        if not imc_data['im1'][i] in gt:
            gt[imc_data['im1'][i]] = {}
        
        gt[imc_data['im1'][i]][imc_data['im2'][i]] = {
            'K1': imc_data['K1'][i],
            'K2': imc_data['K2'][i],
            'R': imc_data['R'][i],
            'T': imc_data['T'][i],
            'image_pair_scale': imc_data['im_pair_scale'][i],
            'scene_scale': imc_data['scene_scales'][i],
            'covisibility': imc_data['covisibility'][i],            
            }
                
    data = {'image_pairs': pairs, 'gt': gt, 'image_path': os.path.join(bench_path, bench_imgs)}
    db.add(data_key, data)
    db.close()
        
    return data['image_pairs'], data['gt'], data['image_path']    
    

def scannet_setup(bench_path='bench_data', bench_imgs='imgs', bench_gt='gt_data', db_file='scannet.hdf5', debug_pairs=None, force=False, **dummy_args):        
    db = pickled_hdf5.pickled_hdf5(db_file, mode='a', label_prefix='pickled')    
    data_key = '/scannet'                    

    scannet_data, is_found = db.get(data_key)                    
    if (not is_found) or force:
        download_megadepth(bench_path)        
        megadepth_data = megadepth_1500_list(os.path.join(bench_path, bench_gt, 'megadepth'))
    
        # for debugging, use only first debug_pairs image pairs
        if not (debug_pairs is None):
            for what in megadepth_data.keys():
                megadepth_data[what] = [megadepth_data[what][i] for i in range(debug_pairs)]
    
        megadepth_data = setup_images_megadepth(megadepth_data, bench_path=bench_path, bench_imgs=bench_imgs)    
        
        db.add(data_key, megadepth_data)
        db.close()
        
    return megadepth_data


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
            self.check_img = check_img            
    

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


def finalize_pipeline(pipeline):
    for pipe_module in pipeline:
        if hasattr(pipe_module, 'finalize'):
            pipe_module.finalize()
    

def run_pairs(pipeline, imgs, db_name='database.hdf5', db_mode='a', force=False, add_path=''):    
    db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)

    for pair in go_iter(image_pairs(imgs, add_path=add_path), msg='          processed pairs'):
        run_pipeline(pair, pipeline, db, force=force, show_progress=True)
        
    finalize_pipeline(pipeline)

                
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


    def finalize(self):
        return


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
        

    def finalize(self):
        return

    
    def run(self, **args):
        img = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=device).unsqueeze(0)
        kp, kr = self.detector(img)
        kp, kH = laf2homo(kp.detach().to(device).squeeze(0))

        return {'kp': kp, 'kH': kH, 'kr': kr.detach().to(device).squeeze(0)}


import hz.hz as hz

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


    def finalize(self):
        return

    
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

    
    def finalize(self):
        return

    
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

    
    def finalize(self):
        return

    
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
    
    
    def finalize(self):
        return


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


    def finalize(self):
        return


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


    def finalize(self):
        return


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
    

    def finalize(self):
        return


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
    
    
    def finalize(self):
        finalize_pipeline(self.pipeline)
        
        return


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


    def finalize(self):
        return

        
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

    
    def finalize(self):
        return

        
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


def pipe_union(pipe_block, unique=True, no_unmatched=False, only_matched=False, sampling_mode=None, sampling_scale=1, sampling_offset=0, overlapping_cells=False, preserve_order=False, counter=False):
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
        
        if overlapping_cells:
            kp0_ = ((kp0 + sampling_offset + (sampling_scale / 2) ) / sampling_scale).round() * sampling_scale - sampling_offset - (sampling_scale / 2)
            kp1_ = ((kp1 + sampling_offset + (sampling_scale / 2) ) / sampling_scale).round() * sampling_scale - sampling_offset - (sampling_scale / 2)

            s0 = ((kp0_unsampled - kp0)**2).sum(dim=1) > ((kp0_unsampled - kp0_)**2).sum(dim=1)
            s1 = ((kp1_unsampled - kp1)**2).sum(dim=1) > ((kp1_unsampled - kp1_)**2).sum(dim=1)

            kp0[s0] = kp0_[s0]
            kp1[s1] = kp1_[s1]
                        
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


    def finalize(self):        
        for pipeline in self.pipeline:
            finalize_pipeline(pipeline)

        return


    def run(self, db=None, force=False, pipe_data=None, pipe_name='/'):
        if pipe_data is None: pipe_data = {}

        pipe_data_block = []
        
        for pipeline in self.pipeline:
            pipe_data_in = pipe_data.copy()
            pair = pipe_data['img']
                                       
            pipe_data_out, pipe_name_out = run_pipeline(pair, pipeline, db, force=force, pipe_data=pipe_data_in, pipe_name=pipe_name)        
            pipe_data_block.append(pipe_data_out)
        
        return self.pipe_gather(pipe_data_block)


from lightglue import LightGlue as lg_lightglue, SuperPoint as lg_superpoint, DISK as lg_disk, SIFT as lg_sift, ALIKED as lg_aliked, DoGHardNet as lg_doghardnet
from lightglue.utils import load_image as lg_load_image, rbd as lg_rbd

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
            'desc_cf': 1,                    # 255 to use R2S2 with what='sift'
            'aliked_model': "aliked-n16rot",          # default is "aliked-n16"
            }

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
    
    
    def finalize(self):
        return


    def run(self, **args):
        image0 = K.io.load_image(args['img'][0], K.io.ImageLoadType.GRAY32, device=device)
        image1 = K.io.load_image(args['img'][1], K.io.ImageLoadType.GRAY32, device=device)

        hw1 = image0.shape[1:]
        hw2 = image1.shape[1:]

        if not (self.args['resize'] is None):        
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

        m_mask = m_val > 0

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
            'overlapping_cells': False,
            'sampling_scale': 1,
            'sampling_offset': 0,
            }

        self.id_string, self.args = set_args('sampling', args, self.args)        


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):           
        pipe_data = args

        return pipe_union(pipe_data, unique=self.args['unique'],
                          no_unmatched=self.args['no_unmatched'],
                          only_matched=self.args['only_matched'],
                          sampling_mode=self.args['sampling_mode'],
                          sampling_scale=self.args['sampling_scale'],
                          sampling_offset=self.args['sampling_offset'],
                          overlapping_cells=self.args['overlapping_cells'])    


import colmap_db.database as coldb

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


    def get_camera(self, camera_id):
        cursor = self.execute("SELECT model, width, height, params, prior_focal_length FROM cameras where camera_id=?", (camera_id, ))
        cam = cursor.fetchone()
        if cam is None:
            return None
        else:
            c, w, h, p, f = cam
            p = coldb.blob_to_array(p, np.float64)
            return c, w, h, p, f


    def get_image(self, image_id):
        cursor = self.execute("SELECT name, camera_id FROM images where image_id=?", (image_id, ))
        img = cursor.fetchone()
        if img is None:
            return None
        else:
            return img


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
            if keypoints.shape[0] > 0:
                self.add_keypoints(image_id, keypoints)
        else:
            if keypoints.shape[0] > 0:
                self.execute(
                    "UPDATE keypoints SET rows=?, cols=?, data=? WHERE image_id=?",
                    keypoints.shape + (coldb.array_to_blob(keypoints), ) + (image_id, ),
                    )
            else:
                self.execute(
                    "DELETE FROM keypoints WHERE image_id=?",
                    (image_id, ),
                    )

    def get_matches(self, image_id1, image_id2):
        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)
        cursor = self.execute("SELECT data, rows, cols FROM matches where pair_id=?", (pair_id, ))
        m = cursor.fetchone()
        if m is None:
            return None
        else:
            m, r, c = m
            
            if r == 0: return None
            
            m = np.reshape(coldb.blob_to_array(m, np.uint32), (r, c))

            if image_id1 > image_id2: m = m[:, ::-1]

            return m


    def get_two_view_geometry(self, image_id1, image_id2):
        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)
        cursor = self.execute("SELECT data, rows, cols, config, E, F, H FROM two_view_geometries where pair_id=?", (pair_id, ))
        m = cursor.fetchone()
        if m is None:
            return None, None
        else:
            model = {}
            m, r, c, config, E, F, H = m
            
            if r == 0: return None, None
            
            m = np.reshape(coldb.blob_to_array(m, np.uint32), (r, c))

            if (config == PLANAR) or (config == PANORAMIC) or (config == PLANAR_OR_PANORAMIC):
                model['H'] = np.reshape(coldb.blob_to_array(H, np.float64), (3, 3))

            if (config == CALIBRATED):
                model['E'] = np.reshape(coldb.blob_to_array(E, np.float64), (3, 3))

            if (config == UNCALIBRATED):
                model['F'] = np.reshape(coldb.blob_to_array(F, np.float64), (3, 3))

            if image_id1 > image_id2:
                m = m[:, ::-1]
                if 'H' in model: model['H'] = np.linalg.inv(model['H']) 
                if 'F' in model: model['F'] = np.transpose(model['F']) 
                if 'E' in model: model['E'] = np.transpose(model['E']) 

            return m, model


    def update_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)

        if self.get_matches(image_id1, image_id2) is None:
            if matches.shape[0] > 0:
                self.add_matches(image_id1, image_id2, matches)
        else:
            matches = np.asarray(matches, np.uint32)

            if image_id1 > image_id2:
                matches = matches[:, ::-1]

            if matches.shape[0] > 0:
                self.execute(
                    "UPDATE matches SET rows=?, cols=?, data=? WHERE pair_id=?",
                    matches.shape + (coldb.array_to_blob(matches), ) + (pair_id, ),
                    )
            else:
                self.execute(
                    "DELETE FROM matches WHERE pair_id=?",
                    (pair_id, ),
                    )


    def get_images(self):
        cursor = self.execute("SELECT image_id, name FROM images")
        m = cursor.fetchall()        
        
        return m
    

    def update_two_view_geometry(self, image_id1, image_id2, matches, model=None):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if model is None: model = {}

        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)

        if self.get_two_view_geometry(image_id1, image_id2)[0] is None:
            if matches.shape[0] > 0:
                how_many_models = 0
                
                if 'H' in model:
                    config = PLANAR_OR_PANORAMIC
                    if image_id1 > image_id2: model['H'] = np.linalg.inv(model['H'])                    
                    how_many_models = how_many_models + 1
                if 'F' in model:
                    config = UNCALIBRATED
                    if image_id1 > image_id2: model['F'] = np.tranpose(model['F'])                    
                    how_many_models = how_many_models + 1
                if 'E' in model:
                    config = CALIBRATED
                    if image_id1 > image_id2: model['E'] = np.tranpose(model['E'])                    
                    how_many_models = how_many_models + 1
                if how_many_models != 1:
                    config = MULTIPLE
                
                self.add_two_view_geometry(image_id1, image_id2, matches, config=config, **model)
        else:
            matches = np.asarray(matches, np.uint32)

            if image_id1 > image_id2:
                matches = matches[:, ::-1]
                
                if 'H' in model: model['H'] = np.linalg.inv(model['H'])
                if 'F' in model: model['F'] = np.transpose(model['F'])
                if 'E' in model: model['E'] = np.transpose(model['E'])

            how_many_models = 0
            model_str = ""
            model_tuple =  ()
            if 'H' in model:
                config = PLANAR_OR_PANORAMIC
                how_many_models = how_many_models + 1
                model_str = model_str + "H=?, "
                model_tuple = model_tuple + (coldb.array_to_blob(model['H']), )
            if 'F' in model:
                config = UNCALIBRATED
                how_many_models = how_many_models + 1
                model_str = model_str + "F=?, "
                model_tuple = model_tuple + (coldb.array_to_blob(model['F']), )
            if 'E' in model:
                config = CALIBRATED
                how_many_models = how_many_models + 1
                model_str = model_str + "E=?, "                    
                model_tuple = model_tuple + (coldb.array_to_blob(model['E']), )
            if how_many_models != 1:
                config = MULTIPLE
                    
            query_str = "UPDATE two_view_geometries SET rows=?, cols=?, data=?, " + model_str + "config=? WHERE pair_id=?"

            if matches.shape[0] > 0:
                self.execute(
                    query_str,
                    matches.shape + (coldb.array_to_blob(matches), ) + model_tuple + (config, pair_id, ),
                    )
            else:
                self.execute(
                    "DELETE FROM two_view_geometries WHERE pair_id=?",
                    (pair_id, ),
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
            'only_matched': False,
            'no_unmatched': True,
            'include_two_view_geometry': True,
            'sampling_mode': 'raw',
            'overlapping_cells' : False,
            'sampling_scale': 1,
            'sampling_offset': 0,
        }
        
        self.id_string, self.args = set_args('to_colmap' , args, self.args)

        self.db = coldb_ext(self.args['db'])
        self.db.create_tables()
        self.aux_hdf5 = None
        if (self.args['sampling_mode'] == 'avg_all_matches') or (self.args['sampling_mode'] == 'avg_inlier_matches'):         
            self.aux_hdf5 = pickled_hdf5.pickled_hdf5(self.args['aux_hdf5'], mode='a', label_prefix='pickled/' + self.id_string)
                

    def finalize(self):
        self.db.commit()
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

        m_idx_old = torch.zeros((0, 2), device=device, dtype=torch.int)        
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
        pipe_out = pipe_union([pipe_old, args], unique=self.args['unique'], no_unmatched=self.args['no_unmatched'], only_matched=self.args['only_matched'], sampling_mode=self.args['sampling_mode'], sampling_scale=self.args['sampling_scale'], sampling_offset=self.args['sampling_offset'], overlapping_cells=self.args['overlapping_cells'], preserve_order=True, counter=counter)

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

            if self.args['include_two_view_geometry']:

                m_idx = pipe_out['m_idx'][pipe_out['m_mask']].to('cpu').numpy()
                models = {}
                for m in ['H', 'E', 'F']:
                    if (m in args):
                        if not (args[m] is None):
                            models[m] = args[m].to('cpu').numpy()
                                
                self.db.update_two_view_geometry(im_ids[0], im_ids[1], m_idx, model=models)

        self.db.commit()
        
        return {}


def kpts_from_colmap(kp): 
    w_ = kp[:, 2:]
    kp = kp[:, :2]
    w = torch.zeros((kp.shape[0], 3, 3), device=device)
    w[:, 2, 2] = 1
    w[:, :2, :2] = w_.reshape(-1, 2, 2)
         
    t = torch.zeros((kp.shape[0], 3, 3), device=device)        
    t[:, [0, 1], 2] = kp
    t[:, 0, 0] = 1
    t[:, 1, 1] = 1
    t[:, 2, 2] = 1           
     
    kH = t.bmm(w).inverse()
     
    kr = torch.full((kp.shape[0], ), np.inf, device=device)    
             
    return kp, kH, kr


class from_colmap_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True

        self.args = {
            'id_more': '',
            'db': 'colmap.db',
            'only_keypoints': False,            
            'include_two_view_geometry': True,
        }
        
        self.id_string, self.args = set_args('from_colmap' , args, self.args)

        self.db = coldb_ext(self.args['db'])
        if self.args['only_keypoints']:
            self.single_image = True
                

    def finalize(self):
        self.db.close()

                
    def get_id(self): 
        return self.id_string

    
    def run(self, **args):   
        if self.single_image:
            im = args['img'][args['idx']]
            _, img = os.path.split(im)           
            im_id = self.db.get_image_id(img)

            if im_id is None:
                kp = torch.zeros((0, 2), device=device)
                kr = torch.zeros((0, ), device=device)
                kH = torch.zeros((0, 3, 3), device=device)
            else:                
                kp_ = self.db.get_keypoints(im_id)
                kp, kH, kr = kpts_from_colmap(torch.tensor(kp_, device=device))

            return {'kp': kp, 'kH': kH, 'kr': kr}
        
        else:
            out_data = {}
            
            im0 = args['img'][0]            
            _, img0 = os.path.split(im0)           
            im0_id = self.db.get_image_id(img0)

            if im0_id is None:
                kp0 = torch.zeros((0, 2), device=device)
                kr0 = torch.zeros((0, ), device=device)
                kH0 = torch.zeros((0, 3, 3), device=device)
            else:                
                kp0_ = self.db.get_keypoints(im0_id)
                kp0, kH0, kr0 = kpts_from_colmap(torch.tensor(kp0_, device=device))

            im1 = args['img'][1]            
            _, img1 = os.path.split(im1)           
            im1_id = self.db.get_image_id(img1)

            if im1_id is None:
                kp1 = torch.zeros((0, 2), device=device)
                kr1 = torch.zeros((0, ), device=device)
                kH1 = torch.zeros((0, 3, 3), device=device)
            else:                
                kp1_ = self.db.get_keypoints(im1_id)
                kp1, kH1, kr1 = kpts_from_colmap(torch.tensor(kp1_, device=device))

            kp = [kp0, kp1]
            kH = [kH0, kH1]
            kr = [kr0, kr1]
            
            out_data['kp'] = kp
            out_data['kH'] = kH
            out_data['kr'] = kr

            if (not (im0_id is None)) and (not (im1_id is None)):
                m_idx = self.db.get_matches(im0_id, im1_id)
                
                if not (m_idx is None):
                    m_idx = torch.tensor(np.copy(m_idx), device=device, dtype=torch.int)
                    
                    if not self.args['include_two_view_geometry']:
                        m_mask = torch.full((m_idx.shape[0],), 1, device=device, dtype=torch.bool)
                        m_val = torch.full((m_idx.shape[0],), np.inf, device=device)
                    
                    else:
                        s_idx, models = self.db.get_two_view_geometry(im0_id, im1_id)
                    
                        if s_idx is None:
                            m_mask = torch.full((m_idx.shape[0],), 1, device=device, dtype=torch.bool)
                        else:
                            s_idx = torch.tensor(np.copy(s_idx), device=device, dtype=torch.int)
                            
                            if len(models.keys()) == 1:
                                for model in ['H', 'F', 'E']:
                                    if model in models: out_data[model] = torch.tensor(models[model], device=device)
                            
                            m_mask = torch.zeros(m_idx.shape[0], device=device, dtype=torch.bool)
                            
                            idx = torch.argsort(m_idx[:, 1].type(torch.int), stable=True)
                            m_idx = m_idx[idx]
                            idx = torch.argsort(m_idx[:, 0].type(torch.int), stable=True)
                            m_idx = m_idx[idx]

                            idx = torch.argsort(s_idx[:, 1].type(torch.int), stable=True)
                            s_idx = s_idx[idx]
                            idx = torch.argsort(s_idx[:, 0].type(torch.int), stable=True)
                            s_idx = s_idx[idx]

                            q0 = 0
                            q1 = 0
                            while (q0 < s_idx.shape[0]) and (q1 < m_idx.shape[0]):                       
                                if (s_idx[q0, 0] < m_idx[q1, 0]) or ((s_idx[q0, 0] == m_idx[q1, 0]) and (s_idx[q0, 1] < m_idx[q1, 1])):
                                    q0 = q0 + 1
                                elif (s_idx[q0, 0] == m_idx[q1, 0]) and (s_idx[q0, 1] == m_idx[q1, 1]):
                                    m_mask[q1] = 1
                                    q0 = q0 + 1
                                    q1 = q1 + 1
                                else:
                                    q1 = q1 + 1

                        m_val = torch.full((m_idx.shape[0],), np.inf, device=device)
                                                    
                    out_data['m_idx'] = m_idx
                    out_data['m_val'] = m_val
                    out_data['m_mask'] = m_mask
        
        return out_data


def relative_pose_error_angular(R_gt, t_gt, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    # t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    # R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, max_iters=10000):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC, maxIters=max_iters)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def relative_pose_error_metric(R_gt, t_gt, R, t, scale_cf=1.0, use_gt_norm=True, t_ambiguity=True):
    t_gt = t_gt * scale_cf
    t = t * scale_cf
    if use_gt_norm: 
        n_gt = np.linalg.norm(t_gt)
        n = np.linalg.norm(t)
        t = t / n * n_gt

    if t_ambiguity:
        t_err = np.minimum(np.linalg.norm(t_gt - t), np.linalg.norm(t_gt + t))
    else:
        t_err = np.linalg.norm(t_gt - t)

    if not isinstance(R, list):
        R = [R]
        
    R_err = []
    for R_ in R:        
        cos = (np.trace(np.dot(R_.T, R_gt)) - 1) / 2
        cos = np.clip(cos, -1., 1.)  # handle numercial errors
        R_err.append(np.rad2deg(np.abs(np.arccos(cos))))
    
    R_err = np.min(R_err)

    return t_err, R_err


def error_auc(errors, thr):
    errors = [0] + sorted(errors)
    recall = list(np.linspace(0, 1, len(errors)))

    last_index = np.searchsorted(errors, thr)
    y = recall[:last_index] + [recall[last_index-1]]
    x = errors[:last_index] + [thr]
    return np.trapezoid(y, x) / thr    


class pairwise_benchmark_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False        
        self.pass_through = True

        self.args = { 
            'id_more': '',
            'gt': None,
            'to_add_path': '',
            'aux_hdf5': 'stats.hdf5',
            'err_th_list': list(range(1,16)),
            'essential_th': 0.5,
            'use_fundamental': True,
            'use_metric': False,
            'angular_thresholds': [5, 10, 20],
            'metric_thresholds': [0.5, 1, 2],
            'am_scaling' : 10, # current metric error requires that angular_thresholds[i] / metric_thresholds[i] = am_scaling
            'save_to': None,
            }
                                
        self.id_string, self.args = set_args('pairwise_benchmark', args, self.args)    
        
        if self.args['gt'] is None:
            warnings.warn("no gt data given!")

        self.args['to_add_path_size'] = len(self.args['to_add_path'])    

        if self.args['err_th_list'] is None:
            self.args['err_th_list'] = list(range(1,16))            
                        
        self.aux_hdf5 = pickled_hdf5.pickled_hdf5(self.args['aux_hdf5'], mode='a', label_prefix='pickled/' + self.id_string)
                

    def finalize(self):
        keys = self.aux_hdf5.get_keys()

        if self.args['use_fundamental']:
            fe = 'F'
        else:
            fe = 'E'
            
        if not (self.args['save_to'] is None):
            f = open(self.args['save_to'], 'w')

        R_error = []
        t_error = []
        n = 0
        inliers = torch.zeros(len(self.args['err_th_list']), device=device, dtype=torch.int)
        auc = []
        acc = []

        for key in keys:      
            val, is_found = self.aux_hdf5.get(key)

            R_error.append(val['R_error'])
            t_error.append(val['t_error'])
            n = n + val['n']
            inliers = inliers + val['inliers']
                        
            aux = np.asarray([R_error, t_error]).T
            if self.args['use_metric']:
                aux[:, 1] = aux[:, 1] * self.args['am_scaling']
            
            max_Rt_err = np.max(aux, axis=1)        
            tmp = np.concatenate((aux, np.expand_dims(max_Rt_err, axis=1)), axis=1)

        if not self.args['use_metric']:    
            for a in self.args['angular_thresholds']:       
                auc_R = error_auc(R_error, a).item()
                auc_t = error_auc(t_error, a).item()
                auc_max_Rt = error_auc(max_Rt_err, a).item()
                acc_ = np.sum(tmp < a, axis=0)/np.shape(tmp)[0]
    
                auc.append([a, auc_R, auc_t, auc_max_Rt])
                acc.append([a, acc_[0].item(), acc_[1].item(), acc_[2].item()])

            avg_inliers = inliers.type(torch.float).mean().to('cpu').numpy().item()
            avg_precision = (inliers / n).type(torch.float).mean().to('cpu').numpy().item()

            if self.args['save_to'] is None:
                print("             R        t        max(R,t)")
                for i, a in enumerate(self.args['angular_thresholds']):       
                    print(f"AUC@{str(a).ljust(2,' ')} ({fe}) : {auc[i][1]*100: >6.2f}%, {auc[i][2]*100: >6.2f}%, {auc[i][3]*100: >6.2f}%")        
                for i, a in enumerate(self.args['angular_thresholds']):       
                    print(f"Acc@{str(a).ljust(2,' ')} ({fe}) : {acc[i][1]*100: >6.2f}%, {acc[i][2]*100: >6.2f}%, {acc[i][3]*100: >6.2f}%")        
                print(f"Prec ({fe}) : {avg_inliers: .0f} / {n} = {avg_precision*100: >6.2f}%")
            else:
                print("what; angular th; metric th; F/E; R; t; max(R,t); inliers; matches; prec", file=f)
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"AUC; {str(a)}; nan; {fe}; {auc[i][1]}; {auc[i][2]}; {auc[i][3]}; nan; nan; nan", file=f)    
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"Acc; {str(a)}; nan; {fe}; {acc[i][1]}; {acc[i][2]}; {acc[i][3]}; nan; nan; nan", file=f)    
                print(f"Prec; nan; nan; {fe}; nan; nan; nan; {avg_inliers}; {n}; {avg_precision}", file=f)
        else:
            for a, m in zip(self.args['angular_thresholds'], self.args['metric_thresholds']):       
                auc_R = error_auc(R_error, a).item()
                auc_t = error_auc(t_error, m).item()
                auc_max_Rt = error_auc(max_Rt_err, a).item()
                                
                aa = (aux[:, 0] < a)[:, np.newaxis]
                mm = (aux[:, 1] < m)[:, np.newaxis]
                tmp = np.concatenate((aa, mm, aa & mm), axis=1)
                acc_ = np.sum(tmp, axis=0) / np.shape(tmp)[0]

                auc.append([a, auc_R, auc_t, auc_max_Rt])
                acc.append([a, acc_[0].item(), acc_[1].item(), acc_[2].item()])

            avg_inliers = inliers.type(torch.float).mean().to('cpu').numpy().item()
            avg_precision = (inliers / n).type(torch.float).mean().to('cpu').numpy().item()

            if self.args['save_to'] is None:    
                print("                 R        t        max(R,t)")
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"@AUC{str(a).ljust(2,' ')},{str(m).ljust(3,' ')} ({fe}) : {auc[i][1]*100: >6.2f}%, {auc[i][2]*100: >6.2f}%, {auc[i][3]*100: >6.2f}%")    
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"@Acc{str(a).ljust(2,' ')},{str(m).ljust(3,' ')} : {acc[i][1]*100: >6.2f}%, {acc[i][2]*100: >6.2f}%, {acc[i][3]*100: >6.2f}%")    
                print(f"Prec ({fe}) : {avg_inliers: .0f} / {n} = {avg_precision*100: >6.2f}%")
            else:
                print("what; angular th; metric th; F/E; R; t; max(R,t); inliers; matches; prec", file=f)
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"AUC; {str(a)}; {str(m)}; {fe}; {auc[i][1]}; {auc[i][2]}; {auc[i][3]}; nan; nan; nan", file=f)    
                for i, am in enumerate(zip(self.args['angular_thresholds'], self.args['metric_thresholds'])):       
                    a = am[0]
                    m = am[1]
                    print(f"Acc; {str(a)}; {str(m)}; {fe}; {acc[i][1]}; {acc[i][2]}; {acc[i][3]}; nan; nan; nan", file=f)    
                print(f"Prec; nan; nan; {fe}; nan; nan; nan; {avg_inliers}; {n}; {avg_precision}", file=f)

        self.aux_hdf5.close()

        if not (self.args['save_to'] is None):
            f.close()


    def get_id(self): 
        return self.id_string


    def run(self, **args):
        if self.args['use_fundamental']:
            return self.run_fundamental(**args)
        else:
            return self.run_essential(**args)
            
    
    def run_fundamental(self, **args):
        err_th_list = self.args['err_th_list']
        
        img1 = args['img'][0]
        img2 = args['img'][1]
        
        if self.args['use_metric']:
            key_metric = '_metric'
        else:
            key_metric = ''
        
        data_key = '/' + os.path.split(img1)[-1] + '/' + os.path.split(img2)[-1] + '/fundamental' + key_metric
        
        out_data, is_found = self.aux_hdf5.get(data_key)
        if is_found:
            return {}

        cannot_do = False

        img1_key = img1[self.args['to_add_path_size'] + 1:]
        img2_key = img2[self.args['to_add_path_size'] + 1:]
        
        if cannot_do or (not (img1_key in self.args['gt'])):
            cannot_do = True
                        
        if cannot_do or (not (img2_key in self.args['gt'][img1_key])):
            cannot_do = None
        
        if not cannot_do:
            gt = self.args['gt'][img1_key][img2_key]
        else:
            gt = None

        if not (gt is None):
            K1 = gt['K1']
            K2 = gt['K2']    
            R_gt = gt['R']
            t_gt = gt['T']
            
            if self.args['use_metric']:
                scene_scale = gt['scene_scale']
    
            mm = args['m_idx'][args['m_mask']]
        
            pts1 = args['kp'][0][mm[:, 0]]
            pts2 = args['kp'][1][mm[:, 1]]
                       
            if torch.is_tensor(pts1):
                pts1 = pts1.detach().cpu().numpy()
                pts2 = pts2.detach().cpu().numpy()
        
            scales = gt['image_pair_scale']
        
            pts1 = pts1 * scales[0]
            pts2 = pts2 * scales[1]
        
            nn = pts1.shape[0]

            inl_sum = torch.zeros(len(err_th_list), device=device, dtype=torch.int)
        
            if nn < 8:
                Rt_ = None
            else:
                if 'F' in args:
                    s1 = torch.eye(3, device=device)
                    s2 = torch.eye(3, device=device)

                    s1[0, 0] = 1 / scales[0, 0]
                    s1[1, 1] = 1 / scales[0, 1]

                    s2[0, 0] = 1 / scales[1, 0]
                    s2[1, 1] = 1 / scales[1, 1]

                    F = s2 @ args['F'].type(torch.float) @ s1
                    F = F / F[2, 2]
                    F = F.to('cpu').numpy()
                else:
                    F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]

                if F is None:
                    Rt_ = None
                else:
                    E = K2.T @ F @ K1
                    Rt_ = cv2.decomposeEssentialMat(E)
        
            if nn > 0:
                F_gt = torch.tensor(K2.T, device=device, dtype=torch.float64).inverse() @ \
                       torch.tensor([[0, -t_gt[2], t_gt[1]],
                                    [t_gt[2], 0, -t_gt[0]],
                                    [-t_gt[1], t_gt[0], 0]], device=device) @ \
                       torch.tensor(R_gt, device=device) @ \
                       torch.tensor(K1, device=device, dtype=torch.float64).inverse()
                F_gt = F_gt / F_gt.sum()
        
                pt1_ = torch.vstack((torch.tensor(pts1.T, device=device), torch.ones((1, nn), device=device)))
                pt2_ = torch.vstack((torch.tensor(pts2.T, device=device), torch.ones((1, nn), device=device)))
        
                l1_ = F_gt @ pt1_
                d1 = pt2_.permute(1,0).unsqueeze(-2).bmm(l1_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l1_[:2]**2).sum(0).sqrt()
        
                l2_ = F_gt.T @ pt2_
                d2 = pt1_.permute(1,0).unsqueeze(-2).bmm(l2_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l2_[:2]**2).sum(0).sqrt()
        
                epi_max_err = torch.maximum(d1, d2)
                inl_sum = (epi_max_err.unsqueeze(-1) < torch.tensor(err_th_list, device=device).unsqueeze(0)).sum(dim=0).type(torch.int)        
            
            if Rt_ is None:
                R_error = np.inf
                t_error = np.inf
            else:
                R_a, t_a, = Rt_[0], Rt_[2].squeeze()
                R_b, t_b, = Rt_[1], Rt_[2].squeeze()

                if not self.args['use_metric']:
                    t_err_a, R_err_a = relative_pose_error_angular(R_gt, t_gt, R_a, t_a)
                    t_err_b, R_err_b = relative_pose_error_angular(R_gt, t_gt, R_b, t_b)
            
                    if max(R_err_a, t_err_a) < max(R_err_b, t_err_b):
                        R_err, t_err = R_err_a, t_err_b
                    else:
                        R_err, t_err = R_err_b, t_err_b
                else:
                    t_err, R_err = relative_pose_error_metric(R_gt, t_gt, [Rt_[0], Rt_[1]], Rt_[2].squeeze(), scale_cf=scene_scale)
        
                R_error = R_err
                t_error = t_err
                
            out_data = {'R_error': R_error.item(), 't_error': t_error.item(), 'n': nn, 'inliers': inl_sum}
        else:
            warnings.warn("image pair not in gt data!")            
            out_data = None

        self.aux_hdf5.add(data_key, out_data)
        return {}


    def run_essential(self, **args):        
        img1 = args['img'][0]
        img2 = args['img'][1]
        
        if self.args['use_metric']:
            key_metric = '_metric'
        else:
            key_metric = ''
                
        data_key = '/' + os.path.split(img1)[-1] + '/' + os.path.split(img2)[-1] + '/essential' + key_metric
        
        out_data, is_found = self.aux_hdf5.get(data_key)
        if is_found:
            return {}

        cannot_do = False

        img1_key = img1[self.args['to_add_path_size'] + 1:]
        img2_key = img2[self.args['to_add_path_size'] + 1:]
        
        if cannot_do or (not (img1_key in self.args['gt'])):
            cannot_do = True
                        
        if cannot_do or (not (img2_key in self.args['gt'][img1_key])):
            cannot_do = None
        
        if not cannot_do:
            gt = self.args['gt'][img1_key][img2_key]
        else:
            gt = None

        if not (gt is None):
            K1 = gt['K1']
            K2 = gt['K2']    
            R_gt = gt['R']
            t_gt = gt['T']

            if self.args['use_metric']:
                scene_scale = gt['scene_scale']
    
            mm = args['m_idx'][args['m_mask']]
        
            pts1 = args['kp'][0][mm[:, 0]]
            pts2 = args['kp'][1][mm[:, 1]]
                       
            if torch.is_tensor(pts1):
                pts1 = pts1.detach().cpu().numpy()
                pts2 = pts2.detach().cpu().numpy()
        
            scales = gt['image_pair_scale']
        
            pts1 = pts1 * scales[0]
            pts2 = pts2 * scales[1]
        
            nn = pts1.shape[0]

            inl_sum = 0
        
            if nn < 5:
                Rt = None
            else:
                Rt = estimate_pose(pts1, pts2, K1, K2, self.args['essential_th'])                                                        

            if Rt is None:
                R_error = np.inf
                t_error = np.inf                          
            else:
                R, t, inliers = Rt

                if not self.args['use_metric']:
                    t_err, R_err = relative_pose_error_angular(R_gt, t_gt, R, t)
                else:
                    t_err, R_err = relative_pose_error_metric(R_gt, t_gt, R, t, scale_cf=scene_scale)
        
                R_error = R_err
                t_error = t_err
                inl_sum = inliers.sum()
                
            out_data = {'R_error': R_error.item(), 't_error': t_error.item(), 'n': nn, 'inliers': inl_sum}
        else:
            warnings.warn("image pair not in gt data!")            
            out_data = None

        self.aux_hdf5.add(data_key, out_data)
        return {}


from romatch import roma_outdoor, roma_indoor, tiny_roma_v1_outdoor

class roma_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
                        
        self.args = {
            'id_more': '',
            'outdoor': True,
            'use_tiny': False,
            'coarse_resolution': 280,
            'upsample_resolution': 432,
            'max_keypoints': 2000,
            'patch_radius': 16,
            }

        self.id_string, self.args = set_args('roma', args, self.args)        

        roma_args = {}
        if not (self.args['coarse_resolution'] is None):
            roma_args['coarse_res'] = self.args['coarse_resolution']
        if not (self.args['upsample_resolution'] is None):
            roma_args['upsample_res'] = self.args['upsample_resolution']

        if self.args['use_tiny']:
            self.roma_model = tiny_roma_v1_outdoor(device=device)            
        else:
            if self.args['outdoor'] == True:
                self.roma_model = roma_outdoor(device=device, **roma_args)
            else:
                self.roma_model = roma_indoor(device=device, **roma_args)


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):
        W_A, H_A = Image.open(args['img'][0]).size
        W_B, H_B = Image.open(args['img'][1]).size
    
        # Match
        if self.args['use_tiny']:
            warp, certainty = self.roma_model.match(args['img'][0], args['img'][1])
        else:
            warp, certainty = self.roma_model.match(args['img'][0], args['img'][1], device=device)
        # Sample matches for estimation
        
        sampling_args = {}
        if not (self.args['max_keypoints'] is None):
            sampling_args['num'] = self.args['max_keypoints']
        
        matches, certainty = self.roma_model.sample(warp, certainty, **sampling_args)
        kpts1, kpts2 = self.roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    

        kps1 = kpts1.detach().to(device)
        kps2 = kpts2.detach().to(device)
        
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

        m_val = certainty.detach().to(device)        

        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}


conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'r2d2'))

from r2d2.tools import common as r2d2_common
from r2d2.tools.dataloader import norm_RGB as r2d2_norm_RGB
import r2d2.nets.patchnet as r2d2_patchnet 

class r2d2_module:
    def load_network(model_fn): 
        checkpoint = torch.load(model_fn, weights_only=False)
        # print("\n>> Creating net = " + checkpoint['net']) 
        net = eval('r2d2_patchnet.' + checkpoint['net'])
        # nb_of_weights = r2d2_common.model_size(net)
        # print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")
    
        # initialization
        weights = checkpoint['state_dict']
        net.load_state_dict({k.replace('module.',''): v for k,v in weights.items()})
        return net.eval()
    
    
    class NonMaxSuppression(torch.nn.Module):
        def __init__(self, rel_thr=0.7, rep_thr=0.7):
            r2d2_patchnet.nn.Module.__init__(self)
            self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.rel_thr = rel_thr
            self.rep_thr = rep_thr

        
        def forward(self, reliability, repeatability, **kw):
            assert len(reliability) == len(repeatability) == 1
            reliability, repeatability = reliability[0], repeatability[0]
    
            # local maxima
            maxima = (repeatability == self.max_filter(repeatability))
    
            # remove low peaks
            maxima *= (repeatability >= self.rep_thr)
            maxima *= (reliability   >= self.rel_thr)
    
            return maxima.nonzero().t()[2:4]
    
    
    def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                            min_scale=0.0, max_scale=1, 
                            min_size=256, max_size=1024, 
                            verbose=False):
        old_bm = torch.backends.cudnn.benchmark 
        torch.backends.cudnn.benchmark = False # speedup
        
        # extract keypoints at multiple scales
        B, three, H, W = img.shape
        assert B == 1 and three == 3, "should be a batch with a single RGB image"
        
        assert max_scale <= 1
        s = 1.0 # current scale factor
        
        X,Y,S,C,Q,D = [],[],[],[],[],[]
        while  s+0.001 >= max(min_scale, min_size / max(H,W)):
            if s-0.001 <= min(max_scale, max_size / max(H,W)):
                nh, nw = img.shape[2:]
                if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
                # extract descriptors
                with torch.no_grad():
                    res = net(imgs=[img])
                    
                # get output and reliability map
                descriptors = res['descriptors'][0]
                reliability = res['reliability'][0]
                repeatability = res['repeatability'][0]
    
                # normalize the reliability for nms
                # extract maxima and descs
                y,x = detector(**res) # nms
                c = reliability[0,0,y,x]
                q = repeatability[0,0,y,x]
                d = descriptors[0,:,y,x].t()
                n = d.shape[0]
    
                # accumulate multiple scales
                X.append(x.float() * W/nw)
                Y.append(y.float() * H/nh)
                S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
                C.append(c)
                Q.append(q)
                D.append(d)
            s /= scale_f
    
            # down-scale the image for next iteration
            nh, nw = round(H*s), round(W*s)
            img = r2d2_patchnet.F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)
    
        # restore value
        torch.backends.cudnn.benchmark = old_bm
    
        Y = torch.cat(Y)
        X = torch.cat(X)
        S = torch.cat(S) # scale
        scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
        XYS = torch.stack([X,Y,S], dim=-1)
        D = torch.cat(D)
        return XYS, D, scores


    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False
        self.pass_through = False
                        
        self.args = { 
            'id_more': '',
            'patch_radius': 16,            
            'top-k': 5000,
            'scale_f': 2**0.25,
            'min_size': 256,
            'max_size': 1024,
            'min_scale': 0,
            'max_scale': 1,
            'reliability-thr': 0.7,
            'repeatability-thr': 0.7,
            'model': 'r2d2/models/r2d2_WAF_N16.pt',
            }
                
        self.id_string, self.args = set_args('r2d2', args, self.args)        

        if device.type == 'cuda':
            cuda = 0
        else:
            cuda = -1

        self.iscuda = r2d2_common.torch_set_gpu(cuda)
    
        # load the network...
        self.net = r2d2_module.load_network(self.args['model'])
        if self.iscuda: self.net = self.net.cuda()
    
        # create the non-maxima detector
        self.detector = r2d2_module.NonMaxSuppression(
            rel_thr = self.args['reliability-thr'], 
            rep_thr = self.args['repeatability-thr'])


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):
        img = Image.open(args['img'][args['idx']]).convert('RGB')
        W, H = img.size
        img = r2d2_norm_RGB(img)[None] 
        if self.iscuda: img = img.cuda()
        
        # extract keypoints/descriptors for a single image
        xys, desc, scores = r2d2_module.extract_multiscale(self.net, img, self.detector,
            scale_f   = self.args['scale_f'], 
            min_scale = self.args['min_scale'], 
            max_scale = self.args['max_scale'],
            min_size  = self.args['min_size'], 
            max_size  = self.args['max_size'], 
            verbose = False)

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[-self.args['top-k'] or None:]
        
        keypoints = torch.tensor(xys[idxs], device=device) 
        descriptors = torch.tensor(desc[idxs], device=device) 
        scores = torch.tensor(scores[idxs], device=device)

        kp = keypoints[:, :2]       
        desc = descriptors
        scales = keypoints[:, 2] / 2

        kH = torch.zeros((kp.shape[0], 3, 3), device=device)        
        kH[:, [0, 1], 2] = -kp / self.args['patch_radius']
        kH[:, 0, 0] = 1 / scales
        kH[:, 1, 1] = 1 / scales
        kH[:, 2, 2] = 1

        kr = scores        
        
        # todo: add feats['keypoint_scores'] as kr        
        return {'kp': kp, 'kH': kH, 'kr': kr, 'desc': desc}


if enable_quadtree:
    cur_dir = os.getcwd()
    os.chdir(os.path.join(os.path.split(__file__)[0], 'quadtreeattention/FeatureMatching'))    
    from src.config.default import qta_get_cfg_defaults
    from src.utils.misc import qta_lower_config
    from src.loftr import qta_LoFTR
    os.chdir(cur_dir)
    
    class quadtreeattention_module:
        def __init__(self, **args):
            self.single_image = False
            self.pipeliner = False   
            self.pass_through = False
                            
            self.args = {
                'id_more': '',
                'outdoor': True,
                'resize': None,                      # self.resize = [800, 600]
                'patch_radius': 16,
                }
    
            self.id_string, self.args = set_args('quadtreeattention', args, self.args)        
    
            download_quadtreeattention()
    
            if self.args['outdoor'] == True:
                self.weights = '../weights/quadtreeattention/outdoor.ckpt'
                self.config_path = 'quadtreeattention/FeatureMatching/configs/loftr/outdoor/loftr_ds_quadtree.py'
            else:
                self.weights = '../weights/quadtreeattention/indoor.ckpt'
                self.config_path = 'quadtreeattention/FeatureMatching/configs/loftr/indoor/loftr_ds_quadtree.py'
    
            parser = argparse.ArgumentParser(description='QuadTreeAttention online demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser.add_argument('--weight', type=str, default=self.weights, help="Path to the checkpoint.")
            parser.add_argument('--config_path', type=str, default=self.config_path, help="Path to the config.")
    
            opt = parser.parse_args()
        
            # init default-cfg and merge it with the main- and data-cfg
            config = qta_get_cfg_defaults()
            config.merge_from_file(opt.config_path)
            _config = qta_lower_config(config)
        
            # Matcher: LoFTR
            self.matcher = qta_LoFTR(config=_config['loftr'])
            state_dict = torch.load(opt.weight, map_location='cpu', weights_only=False)['state_dict']
            self.matcher.load_state_dict(state_dict, strict=True)

            self.matcher.eval()

            if device.type == 'cuda':        
                self.matcher.to('cuda')
    
        def get_id(self): 
            return self.id_string
        
        
        def finalize(self):
            return
    
    
        def run(self, **args):
            image0 = K.io.load_image(args['img'][0], K.io.ImageLoadType.GRAY32, device=device)
            image1 = K.io.load_image(args['img'][1], K.io.ImageLoadType.GRAY32, device=device)
    
            hw1 = image0.shape[1:]
            hw2 = image1.shape[1:]
    
            if not (self.args['resize'] is None):        
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
    
            batch = {
                "image0": image0.unsqueeze(0),    # LofTR works on grayscale images
                "image1": image1.unsqueeze(0),
            }
    
            self.matcher(batch)
            kps1 = batch['mkpts0_f'].detach().to(device).squeeze()
            kps2 = batch['mkpts1_f'].detach().to(device).squeeze()
            m_val = batch['mconf'].detach().to(device)
            m_mask = m_val > 0
    
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
    
            return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}


conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'matchformer'))

from matchformer.model.matchformer import Matchformer
from matchformer.model.utils.misc import lower_config as mf_lower_config
from matchformer.config.defaultmf import get_cfg_defaults as mf_get_cfg_defaults 
  
class matchformer_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
                        
        self.args = {
            'id_more': '',
            'model': 'outdoor-large-LA',
            'resize': None,                          # self.resize = [800, 600]
            'patch_radius': 16,
            }

        self.id_string, self.args = set_args('matchformer', args, self.args)        

        download_matchformer()
        cfgs = os.listdir('../data/matchfomer_cfgs')
        for fname in cfgs:
            dst = os.path.join('matchformer/config', fname)
            src = os.path.join('../data/matchfomer_cfgs', fname)
            if not os.path.isfile(dst):
                shutil.copy(src, dst)

        self.weights = os.path.join('../weights/matchformer/', self.args['model'] + '.ckpt')
        self.config_path = os.path.join('matchformer/config', self.args['model'] + '.py')
        self.base_cfg = 'matchformer/config/defaultmf.py'


        parser = argparse.ArgumentParser(description='MatchFormer online demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--weight', type=str, default=self.weights, help="Path to the checkpoint.")
        parser.add_argument('--config_path', type=str, default=self.base_cfg, help="Path to the config.")
        parser.add_argument('--config_path_other', type=str, default=self.config_path, help="Path to the config.")

        opt = parser.parse_args()
    
        # init default-cfg and merge it with the main- and data-cfg
        config = mf_get_cfg_defaults()
        config.merge_from_file(opt.config_path_other)
        _config = mf_lower_config(config)
    
        # Matcher: LoFTR
        self.matcher = Matchformer(config=_config['matchformer'])        
        self.matcher.load_state_dict({k.replace('matcher.',''):v  for k,v in torch.load(opt.weight, map_location='cpu', weights_only=False).items()})

        self.matcher.eval()
        
        if device.type == 'cuda':        
            self.matcher.to('cuda')

    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):
        image0 = K.io.load_image(args['img'][0], K.io.ImageLoadType.GRAY32, device=device)
        image1 = K.io.load_image(args['img'][1], K.io.ImageLoadType.GRAY32, device=device)

        hw1 = image0.shape[1:]
        hw2 = image1.shape[1:]

        if not (self.args['resize'] is None):        
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

        batch = {
            "image0": image0.unsqueeze(0),    # LofTR works on grayscale images
            "image1": image1.unsqueeze(0),
        }

        self.matcher(batch)
        kps1 = batch['mkpts0_f'].detach().to(device).squeeze()
        kps2 = batch['mkpts1_f'].detach().to(device).squeeze()
        m_val = batch['mconf'].detach().to(device)
        m_mask = m_val > 0

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

        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}


conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'aspanformer'))

from aspanformer.src.ASpanFormer.aspanformer import ASpanFormer 
from aspanformer.src.config.default import get_cfg_defaults as as_get_cfg_defaults
from aspanformer.src.utils.misc import lower_config as as_lower_config
  
class aspanformer_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
                        
        self.args = {
            'id_more': '',
            'outdoor': True,
            'resize': None,                              # default [1024, 1024]
            'patch_radius': 16,
            }

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

        config = as_get_cfg_defaults()
        config.merge_from_file(as_args.config_path)
        _config = as_lower_config(config)
        self.matcher = ASpanFormer(config=_config['aspan'])
        state_dict = torch.load(as_args.weights_path, map_location='cpu', weights_only=False)['state_dict']
        self.matcher.load_state_dict(state_dict,strict=False)

        if device.type == 'cuda':        
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

        image0 = K.io.load_image(args['img'][0], K.io.ImageLoadType.GRAY32, device=device)
        image1 = K.io.load_image(args['img'][1], K.io.ImageLoadType.GRAY32, device=device)

        hw1 = image0.shape[1:]
        hw2 = image1.shape[1:]

        if not (self.args['resize'] is None):        
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

        kps1 = data['mkpts0_f'].detach().to(device).squeeze()
        kps2 = data['mkpts1_f'].detach().to(device).squeeze()
        m_val = data['mconf'].detach().to(device)
        m_mask = m_val > 0

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
                
        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}


import lpm.LPM as lpm

class lpm_module:
    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
                
        self.args = {
            'id_more': '',
            }
        
        self.id_string, self.args = set_args('lpm', args, self.args)        


    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args):  
        pt1_ = args['kp'][0]
        pt2_ = args['kp'][1]
                
        mi = args['m_idx']
        mm = args['m_mask']
        
        pt1 = pt1_[mi[mm][:, 0]]
        pt2 = pt2_[mi[mm][:, 1]]
        
        mask = lpm.LPM_filter(pt1.to('cpu').numpy(), pt2.to('cpu').numpy())        
        mask = torch.tensor(mask, device=device, dtype=torch.bool)
 
        aux = mm.clone()
        mm[aux] = mask
        
        return {'m_mask': mm}


import gms.python.gms_matcher as gms

class gms_module:
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
                
        self.args = {
            'id_more': '',
            }
        
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

        _, mask = gms.compute_matches(sz1r, sz1c, sz2r, sz2c);

        mask = torch.tensor(mask, device=device, dtype=torch.bool)
 
        aux = mm.clone()
        mm[aux] = mask
        
        return {'m_mask': mm}


import adalam.adalam.adalam as adalam

class adalam_module:
    class adalamfilter_custom(adalam.AdalamFilter):
        def __init__(self, custom_config=None):         
            super().__init__(custom_config=custom_config)
            

        def match_and_filter(self, k1, k2, im1shape=None, im2shape=None, o1=None, o2=None, s1=None, s2=None, putative_matches=None, scores=None, mnn=None):    
            if s1 is None or s2 is None:
                if self.config['scale_rate_threshold'] is not None:
                    raise AttributeError("Current configuration considers keypoint scales for filtering, but scales have not been provided.\n"
                                         "Please either provide scales or set 'scale_rate_threshold' to None to disable scale filtering")
            if o1 is None or o2 is None:
                if self.config['orientation_difference_threshold'] is not None:
                    raise AttributeError(
                        "Current configuration considers keypoint orientations for filtering, but orientations have not been provided.\n"
                        "Please either provide orientations or set 'orientation_difference_threshold' to None to disable orientations filtering")
       
            if not self.config['force_seed_mnn']:
                mnn = None
    
            return self.filter_matches(k1, k2, putative_matches, scores, mnn, im1shape, im2shape, o1, o2, s1, s2)
        

    def __init__(self, **args):       
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
                
        self.args = {
            'id_more': '',
            'adalam_params': {
                    'area_ratio': 100,
                    'search_expansion': 4,
                    'ransac_iters': 128,
                    'min_inliers': 6,
                    'min_confidence': 200,
                    'orientation_difference_threshold': 30, 
                    'scale_rate_threshold': 1.5, 
                    'detected_scale_rate_threshold': 5, 
                    'refit': True, 
                    'force_seed_mnn': False,
                    'device': device,
                    'th': 0.8 **2,
                },   
            }
        
        self.id_string, self.args = set_args('adalam', args, self.args)        
        if self.args['adalam_params']['device'] is None:
            self.args['adalam_params']['device'] = device
        
        self.matcher = self.adalamfilter_custom(self.args['adalam_params'])


    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args):  
        sz1 = Image.open(args['img'][0]).size
        sz2 = Image.open(args['img'][1]).size

        k1 = args['kp'][0]
        k2 = args['kp'][1]
        
        lafs1 = homo2laf(args['kp'][0], args['kH'][0])
        lafs2 = homo2laf(args['kp'][1], args['kH'][1])
        
        kp1 = opencv_kpts_from_laf(lafs1)
        kp2 = opencv_kpts_from_laf(lafs2)

        o1 = torch.tensor([kp.angle for kp in kp1], device=device)
        o2 = torch.tensor([kp.angle for kp in kp2], device=device)

        s1 = torch.tensor([kp.size for kp in kp1], device=device)
        s2 = torch.tensor([kp.size for kp in kp2], device=device)

        mi = args['m_idx']
        mm = args['m_mask']
        mv = args['m_val']

        m12 = mi[mm]
        scores = mv[mm]
        
        k1 = k1[m12[:, 0]]
        k2 = k2[m12[:, 1]]

        s1 = s1[m12[:, 0]]
        s2 = s2[m12[:, 1]]

        o1 = o1[m12[:, 0]]
        o2 = o2[m12[:, 1]]
        
        puta_match = torch.arange(0, m12.shape[0], device=device)
        mask = self.matcher.match_and_filter(k1, k2, im1shape=sz1, im2shape=sz2, o1=o1, o2=o2, s1=s1, s2=s2, putative_matches=puta_match, scores=scores, mnn=None)
        
        mask_aux = torch.zeros(m12.shape[0], device=device, dtype=torch.bool)
        mask_aux[mask[:, 0]] = True
         
        aux = mm.clone()
        mm[aux] = mask_aux
        
        return {'m_mask': mm}


def download_fcgnn(weight_path='../weights/fcgnn'):
    file = 'fcgnn.model'    
    url = "https://github.com/xuy123456/fcgnn/releases/download/v0/fcgnn.model"

    os.makedirs(weight_path, exist_ok=True)   

    file_to_download = os.path.join(weight_path, file)    
    if not os.path.isfile(file_to_download):    
        wget.download(url, file_to_download)


import fcgnn.fcgnn as fcgnn

class fcgnn_module:
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
                
        self.args = {
            'id_more': '',
            'thd': 0.999,
            'min_matches': 10,
            }
        
        self.id_string, self.args = set_args('fcgnn', args, self.args)     
        self.fcgnn_refiner = self.fcgnn_custom().to(device)        


    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args): 
        img1 = cv2.imread(args['img'][0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(args['img'][1], cv2.IMREAD_GRAYSCALE)
        
        img1_ = torch.tensor(img1.astype('float32') / 255.)[None, None].to(device)
        img2_ = torch.tensor(img2.astype('float32') / 255.)[None, None].to(device)        
        
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


def download_oanet(weight_path='../weights/oanet'):
    url = "https://drive.google.com/file/d/1Yuk_ZBlY_xgUUGXCNQX-eh8BO2ni_qhm/view?usp=sharing"

    os.makedirs(os.path.join(weight_path, 'download'), exist_ok=True)   

    file_to_download = os.path.join(weight_path, 'download', 'sift-gl3d.tar.gz')    
    if not os.path.isfile(file_to_download):    
        gdown.download(url, file_to_download, fuzzy=True)

    model_file = os.path.join(weight_path, 'model_best.pth')
    if not os.path.isfile(model_file):
        with tarfile.open(file_to_download, "r") as tar_ref:
            tar_ref.extract('gl3d/sift-4000/model_best.pth', path=weight_path)
        
        shutil.copy(os.path.join(weight_path, 'gl3d/sift-4000/model_best.pth'), model_file)
        shutil.rmtree(os.path.join(weight_path, 'gl3d'))


import oanet.learnedmatcher_custom as oanet

class oanet_module:
    def __init__(self, **args):  

        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
                
        self.args = {
            'id_more': '',
            'weights': '../weights/oanet/model_best.pth',
            'inlier_threshold': 1,
            }
        
        download_oanet()
        
        self.id_string, self.args = set_args('oanet', args, self.args)                     
        self.lm = oanet.LearnedMatcher(self.args['weights'], inlier_threshold=self.args['inlier_threshold'], use_ratio=0, use_mutual=0, corr_file=-1)        
        
               
    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

        
    def run(self, **args): 
        mi = args['m_idx']
        mm = args['m_mask']

        m12 = mi[mm]

        k1 = args['kp'][0]
        k2 = args['kp'][1]
        
        k1 = k1[m12[:, 0]]
        k2 = k2[m12[:, 1]]        
        
        pt1 = np.ascontiguousarray(k1.detach().cpu())
        pt2 = np.ascontiguousarray(k2.detach().cpu())
                
        l = pt1.shape[0]
        
        if l > 1:
            _, _, _, _, mask = self.lm.infer(pt1, pt2)
            
            mask_aux = torch.tensor(mask, device=device)         
            aux = mm.clone()
            mm[aux] = mask_aux
        
            return {'m_mask': mm}
        else:
            return {'m_mask': args['m_mask']}


import uuid
from acne.config import get_config as acne_get_config
import acne.acne_custom as acne

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def download_acne(weight_path='../weights/acne'):
    os.makedirs(os.path.join(weight_path, 'download'), exist_ok=True)   

    file_to_download = os.path.join(weight_path, 'download', 'acne_weights.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1yluw3u3F8qH3oTB3dxVw1re4HI6a0TuQ/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)        

    file_to_unzip = file_to_download
    model_dir = os.path.join(weight_path, 'logs')    
    if not os.path.isdir(model_dir):    
        with zipfile.ZipFile(file_to_unzip,"r") as zip_ref:
            zip_ref.extractall(path=weight_path)


class acne_module:
    current_net = None
    current_obj_id = None
    
    def __init__(self, **args):
        self.single_image = False    
        self.pipeliner = False     
        self.pass_through = False
                
        self.args = {
            'id_more': '',
            'outdoor': True,
            'what': 'ACNe_F',
            }
        
        download_acne()
        
        self.id_string, self.args = set_args('acne', args, self.args)                     
        self.acne_id = uuid.uuid4()

        if self.args['outdoor']:
            # Model of ACNe_F trained with outdoor dataset.              
            model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_outdoor/models-best"
        else:
            # Model of ACNe_F trained with indoor dataset.                      
             model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_indoor/models-best"        

        self.model_path = os.path.join('../weights/acne', model_path)
        self.acne_id = uuid.uuid4()  
        
        self.outdoor = self.args['outdoor']
        self.prev_outdoor = self.args['outdoor']      
        
        
    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return


    def run(self, **args):
        force_reload = False
        if (self.outdoor != self.prev_outdoor):
            force_reload = True
            warnings.warn("acne modules with both indoor and outdoor model detected, computation will be very slow...")
            self.prev_outdoor = self.outdoor
            
            if self.outdoor:
                # Model of ACNe_F trained with outdoor dataset.              
                model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_outdoor/models-best"
            else:
                # Model of ACNe_F trained with indoor dataset.                      
                model_path = "logs/main.py---gcn_opt=reweight_vanilla_sigmoid_softmax---bn_opt=gn---weight_opt=sigmoid_softmax---loss_multi_logit=1---use_fundamental=2---data_name=oan_indoor/models-best"
            self.model_path = os.path.join('../weights/acne', model_path)

        if (acne_module.current_obj_id != self.acne_id) or force_reload:
            if not (acne_module.current_obj_id is None):
                acne_module.current_net.sess.close()
                tf.reset_default_graph()

            config, unparsed = acne_get_config()
        
            paras = {
                "CNe_E":{
                    "bn_opt":"bn"},
                "ACNe_E":{
                    "gcn_opt":"reweight_vanilla_sigmoid_softmax",  "bn_opt":"gn",
                    "weight_opt":"sigmoid_softmax"},
                "CNe_F":{
                    "bn_opt":"bn", "use_fundamental":2},
                "ACNe_F":{
                    "gcn_opt":"reweight_vanilla_sigmoid_softmax",  "bn_opt":"gn",
                    "weight_opt":"sigmoid_softmax", "use_fundamental":2},
            }
        
            para = paras[self.args['what']]
    
            for ki, vi in para.items():
               setattr(config, ki, vi)
               
            self.use_fundamental = config.use_fundamental # E:0, F:2.
        
            # Instantiate wrapper class
            self.net = acne.NetworkTest(config, self.model_path)
            acne_module.current_net = self.net
            acne_module.current_obj_id = self.acne_id
            
        sz1 = Image.open(args['img'][0]).size
        sz2 = Image.open(args['img'][1]).size

        mi = args['m_idx']
        mm = args['m_mask']

        m12 = mi[mm]

        k1 = args['kp'][0]
        k2 = args['kp'][1]
        
        k1 = k1[m12[:, 0]]
        k2 = k2[m12[:, 1]]
        
        pt1 = np.ascontiguousarray(k1.detach().cpu())
        pt2 = np.ascontiguousarray(k2.detach().cpu())
                
        l = pt1.shape[0]
        
        if l > 0:    
            corrs = np.hstack((pt1, pt2)).astype(np.float32)
        
            K1 = np.array(
                [[1, 0, sz1[0] / 2.0],
                 [0, 1, sz1[1] / 2.0],
                 [0, 0 ,1]])
        
            K2 = np.array(
                [[1, 0, sz2[0] / 2.0],
                 [0, 1, sz2[1] / 2.0],
                 [0, 0, 1]])
        
            # Prepare input. 
            xs, T1, T2 = acne.prepare_xs(corrs, K1, K2, self.use_fundamental)
            xs = np.array(xs).reshape(1, 1, -1, 4) # reconstruct a batch. Bx1xNx4
        
            # Compute Essential/Fundamental matrix
            E, w_com, score_local = self.net.compute_E(xs)
            E = E[0]
            score_local = score_local[0]
            w_com = w_com[0]
        
            mask = w_com > 1e-5
            mask_aux = torch.tensor(mask, device=device)         
            aux = mm.clone()
            mm[aux] = mask_aux

            return {'m_mask': mm}
        else:
            return {'m_mask': args['m_mask']}


def merge_colmap_db(db_names, db_merged_name, img_folder=None, to_filter=None, how_filter=None,
    only_keypoints=False, unique=True, only_matched=False, no_unmatched=True,
    include_two_view_geometry=True, sampling_mode='raw', overlapping_cells=False,
    sampling_scale=1, sampling_offset=0, focal_cf=1.2):                    

    aux_hdf5 = None
    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):         
        aux_hdf5 = pickled_hdf5.pickled_hdf5('tmp.hdf5', mode='a')
                
    db_merged = coldb_ext(db_merged_name)
    db_merged.create_tables()
    
    for i, db_name in enumerate(go_iter(db_names, msg='Merging progress')):
        db = coldb_ext(db_name)
        imgs = db.get_images()
    
        if (to_filter is None) or (how_filter is None):
            current_how = None
            current_filter = None
        else:
            current_how = how_filter[i]
            current_filter = to_filter[i]
            
        if not(current_filter is None):
            if len(current_filter) == 0:
                current_how = None
                current_filter = None

        img_dict = {}
        pair_dict = {}
        if not (current_filter is None):
            for v in current_filter:
                if not(isinstance(v, list) or isinstance(v, tuple)):
                    img_dict[v] = 1
                else:                    
                    if not (v[0] in pair_dict): pair_dict[v[0]] = {}                    
                    pair_dict[v[0]][v[1]] = 1
            
        for in0a, in0b  in enumerate(imgs):
            for in1a, in1b in enumerate(imgs):
                im0_id, im0_ = in0b
                im1_id, im1_ = in1b
                                
                if im0_id == im1_id: continue
                if in1a <= in0a: continue

                im0 = os.path.split(im0_)[-1]
                im1 = os.path.split(im1_)[-1]
                
                if current_how == 'exclude':
                    cond0 = (im0 in img_dict) or (im1 in img_dict) 
                    cond1 = ((im0 in pair_dict) and (im1 in pair_dict[im0])) or ((im1 in pair_dict) and (im0 in pair_dict[im1]))                    
                    if cond0 or cond1: continue                
 
                if current_how == 'include':
                    cond0 = (im0 in img_dict) or (im1 in img_dict) 
                    cond1 = ((im0 in pair_dict) and (im1 in pair_dict[im0])) or ((im1 in pair_dict) and (im0 in pair_dict[im1]))                    
                    if (not cond0) and (not cond1): continue                

                # print((im0, im1))

                im0_id_prev = db_merged.get_image_id(im0)
                if  im0_id_prev is None:
                    im0_name, cam0_id = db.get_image(im0_id)
                    
                    if img_folder is None:
                        cam0 = db.get_camera(cam0_id)                                       
                        cam0_id_prev = db_merged.add_camera(cam0[0], cam0[1], cam0[2], cam0[3], cam0[4])
                    else:
                        w, h = Image.open(os.path.join(img_folder, im0)).size
                        cam0_id_prev = db_merged.add_camera(SIMPLE_RADIAL, w, h, np.array([focal_cf * max(w, h), w / 2, h / 2, 0]))
                       
                    im0_id_prev = db_merged.add_image(im0_name, cam0_id_prev)
                    db_merged.commit()

                im1_id_prev = db_merged.get_image_id(im1)
                if  im1_id_prev is None:
                    im1_name, cam1_id = db.get_image(im1_id)
                    
                    if img_folder is None:
                        cam1 = db.get_camera(cam1_id)
                        cam1_id_prev = db_merged.add_camera(cam1[0], cam1[1], cam1[2], cam1[3], cam1[4])
                    else:
                        w, h = Image.open(os.path.join(img_folder, im1)).size
                        cam1_id_prev = db_merged.add_camera(SIMPLE_RADIAL, w, h, np.array([focal_cf * max(w, h), w / 2, h / 2, 0]))
                                        
                    im1_id_prev = db_merged.add_image(im1_name, cam1_id_prev)
                    db_merged.commit()
         
                kp0 = db.get_keypoints(im0_id)
                kp1 = db.get_keypoints(im1_id)

                if kp0 is None:
                    w0 = torch.zeros((0, 6), device=device)
                    kp0 = torch.zeros((0, 2), device=device)
                    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):            
                        k0_count = torch.zeros(0, device=device)
                else:
                    w0 = torch.tensor(kp0, device=device)
                    kp0 = torch.tensor(kp0[:, :2], device=device)
                    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):            
                        k0_count, _ = aux_hdf5.get(im0)
                    
                kH0 = torch.zeros((kp0.shape[0], 3, 3), device=device)
                kr0 = torch.full((kp0.shape[0], ), torch.inf, device=device)
        
                if kp1 is None:
                    w1 = torch.zeros((0, 6), device=device)
                    kp1 = torch.zeros((0, 2), device=device)
                    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):            
                        k1_count = torch.zeros(0, device=device)
                else:
                    w1 = torch.tensor(kp1, device=device)
                    kp1 = torch.tensor(kp1[:, :2], device=device)
                    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):            
                        k1_count, _ = aux_hdf5.get(im1)
                    
                kH1 = torch.zeros((kp1.shape[0], 3, 3), device=device)
                kr1 = torch.full((kp1.shape[0], ), torch.inf, device=device)

                pipe = {}
                pipe['kp'] = [kp0, kp1]
                pipe['kH'] = [kH0, kH1]
                pipe['kr'] = [kr0, kr1]
                pipe['w'] = [w0, w1]

                kp0_prev = db_merged.get_keypoints(im0_id_prev)
                kp1_prev = db_merged.get_keypoints(im1_id_prev)
                
                if kp0_prev is None:
                    w0_prev = torch.zeros((0, 6), device=device)
                    kp0_prev = torch.zeros((0, 2), device=device)
                    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):            
                        k0_count_prev = torch.zeros(0, device=device)
                else:
                    w0_prev = torch.tensor(kp0_prev, device=device)
                    kp0_prev = torch.tensor(kp0_prev[:, :2], device=device)
                    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):            
                        k0_count_prev, _ = aux_hdf5.get(im0)
                    
                kH0_prev = torch.zeros((kp0_prev.shape[0], 3, 3), device=device)
                kr0_prev = torch.full((kp0_prev.shape[0], ), torch.inf, device=device)
        
                if kp1_prev is None:
                    w1_prev = torch.zeros((0, 6), device=device)
                    kp1_prev = torch.zeros((0, 2), device=device)
                    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):            
                        k1_count_prev = torch.zeros(0, device=device)
                else:
                    w1_prev = torch.tensor(kp1_prev, device=device)
                    kp1_prev = torch.tensor(kp1_prev[:, :2], device=device)
                    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):            
                        k1_count_prev, _ = aux_hdf5.get(im1)
                    
                kH1_prev = torch.zeros((kp1_prev.shape[0], 3, 3), device=device)
                kr1_prev = torch.full((kp1_prev.shape[0], ), torch.inf, device=device)

                pipe_prev = {}
                pipe_prev['kp'] = [kp0_prev, kp1_prev]
                pipe_prev['kH'] = [kH0_prev, kH1_prev]
                pipe_prev['kr'] = [kr0_prev, kr1_prev]
                pipe_prev['w'] = [w0_prev, w1_prev]

                no_matches = False
                if only_keypoints: no_matches = True
 
                matches = None
                two_view_matches = None
                if no_matches == False:
                    matches = db.get_matches(im0_id, im1_id)
                    if not (matches is None) and include_two_view_geometry:
                        two_view_matches, models = db.get_two_view_geometry(im0_id, im1_id)

                if matches is None:
                    m_idx = torch.zeros((0, 2), device=device, dtype=torch.int)        
                    m_val = torch.full((m_idx.shape[0], ), torch.inf, device=device)
                    m_mask = torch.full((m_idx.shape[0], ), 1, device=device, dtype=torch.bool)
                else:                    
                    m_idx = torch.tensor(np.copy(matches), device=device, dtype=torch.int)
                    if two_view_matches is None:
                        m_mask = torch.full((m_idx.shape[0],), 1, device=device, dtype=torch.bool)
                        m_val = torch.full((m_idx.shape[0],), np.inf, device=device)
                    else:                       
                        s_idx = torch.tensor(np.copy(two_view_matches), device=device, dtype=torch.int)
                            
                        if len(models.keys()) == 1:
                            for model in ['H', 'F', 'E']:
                                if model in models: pipe[model] = torch.tensor(models[model], device=device)
                                
                        m_mask = torch.zeros(m_idx.shape[0], device=device, dtype=torch.bool)
                        
                        idx = torch.argsort(m_idx[:, 1].type(torch.int), stable=True)
                        m_idx = m_idx[idx]
                        idx = torch.argsort(m_idx[:, 0].type(torch.int), stable=True)
                        m_idx = m_idx[idx]

                        idx = torch.argsort(s_idx[:, 1].type(torch.int), stable=True)
                        s_idx = s_idx[idx]
                        idx = torch.argsort(s_idx[:, 0].type(torch.int), stable=True)
                        s_idx = s_idx[idx]

                        q0 = 0
                        q1 = 0
                        while (q0 < s_idx.shape[0]) and (q1 < m_idx.shape[0]):                       
                            if (s_idx[q0, 0] < m_idx[q1, 0]) or ((s_idx[q0, 0] == m_idx[q1, 0]) and (s_idx[q0, 1] < m_idx[q1, 1])):
                                q0 = q0 + 1
                            elif (s_idx[q0, 0] == m_idx[q1, 0]) and (s_idx[q0, 1] == m_idx[q1, 1]):
                                m_mask[q1] = 1
                                q0 = q0 + 1
                                q1 = q1 + 1
                            else:
                                q1 = q1 + 1

                        m_val = torch.full((m_idx.shape[0],), np.inf, device=device)

                pipe['m_idx'] = m_idx
                pipe['m_val'] = m_val
                pipe['m_mask'] = m_mask
                
                if ('sampling_mode' == 'avg_all_matches') or ('sampling_mode' == 'avg_inlier_matches'):        
                    pipe['k_counter'] = [k0_count, k1_count]
        
                matches_prev = None
                two_view_matches_prev = None
                if no_matches == False:
                    matches_prev = db_merged.get_matches(im0_id_prev, im1_id_prev)
                    if not (matches_prev is None) and include_two_view_geometry:
                        two_view_matches_prev, models_prev = db.get_two_view_geometry(im0_id_prev, im1_id_prev)

                if matches_prev is None:
                    m_idx = torch.zeros((0, 2), device=device, dtype=torch.int)        
                    m_val = torch.full((m_idx.shape[0], ), torch.inf, device=device)
                    m_mask = torch.full((m_idx.shape[0], ), 1, device=device, dtype=torch.bool)
                else:                    
                    m_idx = torch.tensor(np.copy(matches_prev), device=device, dtype=torch.int)
                    if two_view_matches_prev is None:
                        m_mask = torch.full((m_idx.shape[0],), 1, device=device, dtype=torch.bool)
                        m_val = torch.full((m_idx.shape[0],), np.inf, device=device)
                    else:                       
                        s_idx = torch.tensor(np.copy(two_view_matches_prev), device=device, dtype=torch.int)
                            
                        if len(models_prev.keys()) == 1:
                            for model in ['H', 'F', 'E']:
                                if model in models_prev: pipe_prev[model] = torch.tensor(models_prev[model], device=device)
                                
                        m_mask = torch.zeros(m_idx.shape[0], device=device, dtype=torch.bool)
                        
                        idx = torch.argsort(m_idx[:, 1].type(torch.int), stable=True)
                        m_idx = m_idx[idx]
                        idx = torch.argsort(m_idx[:, 0].type(torch.int), stable=True)
                        m_idx = m_idx[idx]

                        idx = torch.argsort(s_idx[:, 1].type(torch.int), stable=True)
                        s_idx = s_idx[idx]
                        idx = torch.argsort(s_idx[:, 0].type(torch.int), stable=True)
                        s_idx = s_idx[idx]

                        q0 = 0
                        q1 = 0
                        while (q0 < s_idx.shape[0]) and (q1 < m_idx.shape[0]):                       
                            if (s_idx[q0, 0] < m_idx[q1, 0]) or ((s_idx[q0, 0] == m_idx[q1, 0]) and (s_idx[q0, 1] < m_idx[q1, 1])):
                                q0 = q0 + 1
                            elif (s_idx[q0, 0] == m_idx[q1, 0]) and (s_idx[q0, 1] == m_idx[q1, 1]):
                                m_mask[q1] = 1
                                q0 = q0 + 1
                                q1 = q1 + 1
                            else:
                                q1 = q1 + 1

                        m_val = torch.full((m_idx.shape[0],), np.inf, device=device)

                pipe_prev['m_idx'] = m_idx
                pipe_prev['m_val'] = m_val
                pipe_prev['m_mask'] = m_mask
                
                if ('sampling_mode' == 'avg_all_matches') or ('sampling_mode' == 'avg_inlier_matches'):        
                    pipe_prev['k_counter'] = [k0_count_prev, k1_count_prev]
        
                counter = (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches')
                pipe_out = pipe_union([pipe_prev, pipe], unique=unique, no_unmatched=no_unmatched, only_matched=only_matched, sampling_mode=sampling_mode, sampling_scale=sampling_scale, sampling_offset=sampling_offset, overlapping_cells=overlapping_cells, preserve_order=True, counter=counter)

                pts0 = pipe_out['w'][0].to('cpu').numpy()
                pts1 = pipe_out['w'][1].to('cpu').numpy()
                
                if counter:
                    aux_hdf5.add(im0, pipe_out['k_counter'][0])
                    aux_hdf5.add(im1, pipe_out['k_counter'][1])
                
                db_merged.update_keypoints(im0_id_prev, pts0)
                db_merged.update_keypoints(im1_id_prev, pts1)

                if not only_keypoints:
                    m_idx = pipe_out['m_idx'].to('cpu').numpy()
                    db_merged.update_matches(im0_id_prev, im1_id_prev, m_idx)
        
                    if include_two_view_geometry:        
                        m_idx = pipe_out['m_idx'][pipe_out['m_mask']].to('cpu').numpy()
                        models = {}                                        
                        db_merged.update_two_view_geometry(im0_id_prev, im1_id_prev, m_idx, model=models)

                db_merged.commit()
        db.close()

    db_merged.close()
    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):
        aux_hdf5.close()
        if os.path.isfile('tmp.hdf5'): os.remove('tmp.hdf5')


def filter_colmap_reconstruction(input_model_path='../aux/colmap/model', img_path=None, db_path=None, output_model_path='../aux/colmap/output_model', to_filter=None, how_filter='exclude', only_cameras=True, add_3D_points=False, add_as_possible=True):
    model = pycolmap.Reconstruction(input_model_path)

    os.makedirs(output_model_path, exist_ok=True)
    if to_filter is None: to_filter=[]
    
    to_filter_dict = {}
    for image in to_filter: to_filter_dict[image] = 1
    
    model_imgs = [(image_id, model.image(image_id).name) for image_id in model.images]

    for image_id, image in model_imgs:
        if ((image in to_filter_dict) and (how_filter == 'exclude')) or ((not (image in to_filter_dict)) and (how_filter == 'include')):
            model.deregister_image(image_id)
        
    if only_cameras:
        for pts3D in model.point3D_ids(): model.delete_point3D(pts3D)

    if (not only_cameras) and add_3D_points and (not (img_path is None)) and (not (db_path is None)):
        incr_map_opt = pycolmap.IncrementalPipelineOptions()
        if add_as_possible:
            tri_opt = incr_map_opt.triangulation
            tri_opt.ignore_two_view_tracks = False    
        model = pycolmap.triangulate_points(model, db_path, img_path, output_model_path, True, incr_map_opt, False)
        return

    if not (img_path is None):
        model.extract_colors_for_all_images(img_path)

    model.write_binary(output_model_path)


_EPS = np.finfo(float).eps * 4.0

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis."""
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    data *= data
    np.sum(data, axis=axis, out=out)
    np.sqrt(out, out)
    return None


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        # print("special case")
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# based on the 3D registration from https://github.com/cgohlke/transformations
def affine_matrix_from_points(v0, v1, shear=False, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.
    v0 and v1 are shape (ndims, -1) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.
    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean traffansformation matrix
    is returned.
    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.
    The returned matrix performs rotation, translation and uniform scaling
    (if specified)."""
    
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims: 2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        # print (vector_norm(q), np.linalg.norm(q))
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]

    # print("transformation matrix Python Script: ", M)

    return M


# This is the IMC 3D error metric code
def register_by_Horn(ev_coord, gt_coord, ransac_threshold, inl_cf, strict_cf):
    '''Return the best similarity transforms T that registers 3D points pt_ev in <ev_coord> to
    the corresponding ones pt_gt in <gt_coord> according to a RANSAC-like approach for each
    threshold value th in <ransac_threshold>.
    
    Given th, each triplet of 3D correspondences is examined if not already present as strict inlier,
    a correspondence is a strict inlier if <strict_cf> * err_best < th, where err_best is the registration
    error for the best model so far.
    The minimal model given by the triplet is then refined using also its inliers if their total is greater
    than <inl_cf> * ninl_best, where ninl_best is th number of inliers for the best model so far. Inliers
    are 3D correspondences (pt_ev, pt_gt) for which the Euclidean distance |pt_gt-T*pt_ev| is less than th.'''
    
    # remove invalid cameras, the index is returned
    idx_cams = np.all(np.isfinite(ev_coord), axis=0)
    ev_coord = ev_coord[:, idx_cams]
    gt_coord = gt_coord[:, idx_cams]

    # initialization
    n = ev_coord.shape[1]
    r = ransac_threshold.shape[0]
    ransac_threshold = np.expand_dims(ransac_threshold, axis=0)
    ransac_threshold2 = ransac_threshold**2
    ev_coord_1 = np.vstack((ev_coord, np.ones(n)))

    max_no_inl = np.zeros((1, r))
    best_inl_err = np.full(r, np.Inf)
    best_transf_matrix = np.zeros((r, 4, 4))
    best_err = np.full((n, r), np.Inf)
    strict_inl = np.full((n, r), False)
    triplets_used = np.zeros((3, r))

    # run on camera triplets
    for ii in range(n-2):
        for jj in range(ii+1, n-1):
            for kk in range(jj+1, n):
                i = [ii, jj, kk]
                triplets_used_now = np.full((n), False)
                triplets_used_now[i] = True
                # if both ii, jj, kk are strict inliers for the best current model just skip
                if np.all(strict_inl[i]):
                    continue
                # get transformation T by Horn on the triplet camera center correspondences
                transf_matrix = affine_matrix_from_points(ev_coord[:, i], gt_coord[:, i], usesvd=False)
                # apply transformation T to test camera centres
                rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                # compute error and inliers
                err = np.sum((rotranslated - gt_coord)**2, axis=0)
                inl = np.expand_dims(err, axis=1) < ransac_threshold2
                no_inl = np.sum(inl, axis=0)
                # if the number of inliers is close to that of the best model so far, go for refinement
                to_ref = np.squeeze(((no_inl > 2) & (no_inl > max_no_inl * inl_cf)), axis=0)
                for q in np.argwhere(to_ref):                        
                    qq = q[0]
                    if np.any(np.all((np.expand_dims(inl[:, qq], axis=1) == inl[:, :qq]), axis=0)):
                        # already done for this set of inliers
                        continue
                    # get transformation T by Horn on the inlier camera center correspondences
                    transf_matrix = affine_matrix_from_points(ev_coord[:, inl[:, qq]], gt_coord[:, inl[:, qq]])
                    # apply transformation T to test camera centres
                    rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                    # compute error and inliers
                    err_ref = np.sum((rotranslated - gt_coord)**2, axis=0)
                    err_ref_sum = np.sum(err_ref, axis=0)
                    err_ref = np.expand_dims(err_ref, axis=1)
                    inl_ref = err_ref < ransac_threshold2
                    no_inl_ref = np.sum(inl_ref, axis=0)
                    # update the model if better for each threshold
                    to_update = np.squeeze((no_inl_ref > max_no_inl) | ((no_inl_ref == max_no_inl) & (err_ref_sum < best_inl_err)), axis=0)
                    if np.any(to_update):
                        triplets_used[0, to_update] = ii
                        triplets_used[1, to_update] = jj
                        triplets_used[2, to_update] = kk
                        max_no_inl[:, to_update] = no_inl_ref[to_update]
                        best_err[:, to_update] = np.sqrt(err_ref)
                        best_inl_err[to_update] = err_ref_sum
                        strict_inl[:, to_update] = (best_err[:, to_update] < strict_cf * ransac_threshold[:, to_update])
                        best_transf_matrix[to_update] = transf_matrix

    # print("\n")
    # for i in range(r):
    #    print(f'Registered cameras {max_no_inl[0, i]} of {n} for threshold {ransac_threshold[0, i]}')

    best_model = {
        "valid_cams": idx_cams,        
        "no_inl": max_no_inl,
        "err": best_err,
        "triplets_used": triplets_used,
        "transf_matrix": best_transf_matrix}
    return best_model


def evaluate_rec(gt_df, user_df, inl_cf = 0.8, strict_cf=0.5, thresholds=[0.05]):
    ''' Register the <user_df> camera centers to the ground-truth <gt_df> camera centers and
    return the corresponding mAA as the average percentage of registered camera threshold.
    
    For each threshold value in <thresholds>, the best similarity transformation found which
    maximizes the number of registered cameras is employed. A camera is marked as registered
    if after the transformation its Euclidean distance to the corresponding ground-truth camera
    center is less than the mentioned threshold. Current measurements are in meter.
    
    Registration parameters:
    <inl_cf> coefficient to activate registration refinement, set to 1 to refine a new model
    only when it gives more inliers, to 0 to refine a new model always; high values increase
    speed but decrease precision.
    <strict_cf> threshold coefficient to define strict inliers for the best registration so far,
    new minimal models made up of strict inliers are skipped. It can vary from 0 (slower) to
    1 (faster); set to -1 to check exhaustively all the minimal model triplets.'''
    
    # get camera centers
    ucameras = user_df
    gcameras = gt_df    
        
    # get the image list to use
    good_cams = []
    for image_path in gcameras.keys():
        if image_path in ucameras.keys():
            good_cams.append(image_path)
        
    # put corresponding camera centers into matrices
    n = len(good_cams)
    u_cameras = np.zeros((3, n))
    g_cameras = np.zeros((3, n))
    
    ii = 0
    for i in good_cams:
        u_cameras[:, ii] = ucameras[i]
        g_cameras[:, ii] = gcameras[i]
        ii += 1
        
    # Horn camera centers registration, a different best model for each camera threshold
    model = register_by_Horn(u_cameras, g_cameras, np.asarray(thresholds), inl_cf, strict_cf)
    
    # transformation matrix
    # print("Transformation matrix for maximum threshold")
    # T = np.squeeze(model['transf_matrix'][-1])
    # print(T)
    
    return model
    
                
def align_colmap_models(model_path1='../aux/colmap/model0', model_path2='../aux/colmap/model1', imgs_path=None, db_path0=None, db_path1=None,
                        output_db='../aux/colmap/merged_database.db', output_model='../aux/colmap/merged_model', th=None,
                        only_cameras=False, add_as_possible=True, no_force_db_fusion=True):

    model1 = pycolmap.Reconstruction(model_path1)
    model2 = pycolmap.Reconstruction(model_path2)

    if (not only_cameras) and (not (db_path0 is None)) and (not (db_path1 is None)) and (not (imgs_path is None)):
        if (not (os.path.isfile(output_db))) or (not no_force_db_fusion):
            os.makedirs(os.path.split(output_db)[0], exist_ok=True)   
            
            f1 = [model1.image(image_id).name for image_id in model1.images]            
            l1 = []
            for i, name1 in enumerate(f1):
                for name2 in f1[i + 1:]:
                    l1.append([name1, name2])

            f2 = [model2.image(image_id).name for image_id in model2.images]            
            l2 = []
            for i, name1 in enumerate(f2):
                for name2 in f2[i + 1:]:
                    l2.append([name1, name2])
            
            to_filter=[l1, l2]
            how_filter=['include', 'include']
            
            merge_colmap_db([db_path0, db_path1], output_db, to_filter=to_filter, how_filter=how_filter)
    else:
        only_cameras = True

    model1_imgs = {model1.image(image_id).name: model1.image(image_id).projection_center() for image_id in model1.images}
    model2_imgs = {model2.image(image_id).name: model2.image(image_id).projection_center() for image_id in model2.images}

    if th is None:
        c = np.vstack([model1_imgs[im] for im in model1_imgs])        
        th = np.mean(scipy.spatial.distance.pdist(c)) / 100
        warnings.warn(f'setting alignement threshold to {th}')

    align = evaluate_rec(model1_imgs, model2_imgs, thresholds=[th])
    model2.transform(pycolmap.Sim3d(align['transf_matrix'][0, :3, :].astype(np.float64)))
        
    fused_model = pycolmap.Reconstruction()
    
    if not only_cameras:
        fused_db = coldb_ext(output_db)
    
    count = 1
    for image_id in model1.images:
        image = model1.image(image_id)
        camera = model1.camera(image.camera_id)
        
        if only_cameras:
            img_id = count
            cam_id = count
        else:
            img_id = fused_db.get_image_id(image.name)
            cam_id = fused_db.get_image(img_id)[1]
        
        new_camera = pycolmap.Camera()
        new_camera.camera_id = cam_id
        new_camera.model = camera.model
        new_camera.width = camera.width
        new_camera.height = camera.height
        new_camera.params = camera.params
        fused_model.add_camera(new_camera)

        new_image = pycolmap.Image()
        new_image.name = image.name        
        new_image.image_id = img_id
        new_image.camera_id = cam_id
        new_image.cam_from_world = image.cam_from_world        
        fused_model.add_image(new_image)

        count = count + 1        

    for image_id in model2.images:
        if not (model1.find_image_with_name(model2.image(image_id).name) is None): continue

        image = model2.image(image_id)
        camera = model2.camera(image.camera_id)

        if only_cameras:
            img_id = count
            cam_id = count
        else:
            img_id = fused_db.get_image_id(image.name)
            cam_id = fused_db.get_image(img_id)[1]

        new_camera = pycolmap.Camera()
        new_camera.camera_id = cam_id
        new_camera.model = camera.model
        new_camera.width = camera.width
        new_camera.height = camera.height
        new_camera.params = camera.params
        fused_model.add_camera(new_camera)

        new_image = pycolmap.Image()
        new_image.name = image.name
        new_image.image_id = img_id
        new_image.camera_id = cam_id
        new_image.cam_from_world = image.cam_from_world        
        fused_model.add_image(new_image)

        count = count + 1 
        
    if not only_cameras:
        fused_db.close()
        
    if (not only_cameras):
        incr_map_opt = pycolmap.IncrementalPipelineOptions()
        if add_as_possible:
            tri_opt = incr_map_opt.triangulation
            tri_opt.ignore_two_view_tracks = False    
        fused_model = pycolmap.triangulate_points(fused_model, output_db, imgs_path, output_model, True, incr_map_opt, False)
        return

    os.makedirs(output_model, exist_ok=True)
    fused_model.write_binary(output_model)

  
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
#           show_matches_module(id_moreFalse='only', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
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
#           sampling_module(sampling_mode='avg_inlier_matches', overlapping_cells=True, sampling_scale=20),
#           show_matches_module(id_more='second', img_prefix='matches_sampled_', mask_idx=[1, 0], prepend_pair=False),
#       ]

#       pipeline = [
#           loftr_module(),
#           magsac_module(),
#           show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           to_colmap_module(),
#       ]     
        
#       pipeline = [
#           deep_joined_module(what='aliked'),
#           lightglue_module(what='aliked'),
#           magsac_module(),
#           show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           to_colmap_module(),
#       ]         
                
#       imgs = '../data/ET_random_rotated'

#       pipeline = [
#           loftr_module(),
#           magsac_module(),
#           show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           to_colmap_module(),
#       ]     

#       pipeline = [
#           roma_module(),
#           magsac_module(),
#           show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           to_colmap_module(),
#       ]     

#       pipeline = [
#           r2d2_module(),
#         # smnn_module(),
#           lightglue_module(what='sift', desc_cf=255),
#           magsac_module(),
#           show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#       ]   

#       pipeline = [
#           dog_module(),
#           patch_module(),
#           deep_descriptor_module(),
#           smnn_module(),
#           show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0]),
#           acne_module(),
#           show_matches_module(id_more='second', img_prefix='matches_after_filter_', mask_idx=[1, 0]),
#           magsac_module(),
#           show_matches_module(id_more='third', img_prefix='matches_final_', mask_idx=[1, 0]),
#       ]

#       pipeline = [
#           aspanformer_module(),
#           magsac_module(),
#           show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           to_colmap_module(),
#       ]  

#       pipeline = [
#           from_colmap_module(),
#           show_kpts_module(img_prefix='sift_', prepend_pair=False),
#           show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#       ]   

#       imgs = '../data/ET'
#       run_pairs(pipeline, imgs)

#       imgs_megadepth, gt_megadepth, to_add_path_megadepth = benchmark_setup(bench_path='../bench_data', dataset='megadepth')
#       imgs_scannet, gt_scannet, to_add_path_scannet = benchmark_setup(bench_path='../bench_data', dataset='scannet')
#       imgs_imc, gt_imc, to_add_path_imc = benchmark_setup(bench_path='../bench_data', dataset='imc')

#       pipeline = [
#           deep_joined_module(what='aliked'),
#           lightglue_module(what='aliked'),
#           magsac_module(),
#           show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           pairwise_benchmark_module(id_more='megadepth_fundamental', gt=gt_megadepth, to_add_path=to_add_path_megadepth, use_fundamental=True),
#           pairwise_benchmark_module(id_more='megadepth_essential', gt=gt_megadepth, to_add_path=to_add_path_megadepth, use_fundamental=False),
#       ]         

#       imgs = [imgs_megadepth[i] for i in range(10)]
#       run_pairs(pipeline, imgs, add_path=to_add_path_megadepth)

#       pipeline = [
#           deep_joined_module(what='aliked'),
#           lightglue_module(what='aliked'),
#           magsac_module(),
#           show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
#           pairwise_benchmark_module(id_more='megadepth_fundamental', gt=gt_imc, to_add_path=to_add_path_imc, use_fundamental=True, use_metric=False),
#           pairwise_benchmark_module(id_more='megadepth_fundamental_metric', gt=gt_imc, to_add_path=to_add_path_imc, use_fundamental=True, use_metric=True),
#           pairwise_benchmark_module(id_more='megadepth_essential', gt=gt_imc, to_add_path=to_add_path_imc, use_fundamental=False, use_metric=False),
#           pairwise_benchmark_module(id_more='megadepth_essential_metric', gt=gt_imc, to_add_path=to_add_path_imc, use_fundamental=False, use_metric=True),
#       ]         

#       imgs = [imgs_imc[i] for i in range(10)]
#       run_pairs(pipeline, imgs, add_path=to_add_path_imc)


        pipeline = [
            deep_joined_module(what='aliked'),
            lightglue_module(what='aliked'),
            magsac_module(),
            show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
            to_colmap_module(db='aliked.db'),            
        ]         

        imgs = '../data/ET'
        run_pairs(pipeline, imgs)

        pipeline = [
            deep_joined_module(what='superpoint'),
            lightglue_module(what='superpoint'),
            magsac_module(),
            show_matches_module(img_prefix='superpoint_matches_', mask_idx=[1, 0], prepend_pair=False),
            to_colmap_module(db='superpoint.db'),            
        ]         

        imgs = '../data/ET'
        run_pairs(pipeline, imgs)

        merge_colmap_db(['aliked.db', 'superpoint.db'], 'aliked_superpoint.db', img_folder='../data/ET')


#       filter_colmap_reconstruction(input_model_path='../aux/cluster0/model/0', db_path='../aux/cluster0/database.db', img_path='../aux/cluster0/images', output_model_path='../aux/new_model', to_filter=['archive_0004.png', 'archive_0005.png', 'IMG_0281.png', 'IMG_0272.png'], how_filter='include', only_cameras=False, add_3D_points=True)

#       filter_colmap_reconstruction(input_model_path='../aux/cluster0/model/0', db_path='../aux/cluster0/database.db', img_path='../aux/cluster0/images', output_model_path='../aux/new_model_a', to_filter=['archive_0025.png', '3DOM_FBK_IMG_1517.png', 'archive_0004.png', 'archive_0005.png', 'IMG_0281.png', 'IMG_0272.png', 'archive_0013.png', 'archive_0055.png'], how_filter='include', only_cameras=False, add_3D_points=False)
#       filter_colmap_reconstruction(input_model_path='../aux/cluster0/model/0', db_path='../aux/cluster0/database.db', img_path='../aux/cluster0/images', output_model_path='../aux/new_model_b', to_filter=['archive_0013.png', 'archive_0055.png', 'archive_0230.png', 'IMG_0075.png', 'IMG_0205.png', 'archive_0362.png', 'archive_0004.png', 'archive_0005.png', 'IMG_0281.png', 'IMG_0272.png'], how_filter='include', only_cameras=False, add_3D_points=False)
#       align_colmap_models(model_path1='../aux/new_model_a', model_path2='../aux/new_model_b', imgs_path='../aux/cluster0/images', db_path0='../aux/cluster0/database.db', db_path1='../aux/cluster0/database.db', output_db='../aux/merged_database.db', output_model='../aux/merged_model', th=None)

        print('doh!')