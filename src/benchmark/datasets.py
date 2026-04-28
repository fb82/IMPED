import csv
import os
import shutil
import tarfile
import zipfile

import cv2
import gdown
import numpy as np
from PIL import Image
from tqdm import tqdm

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from core import decompress_pickle


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


def benchmark_setup(bench_path='bench_data', bench_imgs='imgs', bench_gt='gt_data', dataset='megadepth', debug_pairs=None,
        force=False, sample_size=800, seed=42, covisibility_range=[0.1, 0.7], new_sample=False, scene_list=None, bench_plot='aux_images',        
        upright=False, max_imgs=6, to_exclude =['graf'], img_ext='.png', save_ext='.png', check_data=True):

    if (dataset == 'megadepth') or (dataset == 'scannet'):
        return megadepth_scannet_setup(bench_path=bench_path, bench_imgs=bench_imgs, bench_gt=bench_gt, dataset=dataset, debug_pairs=debug_pairs, force=force)
    
    if dataset == 'imc':
        return imc_phototourism_setup(bench_path=bench_path, bench_imgs=bench_imgs, dataset=dataset, sample_size=sample_size, seed=seed, covisibility_range=covisibility_range, new_sample=new_sample, force=force)
        
    if (dataset == 'planar'):
        return planar_setup(bench_path=bench_path, bench_imgs=bench_imgs, bench_plot=bench_plot, dataset=dataset, debug_pairs=debug_pairs, force=force,
                            upright=upright, max_imgs=max_imgs, to_exclude=to_exclude, img_ext=img_ext, save_ext=save_ext, check_data=check_data)


def megadepth_scannet_setup(bench_path='bench_data', bench_imgs='imgs', bench_gt='gt_data', dataset='megadepth', debug_pairs=None, force=False, max_sz=1200):  
    """
    Downloads and organizes the MegaDepth or ScanNet datasets for evaluation.

    This function acts as a wrapper that swaps between outdoor (MegaDepth) 
    and indoor (ScanNet) logic. It handles image resizing, camera pose 
    extraction (R, T matrices), and coordinate scaling.

    Args:
        dataset (str): Either 'megadepth' or 'scannet'.
        debug_pairs (int): If set, only loads a small subset of the data 
            to speed up testing and debugging.
        max_sz (int): The maximum dimension (width or height) for images; 
            larger images are automatically downscaled.

    Returns:
        tuple: (image_pairs, ground_truth_dict, base_image_path)
    """      
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
        if debug_pairs is not None:
            for what in data.keys():
                data[what] = [data[what][i] for i in range(debug_pairs)]
    
        data = setup_images(data, bench_path=bench_path, bench_imgs=bench_imgs, max_sz=max_sz)    
        
        pairs = [(im1, im2) for im1, im2 in zip(data['im1'], data['im2'])]
        gt = {}
        
        gt['use_scale'] = True if (dataset == 'megadepth') else False
        
        for i in range(len(data['im1'])):
            if data['im1'][i] not in gt:
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



def imc_phototourism_setup(bench_path='bench_data', bench_imgs='imgs', dataset='imc', sample_size=800, seed=42, covisibility_range=[0.1, 0.7], new_sample=False, force=False):
    """
    Downloads and prepares the IMC Phototourism dataset for benchmarking.

    This function automates the retrieval of landmark images and their 
    corresponding 'Ground Truth' (GT) calibration data. It filters image 
    pairs based on 'covisibility'—ensuring that the pairs selected actually 
    overlap enough to be matched.

    The final data is stored in a pickled HDF5 database for rapid loading 
    in future sessions.

    Returns:
        tuple: (image_pairs, ground_truth_dict, base_image_path)
    """
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
    gt['use_scale'] = False    
    
    for i in range(len(imc_data['im1'])):
        if imc_data['im1'][i] not in gt:
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
        if debug_pairs is not None:
            for what in megadepth_data.keys():
                megadepth_data[what] = [megadepth_data[what][i] for i in range(debug_pairs)]
    
        megadepth_data = setup_images_megadepth(megadepth_data, bench_path=bench_path, bench_imgs=bench_imgs)    
        
        db.add(data_key, megadepth_data)
        db.close()
        
    return megadepth_data


def planar_setup(bench_path='bench_data', bench_imgs='imgs', bench_plot='aux_images', dataset='planar', upright=False, max_imgs=6, to_exclude=['graf'], debug_pairs=None, force=False, img_ext='.png', save_ext='.png', check_data=True):        
    """
    Sets up a planar homography dataset for benchmarking.

    This function prepares a pair-wise evaluation environment by:
    1. Downloading the raw dataset if not present.
    2. Parsing scene directories to find image pairs and their H matrices.
    3. Handling 'upright' vs 'rotated' versions of the images.
    4. Generating 'Full Masks': Identifying the intersection of valid pixels 
       between two images after warping them via the homography.
    5. Saving debugging visualizations (warped overlays) to verify alignment.

    Args:
        bench_path (str): Root directory for all benchmark data.
        dataset (str): Name of the dataset subfolder.
        upright (bool): If True, ignores rotated versions of images.
        max_imgs (int): Maximum number of secondary images to pair with Image 1.
        check_data (bool): If True, saves warped 'check' images to disk.

    Returns:
        tuple: (image_pairs_list, ground_truth_dict, final_image_path)
    """
    os.makedirs(bench_path, exist_ok=True)    
    db_file = os.path.join(bench_path, dataset + '.hdf5')    
    db = pickled_hdf5.pickled_hdf5(db_file, mode='a')    

    data_key = '/' + dataset

    data, is_found = db.get(data_key)                    
    if (not is_found) or force:
        download_planar(bench_path)        
        out_dir = os.path.join(bench_path, dataset)

        in_path = out_dir
        out_path = os.path.join(bench_path, bench_imgs, dataset)
        os.makedirs(out_path, exist_ok=True)

        if check_data:
            check_path = os.path.join(bench_path, bench_plot, dataset)
            os.makedirs(check_path, exist_ok=True)

        imgs = []

        planar_scenes = sorted([scene[:-5] for scene in os.listdir(out_dir) if (scene[-5:]=='1' + img_ext) and (scene[:5] != 'mask_')])        
        for i in to_exclude: planar_scenes.remove(i)

        for scene in planar_scenes:     
            
            if scene[-3:] == 'rot': continue
            
            img1 = scene + '1' + img_ext
            im1s = os.path.join(in_path, img1)

            img1r = scene + 'rot1' + img_ext
            im1sr = os.path.join(in_path, img1r)
    
            for i in range(2, max_imgs+1):
                img2 = scene + str(i) + img_ext
                im2s = os.path.join(in_path, img2)
    
                H12 = scene + '_H1' + str(i) + '.txt'                            
                H12s = os.path.join(in_path, H12)
 
                img2r = scene + 'rot' + str(i) + img_ext
                im2sr = os.path.join(in_path, img2r)
   
                H12r = scene + 'rot_H1' + str(i) + '.txt'                            
                H12sr = os.path.join(in_path, H12r)
                
                if (not os.path.isfile(im1s)) or (not os.path.isfile(im1s)) or (not os.path.isfile(H12s)):
                    continue
                
                if upright:
                    imgs.append((img1, img2, H12))
                else:
                    if (not upright) and (os.path.isfile(im1sr)) and (os.path.isfile(im2sr)) and (os.path.isfile(H12sr)):
                        imgs.append((img1r, img2r, H12r))
                    else:
                        imgs.append((img1, img2, H12))
                            
        # for debugging, use only first debug_pairs image pairs
        if debug_pairs is not None:
            imgs = [imgs[i] for i in range(debug_pairs)]
    
        image_pairs = imgs
        image_path = out_path
        gt = {}
        gt['use_scale'] = False    
                                
        for img1, img2, H in tqdm(image_pairs, desc='planar image setup'):
            im1s = os.path.join(in_path, img1)
            im2s = os.path.join(in_path, img2)

            im1d = os.path.join(out_path, img1)
            im2d = os.path.join(out_path, img2)
 
            shutil.copyfile(im1s, im1d)
            shutil.copyfile(im2s, im2d)

            H_ = np.loadtxt(os.path.join(in_path, H))
            H_inv_ = np.linalg.inv(H_)
            
            im1 = cv2.imread(im1s)
            sz1 = (im1.shape[0], im1.shape[1])
            mask1 = np.full(sz1, 1, dtype=bool)

            mask1s = os.path.join(in_path, 'mask_' + img1)
            if os.path.isfile(mask1s):
                aux = cv2.imread(mask1s, cv2.IMREAD_GRAYSCALE).astype(bool)
                mask1 = mask1 & ~aux

            mask1s = os.path.join(in_path, 'mask_bad_' + img1)
            if os.path.isfile(mask1s):
                aux = cv2.imread(mask1s, cv2.IMREAD_GRAYSCALE).astype(bool)
                mask1 = mask1 & ~aux

            im2 = cv2.imread(im2s)
            sz2 = (im2.shape[0], im2.shape[1])
            mask2 = np.full(sz2, 1, dtype=bool)

            mask2s = os.path.join(in_path, 'mask_' + img2)
            if os.path.isfile(mask2s):
                aux = cv2.imread(mask2s, cv2.IMREAD_GRAYSCALE).astype(bool)
                mask2 = mask2 & ~aux

            mask2s = os.path.join(in_path, 'mask_bad_' + img2)
            if os.path.isfile(mask2s):
                aux = cv2.imread(mask2s, cv2.IMREAD_GRAYSCALE).astype(bool)
                mask2 = mask2 & ~aux

            mask1_ = cv2.warpPerspective(mask2.astype(np.uint8), H_inv_, (sz1[1], sz1[0]), flags=cv2.INTER_LANCZOS4).astype(bool) 
            mask2_ = cv2.warpPerspective(mask1.astype(np.uint8), H_, (sz2[1], sz2[0]), flags=cv2.INTER_LANCZOS4).astype(bool)
            
            if img1 not in gt:
                gt[img1] = {}
                                
            gt[img1][img2] = {'H': H_, 'mask1': mask1, 'mask2': mask2, 'full_mask1': mask1 & mask1_, 'full_mask2': mask2 & mask2_, 'image_pair_scale': np.full((2, 2), 1)}
                        
            if check_data:                            
                im1_ = cv2.warpPerspective(im2, H_inv_, (sz1[1], sz1[0]), flags=cv2.INTER_LANCZOS4)
                im2_ = cv2.warpPerspective(im1, H_, (sz2[1], sz2[0]), flags=cv2.INTER_LANCZOS4)

                mask1_full = np.expand_dims((mask1 & mask1_).astype(np.uint8), axis=-1)
                mask2_full = np.expand_dims((mask2 & mask2_).astype(np.uint8), axis=-1)

                mask1 = np.expand_dims(mask1.astype(np.uint8), axis=-1)
                mask2 = np.expand_dims(mask2.astype(np.uint8), axis=-1)

                mask1_ = np.expand_dims(mask1_.astype(np.uint8), axis=-1)
                mask2_ = np.expand_dims(mask2_.astype(np.uint8), axis=-1)


                im1 = np.concatenate((im1, mask1 * 196 + mask1_full * 59), axis=-1)
                im2 = np.concatenate((im2, mask2 * 196 + mask2_full * 59), axis=-1)

                im1_ = np.concatenate((im1_, mask1_ * 196 + mask1_full * 59), axis=-1)
                im2_ = np.concatenate((im2_, mask2_ * 196 + mask2_full * 59), axis=-1)
                            
                cv2.imwrite(os.path.join(check_path, img1 + '_' + img2 + '_1a' + save_ext), im1)
                cv2.imwrite(os.path.join(check_path, img1 + '_' + img2 + '_1b' + save_ext), im1_)
                cv2.imwrite(os.path.join(check_path, img1 + '_' + img2 + '_2a' + save_ext), im2)
                cv2.imwrite(os.path.join(check_path, img1 + '_' + img2 + '_2b' + save_ext), im2_)
                
        image_pairs = [(img1, img2) for img1, img2, H in image_pairs]
                
        data = {'image_pairs': image_pairs, 'gt': gt, 'image_path': image_path}
        db.add(data_key, data)
        db.close()
        
    return data['image_pairs'], data['gt'], data['image_path']



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




def download_planar(bench_path ='bench_data'):  
    """
    Downloads and extracts the planar homography benchmark dataset.

    This function automates the data acquisition process by:
    1. Creating a 'downloads' directory to keep the workspace clean.
    2. Downloading a compressed ZIP file from a Google Drive share link 
       using the 'gdown' library.
    3. Extracting the contents into a structured 'planar' directory.
    4. Skipping the download/extraction if the files already exist 
       locally (idempotency).

    Args:
        bench_path (str): The root directory where benchmark data and 
            downloads should be stored.

    Returns:
        None: Operates by side-effects (file system modifications).
    """ 
    os.makedirs(os.path.join(bench_path, 'downloads'), exist_ok=True)   

    file_to_download = os.path.join(bench_path, 'downloads', 'planar_data.zip')    
    if not os.path.isfile(file_to_download):    
        url = "https://drive.google.com/file/d/1XkP4RR9KKbCV94heI5JWlue2l32H0TNs/view?usp=drive_link"
        gdown.download(url, file_to_download, fuzzy=True)

    out_dir = os.path.join(bench_path, 'planar')
    if not os.path.isdir(out_dir):    
        with zipfile.ZipFile(file_to_download, "r") as zip_ref:
            zip_ref.extractall(out_dir)    

    return
