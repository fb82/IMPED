import torch
import pycolmap
import numpy as np
import os
import shutil
import csv
import warnings
import math
import uuid

imped_lib = True
try:
    import imped
except:
    warnings.warn("imped unavailable")
    imped_lib = False

_EPS = np.finfo(float).eps * 4.0

    
def arr_to_str(a):
    return ';'.join([str(x) for x in a.reshape(-1)])


def to_csv(datasets, csv_file='../kaggle_raw_data/gt.csv', recopy_in_original=True, recopy_csv='imc2025_gt.csv'):
    '''take raw data contained in <datasets> and put it into the csv <csv_file>,
    csv format is dataset,scene,image,rotation_matrix,translation_vector
    if there is a file imc2025_gt.csv associated to the scene dataset, R,T are taken from this,
    otherwise if there is a file imc2024_gt.csv (old format) associated to the scene dataset, R,T are taken from this,
    otherwise if there is a colmap folder associated to the scene dataset, R,T are taken from this (scale can be included as csv file too);
    if <recopy_in_original> is True the file <recopy_csv> associated to the scene dataset is generated too
    '''

    os.makedirs(os.path.split(csv_file)[0], exist_ok=True)  
    
    with open(csv_file, 'w') as f:        
        f.write('dataset,scene,image,rotation_matrix,translation_vector\n')

        for dataset in datasets:
            dataset_name = dataset['name']
            for i in range(len(dataset['scenes'])):
                if recopy_in_original:
                    recopy_file = os.path.join(dataset['images'][i], recopy_csv)

                    recopy_tmp_file = recopy_file
                    if os.path.isfile(recopy_file):
                        recopy_tmp_file = os.path.join(dataset['images'][i], 'tmp.csv')
                    else:
                        recopy_tmp_file = recopy_csv
                        
                    ff = open(recopy_tmp_file, 'w')
                    ff.write('dataset,scene,image,rotation_matrix,translation_vector\n')

                scene = dataset['scenes'][i]
                imgs = os.listdir(os.path.join(dataset['images'][i], 'images'))            

                if not(dataset['csv'][i] is None):                
                    _, model = check_gt_imc2025(dataset['csv'][i], tolerance=float('inf'), scene_name=dataset['scenes'][i])
                    if (not dataset_name in model) and (not scene in model[dataset_name]):
                        warnings.warn(f"scene {scene} not found in file {dataset['csv'][i]}")
                    else:
                        for img in imgs:
                            if os.path.isfile(os.path.join(dataset['images'][i], 'images', img)):
                                if img in model[dataset_name][scene]:
                                    R = model[dataset_name][scene][img]['R']
                                    T = model[dataset_name][scene][img]['T']
                                    f.write(f'{dataset_name},{scene},{img},{arr_to_str(R)},{arr_to_str(T)}\n')
                                    if recopy_in_original:
                                        ff.write(f'{dataset_name},{scene},{img},{arr_to_str(R)},{arr_to_str(T)}\n')
                                else:
                                    warnings.warn(f'no GT data for image {img} in scene {scene} and dataset {dataset_name}')
                elif not(dataset['csv_other'][i] is None):                
                        _, model = check_gt_imc2024(dataset['csv_other'][i], tolerance=float('inf'), scene_name=dataset['scenes'][i])
                        if not scene in model:
                            warnings.warn(f"scene {scene} not found in file {dataset['csv_other'][i]}")
                        else:
                            for img in imgs:
                                if os.path.isfile(os.path.join(dataset['images'][i], 'images', img)):
                                    if img in model[scene]:
                                        R = model[scene][img]['R']
                                        T = model[scene][img]['T']
                                        f.write(f'{dataset_name},{scene},{img},{arr_to_str(R)},{arr_to_str(T)}\n')
                                        if recopy_in_original:
                                            ff.write(f'{dataset_name},{scene},{img},{arr_to_str(R)},{arr_to_str(T)}\n') 
                                    else:
                                        warnings.warn(f'no GT data for image {img} in scene {scene} and dataset {dataset_name}')                  
                elif not(dataset['model'][i] is None):                
                    model = pycolmap.Reconstruction(dataset['model'][i])

                    scale = 1.0
                    scale_file = os.path.join(dataset['images'][i], 'colmap', 'scales.csv')
                    if os.path.isfile(scale_file):
                        with open(scale_file, newline='\n') as csvfile:    
                            csv_lines = csv.reader(csvfile, delimiter=',')
                            next(csv_lines)
                            for row in csv_lines:
                                if row[0] == scene: 
                                    scale = float(row[1])
                                    break
                    warnings.warn(f'3D model scale for scene {scene} is {str(scale)}')

                    for img in imgs:
                        if os.path.isfile(os.path.join(dataset['images'][i], 'images', img)):
                            im = model.find_image_with_name(img)
                            if not (im is None):
                                R = im.cam_from_world.rotation.matrix()
                                T = scale * np.array(im.cam_from_world.translation)
                                f.write(f'{dataset_name},{scene},{img},{arr_to_str(R)},{arr_to_str(T)}\n')
                                if recopy_in_original:
                                    ff.write(f'{dataset_name},{scene},{img},{arr_to_str(R)},{arr_to_str(T)}\n')
                            else:
                                warnings.warn(f'no GT data for image {img} in scene {scene} and dataset {dataset_name}')
                else:
                    warnings.warn(f'no GT data for all images in scene {scene} and dataset {dataset_name}')
                                                    
                if recopy_in_original:
                    ff.close()
                    if recopy_tmp_file != recopy_file:
                        shutil.copy(recopy_tmp_file, recopy_file)
                        os.remove(recopy_tmp_file)

            if not (dataset['outliers'] is None):
                if recopy_in_original:
                    ff = open(os.path.join(dataset['outliers'], 'imc2025_gt.csv'), 'w')
                    ff.write('dataset,scene,image,rotation_matrix,translation_vector\n')
                
                imgs = os.listdir(os.path.join(dataset['outliers'], 'images'))
                for img in imgs:
                    R = np.full((3, 3), np.nan)
                    T = np.full((3, ), np.nan)
                    f.write(f'{dataset_name},outliers,{img},{arr_to_str(R)},{arr_to_str(T)}\n')
                    if recopy_in_original:
                        ff.write(f'{dataset_name},outliers,{img},{arr_to_str(R)},{arr_to_str(T)}\n')

                if recopy_in_original: ff.close()


def check_gt_imc2025(csv_gt, tolerance=5, warning=False, scene_name=None):
    '''check the imc2025 <csv_gt> csv file, if the csv entry is missing for more than <tolerance> images the check fails,
    if <warning> is True, missing images are shown'''
    
    if not os.path.isfile(csv_gt): return False, None
    
    aux = os.path.split(csv_gt)[0]
    aux_ = os.path.split(aux)[0]
    dataset_name = os.path.split(aux_)[1]
    
    img_folder = os.path.join(aux, 'images')
    img_list = os.listdir(img_folder)
    img_dict = {}
    for img in img_list: img_dict[img] = 1
    
    data = {}

    with open(csv_gt, newline='\n') as csvfile:    
        csv_lines = csv.reader(csvfile, delimiter=',')
    
        header = True
        for row in csv_lines:
            if header:
                header = False
                continue
            
            dataset = row[0]
            scene = row[1]
            image = row[2]
            R = np.array([float(x) for x in (row[3].split(';'))]).reshape(3,3)
            t = np.array([float(x) for x in (row[4].split(';'))]).reshape(3)

            if dataset != dataset_name:
                continue

            if (not (scene_name is None)) and (scene != scene_name):
                continue
            
            if not os.path.isfile(os.path.join(img_folder, image)):
                continue

            if not (dataset in data):
                data[dataset] = {}
            
            if not (scene in data[dataset]):
                data[dataset][scene] = {}
                
            img_dict.pop(image)            
            data[dataset][scene][image] = {'R': R, 'T': t}

    if scene_name is None:
        valid = True
    else:
        if len(img_dict) > 0: warnings.warn(f'scene {scene_name} - missing images {list(img_dict.keys())}')
        valid = len(img_dict) < tolerance

    return valid, data
    

def check_gt_imc2024(csv_gt, tolerance=5, warning=False, scene_name=None):
    '''check the imc2024 <csv_gt> csv file, if the csv entry is missing for more than <tolerance> images the check fails,
    if <warning> is True, missing images are shown'''
    
    if not os.path.isfile(csv_gt): return False, None

    aux = os.path.split(csv_gt)[0]
    
    img_folder = os.path.join(aux, 'images')
    img_list = os.listdir(img_folder)
    img_dict = {}
    for img in img_list: img_dict[img] = 1
    
    data = {}

    with open(csv_gt, newline='\n') as csvfile:    
        csv_lines = csv.reader(csvfile, delimiter=',')
    
        header = True
        for row in csv_lines:
            if header:
                header = False
                continue
            
            scene = row[0]
            image = row[-1]
            R = np.array([float(x) for x in (row[-3].split(';'))]).reshape(3,3)
            t = np.array([float(x) for x in (row[-2].split(';'))]).reshape(3)

            if (not (scene_name is None)) and (scene != scene_name):
                continue
            
            if not os.path.isfile(os.path.join(img_folder, image)):
                continue
            
            if not (scene in data):
                data[scene] = {}
                
            img_dict.pop(image)            
            data[scene][image] = {'R': R, 'T': t}

    if scene_name is None:
        valid = True
    else:
        if len(img_dict) > 0: warnings.warn(f'scene {scene_name} - missing images {list(img_dict.keys())}')
        valid = len(img_dict) < tolerance

    return valid, data


def get_3d_model(db, abs_scene, abs_3d, models):
    '''compute a 3D model, if the colmap matching database <db> is missing, this is computed by ALIKED using imped if available,
    <abs_scene> is the image folder, <models> is the folder were colmap 3D models are saved'''
    
    if not os.path.isdir(models):
        if not os.path.isfile(db):
            if imped_lib:
                os.makedirs(abs_3d, exist_ok=True)  

                with torch.inference_mode():    
                    pipeline = [
                        imped.deep_joined_module(what='aliked'),
                        imped.lightglue_module(what='aliked'),
                        imped.poselib_module(),
                        imped.to_colmap_module(db=db),
                    ]       
                            
                    imped.run_pairs(pipeline, abs_scene, db_name=os.path.splitext(db)[0] + '.hdf5')

        if os.path.isfile(db):
            os.makedirs(models, exist_ok=True)          
            pycolmap.incremental_mapping(database_path=db, image_path=abs_3d, output_path=models)            


def make_gt(folder='../kaggle_raw_data', csv_name='imc2025_gt.csv', csv_other_name='imc2024_gt.csv', min_model_size=3):
    '''generate raw gt data, the <folder> structure is on the form:
    <folder>/dataset/scene - scene for each dataset
    <folder>/dataset/scene/images - images for the specific dataset, scene
    <folder>/dataset/scene/<csv_name> - gt (rotation and traslation) data in the IMC2025 format for the specific dataset, scene
    <folder>/dataset/scene/<csv_other_name> - gt (rotation and traslation) data in the IMC2024  format for the specific dataset, scene
    <folder>/dataset/scene/colmap - gt data as colmap reconstruction for the specific dataset, scene
    <folder>/dataset/scene/colmap/database.db - colmap database for the specific dataset, scene
    <folder>/dataset/scene/colmap/models - colmap 3D models for the specific dataset, scene
    <folder>/dataset/outliers/images - distractors images - will get nan array for rotation and translation;
    if no gt is available it will try to buld a 3D model in colmap format by imped if available,
    the 3D model with the highest number of 3D scene is select
    if the images for the gt are less than <min_model_size> the scene is excluded from the kaggle data
    '''
    
    data = []

    datasets = os.listdir(folder)
    for dataset in datasets:
        abs_dataset = os.path.join(folder, dataset)
        
        if os.path.isdir(abs_dataset):
            tmp_dataset = {}
            tmp_dataset['name'] = dataset
            tmp_dataset['scenes'] = []
            tmp_dataset['images'] = []
            tmp_dataset['model'] = []
            tmp_dataset['csv'] = []
            tmp_dataset['csv_other'] = []
            tmp_dataset['outliers'] = None
            
            scenes = os.listdir(abs_dataset)

            for scene in scenes:
                abs_scene = os.path.join(folder, dataset, scene)
                
                if os.path.isdir(abs_scene):
                    if scene == 'outliers':
                        tmp_dataset['outliers'] = os.path.join(abs_dataset, 'outliers')
                    else:
                        csv_gt_other = os.path.join(folder, dataset, scene, csv_other_name)                        
                        csv_gt = os.path.join(folder, dataset, scene, csv_name)                        
                        abs_3d = os.path.join(folder, dataset, scene, 'colmap')
                        db = os.path.join(abs_3d, 'database.db')
                        models = os.path.join(abs_3d, 'models')    

                        cur_csv_gt_other = None                        
                        cur_csv_gt = None                        
                        cur_model = None
                        
                        csv_check, _ = check_gt_imc2025(csv_gt, warning=True, scene_name=scene)
                        if csv_check:
                            cur_csv_gt = csv_gt
                        else:
                            csv_check_other, _ = check_gt_imc2024(csv_gt_other, warning=True, scene_name=scene)
 
                        if (not csv_check) and csv_check_other:
                            cur_csv_gt_other = csv_gt_other
                                                        
                        if not csv_check and not csv_check_other:                            
                            get_3d_model(db, os.path.join(abs_scene, 'images'), abs_3d, models)

                            best_n = 0
                            best_model = None
                            for model in os.listdir(models):
                                abs_model = os.path.join(models, model)
                                
                                n = pycolmap.Reconstruction(abs_model).num_images()
                                if n > best_n:
                                    best_n = n
                                    best_model = abs_model
                            
                            if best_n > min_model_size:
                                cur_model = best_model

                                scale_file = os.path.join(abs_3d, 'scales.csv')
                                if not os.path.isfile(scale_file):
                                    with open(scale_file, 'w') as f:        
                                        f.write('scene,scale\n')            
                                        f.write(f'{scene},1.0\n')            
                                                                                                
                        tmp_dataset['scenes'].append(scene)
                        tmp_dataset['images'].append(abs_scene)                                    
                        tmp_dataset['model'].append(cur_model)   
                        tmp_dataset['csv_other'].append(cur_csv_gt_other)   
                        tmp_dataset['csv'].append(cur_csv_gt)
                        
            data.append(tmp_dataset)
            
    return data                
    

def make_todo(img_file='../kaggle_data', rec_file='../kaggle_submission', min_model_size=3):
    '''generate a submission <rec_file> by exahustive brute force, for each dataset in <img_file> colmap is used with imped to generate a 3D reconstruction,
    the best model with highest number of registered images is selected, the process is repeated removing  these images to generate the next cluster,
    the process is repeated untile the number of registered images is greater than <min_model_size>,
    remaining images are put in the outlier cluster'''
    
    data = []

    datasets = os.listdir(img_file)
    for dataset in datasets:
        abs_dataset = os.path.join(img_file, dataset)
        
        if os.path.isdir(abs_dataset):
            cluster = 0
            processed_images = {}
            
            tmp_dataset = {}
            tmp_dataset['name'] = dataset
            tmp_dataset['scenes'] = []
            tmp_dataset['images'] = []
            tmp_dataset['model'] = []
            tmp_dataset['csv_other'] = []
            tmp_dataset['csv'] = []
            tmp_dataset['outliers'] = None            
            
            while True:            
                img_path = os.path.join(rec_file, dataset, 'cluster' + str(cluster), 'images')
                db = os.path.join(rec_file, dataset, 'cluster' + str(cluster), 'database.db')
                abs_3d = os.path.join(rec_file, dataset, 'cluster' + str(cluster))
                models = os.path.join(abs_3d, 'model')
    
                os.makedirs(img_path, exist_ok=True)
                
                for img in os.listdir(abs_dataset):
                    if not img in processed_images:
                        shutil.copy(os.path.join(abs_dataset, img), os.path.join(img_path, img))
                        
                if len(os.listdir(img_path)) < min_model_size:
                    break
                        
                if not imped_lib:                    
                    warnings.warn('imped not found, cannot match images and generate 3d model')
                    break

                get_3d_model(db, img_path, abs_3d, models)

                best_n = 0
                best_model = None
                for model in os.listdir(models):
                    abs_model = os.path.join(models, model)
                    
                    n = pycolmap.Reconstruction(abs_model).num_images()
                    if n > best_n:
                        best_n = n
                        best_model = abs_model
                                        
                if best_model is None:
                    break
    
                current_3d = pycolmap.Reconstruction(best_model)
    
                if current_3d.num_images() < min_model_size:
                    break
                
                tmp_dataset['scenes'].append('cluster' + str(cluster))
                tmp_dataset['images'].append(os.path.join(rec_file, dataset,'cluster' + str(cluster)))
                tmp_dataset['model'].append(best_model)
                tmp_dataset['csv_other'].append(None)
                tmp_dataset['csv'].append(None)
                                
                for img in os.listdir(img_path):
                    im = current_3d.find_image_with_name(img)
                    
                    if im is None:
                        os.remove(os.path.join(img_path, img))
                    else:
                        processed_images[img] = True
                        
                cluster = cluster + 1
                
            outlier_path = os.path.join(rec_file, dataset, 'outliers')
            os.makedirs(os.path.join(outlier_path, 'images'), exist_ok=True)

            for img in os.listdir(abs_dataset):
                if not img in processed_images:
                    shutil.copy(os.path.join(abs_dataset, img), os.path.join(outlier_path, 'images', img))
                    
            if len(os.listdir(outlier_path)) > 0:
                tmp_dataset['outliers'] = outlier_path

            data.append(tmp_dataset)
    
    return data


def make_input_data(csv_gt='../kaggle_raw_data/gt.csv', input_folder='../kaggle_raw_data', input_tth='../kaggle_raw_data/thresholds.csv', out_folder='../kaggle_data', mapping_csv='../kaggle_data/mapping.csv', final_gt_csv='../kaggle_data/gt.csv', final_tth='../kaggle_data/thresholds.csv', obfuscate_data=False):
    '''take the images in <input_folder> for each dataset, scene, the gt in <csv_gt>, the thresholds in <input_tth>,
    and generate the kaggle data as dataset/images in <out_folder>, the corresponding kaggle gt and threshold csv files <final_gt_csv> and <final_tth_csv>,
    together with the data mapping csv file <mapping_csv>, when <obfuscate_data> is true the generated data are obfuscated,
    in any case if two images in the same dataset but different scene have the same name, one is renamed (the mapping is put in <mapping_csv>)'''
    
    data = {}
    with open(csv_gt, newline='\n') as csvfile:    
        csv_lines = csv.reader(csvfile, delimiter=',')
    
        header = True
        for row in csv_lines:
            if header:
                header = False
                continue
            
            dataset, scene, image, R, T = row

            if not dataset in data: data[dataset] = {}
            if not scene in data[dataset]: data[dataset][scene] = {}
            if not image in data[dataset][scene]: data[dataset][scene][image] = {'R': R, 'T': T}

    tth = tth_from_csv(input_tth)

    if isinstance(tth, dict):
        tth_new = {}
        tth_new['default'] = tth['default']
    else:
        tth_new = tth
        
    os.makedirs(os.path.split(final_gt_csv)[0], exist_ok=True)
    f_map = open(mapping_csv, 'w')        
    f_map.write('dataset_raw,scene_raw,image_raw,dataset,scene,image\n')            

    os.makedirs(os.path.split(mapping_csv)[0], exist_ok=True)    
    f_gt = open(final_gt_csv, 'w')        
    f_gt.write('dataset,scene,image,rotation_matrix,translation_vector\n')            

    dataset_dict = {}

    for dataset in data:
        img_dict = {}
                
        if obfuscate_data:
            out_dataset = uuid.uuid4().hex[:16]
            while out_dataset in dataset_dict:
                out_dataset = uuid.uuid4().hex[:16]
            dataset_dict[out_dataset] = 1
        else:
            out_dataset = dataset

        scene_dict = {}
                
        for scene in data[dataset]:

            if obfuscate_data and (scene != 'outliers'):
                out_scene = uuid.uuid4().hex[:16]
                while out_scene in scene_dict:
                    out_scene = uuid.uuid4().hex[:16]
                scene_dict[out_scene] = 1                    
            else:
                out_scene = scene

            if isinstance(tth, dict):
                if (dataset in tth) and (scene in tth[dataset]): 
                    if not (out_dataset in tth_new): tth_new[out_dataset] = {}
                    tth_new[out_dataset][out_scene] = tth[dataset][scene]

            for image_src in data[dataset][scene]:

                if obfuscate_data:
                    image_dst = uuid.uuid4().hex[:16]
                    while image_dst in img_dict:
                        image_dst = uuid.uuid4().hex[:16]
                else: 
                    suffix = ''
                    while (image_src + suffix) in img_dict: 
                        if suffix == '':
                            suffix = '0'
                        else:
                            suffix = str(int(suffix) + 1)
                    image_dst = image_src + suffix
                
                img_dict[image_dst] = 1
                
                os.makedirs(os.path.join(out_folder, out_dataset), exist_ok=True)
                im_src = os.path.join(input_folder, dataset, scene, 'images', image_src)
                im_dst = os.path.join(out_folder, out_dataset, image_dst)
                
                shutil.copy(im_src, im_dst)

                f_map.write(f"{dataset},{scene},{image_src},{out_dataset},{out_scene},{image_dst}\n")                
                f_gt.write(f"{out_dataset},{out_scene},{image_dst},{data[dataset][scene][image_src]['R']},{data[dataset][scene][image_src]['T']}\n")                

    f_map.close()
    f_gt.close()
    
    tth_to_csv(tth_new, csv_file=final_tth)


def vector_norm(data, axis=None, out=None):
    '''Return length, i.e. Euclidean norm, of ndarray along axis.'''

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
    '''Return homogeneous rotation matrix from quaternion.'''
    
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
    '''Return affine transform matrix to register two point sets.
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
    (if specified).'''
    
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
    best_inl_err = np.full(r, np.inf)
    best_transf_matrix = np.zeros((r, 4, 4))
    best_err = np.full((n, r), np.inf)
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

    # for i in range(r):
    #     print(f'Registered cameras {int(max_no_inl[0, i])}/{n} for threshold {ransac_threshold[0, i]}')

    best_model = {
        "valid_cams": idx_cams,        
        "no_inl": max_no_inl,
        "err": best_err,
        "triplets_used": triplets_used,
        "transf_matrix": best_transf_matrix}
    return best_model


def read_csv(filename):
    '''IMC2025 read gt/submission csv file (not the same of the IMC2024)'''

    data = {}

    with open(filename, newline='\n') as csvfile:    
        csv_lines = csv.reader(csvfile, delimiter=',')
    
        header = True
        for row in csv_lines:
            if header:
                header = False
                continue
            
            dataset = row[0]
            scene = row[1]
            image = row[2]
            R = np.array([float(x) for x in (row[3].split(';'))]).reshape(3,3)
            t = np.array([float(x) for x in (row[4].split(';'))]).reshape(3)
            c = -R.T @ t

            if not (dataset in data):
                data[dataset] = {}
            
            if not (scene in data[dataset]):
                data[dataset][scene] = {}
                
            data[dataset][scene][image] = {'R': R, 't': t, 'c': c}

    return data


# mAA computation
def mAA_on_cameras(err, thresholds, n, skip_top_thresholds, to_dec=3):
    '''mAA is the mean of mAA_i, where for each threshold th_i in <thresholds>, excluding the first <skip_top_thresholds values>,
    mAA_i = max(0, sum(err_i < th_i) - <to_dec>) / (n - <to_dec>)
    where <n> is the number of ground-truth cameras and err_i is the camera registration error for the best 
    registration corresponding to threshold th_i'''
    
    aux = err[:, skip_top_thresholds:] < np.expand_dims(np.asarray(thresholds[skip_top_thresholds:]), axis=0)
    return np.sum(np.maximum(np.sum(aux, axis=0) - to_dec, 0)) / (len(thresholds[skip_top_thresholds:]) * (n - to_dec))


def mAA_on_cameras_per_th(err, thresholds, n, to_dec=3):
    '''as mAA_on_cameras, to be used in score_all_ext with per_th=True'''
    aux = err < np.expand_dims(np.asarray(thresholds), axis=0)
    return np.maximum(np.sum(aux, axis=0) - to_dec, 0) / (n - to_dec)


def check_data(gt_data, user_data, print_error=False):    
    '''check if the gt/submission data are correct -
    <gt_data> - images in different scenes in the same dataset cannot have the same name
    <user_data> - there must be exactly an entry for each dataset, scene, image entry in the gt
    <print_error> - print the error *ATTENTION: must be disable when called from score_all_ext to avoid possible data leaks!*
    '''
    
    for dataset in gt_data.keys():
        aux = {}
        for scene in gt_data[dataset].keys():
            for image in gt_data[dataset][scene].keys():
                if image in aux:
                    if print_error: warnings.warn(f'image {image} found duplicated in the GT dataset {dataset}')
                    return False
                else:
                    aux[image] = 1

        if not dataset in user_data.keys():
            if print_error: warnings.warn(f'dataset {dataset} not found in submission')
            return False
        
        for scene in user_data[dataset].keys():
            for image in user_data[dataset][scene].keys():
                if not (image in aux):
                    if print_error: warnings.warn(f'image {image} does not belong to the GT dataset {dataset}')
                    return False
                else:
                    aux.pop(image)
 
        if len(aux) > 0:
            if print_error:  warnings.warn(f'submission dataset {dataset} missing some GT images')            
            return False           

    return True


def score_all_ext(gt_csv, user_csv, combo_mode='harmonic', strict_cluster=False, per_th=False, inl_cf = 0.8, strict_cf=0.5, skip_top_thresholds=2, to_dec=3, thresholds=None):
    '''compute the score: <gt_csv>/<user_csv> - gt/submission csv file;
    <combo_mode> - how to mix mAA_score and clusterness score ["harmonic", "geometric", "arithmetic"];
    <strict_cluster> - if True the image must be correctly registered to be accounted for a cluster beside beloing to the cluster;
    <per_th> - if True the greedy cluster assignment is done for each threshold instead of considering the average among thresholds;
    <inl_cf>, <strict_cf>, <skip_threshold>, <to_dec> - parameters to be passed to mAA computation;
    <thresholds> - the threshold dict tth'''
        
    gt_data = read_csv(gt_csv)    
    user_data = read_csv(user_csv)

    assert check_data(gt_data, user_data, print_error=True)

    if thresholds is None:
        thresholds = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2])
    else:
        thresholds = tth_from_csv(thresholds)
    
    if isinstance(thresholds, list):            
        th_n = len(thresholds)
    else:
        th_n = thresholds['default'].shape[0]

    lt = th_n - skip_top_thresholds
    if per_th:
        ll = lt
    else:
        ll = 1

    stat_score = []
    stat_mAA = []
    stat_clusterness = []
        
    for dataset in gt_data.keys():
        gt_dataset = gt_data[dataset]
        user_dataset = user_data[dataset]

        lg = len(gt_dataset)
        lu = len(user_dataset)
               
        model_table = []
        registered_table = np.zeros((lt, lg, lu))
        mAA_table = np.zeros((lg, lu))
        mAA_th_table = np.zeros((lt, lg, lu))
        cluster_table = np.zeros((lg, lu))
        gt_scene_sum_table = np.zeros((lg, lu))
        user_scene_sum_table = np.zeros((lg, lu))

        best_gt_scene = [[] for k in range(ll)]
        best_user_scene = [[] for k in range(ll)]
        best_model = [[] for k in range(ll)]
        best_registered = [np.zeros((lt, lg)) for k in range(ll)]
        best_mAA = [np.zeros(lg) for k in range(ll)]
        best_mAA_th = [np.zeros((lt, lg)) for k in range(ll)]
        best_cluster = [np.zeros(lg) for k in range(ll)]
        best_gt_scene_sum = [np.zeros(lg) for k in range(ll)]
        best_user_scene_sum = [np.zeros(lg) for k in range(ll)]

        # all possible gt/submission cluster association per dataset
        gt_scene_list = []        
        for i, gt_scene in enumerate(gt_dataset.keys()):
            gt_scene_list.append(gt_scene)

            model_row = []
            user_scene_list = []
            for j, user_scene in enumerate(user_dataset.keys()):                
                user_scene_list.append(user_scene)

                if (gt_scene == 'outliers') or (user_scene == 'outliers'):
                    model_row.append([])
                    continue
                
                if not isinstance(thresholds, dict):
                    ths = thresholds
                else:
                    if (dataset in thresholds) and (gt_scene in thresholds[dataset]):                  
                        ths = thresholds[dataset][gt_scene]
                    else:
                        ths = thresholds['default']
                
                gt_cams = gt_data[dataset][gt_scene]
                user_cams = user_data[dataset][user_scene]
                                
                # the denominator for mAA ratio
                m = len(gt_cams)
                
                # get the image list to use
                good_cams = []
                for image_path in gt_cams.keys():
                    if image_path in user_cams.keys():
                        good_cams.append(image_path)
                    
                # put corresponding camera centers into matrices
                n = len(good_cams)
                u_cameras = np.zeros((3, n))
                g_cameras = np.zeros((3, n))
                
                ii = 0
                for k in good_cams:
                    u_cameras[:, ii] = user_cams[k]['c']
                    g_cameras[:, ii] = gt_cams[k]['c']
                    ii += 1
                    
                # Horn camera centers registration, a different best model for each camera threshold
                model = register_by_Horn(u_cameras, g_cameras, np.asarray(ths), inl_cf, strict_cf)

                # mAA                
                mAA = mAA_on_cameras(model["err"], ths, m, skip_top_thresholds, to_dec)
                mAA_th = mAA_on_cameras_per_th(model["err"], ths, m, to_dec)
                
                registered_table[:, i, j] = model['no_inl'][:, skip_top_thresholds:]
                mAA_table[i, j] = mAA
                mAA_th_table[:, i, j] = mAA_th[skip_top_thresholds:]
                
                if not strict_cluster:                
                    cluster_table[i, j] = n
                else:
                    cluster_table[i, j] = np.mean(model['no_inl'])

                gt_scene_sum_table[i, j] = m
                user_scene_sum_table[i, j] = len(user_data[dataset][user_scene])
                
                model_row.append(model)
    
            model_table.append(model_row)

        # best greedy cluster association per dataset
        for t in range(ll):  
            for i, gt_scene in enumerate(gt_dataset.keys()):                                  
                if per_th:
                    if strict_cluster:
                        aux = cluster_table[i]
                    else:
                        aux = registered_table[t, i]
                    best_ind = np.lexsort((-mAA_th_table[t, i], -aux))[0]
                else:
                    best_ind = np.lexsort((-mAA_table[i], -cluster_table[i]))[0]

                best_gt_scene[t].append(gt_scene)
                best_user_scene[t].append(user_scene_list[best_ind])
                best_model[t].append(model_table[i][best_ind])
                best_registered[t][:, i] = registered_table[:, i, best_ind]
                best_mAA[t][i] = mAA_table[i, best_ind]
                best_mAA_th[t][:, i] = mAA_th_table[:, i, best_ind]
                best_cluster[t][i] = cluster_table[i, best_ind]
                best_gt_scene_sum[t][i] = gt_scene_sum_table[i, best_ind]
                best_user_scene_sum[t][i] = user_scene_sum_table[i, best_ind]

            # exclude outliers cluster            
            outlier_idx = -1
            for i, scene in enumerate(best_gt_scene[t]):
                if scene == 'outliers':
                    outlier_idx = i
                    break            
                
            if outlier_idx > -1:
                best_gt_scene[t].pop(outlier_idx)
                best_user_scene[t].pop(outlier_idx)
                best_model[t].pop(outlier_idx)
                best_registered[t] = np.delete(best_registered[t], outlier_idx, axis=1)            
                best_mAA[t] = np.delete(best_mAA[t], outlier_idx)            
                best_mAA_th[t] = np.delete(best_mAA_th[t], outlier_idx, axis=1)            
                best_cluster[t] = np.delete(best_cluster[t], outlier_idx)            
                best_gt_scene_sum[t] = np.delete(best_gt_scene_sum[t], outlier_idx)            
                best_user_scene_sum[t] = np.delete(best_user_scene_sum[t], outlier_idx)

        # compute the clusterness score
        # basically the precision: images in the both  gt and user cluster / images in the user cluster only
        if per_th:
            n = 0
            m = 0
            for t in range(th_n - skip_top_thresholds):
                n = n + np.sum(best_cluster[t])
                m = m + np.sum(best_user_scene_sum[t])
        else:            
            n = np.sum(best_cluster[0])
            m = np.sum(best_user_scene_sum[0])

        cluster_score = n / m

        # compute the mAA score
        # basically the recall: images in the both gt and user cluster correctly registered / images in the gt cluster only
        if per_th:    
            a = 0
            b = 0
            for t in range(th_n - skip_top_thresholds):                
                n = np.sum(best_gt_scene_sum[t])
    
                for i, scene in enumerate(best_gt_scene[t]):
                    if not isinstance(thresholds, dict):
                        ths = thresholds
                    else:
                        if (dataset in thresholds) and (gt_scene in thresholds[dataset]):                  
                            ths = thresholds[dataset][gt_scene]
                        else:
                            ths = thresholds['default']      
                            
                    tmp = best_model[t][i]['err'][:, skip_top_thresholds + t] < ths[skip_top_thresholds+t]
                    a = a + np.maximum(np.sum(tmp) - to_dec, 0)
    
                b = b + (n - len(best_gt_scene[t]) * to_dec) 
        else:
            n = np.sum(best_gt_scene_sum[0])
            a = 0
            for i, scene in enumerate(best_gt_scene[0]):
                if not isinstance(thresholds, dict):
                    ths = thresholds
                else:
                    if (dataset in thresholds) and (gt_scene in thresholds[dataset]):                  
                        ths = thresholds[dataset][gt_scene]
                    else:
                        ths = thresholds['default']
                        
                tmp = best_model[0][i]['err'][:, skip_top_thresholds:] < np.expand_dims(np.asarray(ths[skip_top_thresholds:]), axis=0)
                a = a + np.sum(np.maximum(np.sum(tmp, axis=0) - to_dec, 0))
    
            b = lt * (n - len(best_gt_scene[0]) * to_dec) 

        mAA_score = a / b
        
        if combo_mode =='harmonic':
            # it is basically the F1 score
            score = 2 * mAA_score * cluster_score / (mAA_score + cluster_score)
        elif combo_mode == 'geometric':
            score = (mAA_score * cluster_score) ** 0.5
        elif combo_mode == 'arithmetic':
            # to be avoided, since if one of the mAA or clusterness score is zero is not zero
            score = (mAA_score + cluster_score) * 0.5
        elif combo_mode == 'mAA':
            score = mAA_score
        elif combo_mode == 'clusterness':
            score = cluster_score
                    
        print(f'{dataset}: mAA = {mAA_score * 100: .2f} %, clusterness = {cluster_score * 100: .2f} %, combined = {score* 100: .2f} %')

        stat_mAA.append(mAA)
        stat_clusterness.append(cluster_score)
        stat_score.append(score)

    final_score = np.mean(stat_score)
    final_mAA = np.mean(stat_mAA)
    final_clusterness = np.mean(stat_clusterness)

    print(f'averaged on datasets: mAA = {final_mAA * 100: .2f} %, clusterness = {final_clusterness * 100: .2f} %, combined = {final_score* 100: .2f} %')

    return final_score


# mAA evaluation thresholds per dataset and scene
# if not included, default is used
# all threshold vectors must have the same length
tth = {
    'ETs':
    {
       'dioscuri': np.array([0.002,  0.005,  0.008,  0.01,  0.02,  0.04]),
             'ET': np.array([0.002,  0.005,  0.008,  0.01,  0.02,  0.04]),
     'another_ET': np.array([0.002,  0.005,  0.008,  0.01,  0.02,  0.04]),
    },    
    'kermits':
    {
         'kermit': np.array([0.002,  0.005,  0.008,  0.01,  0.02,  0.04]),
       'dioscuri': np.array([0.002,  0.005,  0.008,  0.01,  0.02,  0.04]),
    },
    'default': np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2]),
}

    
def tth_to_csv(tth, csv_file='../kaggle_raw_data/thresholds.csv'):    
    '''save thresholds to csv file <csv_file>'''

    os.makedirs(os.path.split(csv_file)[0], exist_ok=True)  
    
    with open(csv_file, 'w') as f:        
        f.write('dataset,scene,thresholds\n')

        for dataset in tth:
            if dataset == 'default':
                f.write(f"default,default,{arr_to_str(tth['default'])}\n")
            else:
                for scene in tth[dataset]:
                    f.write(f'{dataset},{scene},{arr_to_str(tth[dataset][scene])}\n')
    
    
def tth_from_csv(csv_file='../kaggle_raw_data/thresholds.csv'):    
    '''read thresholds from csv file <csv_file>'''

    tth = {}

    with open(csv_file, newline='\n') as csvfile:    
        csv_lines = csv.reader(csvfile, delimiter=',')
    
        header = True
        for row in csv_lines:
            if header:
                header = False
                continue
            
            dataset = row[0]
            scene = row[1]
            th = np.array([float(x) for x in (row[2].split(';'))])

            if dataset == 'default':
                tth['default'] = th
            else:
                if not dataset in tth: tth[dataset] = {}
                tth[dataset][scene] = th
                
    return tth


if __name__ == '__main__':   
    # generate raw gt

    # input structure    
    # kaggle_raw_data/dataset/scene - scene for each dataset *required*
    # kaggle_raw_data/dataset/scene/images - images for the specific dataset, scene *required*
    # kaggle_raw_data/dataset/scene/imc2025_gt.csv - gt (rotation and traslation) data in the IMC2025 format for the specific dataset, scene *generated by make_gt()*
    # kaggle_raw_data/dataset/scene/imc2024_gt.csv - gt (rotation and traslation) data in the IMC2024  format for the specific dataset, scene *optionally provided* 
    # kaggle_raw_data/dataset/scene/colmap - gt data as colmap reconstruction for the specific dataset, scene *optional, or computed by make_gt()*
    # kaggle_raw_data/dataset/scene/colmap/database.db - colmap database for the specific dataset, scene *optional, provided or computed by imped in make_gt()*
    # kaggle_raw_data/dataset/scene/colmap/models - colmap 3D models for the specific dataset, scene *optional, provided or computed by pycolmap in make_gt()*
    # kaggle_raw_data/dataset/scene/colmap/scales.csv - colmap 3D models scale factor for the specific dataset, scene *optionally provided, otherwise scale is set to 1*
    # kaggle_raw_data/dataset/outliers - distractor folder for the specific dataset, scene *optional*
    # kaggle_raw_data/dataset/outliers/images - distractor images for the specific dataset, scene *optional*    

    # if imc2025_gt.csv is present, this is used for the gt
    # else if imc2024_gt.csv is present, this is used for the gt
    # else if colmap 3D is present, this is used for the gt
    # else if colmap database is present, this is used to generate the 3D model
    # else if imped is available the 3D model is generated
    tth_to_csv(tth)
    gt_data = make_gt()
    to_csv(gt_data)   


    # generate gt data for kaggle    

    # output structure
    # kaggle_data/dataset - contains images of all the scenes for the specific dataset
    # kaggle_data/gt.csv - contains gt data
    # kaggle_data/mapping.csv - contains gt mapping data from kaggle_raw_data to kaggle_data
    # kaggle_data/thresholds.csv - contains the thresholds
    
    # if obfuscate_data is True, each run will generate new kaggle data, so run only once
    already_run = True # I've laready run it!
    if not already_run: make_input_data(obfuscate_data=True)


    # generate a simple greedy submission by exahustive brute force

    # for each dataset in kaggle_data colmap is used with imped to generate a 3D reconstruction,
    # the best model with highest number of registered images is selected,
    # the process is repeated removing these images to generate the next cluster
    # until the number of registered images is greater than the min model size,
    # remaining images are put in the outlier cluster
    submission = make_todo()
    to_csv(submission, csv_file='../kaggle_submission/submission.csv')    


    # check gt 
    print('GT vs GT')
    score_all_ext('../kaggle_data/gt.csv', '../kaggle_data/gt.csv', thresholds='../kaggle_data/thresholds.csv', per_th=False, strict_cluster=False)

    # submission score
    print('GT vs submission')
    score_all_ext('../kaggle_data/gt.csv', '../kaggle_submission/submission.csv', thresholds='../kaggle_data/thresholds.csv', per_th=False, strict_cluster=False)
