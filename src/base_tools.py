import os
import warnings
import pickled_hdf5.pickled_hdf5 as pickled_hdf5
import time

from PIL import Image
from tqdm import tqdm


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
        run_pipeline(pair, pipeline, db, force=force)

                
def run_pipeline(pair, pipeline, db, force=False, pipe_name='', pipe_data={}):

    for pipe_module in pipeline:
        if not isinstance(pipe_module, list):
            pipe_name = pipe_name + '/' + pipe_module.get_id()
            
            if hasattr(pipe_module, 'single_image') and pipe_module.single_image:
                for n in len(pair):
                    im = os.path.split(pair[n])[-1]
                    data_key = '/' + im + '/' + pipe_name + '/data'                    
    
                    out_data, is_found = db.get(data_key)                    
                    if (not is_found) or force:
                        start_time = time.time()
                        out_data = pipe_module.run(**pipe_data)
                        stop_time = time.time()
                        out_data['running_time'] = stop_time - start_time
                        db.add(data_key, out_data)
    
                    for k, v in out_data.items():
                        if k in pipe_data:
                            pipe_data[k].append(v)
                        else:
                            pipe_data[k] = [v]
                            
            else:
                im0 = os.path.split(pair[0])[-1]
                im1 = os.path.split(pair[1])[-1]
                data_key = '/' + im0 + '/' + im1 + '/' + pipe_name + '/data'   
    
                out_data, is_found = db.get(data_key)                    
                if (not is_found) or force:
                    start_time = time.time()
                    out_data = pipe_module.run(**pipe_data)
                    stop_time = time.time()
                    out_data['running_time'] = stop_time - start_time
                    db.add(data_key, out_data)
    
                for k, v in out_data.items(): pipe_data[k] = v
        else:
            pipe_name = pipe_name + '/' + pipe_module[0]            
            im0 = os.path.split(pair[0])[-1]
            im1 = os.path.split(pair[1])[-1]
            data_key = '/' + im0 + '/' + im1 + '/' + pipe_name + '/data'               

            out_data, is_found = db.get(data_key)                    
            if (not is_found) or force:
                branch_data = []
                for pipeline_branch in pipe_module[1:]:
                    branch_data.append(run_pipeline(pair, pipeline_branch, db, force=force, pipe_name=pipe_name, pipe_data=pipe_data)[0])
                    # to continue here!
                
            for k, v in out_data.items(): pipe_data[k] = v
                
    return pipe_data, pipe_name


if __name__ == '__main__':
    im_list = []

    for i in tqdm(image_pairs('/home/warlock/Scaricati/villa_giulia2/imgs'), desc='Image pair: '):
        im_list.append(i)