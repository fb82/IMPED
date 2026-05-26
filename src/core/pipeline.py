import os
import time

import torch
from tqdm import tqdm

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from image_pairs import image_pairs

from .device import device, show_progress


def go_iter(to_iter, msg='', active=True, params=None):
    if params is None: params = {}
    
    if show_progress and active:
        return tqdm(to_iter, desc=msg, **params)
    else:
        return to_iter 


def finalize_pipeline(pipeline):
    for pipe_module in pipeline:
        if hasattr(pipe_module, 'finalize'):
            pipe_module.finalize()
    
def run_pairs(pipeline, imgs, db_name='database.hdf5', db_mode='a', force=False, add_path='', colmap_db_or_list=None, mode='exclude', colmap_req='geometry', colmap_min_matches=0):    
    db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)

    if isinstance(imgs, str):
        imgs = [
            os.path.join(imgs, f)
            for f in os.listdir(imgs)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

    imgs = list(imgs)

    if imgs and isinstance(imgs[0], tuple):
        if add_path:
            imgs = [(os.path.join(add_path, p0), os.path.join(add_path, p1)) for p0, p1 in imgs]
        for pair in go_iter(imgs, msg='          processed pairs'):
            run_pipeline(pair, pipeline, db, force=force, show_progress=True)
        finalize_pipeline(pipeline)
        return

    img_map = {
        os.path.basename(p): p
        for p in imgs
    }

    colmap_db_path = None
    for m in pipeline:
        if hasattr(m, 'args') and 'db' in m.args:
            colmap_db_path = m.args['db']
            break

    existing_images = set()

    if colmap_db_path is not None:
        try:
            from colmap_fun.colmap_ext import coldb_ext

            colmap_db = coldb_ext(colmap_db_path)
            images = colmap_db.get_images()  # (id, name)

            for _, name in images:
                existing_images.add(name)

            colmap_db.close()

        except Exception as e:
            print("Warning: failed to read COLMAP DB, fallback to full pairing:", e)
            existing_images = set()

    existing = [
        img_map[name]
        for name in existing_images
        if name in img_map
    ]

    new = [
        p for p in imgs
        if os.path.basename(p) not in existing_images
    ]

    print(f"Total imgs: {len(imgs)}")
    print(f"Existing: {len(existing)}")
    print(f"New: {len(new)}")


    if colmap_db_path is None or len(existing_images) == 0:

        pairs_iter = image_pairs(
            imgs,
            add_path=add_path,
            colmap_db_or_list=colmap_db_or_list,
            mode=mode,
            colmap_req=colmap_req,
            colmap_min_matches=colmap_min_matches
        )
    else:
        # Incremental mode
        def gen_pairs():
            # new vs existing
            for n in new:
                for e in existing:
                    yield (n, e)

            # new vs new
            for i in range(len(new)):
                for j in range(i + 1, len(new)):
                    yield (new[i], new[j])

        pairs_iter = gen_pairs()


    for pair in go_iter(pairs_iter, msg='          processed pairs'):
        run_pipeline(pair, pipeline, db, force=force, show_progress=True)

    finalize_pipeline(pipeline)



def _move_to_device(x, target_device):
    if torch.is_tensor(x):
        return x.to(target_device)
    if isinstance(x, list):
        return [_move_to_device(v, target_device) for v in x]
    if isinstance(x, dict):
        return {k: _move_to_device(v, target_device) for k, v in x.items()}
    return x


def _align_pipe_data_device(pipe_data, target_device):
    tensor_keys = [
        'warp', 'kp', 'kH', 'kr', 'desc',
        'm_idx', 'm_val', 'm_mask',
        'F', 'E', 'H'
    ]
    for k in tensor_keys:
        if k in pipe_data:
            pipe_data[k] = _move_to_device(pipe_data[k], target_device)


def run_pipeline(pair, pipeline, db, force=False, pipe_data=None, pipe_name='/', show_progress=False):  
    """
    Executes a sequence of image processing modules on a pair of images.

    This function iterates through a list of 'pipeline' modules, handles 
    data dependencies, and manages persistent storage (db). It distinguishes 
    between 'single_image' tasks (like keypoint detection) and 'pair' tasks 
    (like feature matching).

    Key Features:
    - Smart Caching: Checks the 'db' for existing results based on a unique 
      hierarchical key before running a module.
    - Data Propagation: Updates a shared 'pipe_data' dictionary that grows 
      as images move through the pipeline.
    - Hierarchical Naming: Builds a 'pipe_name' string (e.g., /sift/smnn/magsac) 
      to track the specific lineage of the data.
    """
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

                    target_device = getattr(pipe_module, 'device', device)
                    _align_pipe_data_device(pipe_data, target_device)


                    out_data = pipe_module.run(idx=n, **pipe_data)
                    stop_time = time.time()
                    out_data['running_time'] = stop_time - start_time
                    if pipe_module.add_to_cache: db.add(data_key, out_data)
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

                target_device = getattr(pipe_module, 'device', device)
                _align_pipe_data_device(pipe_data, target_device)

                if hasattr(pipe_module, 'pipeliner') and pipe_module.pipeliner:
                    out_data = pipe_module.run(pipe_data=pipe_data, pipe_name=pipe_name_prev, db=db, force=force)
                else:
                    out_data = pipe_module.run(**pipe_data)

                stop_time = time.time()
                out_data['running_time'] = stop_time - start_time
                if pipe_module.add_to_cache: db.add(data_key, out_data)
            out_data['running_time']
                
            
            for k, v in out_data.items(): pipe_data[k] = v
                
    return pipe_data, pipe_name

