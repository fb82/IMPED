import os
import time
import traceback

import h5py
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

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


def resolve_image_folder(folder):
    """List a folder's contents, keeping only files PIL can open as images."""
    imgs = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        try:
            Image.open(p).verify()
        except Exception:
            continue
        imgs.append(p)
    return imgs


def run_pairs(pipeline, imgs, db_name='database.hdf5', db_mode='a', force=False, add_path='', colmap_db_or_list=None, mode='exclude', colmap_req='geometry', colmap_min_matches=0):
    """
    Runs `pipeline` once over pairs built from `imgs` (or the explicit list
    of pairs passed in), then calls `finalize_pipeline(pipeline)`.
    """
    if isinstance(imgs, str):
        imgs = resolve_image_folder(imgs)

    imgs = list(imgs)

    if imgs and isinstance(imgs[0], tuple):
        db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)
        if add_path:
            imgs = [(os.path.join(add_path, p0), os.path.join(add_path, p1)) for p0, p1 in imgs]
        for pair in go_iter(imgs, msg='          processed pairs'):
            run_pipeline(pair, pipeline, db, force=force, show_progress=True)
        db.close()
        finalize_pipeline(pipeline)
        return

    if colmap_db_or_list is None:
        for m in pipeline:
            if hasattr(m, 'args') and 'db' in m.args:
                colmap_db_or_list = m.args['db']
                break

    pairs = image_pairs(
        imgs,
        add_path=add_path,
        colmap_db_or_list=colmap_db_or_list,
        mode=mode,
        colmap_req=colmap_req,
        colmap_min_matches=colmap_min_matches,
    )

    db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)
    total = len(pairs)

    for k, pair in enumerate(go_iter(pairs, msg='          processed pairs')):
        img0 = os.path.basename(pair[0])
        img1 = os.path.basename(pair[1])

        msg = f'pair {k + 1}/{total}: {img0} <-> {img1}'
        tqdm.write(msg) if show_progress else print(msg)
        try:
            run_pipeline(pair, pipeline, db, force=force, show_progress=True)
        except Exception as e:
            msg = f'  skipping pair ({img0}, {img1}): {e}\n{traceback.format_exc()}'
            tqdm.write(msg) if show_progress else print(msg)

    db.close()

    finalize_pipeline(pipeline)



def _cached_single_image(db, module, img):
    """
    Runs a single-image module on one image, caching the result under the
    same key run_pipeline would use if `module` were the first entry of a
    pipeline (pipe_name starts at '/') — so a later run_pairs() call on a
    pipeline starting with the same module reuses this cached entry instead
    of recomputing it.
    """
    data_key = f'/{os.path.basename(img)}//{module.get_id()}/data'

    out, found = db.get(data_key)
    if not found:
        start_time = time.time()
        out = module.run(idx=0, img=[img])
        out['running_time'] = time.time() - start_time
        if module.add_to_cache:
            db.add(data_key, out)
    elif 'running_time' not in out:
        # repairs entries cached before 'running_time' was tracked here, so
        # run_pipeline's unconditional del/access on this key doesn't crash
        # when it later re-reads this same cache entry.
        out['running_time'] = 0.0
        if module.add_to_cache:
            db.add(data_key, out)

    return out


def _cached_pair_similarity(db, descriptor, similarity, img_a, img_b, desc_a, desc_b):
    """
    Runs a pair similarity module on one pair, caching the result under the
    same key run_pipeline would use for `similarity` as the second entry of
    a pipeline starting with `descriptor` — so a later run_pairs() call on
    that same pipeline reuses this cached 'pair_sim' instead of recomputing
    it (e.g. a costly pairwise BFMatcher/similarity pass).
    """
    im0 = os.path.basename(img_a)
    im1 = os.path.basename(img_b)
    data_key = f'/{im0}/{im1}//{descriptor.get_id()}/{similarity.get_id()}/data'

    out, found = db.get(data_key)
    if not found:
        start_time = time.time()
        out = similarity.run(global_desc=[desc_a, desc_b])
        out['running_time'] = time.time() - start_time
        if similarity.add_to_cache:
            db.add(data_key, out)
    elif 'running_time' not in out:
        out['running_time'] = 0.0
        if similarity.add_to_cache:
            db.add(data_key, out)

    return out['pair_sim']


def _move_to_device(x, target_device):
    if torch.is_tensor(x):
        return x.to(target_device)
    if isinstance(x, list):
        return [_move_to_device(v, target_device) for v in x]
    if isinstance(x, dict):
        return {k: _move_to_device(v, target_device) for k, v in x.items()}
    return x


def _align_pipe_data_device(pipe_data, target_device):
    for k in pipe_data:
        pipe_data[k] = _move_to_device(pipe_data[k], target_device)


def run_pipeline(pair, pipeline, db, force=False, pipe_data=None, pipe_name='/', show_progress=False):  
    """
    Executes a sequence of image processing modules on a pair of images.

    This function iterates through a list of 'pipeline' modules, handles 
    data dependencies, and manages persistent storage (db). It distinguishes 
    between 'single_image' tasks (like keypoint detection) and 'pair' tasks 
    (like feature matching).
    """
    if pipe_data is None: pipe_data = {}

    if not pipe_data:
        pipe_data['img'] = [pair[0], pair[1]]
        pipe_data['warp'] = [torch.eye(3, device=device, dtype=torch.float), torch.eye(3, device=device, dtype=torch.float)]
        pipe_data['stop'] = [False, False]

    for pipe_module in go_iter(pipeline, msg='current pipeline progress', active=show_progress, params={'leave': False}):
        stop = pipe_data.get('stop', False)
        if any(stop) if isinstance(stop, (list, tuple)) else stop:
            break

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


def split_images(imgs, n_chunks, chunk_idx):
    if isinstance(imgs, str):
        imgs = resolve_image_folder(imgs)
    imgs = sorted(imgs)
    chunk_size = (len(imgs) + n_chunks - 1) // n_chunks
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, len(imgs))
    return imgs[start:end]


def merge_hdf5(db_paths, output_path, prefix='pickled'):
    with h5py.File(output_path, 'a') as dst:
        if prefix not in dst:
            dst.create_group(prefix)
        dst_root = dst[prefix]

        for db_path in db_paths:
            with h5py.File(db_path, 'r') as src:
                if prefix not in src:
                    continue
                src_root = src[prefix]
                for key in src_root.keys():
                    if key in dst_root:
                        print(f'Warning: skipping conflicting key "{key}" from {db_path}')
                        continue
                    src.copy(f'{prefix}/{key}', dst_root)

