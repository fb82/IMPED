import os
import time

import h5py
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
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
            colmap_min_matches=colmap_min_matches,
        )
    else:
        # Incremental mode
        def gen_pairs():
            if mode == 'include':
                # existing vs existing
                for i in range(len(existing)):
                    for j in range(i + 1, len(existing)):
                        yield (existing[i], existing[j])

            # new vs existing
            for n in new:
                for e in existing:
                    yield (n, e)

            # new vs new
            for i in range(len(new)):
                for j in range(i + 1, len(new)):
                    yield (new[i], new[j])

        pairs_iter = gen_pairs()

        n_existing_vs_existing = len(existing) * (len(existing) - 1) // 2 if mode == 'include' else 0
        n_new_vs_existing = len(new) * len(existing)
        n_new_vs_new = len(new) * (len(new) - 1) // 2
        total = n_existing_vs_existing + n_new_vs_existing + n_new_vs_new


    total = len(pairs_iter) if hasattr(pairs_iter, '__len__') else total

    computed_pairs = set()
    if not force:
        hdf5 = db.get_hdf5()
        if hdf5 is not None and db.label_prefix in hdf5:
            root = hdf5[db.label_prefix]
            for im0, item0 in root.items():
                if hasattr(item0, 'items'):
                    for im1, item1 in item0.items():
                        if hasattr(item1, 'items'):
                            computed_pairs.add((im0, im1))

    for k, pair in enumerate(go_iter(pairs_iter, msg='          processed pairs')):
        img0 = os.path.basename(pair[0])
        img1 = os.path.basename(pair[1])

        if not force:
            if (img0, img1) in computed_pairs or (img1, img0) in computed_pairs:
                tqdm.write(f'  skipping already computed pair ({img0}, {img1})') if show_progress else print(f'  skipping already computed pair ({img0}, {img1})')
                continue

        msg = f'pair {k + 1}/{total}: {img0} <-> {img1}'
        tqdm.write(msg) if show_progress else print(msg)
        try:
            run_pipeline(pair, pipeline, db, force=force, show_progress=True)
        except Exception as e:
            tqdm.write(f'  skipping pair ({img0}, {img1}): {e}') if show_progress else print(f'  skipping pair ({img0}, {img1}): {e}')

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


_SALAD_TRANSFORM = transforms.Compose([
    transforms.Resize((322, 322), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_salad_model = None


_SALAD_CONFLICTING_MODULES = ('models', 'vpr_model', 'utils')


def _get_salad_model(salad_path, salad_device):
    global _salad_model
    if _salad_model is None:
        import sys

        # Other submodules (e.g. mast3r/dust3r) may have already cached a 'models'
        # or 'utils' package in sys.modules, which shadows SALAD's own modules.
        # We temporarily evict those entries, do the import, then restore them.
        saved = {}
        for key in list(sys.modules):
            if key in _SALAD_CONFLICTING_MODULES or any(
                key.startswith(m + '.') for m in _SALAD_CONFLICTING_MODULES
            ):
                saved[key] = sys.modules.pop(key)

        sys.path.insert(0, salad_path)
        try:
            from vpr_model import VPRModel
            from models.backbones.dinov2 import DINOV2_ARCHS
            import utils as salad_utils

            # get_loss / get_miner require pytorch_metric_learning which is a
            # training-only dependency. Stub them out — they are never called
            # during inference (forward() only uses backbone + aggregator).
            _orig_get_loss = salad_utils.get_loss
            _orig_get_miner = salad_utils.get_miner
            salad_utils.get_loss = lambda *a, **kw: None
            salad_utils.get_miner = lambda *a, **kw: None

            backbone = 'dinov2_vitb14'
            try:
                model = VPRModel(
                    backbone_arch=backbone,
                    backbone_config={
                        'num_trainable_blocks': 4,
                        'return_token': True,
                        'norm_layer': True,
                    },
                    agg_arch='SALAD',
                    agg_config={
                        'num_channels': DINOV2_ARCHS[backbone],
                        'num_clusters': 64,
                        'cluster_dim': 128,
                        'token_dim': 256,
                    },
                )
            finally:
                salad_utils.get_loss = _orig_get_loss
                salad_utils.get_miner = _orig_get_miner
            model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    'https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt',
                    map_location=torch.device('cpu'),
                )
            )
        finally:
            sys.path.remove(salad_path)
            # Evict the SALAD-specific entries we just imported, then restore the originals.
            for key in list(sys.modules):
                if key in _SALAD_CONFLICTING_MODULES or any(
                    key.startswith(m + '.') for m in _SALAD_CONFLICTING_MODULES
                ):
                    del sys.modules[key]
            sys.modules.update(saved)

        _salad_model = model
        _salad_model.eval()
        _salad_model.to(salad_device)
    return _salad_model


def _compute_global_descriptors(imgs, salad_path, salad_device):
    model = _get_salad_model(salad_path, salad_device)
    descriptors = []
    for img_path in go_iter(imgs, msg='computing global descriptors'):
        img = Image.open(img_path).convert('RGB')
        x = _SALAD_TRANSFORM(img).unsqueeze(0).to(salad_device)
        with torch.no_grad():
            desc = model(x)
        descriptors.append(desc.squeeze(0).cpu())
    return torch.stack(descriptors)  # (N, D)


def run_close_pairs(pipeline, imgs, n=10, db_name='database.hdf5', db_mode='a', force=False,
                    add_path='', colmap_db_or_list=None, mode='exclude', colmap_req='geometry',
                    colmap_min_matches=0, salad_device=None, n_chunks=1, chunk_idx=0,
                    salad_cache=None):
    """Like run_pairs but only matches each image against its n closest neighbours.

    Global descriptors are computed with DINOv2 SALAD (serizba/salad) to rank
    image similarity before running the feature-matching pipeline.

    Args:
        n (int): Number of nearest neighbours to pair each image with.
        n_chunks (int): Total number of independent workers sharing the dataset.
            Set to 1 (default) for single-machine use.
        chunk_idx (int): Zero-based index of this worker (0 … n_chunks-1).
            Pair generation is identical on every worker (deterministic, read-only);
            the pair list is then split round-robin so each worker gets a disjoint,
            balanced subset with no coordination or locking required.
        salad_cache (str | None): Path to a .pt file for caching SALAD descriptors.
            On first run the descriptors are saved there; subsequent runs (including
            other chunks on other machines) load from it instead of recomputing.
            Set to None (default) to disable caching.
        salad_device: Torch device for SALAD inference. Defaults to the project device.
        All other args are forwarded to run_pairs.
    """
    if salad_device is None:
        salad_device = device

    salad_path = os.path.join(os.path.dirname(__file__), '..', 'salad')
    salad_path = os.path.normpath(salad_path)

    if isinstance(imgs, str):
        imgs = [
            os.path.join(imgs, f)
            for f in os.listdir(imgs)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
    if add_path:
        imgs = [os.path.join(add_path, f) if not os.path.isabs(f) else f for f in imgs]

    imgs = sorted(imgs)

    if salad_cache is not None and os.path.exists(salad_cache):
        cached = torch.load(salad_cache, map_location='cpu', weights_only=False)
        if cached.get('imgs') == imgs:
            print(f'Loading cached SALAD descriptors from {salad_cache}')
            descs = cached['descs']
        else:
            print(f'SALAD cache image list mismatch, recomputing.')
            descs = _compute_global_descriptors(imgs, salad_path, salad_device)
            torch.save({'imgs': imgs, 'descs': descs}, salad_cache)
    else:
        descs = _compute_global_descriptors(imgs, salad_path, salad_device)
        if salad_cache is not None:
            torch.save({'imgs': imgs, 'descs': descs}, salad_cache)
            print(f'SALAD descriptors saved to {salad_cache}')

    # L2-normalise then dot product == cosine similarity
    descs = descs / descs.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sim = descs @ descs.T  # (N, N)

    k = min(n, len(imgs) - 1)
    pairs = set()
    for i in range(len(imgs)):
        sim[i, i] = -1.0  # exclude self
        top_k = torch.topk(sim[i], k=k).indices.tolist()
        for j in top_k:
            pair = (min(i, j), max(i, j))
            pairs.add(pair)

    # SALAD is only needed for pair generation — release it now so the
    # ~1.3 GB DINOv2 backbone doesn't occupy VRAM during matching.
    global _salad_model
    _salad_model = None
    torch.cuda.empty_cache()

    pair_list = [(imgs[i], imgs[j]) for i, j in sorted(pairs)]

    # Round-robin split across workers. Pair generation is identical on every
    # machine (deterministic, read-only), so no coordination is needed.
    if n_chunks > 1:
        pair_list = pair_list[chunk_idx::n_chunks]

    n_candidates = len(pair_list)

    # When run_pairs receives a list of tuples it skips the colmap_db_or_list
    # filter entirely, so we apply it here manually before handing off.
    n_skipped_colmap = 0
    if colmap_db_or_list is not None and not force:
        ip = image_pairs(
            pair_list,
            add_path=add_path,
            colmap_db_or_list=colmap_db_or_list,
            mode=mode,
            colmap_req=colmap_req,
            colmap_min_matches=colmap_min_matches,
        )
        pair_list = list(ip)
        n_skipped_colmap = n_candidates - len(pair_list)

    # Check how many are already cached in the HDF5 db.
    n_skipped_hdf5 = 0
    if db_name is not None and not force:
        db_tmp = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)
        hdf5 = db_tmp.get_hdf5()
        if hdf5 is not None and db_tmp.label_prefix in hdf5:
            root = hdf5[db_tmp.label_prefix]
            cached = {
                (im0, im1)
                for im0, item0 in root.items() if hasattr(item0, 'items')
                for im1, item1 in item0.items() if hasattr(item1, 'items')
            }
            before = len(pair_list)
            pair_list = [
                (p0, p1) for p0, p1 in pair_list
                if (os.path.basename(p0), os.path.basename(p1)) not in cached
                and (os.path.basename(p1), os.path.basename(p0)) not in cached
            ]
            n_skipped_hdf5 = before - len(pair_list)

    n_to_compute = len(pair_list)
    n_all_vs_all = len(imgs) * (len(imgs) - 1) // 2

    print("\nClose-pairs summary:")
    print(f"  Images           : {len(imgs)}")
    print(f"  All-vs-all       : {n_all_vs_all}")
    if n_chunks > 1:
        print(f"  Chunk            : {chunk_idx} of {n_chunks}")
    print(f"  Candidates (n={n:2d}): {n_candidates}")
    if n_skipped_colmap:
        print(f"  Skipped (colmap) : {n_skipped_colmap}")
    if n_skipped_hdf5:
        print(f"  Skipped (hdf5)   : {n_skipped_hdf5}")
    print(f"  To compute       : {n_to_compute}")
    print()

    run_pairs(
        pipeline, pair_list,
        db_name=db_name, db_mode=db_mode, force=force,
    )


def split_images(imgs, n_chunks, chunk_idx):
    if isinstance(imgs, str):
        imgs = [
            os.path.join(imgs, f)
            for f in os.listdir(imgs)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
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

