import itertools
import os
import time

import cv2
import h5py
import networkx as nx
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


def run_pairs(pipeline, imgs, db_name='database.hdf5', db_mode='a', force=False, add_path='', colmap_db_or_list=None, mode='exclude', colmap_req='geometry', colmap_min_matches=0, on_pair=None):
    db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)

    if isinstance(imgs, str):
        imgs = resolve_image_folder(imgs)

    imgs = list(imgs)

    if imgs and isinstance(imgs[0], tuple):
        if add_path:
            imgs = [(os.path.join(add_path, p0), os.path.join(add_path, p1)) for p0, p1 in imgs]
        for pair in go_iter(imgs, msg='          processed pairs'):
            pipe_data, _ = run_pipeline(pair, pipeline, db, force=force, show_progress=True)
            if on_pair is not None:
                on_pair(pair, pipe_data)
        finalize_pipeline(pipeline)
        return

    if colmap_db_or_list is None:
        for m in pipeline:
            if hasattr(m, 'args') and 'db' in m.args:
                colmap_db_or_list = m.args['db']
                break

    pairs_iter = image_pairs(
        imgs,
        add_path=add_path,
        colmap_db_or_list=colmap_db_or_list,
        mode=mode,
        colmap_req=colmap_req,
        colmap_min_matches=colmap_min_matches,
    )

    total = len(pairs_iter)

    for k, pair in enumerate(go_iter(pairs_iter, msg='          processed pairs')):
        img0 = os.path.basename(pair[0])
        img1 = os.path.basename(pair[1])

        msg = f'pair {k + 1}/{total}: {img0} <-> {img1}'
        tqdm.write(msg) if show_progress else print(msg)
        try:
            pipe_data, _ = run_pipeline(pair, pipeline, db, force=force, show_progress=True)
            if on_pair is not None:
                on_pair(pair, pipe_data)
        except Exception as e:
            tqdm.write(f'  skipping pair ({img0}, {img1}): {e}') if show_progress else print(f'  skipping pair ({img0}, {img1}): {e}')

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
    it (e.g. a costly BFMatcher pass for standard_similarity_module).
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
    - Early Exit: Any module may return a 'stop' key (True for either image)
      to signal that the pair is no longer worth processing. Once set, all
      remaining modules in the pipeline are skipped for this pair.
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


def _resolve_and_sort_imgs(imgs, add_path):
    if isinstance(imgs, str):
        imgs = resolve_image_folder(imgs)
    if add_path:
        imgs = [os.path.join(add_path, f) if not os.path.isabs(f) else f for f in imgs]
    return sorted(imgs)


def _compute_salad_similarity(imgs, salad_device, salad_cache):
    """Returns the full (N, N) cosine-similarity matrix between imgs' SALAD
    global descriptors, loading/saving salad_cache if given. Releases the
    SALAD model afterwards so its ~1.3 GB DINOv2 backbone doesn't occupy VRAM
    during matching.
    """
    salad_path = os.path.join(os.path.dirname(__file__), '..', 'salad')
    salad_path = os.path.normpath(salad_path)

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

    global _salad_model
    _salad_model = None
    torch.cuda.empty_cache()

    # L2-normalise then dot product == cosine similarity
    descs = descs / descs.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return descs @ descs.T  # (N, N)


def _dispatch_pair_list(pipeline, imgs, pair_list, db_name, db_mode, force, add_path,
                         colmap_db_or_list, mode, colmap_req, colmap_min_matches,
                         n_chunks, chunk_idx, summary_lines):
    """Shared tail for the close-pairs style entry points: splits pair_list across
    workers, filters out pairs already satisfied by colmap/hdf5, prints a summary
    and hands the remaining pairs off to run_pairs.
    """
    # Round-robin split across workers. Pair generation is identical on every
    # machine (deterministic, read-only), so no coordination is needed.
    if n_chunks > 1:
        chunk_ip = image_pairs(
            pair_list,
            check_img=False,
            chunk_id=chunk_idx,
            n_chunk=n_chunks,
        )
        pair_list = list(chunk_ip)

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
        print(f"  Chunk            : {chunk_idx+1} of {n_chunks}")
    for line in summary_lines:
        print(f"  {line}")
    print(f"  Candidates       : {n_candidates}")
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

    imgs = _resolve_and_sort_imgs(imgs, add_path)
    sim = _compute_salad_similarity(imgs, salad_device, salad_cache)

    k = min(n, len(imgs) - 1)
    pairs = set()
    for i in range(len(imgs)):
        sim[i, i] = -1.0  # exclude self
        top_k = torch.topk(sim[i], k=k).indices.tolist()
        for j in top_k:
            pair = (min(i, j), max(i, j))
            pairs.add(pair)

    pair_list = [(imgs[i], imgs[j]) for i, j in sorted(pairs)]

    _dispatch_pair_list(
        pipeline, imgs, pair_list, db_name, db_mode, force, add_path,
        colmap_db_or_list, mode, colmap_req, colmap_min_matches,
        n_chunks, chunk_idx, summary_lines=[f"Neighbours per image (n={n})"],
    )


def run_transitive_pairs(pipeline, imgs, initial_selection_module, max_rounds=None, db_name='database.hdf5', db_mode='a',
                          force=False, add_path='', on_round=None, on_pair=None, on_candidates=None):
    """
    Incrementally builds the pair list to run `pipeline` on, driven by the
    global descriptor already present in `pipeline` (e.g. salad_module,
    standard_descriptor_module, or any compatible module) and by transitive
    closure over confirmed pairs.

    `pipeline` must contain:
    - a single-image module producing 'global_desc' (the global descriptor),
    - a pair module consuming two 'global_desc' and producing 'pair_sim'
      (the similarity module),
    - a module exposing a `_table` list of (img0, img1) pairs confirmed so
      far, updated as the pipeline runs (conf_module or compatible), and an
      `args['threshold']` used below the same way conf_module uses it.

    `initial_selection_module` decides which pairs to try in the first round, since that
    round has no confirmed pairs yet to build candidates from and has to
    pick out of every possible pair instead (e.g.
    transitive_initial_selection.percentage_module, keeping the top
    fraction of all pairs, or transitive_initial_selection.max_uses_module,
    capping how many times any single image is paired). It must expose
    `select(scored, threshold)`, with `scored` a list of (pair_sim, (a, b))
    sorted by pair_sim descending, returning the list of (a, b) pairs to
    try; it is expected to honor `threshold` itself and never return a pair
    at or below it.

    Global descriptors are computed once per image and cached in `db_name`
    under the same key run_pairs() would use for that module, so later
    rounds (and any other pipeline sharing the same db) reuse them instead
    of recomputing. The descriptor module in use is printed to the terminal
    before computing them.

    Right after the global descriptors are computed, a networkx.Graph is
    built with one node per image and no edges; every round adds an edge
    per newly confirmed pair. The graph is what candidate generation reads
    for transitive closure, it is passed to `on_round(graph, n_round, n_new)`
    after every round if given, and it is returned at the end. For a live
    view as the run progresses, see visualization.live_pair_graph and wire
    it up via `on_pair`/`on_candidates` below instead. `on_pair(pair,
    pipe_data)` if given is called right after
    every single pair is run through `pipeline` (before the round's
    confirmations are known), with `pipe_data` the dict run_pipeline built
    for that pair (e.g. `pipe_data.get('pair_conf')` for conf_module's
    output) — e.g. to feed visualization.live_pair_graph. Note that, given
    the filtering below, `pipeline` is only ever run on pairs already known
    to score above `threshold`, so on_pair effectively never sees a
    rejected pair. `on_candidates(scored, threshold)` if given is called
    once per round, right after ranking, with the full `scored` list before
    that filtering — including the below-threshold pairs that never make it
    to `pipeline` at all — e.g. to draw those as a distinct "considered but
    rejected" edge in visualization.live_pair_graph.

    Each round:
    - If no pair has been confirmed yet, `initial_selection_module` picks which pairs
      (out of every possible pair) to try.
    - Otherwise, candidates are generated transitively from the confirmed
      pairs (A-C whenever A-B and B-C are both confirmed, and A-C hasn't
      been tried yet), and every one of them scoring above the conf_module
      threshold is tried — there's no further cap here since transitive
      closure candidates are already bounded by the confirmed graph.

    Rounds repeat until a round tries no pairs (every remaining candidate,
    seed or transitive, scores at or below threshold) or confirms no new
    pair, or until `max_rounds` is reached (None runs until convergence).
    """
    cv2.setNumThreads(20)

    imgs = _resolve_and_sort_imgs(imgs, add_path)

    descriptor = next((m for m in pipeline if getattr(m, 'single_image', False)), None)
    assert descriptor is not None, \
        "pipeline must include a single-image global descriptor module (e.g. salad_module, standard_descriptor_module)"

    conf = next((m for m in pipeline if hasattr(m, '_table')), None)
    assert conf is not None, \
        "pipeline must end with a conf_module (or compatible module exposing '_table')"

    similarity = next((m for m in pipeline if m is not descriptor and m is not conf), None)
    assert similarity is not None, \
        "pipeline must include a pair similarity module consuming 'global_desc' (e.g. cosine_similarity_module, standard_similarity_module)"

    desc_db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)

    print(f"run_transitive_pairs: computing global descriptors with '{descriptor.get_id()}'")
    global_desc = {
        img: _cached_single_image(desc_db, descriptor, img)['global_desc']
        for img in go_iter(imgs, msg='computing global descriptors')
    }

    # run_pairs() below opens its own h5py.File handle on db_name each round;
    # keeping this one open concurrently (same file, no SWMR) risks the other
    # handle seeing stale/partial writes, so release it now.
    desc_db.close()

    graph = nx.Graph()
    graph.add_nodes_from(imgs)

    threshold = conf.args['threshold']
    tried = set()

    def rank(candidates):
        sim_db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)
        scored = [
            (
                _cached_pair_similarity(sim_db, descriptor, similarity, a, b, global_desc[a], global_desc[b]),
                (a, b),
            )
            for a, b in go_iter(candidates, msg='ranking candidate pairs')
        ]
        # released before run_pairs() opens its own handle on db_name below
        sim_db.close()

        scored.sort(key=lambda x: x[0], reverse=True)
        if on_candidates is not None:
            on_candidates(scored, threshold)
        return scored

    def candidates_first_round():
        candidates = [
            (imgs[i], imgs[j])
            for i in range(len(imgs))
            for j in range(i + 1, len(imgs))
            if (imgs[i], imgs[j]) not in tried
        ]
        return initial_selection_module.select(rank(candidates), threshold)

    def candidates_transitive():
        candidates = set()
        for b in graph.nodes:
            for a, c in itertools.combinations(graph.neighbors(b), 2):
                pair = (min(a, c), max(a, c))
                if pair not in tried:
                    candidates.add(pair)

        return [pair for sim, pair in rank(list(candidates)) if sim > threshold]

    n_rounds = 0
    while max_rounds is None or n_rounds < max_rounds:
        pairs = candidates_first_round() if graph.number_of_edges() == 0 else candidates_transitive()
        if not pairs:
            break

        tried.update(pairs)
        n_before = len(conf._table)
        run_pairs(pipeline, pairs, db_name=db_name, db_mode=db_mode, force=force, on_pair=on_pair)
        n_new = len(conf._table) - n_before
        graph.add_edges_from(conf._table[n_before:])
        n_rounds += 1

        print(f"run_transitive_pairs: round {n_rounds}, {len(pairs)} pairs tried, {n_new} newly confirmed")

        if on_round is not None:
            on_round(graph, n_rounds, n_new)

        if n_new == 0:
            break

    return graph


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

