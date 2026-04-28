import os
import time
import warnings

import numpy as np
import pycolmap
import scipy
import torch
from PIL import Image
from tqdm import tqdm

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from core import device
from ensemble import pipe_union

from .colmap_ext import coldb_ext


def merge_colmap_db(db_names, db_merged_name, img_folder=None, to_filter=None, how_filter=None,
    only_keypoints=False, unique=True, only_matched=False, no_unmatched=True,
    include_two_view_geometry=True, sampling_mode='raw', overlapping_cells=False,
    sampling_scale=1, sampling_offset=0, focal_cf=1.2, profile=False, return_profile=False):              
    """
    Merges multiple COLMAP SQLite databases into a single unified database.

    This function iterates through a list of source databases and performs:
    1. Image/Camera Synchronization: Ensures that if an image appears in 
       multiple databases, it is only created once in the merged database 
       with a consistent Camera ID.
    2. Keypoint Union: Uses `pipe_union` to combine keypoints from different 
       sources. It can handle duplicates, average positions, or filter by 
       sampling modes (e.g., keeping only points with high inlier counts).
    3. Match Consolidation: Merges 2D-2D match indices and Two-View 
       Geometries (Fundamental/Essential matrices).
    4. Advanced Filtering: Supports including or excluding specific image 
       pairs during the merge process.

    Args:
        db_names (list): List of paths to source .db files.
        db_merged_name (str): Path where the final merged database will be saved.
        sampling_mode (str): How to handle overlapping keypoints 
            ('raw', 'avg_all_matches', etc.).
        include_two_view_geometry (bool): Whether to preserve the calculated 
            geometric verification (F, E, H matrices).

    Returns:
        None or dict: Side-effect is the creation of a merged COLMAP database.
        If return_profile=True, returns a timing dictionary.
    """      
    from core import go_iter

    from .colmap_ext import SIMPLE_RADIAL

    if device.type != 'cpu':
        warnings.warn('device is not set to cpu, computation will be *very slow*')

    t_all_start = time.perf_counter()
    profile_data = {
        'total_s': 0.0,
        'db_open_s': 0.0,
        'pair_loop_s': 0.0,
        'filter_s': 0.0,
        'image_sync_s': 0.0,
        'keypoint_read_s': 0.0,
        'match_read_s': 0.0,
        'pipe_union_s': 0.0,
        'db_write_match_s': 0.0,
        'db_write_kp_flush_s': 0.0,
        'db_commit_s': 0.0,
        'pairs_total': 0,
        'pairs_processed': 0,
        'pairs_skipped_filter': 0,
        'pairs_missing_name': 0,
        'per_db': [],
    }
    
    aux_hdf5 = None
    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):         
        aux_hdf5 = pickled_hdf5.pickled_hdf5('tmp.hdf5', mode='a')
                
    counter = (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches')

    def _init_image_state(kp_np, image_name):
        if kp_np is None:
            state = {
                'w': torch.zeros((0, 6), device=device),
                'kp': torch.zeros((0, 2), device=device),
                'kH': torch.zeros((0, 3, 3), device=device),
                'kr': torch.full((0,), torch.inf, device=device),
            }
            if counter:
                state['k_counter'] = torch.zeros(0, device=device)
            return state

        w = torch.tensor(kp_np, device=device)
        kp = torch.tensor(kp_np[:, :2], device=device)
        state = {
            'w': w,
            'kp': kp,
            'kH': torch.zeros((kp.shape[0], 3, 3), device=device),
            'kr': torch.full((kp.shape[0],), torch.inf, device=device),
        }
        if counter:
            k_count, _ = aux_hdf5.get(image_name)
            state['k_counter'] = k_count
        return state

    db_merged = coldb_ext(db_merged_name)
    db_merged.create_tables()
    merged_image_cache = {os.path.split(name)[-1]: image_id for image_id, name in db_merged.get_images()}
    
    for i, db_name in enumerate(go_iter(db_names, msg='         merging progress')):
        t_db0 = time.perf_counter()
        db_stats = {
            'db_name': db_name,
            'pairs_total': 0,
            'pairs_processed': 0,
            'pairs_skipped_filter': 0,
            'pairs_missing_name': 0,
            'open_s': 0.0,
            'pair_loop_s': 0.0,
            'filter_s': 0.0,
            'image_sync_s': 0.0,
            'keypoint_read_s': 0.0,
            'match_read_s': 0.0,
            'pipe_union_s': 0.0,
            'db_write_match_s': 0.0,
            'db_write_kp_flush_s': 0.0,
            'db_commit_s': 0.0,
        }
        t_open0 = time.perf_counter()
        db = coldb_ext(db_name)
        imgs = db.get_images()
        img_name_by_id = {image_id: os.path.split(name)[-1] for image_id, name in imgs}
        match_pairs = db.get_match_image_pairs(include_two_view_geometry=include_two_view_geometry)
        db_stats['open_s'] += time.perf_counter() - t_open0
        db_stats['pairs_total'] = len(match_pairs)
        profile_data['pairs_total'] += len(match_pairs)
    
        if (to_filter is None) or (how_filter is None):
            current_how = None
            current_filter = None
        else:
            current_how = how_filter[i]
            current_filter = to_filter[i]
            
        if current_filter is not None:
            if len(current_filter) == 0:
                current_how = None
                current_filter = None

        img_dict = {}
        pair_dict = {}
        if current_filter is not None:
            for v in current_filter:
                if not(isinstance(v, list) or isinstance(v, tuple)):
                    img_dict[v] = 1
                else:                    
                    if v[0] not in pair_dict: pair_dict[v[0]] = {}                    
                    pair_dict[v[0]][v[1]] = 1
            
            
        src_state_cache = {}
        merged_state_cache = {}
        dirty_merged_ids = set()

        pbar = tqdm(total=len(match_pairs), desc='current database progress', leave=False)
        t_pair_loop0 = time.perf_counter()
        for im0_id, im1_id in match_pairs:
                pbar.update()

                im0 = img_name_by_id.get(im0_id)
                im1 = img_name_by_id.get(im1_id)
                if (im0 is None) or (im1 is None):
                    db_stats['pairs_missing_name'] += 1
                    profile_data['pairs_missing_name'] += 1
                    continue
                
                t_filter0 = time.perf_counter()
                skipped = False
                if current_how == 'exclude':
                    cond0 = (im0 in img_dict) or (im1 in img_dict) 
                    cond1 = ((im0 in pair_dict) and (im1 in pair_dict[im0])) or ((im1 in pair_dict) and (im0 in pair_dict[im1]))                    
                    if cond0 or cond1:
                        skipped = True
 
                if current_how == 'include':
                    cond0 = (im0 in img_dict) or (im1 in img_dict) 
                    cond1 = ((im0 in pair_dict) and (im1 in pair_dict[im0])) or ((im1 in pair_dict) and (im0 in pair_dict[im1]))                    
                    if (not cond0) and (not cond1):
                        skipped = True
                dt_filter = time.perf_counter() - t_filter0
                db_stats['filter_s'] += dt_filter
                if skipped:
                    db_stats['pairs_skipped_filter'] += 1
                    profile_data['pairs_skipped_filter'] += 1
                    continue

                # print((im0, im1))
                t_img_sync0 = time.perf_counter()
                im0_id_prev = merged_image_cache.get(im0)
                if  im0_id_prev is None:
                    im0_name, cam0_id = db.get_image(im0_id)
                    
                    if img_folder is None:
                        cam0 = db.get_camera(cam0_id)                                       
                        cam0_id_prev = db_merged.add_camera(cam0[0], cam0[1], cam0[2], cam0[3], cam0[4])
                    else:
                        w, h = Image.open(os.path.join(img_folder, im0)).size
                        cam0_id_prev = db_merged.add_camera(SIMPLE_RADIAL, w, h, np.array([focal_cf * max(w, h), w / 2, h / 2, 0]))
                       
                    im0_id_prev = db_merged.add_image(im0_name, cam0_id_prev)
                    merged_image_cache[im0] = im0_id_prev

                im1_id_prev = merged_image_cache.get(im1)
                if  im1_id_prev is None:
                    im1_name, cam1_id = db.get_image(im1_id)
                    
                    if img_folder is None:
                        cam1 = db.get_camera(cam1_id)
                        cam1_id_prev = db_merged.add_camera(cam1[0], cam1[1], cam1[2], cam1[3], cam1[4])
                    else:
                        w, h = Image.open(os.path.join(img_folder, im1)).size
                        cam1_id_prev = db_merged.add_camera(SIMPLE_RADIAL, w, h, np.array([focal_cf * max(w, h), w / 2, h / 2, 0]))
                                        
                    im1_id_prev = db_merged.add_image(im1_name, cam1_id_prev)
                    merged_image_cache[im1] = im1_id_prev
                db_stats['image_sync_s'] += time.perf_counter() - t_img_sync0
         
                t_kp_read0 = time.perf_counter()
                if im0_id not in src_state_cache:
                    src_state_cache[im0_id] = _init_image_state(db.get_keypoints(im0_id), im0)
                if im1_id not in src_state_cache:
                    src_state_cache[im1_id] = _init_image_state(db.get_keypoints(im1_id), im1)
                src0 = src_state_cache[im0_id]
                src1 = src_state_cache[im1_id]

                if im0_id_prev not in merged_state_cache:
                    merged_state_cache[im0_id_prev] = _init_image_state(db_merged.get_keypoints(im0_id_prev), im0)
                if im1_id_prev not in merged_state_cache:
                    merged_state_cache[im1_id_prev] = _init_image_state(db_merged.get_keypoints(im1_id_prev), im1)
                prev0 = merged_state_cache[im0_id_prev]
                prev1 = merged_state_cache[im1_id_prev]
                db_stats['keypoint_read_s'] += time.perf_counter() - t_kp_read0

                pipe = {
                    'kp': [src0['kp'], src1['kp']],
                    'kH': [src0['kH'], src1['kH']],
                    'kr': [src0['kr'], src1['kr']],
                    'w': [src0['w'], src1['w']],
                }

                pipe_prev = {
                    'kp': [prev0['kp'], prev1['kp']],
                    'kH': [prev0['kH'], prev1['kH']],
                    'kr': [prev0['kr'], prev1['kr']],
                    'w': [prev0['w'], prev1['w']],
                }

                no_matches = False
                if only_keypoints: no_matches = True
 
                matches = None
                two_view_matches = None
                if no_matches == False:
                    t_match_read0 = time.perf_counter()
                    matches = db.get_matches(im0_id, im1_id)
                    if matches is not None and include_two_view_geometry:
                        two_view_matches, models = db.get_two_view_geometry(im0_id, im1_id)
                    db_stats['match_read_s'] += time.perf_counter() - t_match_read0

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
                
                if counter:
                    pipe['k_counter'] = [src0['k_counter'], src1['k_counter']]
        
                matches_prev = None
                two_view_matches_prev = None
                if no_matches == False:
                    t_match_read1 = time.perf_counter()
                    matches_prev = db_merged.get_matches(im0_id_prev, im1_id_prev)
                    if matches_prev is not None and include_two_view_geometry:
                        two_view_matches_prev, models_prev = db_merged.get_two_view_geometry(im0_id_prev, im1_id_prev)
                    db_stats['match_read_s'] += time.perf_counter() - t_match_read1

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
                
                if counter:
                    pipe_prev['k_counter'] = [prev0['k_counter'], prev1['k_counter']]

                t_union0 = time.perf_counter()
                pipe_out = pipe_union([pipe_prev, pipe], unique=unique, no_unmatched=no_unmatched, only_matched=only_matched, sampling_mode=sampling_mode, sampling_scale=sampling_scale, sampling_offset=sampling_offset, overlapping_cells=overlapping_cells, preserve_order=True, counter=counter)
                db_stats['pipe_union_s'] += time.perf_counter() - t_union0

                if counter:
                    aux_hdf5.add(im0, pipe_out['k_counter'][0])
                    aux_hdf5.add(im1, pipe_out['k_counter'][1])
                    prev0['k_counter'] = pipe_out['k_counter'][0]
                    prev1['k_counter'] = pipe_out['k_counter'][1]

                prev0['w'] = pipe_out['w'][0]
                prev1['w'] = pipe_out['w'][1]
                prev0['kp'] = pipe_out['kp'][0]
                prev1['kp'] = pipe_out['kp'][1]
                prev0['kH'] = pipe_out['kH'][0]
                prev1['kH'] = pipe_out['kH'][1]
                prev0['kr'] = pipe_out['kr'][0]
                prev1['kr'] = pipe_out['kr'][1]
                dirty_merged_ids.add(im0_id_prev)
                dirty_merged_ids.add(im1_id_prev)

                if not only_keypoints:
                    t_write_match0 = time.perf_counter()
                    m_idx = pipe_out['m_idx'].to('cpu').numpy()
                    db_merged.update_matches(im0_id_prev, im1_id_prev, m_idx)
        
                    if include_two_view_geometry:        
                        m_idx = pipe_out['m_idx'][pipe_out['m_mask']].to('cpu').numpy()
                        models = {}                                        
                        db_merged.update_two_view_geometry(im0_id_prev, im1_id_prev, m_idx, model=models)
                    db_stats['db_write_match_s'] += time.perf_counter() - t_write_match0

                t_commit0 = time.perf_counter()
                db_merged.commit()
                db_stats['db_commit_s'] += time.perf_counter() - t_commit0
                db_stats['pairs_processed'] += 1
                profile_data['pairs_processed'] += 1
        db_stats['pair_loop_s'] += time.perf_counter() - t_pair_loop0

        t_kp_flush0 = time.perf_counter()
        for merged_id in dirty_merged_ids:
            pts = merged_state_cache[merged_id]['w'].to('cpu').numpy()
            db_merged.update_keypoints(merged_id, pts)
        db_stats['db_write_kp_flush_s'] += time.perf_counter() - t_kp_flush0
        t_commit1 = time.perf_counter()
        db_merged.commit()
        db_stats['db_commit_s'] += time.perf_counter() - t_commit1

        db.close()
        pbar.close()
        db_stats['pair_loop_s'] += 0.0
        profile_data['per_db'].append(db_stats)
        profile_data['db_open_s'] += db_stats['open_s']
        profile_data['pair_loop_s'] += db_stats['pair_loop_s']
        profile_data['filter_s'] += db_stats['filter_s']
        profile_data['image_sync_s'] += db_stats['image_sync_s']
        profile_data['keypoint_read_s'] += db_stats['keypoint_read_s']
        profile_data['match_read_s'] += db_stats['match_read_s']
        profile_data['pipe_union_s'] += db_stats['pipe_union_s']
        profile_data['db_write_match_s'] += db_stats['db_write_match_s']
        profile_data['db_write_kp_flush_s'] += db_stats['db_write_kp_flush_s']
        profile_data['db_commit_s'] += db_stats['db_commit_s']
        _ = time.perf_counter() - t_db0

    db_merged.close()
    if (sampling_mode == 'avg_all_matches') or (sampling_mode == 'avg_inlier_matches'):
        aux_hdf5.close()
        if os.path.isfile('tmp.hdf5'): os.remove('tmp.hdf5')

    profile_data['total_s'] = time.perf_counter() - t_all_start
    if profile:
        print('merge_colmap_db profile:')
        print(f"  total_s={profile_data['total_s']:.3f} pairs_processed={profile_data['pairs_processed']}/{profile_data['pairs_total']} skipped_filter={profile_data['pairs_skipped_filter']} missing_name={profile_data['pairs_missing_name']}")
        print(f"  db_open_s={profile_data['db_open_s']:.3f} pair_loop_s={profile_data['pair_loop_s']:.3f} filter_s={profile_data['filter_s']:.3f}")
        print(f"  image_sync_s={profile_data['image_sync_s']:.3f} keypoint_read_s={profile_data['keypoint_read_s']:.3f} match_read_s={profile_data['match_read_s']:.3f}")
        print(f"  pipe_union_s={profile_data['pipe_union_s']:.3f} db_write_match_s={profile_data['db_write_match_s']:.3f} db_write_kp_flush_s={profile_data['db_write_kp_flush_s']:.3f} db_commit_s={profile_data['db_commit_s']:.3f}")
        for dbs in profile_data['per_db']:
            print(f"  per_db: {dbs['db_name']} processed={dbs['pairs_processed']}/{dbs['pairs_total']} pair_loop_s={dbs['pair_loop_s']:.3f} pipe_union_s={dbs['pipe_union_s']:.3f} keypoint_read_s={dbs['keypoint_read_s']:.3f} match_read_s={dbs['match_read_s']:.3f} writes_match_s={dbs['db_write_match_s']:.3f} writes_kp_s={dbs['db_write_kp_flush_s']:.3f} commit_s={dbs['db_commit_s']:.3f}")
    if return_profile:
        return profile_data
        

def filter_colmap_reconstruction(input_model_path='../aux/colmap/model', img_path=None, db_path=None, output_model_path='../aux/colmap/output_model', to_filter=None, how_filter='exclude', only_cameras=True, add_3D_points=False, add_as_possible=True):
    """
    Subsets a COLMAP reconstruction by including or excluding specific images.

    This function modifies a sparse 3D model by deregistering images based on 
    a filter list. It can either simply save the updated camera poses or 
    perform a full re-triangulation of 3D points to ensure the point cloud 
    matches the new subset of cameras.

    Args:
        input_model_path (str): Path to the source COLMAP sparse model.
        img_path (str, optional): Path to the original image folder (required 
            for point triangulation or color extraction).
        db_path (str, optional): Path to the COLMAP SQLite database.
        output_model_path (str): Directory where the filtered model will be saved.
        to_filter (list): A list of image filenames to act upon.
        how_filter (str): 'exclude' (removes images in the list) or 
            'include' (keeps only images in the list).
        only_cameras (bool): If True, deletes all 3D points and keeps only poses.
        add_3D_points (bool): If True, triggers `pycolmap.triangulate_points` 
            to reconstruct the scene geometry for the remaining images.

    Returns:
        None: Saves the processed model to the output path.
    """
    model = pycolmap.Reconstruction(input_model_path)

    os.makedirs(output_model_path, exist_ok=True)
    if to_filter is None: to_filter=[]
    
    to_filter_dict = {}
    for image in to_filter: to_filter_dict[image] = 1
    
    model_imgs = [(image_id, model.image(image_id).name) for image_id in model.images]

    for image_id, image in model_imgs:
        if ((image in to_filter_dict) and (how_filter == 'exclude')) or ((image not in to_filter_dict) and (how_filter == 'include')):
            #model.deregister_image(image_id)
            image = model.image(image_id)
            frame_id = image.frame_id
            if model.exists_frame(frame_id):
                model.deregister_frame(frame_id)
        
    if only_cameras:
        for pts3D in model.point3D_ids(): model.delete_point3D(pts3D)

    if (not only_cameras) and add_3D_points and (img_path is not None) and (db_path is not None):
        incr_map_opt = pycolmap.IncrementalPipelineOptions()
        if add_as_possible:
            tri_opt = incr_map_opt.triangulation
            tri_opt.ignore_two_view_tracks = False    
        model = pycolmap.triangulate_points(model, db_path, img_path, output_model_path, True, incr_map_opt, False)
        return

    if img_path is not None:
        model.extract_colors_for_all_images(img_path)

    model.write_binary(output_model_path)


_EPS = np.finfo(float).eps * 4.0


               
def align_colmap_models(model_path1='../aux/colmap/model0', model_path2='../aux/colmap/model1', imgs_path=None, db_path0=None, db_path1=None,
                        output_db='../aux/colmap/merged_database.db', output_model='../aux/colmap/merged_model', th=None,
                        only_cameras=False, add_as_possible=True, no_force_db_fusion=True):
    """
        Aligns and merges two separate COLMAP reconstruction models into one.

        The function performs three main steps:
        1. Database Fusion: Combines the SQL databases containing image matches.
        2. Similarity Transformation (Sim3): Uses shared camera locations to 
        calculate the rotation, translation, and scale needed to move Model 2 
        into Model 1's coordinate space.
        3. Incremental Triangulation: Re-computes 3D points in the shared space 
        to create a single, dense point cloud from the merged camera poses.

        Args:
            model_path1/2 (str): Paths to the input COLMAP sparse models.
            output_db (str): Path to the new merged SQLite database.
            output_model (str): Directory where the final aligned model is saved.
            th (float): Distance threshold for alignment; if None, it is 
                automatically calculated based on average camera distance.
            only_cameras (bool): If True, only camera poses are merged (no 3D points).

        Returns:
            None: Saves the resulting merged model and database to disk.
        """
    from benchmark import evaluate_rec

    model1 = pycolmap.Reconstruction(model_path1)
    model2 = pycolmap.Reconstruction(model_path2)

    if (not only_cameras) and (db_path0 is not None) and (db_path1 is not None) and (imgs_path is not None):
        if (not (os.path.isfile(output_db))) or (not no_force_db_fusion):
            
            db_path = os.path.split(output_db)[0]
            if db_path != '': os.makedirs(db_path, exist_ok=True)   
            
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
        fused_model.add_camera_with_trivial_rig(new_camera)
        new_image = pycolmap.Image()
        new_image.name = image.name
        new_image.image_id = img_id
        new_image.camera_id = cam_id
        fused_model.add_image_with_trivial_frame(new_image)
        fused_model.image(img_id).frame.set_cam_from_world(cam_id, image.cam_from_world())
        fused_model.register_frame(fused_model.image(img_id).frame_id)
        count = count + 1

    for image_id in model2.images:
        if model1.find_image_with_name(model2.image(image_id).name) is not None: continue

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
        fused_model.add_camera_with_trivial_rig(new_camera)
        new_image = pycolmap.Image()
        new_image.name = image.name
        new_image.image_id = img_id
        new_image.camera_id = cam_id
        fused_model.add_image_with_trivial_frame(new_image)
        fused_model.image(img_id).frame.set_cam_from_world(cam_id, image.cam_from_world())
        fused_model.register_frame(fused_model.image(img_id).frame_id)
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

  
