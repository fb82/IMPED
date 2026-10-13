import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
import inspect
from functools import partial

project_root = Path(__file__).parent.parent.resolve()
external_root = project_root.parent / "external"

if str(external_root) not in sys.path:
    sys.path.insert(0, str(external_root))

modified_root = external_root / "modified"
if str(modified_root) not in sys.path:
    sys.path.insert(0, str(modified_root))

import h5py
import networkx as nx
import pycolmap
import torch

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from core import enable_quadtree, run_pairs, split_images, merge_hdf5, resolve_image_folder

extra_paths = [
    external_root / "r2d2",
    external_root / "mast3r",
    external_root / "matchformer",
    external_root / "aspanformer" / "src",
    external_root / "miho" / "src",
    external_root / "romav2" / "src",
    external_root / "loma" / "src",
    external_root / "vggt-omega",
    external_root / "map-anything",
    external_root / "gsm"
]



for p in extra_paths:
    if p.exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


from descriptors import deep_descriptor_module, dino_descriptor_module, patch_module, sift_module
from global_descriptors import salad_module
from similarity import (
    cosine_similarity_module,
    l2_similarity_module,
    n_matches_similarity_module,
)
from confidence import conf_module
from transitive import kfc_module, transitive_module
from core.image_pairs import image_pairs
from detectors import dog_module, hz_module, r2d2_module
from matchers import (
    aspanformer_module,
    blob_matching_module,
    deep_joined_module,
    dust3r_module,
    lightglue_module,
    loftr_module,
    mast3r_module,
    loma_module,
    vggt_omega_module,
    mapanything_module,
    roma_module,
    romav2_module,
    similarity_matrix_module,
    smnn_extended_module,
    smnn_module,
)

if enable_quadtree:
    pass

from benchmark import benchmark_setup, pairwise_benchmark_module
from colmap_fun import (
    align_colmap_models,
    filter_colmap_reconstruction,
    from_colmap_module,
    merge_colmap_db,
    to_colmap_module,
)
from ensemble import (
    image_muxer_module,
    pair_pyramid,
    pair_resize,
    pair_rot4,
    pipe_identity,
    pipe_sm_product,
    pipe_max_matches,
    pipe_union,
    pipeline_muxer_module,
    sampling_module,
)
from reconstruct import hierarchical_reconstruct_module, reconstruct_module
from filters import acne_module, dtm_module, magsac_module, mop_miho_ncc_module, oanet_module, dino_filter_module
from segmentators import segformer_module
from visualization import (
    live_pair_graph,
    show_homography_module,
    show_kpts_module,
    show_matches_module,
    show_patches_module,
)

OUTPUT_DIR = 'output_tests'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pipeline1():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
            dog_module(),
            # show_kpts_module(id_more='first', prepend_pair=False),
            patch_module(),
            # show_kpts_module(id_more='second', img_prefix='orinet_affnet_', prepend_pair=True),
            deep_descriptor_module(),
            smnn_module(),
            magsac_module(),
            # show_kpts_module(id_more='third', img_prefix='ransac_', prepend_pair=True, mask_idx=[0, 1]),
            # show_matches_module(id_more='forth', img_prefix='matches_', mask_idx=[1, 0]),
            # show_matches_module(id_more='fifth', img_prefix='matches_inliers_', mask_idx=[1]),
            # show_matches_module(id_more='sixth', img_prefix='matches_all_', mask_idx=-1),
            show_matches_module(id_moreFalse='only', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline2():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        loftr_module(),
        show_kpts_module(id_more='first', prepend_pair=False),
        magsac_module(),
        show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline3():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        deep_joined_module(),
        show_kpts_module(id_more='first', prepend_pair=False),
        lightglue_module(),
        magsac_module(),
        show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)

def pipeline4():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        image_muxer_module(pair_generator=pair_rot4, pipe_gather=pipe_max_matches, pipeline=[
            hz_module(),
            patch_module(sift_orientation=True, orinet=False),
            deep_descriptor_module(),
            show_kpts_module(id_more='first', prepend_pair=False),
            smnn_module(),
            magsac_module(),
            show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        ]),
        show_kpts_module(id_more='third', img_prefix='best_rot_', prepend_pair=False),
        show_matches_module(id_more='fourth', img_prefix='best_rot_matches_', mask_idx=[1, 0], prepend_pair=False),            
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)

def pipeline5():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        image_muxer_module(pair_generator=pair_rot4, pipe_gather=pipe_max_matches, pipeline=[
            deep_joined_module(),
            show_kpts_module(id_more='first', prepend_pair=False),
            lightglue_module(),
            magsac_module(),
            show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        ]),
        show_kpts_module(id_more='third', img_prefix='best_rot_', prepend_pair=False),
        show_matches_module(id_more='fourth', img_prefix='best_rot_matches_', mask_idx=[1, 0], prepend_pair=False),            
    ]
    imgs = '../data/ET_random_rotated'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)

def pipeline6():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                loftr_module(),
                show_kpts_module(id_more='a_first', img_prefix='a_', prepend_pair=False),
                magsac_module(),
                show_matches_module(id_more='a_second', img_prefix='a_matches_', mask_idx=[1, 0], prepend_pair=False),
            ],
            [
                deep_joined_module(),
                show_kpts_module(id_more='b_first', img_prefix='b_', prepend_pair=False),
                lightglue_module(),
                magsac_module(),
                show_matches_module(id_more='b_second', img_prefix='b_matches_', mask_idx=[1, 0], prepend_pair=False),                    
            ],
        ]),
        show_kpts_module(id_more='third', img_prefix='union_', prepend_pair=False),
        show_matches_module(id_more='fourth', img_prefix='union_matches_', mask_idx=[1, 0], prepend_pair=False),            
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline7():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")   
    pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                deep_joined_module(),
                show_kpts_module(id_more='a_first', img_prefix='a_', prepend_pair=False),
                lightglue_module(),
                magsac_module(),
                show_matches_module(id_more='a_second', img_prefix='a_matches_', mask_idx=[1, 0], prepend_pair=False),                    
            ],
            [
                deep_joined_module(),
                show_kpts_module(id_more='b_first', img_prefix='b_', prepend_pair=False),
                lightglue_module(),
                magsac_module(),
                show_matches_module(id_more='b_second', img_prefix='b_matches_', mask_idx=[1, 0], prepend_pair=False),                    
            ],
        ]),
        show_kpts_module(id_more='third', img_prefix='union_', prepend_pair=False),
        show_matches_module(id_more='fourth', img_prefix='union_matches_', mask_idx=[1, 0], prepend_pair=False),            
    ]    
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)    

def pipeline8():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        loftr_module(),
        magsac_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        sampling_module(sampling_mode='avg_inlier_matches', overlapping_cells=True, sampling_scale=20),
        show_matches_module(id_more='second', img_prefix='matches_sampled_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)   

def pipeline9():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        loftr_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)       


def pipeline10():  
    name_example = inspect.currentframe().f_code.co_name 
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)  

def pipeline11(): 
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        loftr_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)  


def pipeline12():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        roma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),
    ]    
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)  
 
def pipeline13():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        r2d2_module(),
        # smnn_module(),
        lightglue_module(what='sift', desc_cf=255),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)   

def pipeline14():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0]),
        acne_module(),
        show_matches_module(id_more='second', img_prefix='matches_after_filter_', mask_idx=[1, 0]),
        magsac_module(),
        show_matches_module(id_more='third', img_prefix='matches_final_', mask_idx=[1, 0]),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)  

def pipeline15():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        aspanformer_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)

def pipeline16():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        from_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),
        show_kpts_module(img_prefix='sift_', prepend_pair=False),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)
 
def pipeline17():
    name_example = inspect.currentframe().f_code.co_name 
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    imgs_megadepth, gt_megadepth, to_add_path_megadepth = benchmark_setup(bench_path='../bench_data', dataset='megadepth')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(id_more='megadepth_fundamental', gt=gt_megadepth, to_add_path=to_add_path_megadepth, mode='fundamental'),
        pairwise_benchmark_module(id_more='megadepth_essential', gt=gt_megadepth, to_add_path=to_add_path_megadepth, mode='essential'),
    ]         
    imgs = [imgs_megadepth[i] for i in range(10)]
    run_pairs(pipeline, imgs, add_path=to_add_path_megadepth, db_name=os.path.join(OUTPUT_DIR, 'database.hdf5'))

def pipeline18():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    imgs_scannet, gt_scannet, to_add_path_scannet = benchmark_setup(bench_path='../bench_data', dataset='scannet')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(id_more='scannet_fundamental', gt=gt_scannet, to_add_path=to_add_path_scannet, mode='fundamental'),
        pairwise_benchmark_module(id_more='scannet_essential', gt=gt_scannet, to_add_path=to_add_path_scannet, mode='essential'),
    ]
    imgs = [imgs_scannet[i] for i in range(10)]
    run_pairs(pipeline, imgs, add_path=to_add_path_scannet, db_name=os.path.join(OUTPUT_DIR, 'database.hdf5'))

def pipeline19():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    imgs_imc, gt_imc, to_add_path_imc = benchmark_setup(bench_path='../bench_data', dataset='imc')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(id_more='megadepth_fundamental', gt=gt_imc, to_add_path=to_add_path_imc, mode='fundamental', metric=False),
        pairwise_benchmark_module(id_more='megadepth_fundamental_metric', gt=gt_imc, to_add_path=to_add_path_imc, mode='fundamental', metric=True),
        pairwise_benchmark_module(id_more='megadepth_essential', gt=gt_imc, to_add_path=to_add_path_imc, mode='essential', metric=False),
        pairwise_benchmark_module(id_more='megadepth_essential_metric', gt=gt_imc, to_add_path=to_add_path_imc, mode='essential', metric=True),
    ]         
    imgs = [imgs_imc[i] for i in range(10)]
    run_pairs(pipeline, imgs, add_path=to_add_path_imc, db_name=os.path.join(OUTPUT_DIR, 'database.hdf5'))

def pipeline20():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    imgs = '../data/ET'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, f'{name_example}_aliked.db')),
    ]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}_aliked.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)
    #
    pipeline = [
        deep_joined_module(what='superpoint'),
        lightglue_module(what='superpoint'),
        magsac_module(),
        show_matches_module(img_prefix='superpoint_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, f'{name_example}_superpoint.db')),
    ]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}_superpoint.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)
    #
    device = torch.device('cpu')
    merge_colmap_db([os.path.join(OUTPUT_DIR, f'{name_example}_aliked.db'), os.path.join(OUTPUT_DIR, f'{name_example}_superpoint.db')], os.path.join(OUTPUT_DIR, f'{name_example}_aliked_superpoint.db'), img_folder='../data/ET')

def pipeline21():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, f'database_{name_example}_aliked.db')),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)
    aliked_db = os.path.join(OUTPUT_DIR, f'database_{name_example}_aliked.db')
    aliked_models = os.path.join(OUTPUT_DIR, 'aliked_colmap_models')
    os.makedirs(aliked_models, exist_ok=True)
    pycolmap.incremental_mapping(database_path=aliked_db, image_path=imgs, output_path=aliked_models)
    filter_colmap_reconstruction(input_model_path=os.path.join(aliked_models, '0'), db_path=aliked_db, img_path=imgs, output_model_path=os.path.join(aliked_models, 'filtered_model'), to_filter=['et002.jpg', 'et005.jpg'], how_filter='exclude', only_cameras=False, add_3D_points=True)


def pipeline21bis():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    import os
    from pathlib import Path

    src_dir = Path(__file__).parent.parent
    base_dir = src_dir / OUTPUT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"__file__: {__file__}")
    print(f"base_dir: {base_dir}")
    print(f"CWD: {os.getcwd()}")
    print(f"aliked_colmap_models exists: {(base_dir / 'aliked_colmap_models').exists()}")
    print(f"aliked_colmap_models/0 exists: {(base_dir / 'aliked_colmap_models' / '0').exists()}")
    print(f"Contents of aliked_colmap_models: {list((base_dir / 'aliked_colmap_models').iterdir()) if (base_dir / 'aliked_colmap_models').exists() else 'DIR NOT FOUND'}")

    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=str(base_dir / f"{name_example}_aliked.db")),
    ]
    imgs = str(src_dir.parent / 'data' / 'ET')
    
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)
    os.makedirs(base_dir / 'aliked_colmap_models', exist_ok=True)    
      
    pycolmap.incremental_mapping(
        database_path=str(base_dir / f"{name_example}_aliked.db"),
        image_path=imgs,
        output_path=str(base_dir / 'aliked_colmap_models')
    )            
    
    filter_colmap_reconstruction(
        input_model_path=str(base_dir / 'aliked_colmap_models' / '0'),
        db_path=str(base_dir / f"{name_example}_aliked.db"),
        img_path=imgs,
        output_model_path=str(base_dir / 'aliked_colmap_models' / 'filtered_model'),
        to_filter=['et002.jpg', 'et005.jpg'],
        how_filter='exclude',
        only_cameras=False,
        add_3D_points=True
    )

def pipeline22():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    imgs = '../data/ET'
    aliked_db = os.path.join(OUTPUT_DIR, 'aliked.db')
    aliked_models = os.path.join(OUTPUT_DIR, 'aliked_colmap_models')
    superpoint_db = os.path.join(OUTPUT_DIR, 'superpoint.db')
    superpoint_models = os.path.join(OUTPUT_DIR, 'superpoint_colmap_models')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=aliked_db),
    ]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}_aliked.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)
    os.makedirs(aliked_models, exist_ok=True)
    pycolmap.incremental_mapping(database_path=aliked_db, image_path=imgs, output_path=aliked_models)
    filter_colmap_reconstruction(input_model_path=os.path.join(aliked_models, '0'), db_path=aliked_db, img_path=imgs, output_model_path=os.path.join(aliked_models, 'filtered_model'), to_filter=['et002.jpg', 'et005.jpg'], how_filter='exclude', only_cameras=False, add_3D_points=True)
    #
    pipeline = [
        deep_joined_module(what='superpoint'),
        lightglue_module(what='superpoint'),
        magsac_module(),
        show_matches_module(img_prefix='superpoint_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=superpoint_db),
    ]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}_superpoint.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)
    os.makedirs(superpoint_models, exist_ok=True)
    pycolmap.incremental_mapping(database_path=superpoint_db, image_path=imgs, output_path=superpoint_models)
    filter_colmap_reconstruction(input_model_path=os.path.join(superpoint_models, '0'), db_path=superpoint_db, img_path=imgs, output_model_path=os.path.join(superpoint_models, 'filtered_model'), to_filter=['et001.jpg', 'et002.jpg', 'et003.jpg', 'et004.jpg', 'et005.jpg'], how_filter='include', only_cameras=False, add_3D_points=True)
    #
    device = torch.device('cpu')
    align_colmap_models(model_path1=os.path.join(aliked_models, 'filtered_model'), model_path2=os.path.join(superpoint_models, 'filtered_model'), imgs_path=imgs, db_path0=aliked_db, db_path1=superpoint_db, output_db=os.path.join(OUTPUT_DIR, 'aliked_superpoint.db'), output_model=os.path.join(OUTPUT_DIR, 'merged_model'), th=None)

def pipeline23():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        deep_joined_module(),
        lightglue_module(),
        magsac_module(),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),            
        show_matches_module(mask_idx=[1], prepend_pair=False),
    ]
    imgs = '../data/ET'
    # no hdf5 cache with db_name=None
    run_pairs(pipeline, imgs, db_name=None)

def pipeline24():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    aliked_db = os.path.join(OUTPUT_DIR, 'aliked.db')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(id_more='1st', img_prefix='aliked_matches_1st_', mask_idx=[1], prepend_pair=False),
        to_colmap_module(db=aliked_db),
    ]
    # imgs = '../data/ET'
    # run_pairs(pipeline, imgs, colmap_db_or_list=['et000.jpg', 'et001.jpg', 'et003.jpg', 'et006.jpg', 'et007.jpg', 'et008.jpg'], mode='exclude')
    imgs = ['et000.jpg', 'et001.jpg', 'et003.jpg', 'et006.jpg', 'et007.jpg', 'et008.jpg']
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}_aliked1.hdf5")
    run_pairs(pipeline, imgs, add_path='../data/ET', db_name= name_db)
    # now the remaining mathing pairs only
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(id_more='2nd', img_prefix='aliked_matches_2nd_', mask_idx=[1], prepend_pair=False),
        to_colmap_module(db=aliked_db),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}_aliked2.hdf5")
    run_pairs(pipeline, imgs, colmap_db_or_list=aliked_db, mode='exclude', colmap_req='matches', db_name=name_db)

def pipeline25():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                deep_joined_module(what='aliked'),
                lightglue_module(what='aliked'),
            ],
            [
                deep_joined_module(what='superpoint'),
                lightglue_module(what='superpoint'),
            ],                
            [
                dog_module(),
                patch_module(),
                deep_descriptor_module(),
                smnn_module(),
            ],
    #      [
    #          roma_module(),
    #      ]
        ]),
        magsac_module(),            
        show_matches_module(img_prefix='union_', prepend_pair=False),  
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),                       
    ]    
    imgs = '../data/ET'
    run_pairs(pipeline, imgs, db_name=None)  

def pipeline26():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        deep_joined_module(),
        lightglue_module(),
        magsac_module(mode='homography_matrix'),
        show_homography_module(prepend_pair=False),
    ]
    imgs = '../data/graffiti'
    # no hdf5 cache with db_name=None
    run_pairs(pipeline, imgs, db_name=None)


def pipeline27():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    imgs_planar, gt_planar, to_add_path_planar = benchmark_setup(bench_path='../bench_data', dataset='planar')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(mode='homography', id_more='H_mode'),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_planar, to_add_path=to_add_path_planar, mode='homography'),
    ]         
    imgs = [imgs_planar[i] for i in range(20)]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, add_path=to_add_path_planar, force=True, db_name=name_db)   

def pipeline28():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    imgs_imc, gt_imc, to_add_path_imc = benchmark_setup(bench_path='../bench_data', dataset='imc')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_imc, to_add_path=to_add_path_imc, mode='epipolar'),
    ]         
    imgs = [imgs_imc[i] for i in range(10)]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, add_path=to_add_path_imc, db_name=name_db)   

def pipeline29():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    imgs_megadepth, gt_megadepth, to_add_path_megadepth = benchmark_setup(bench_path='../bench_data', dataset='megadepth')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_megadepth, to_add_path=to_add_path_megadepth, mode='epipolar'),
    ]         
    imgs = [imgs_megadepth[i] for i in range(10)]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, add_path=to_add_path_megadepth, db_name = name_db)   

def pipeline30():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    imgs_scannet, gt_scannet, to_add_path_scannet = benchmark_setup(bench_path='../bench_data', dataset='scannet')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_scannet, to_add_path=to_add_path_scannet, mode='epipolar'),
    ]         
    imgs = [imgs_scannet[i] for i in range(10)]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, add_path=to_add_path_scannet, db_name = name_db)   

def pipeline31():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0]),
        show_kpts_module(id_more='first', img_prefix='patches_', mask_idx=[1, 0], prepend_pair=True),
        mop_miho_ncc_module(),
        show_matches_module(id_more='second', img_prefix='matches_after_filter_', mask_idx=[1, 0]),
        show_kpts_module(id_more='second', img_prefix='patches_after_filter_', mask_idx=[1, 0], prepend_pair=True),
        show_patches_module(id_more='first', img_prefix='block_patches_', prepend_pair=True),
        magsac_module(),
        show_matches_module(id_more='third', img_prefix='matches_final_', mask_idx=[1, 0]),
        show_kpts_module(id_more='third', img_prefix='patches_after_final_', mask_idx=[1, 0], prepend_pair=True),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline32():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                dog_module(),
                patch_module(),
                deep_descriptor_module(),
                blob_matching_module(),   
            ],
            [
                hz_module(),
                patch_module(),
                deep_descriptor_module(),
                blob_matching_module(),                      
            ],
        ]),
        mop_miho_ncc_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_final_', mask_idx=[1]),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline33():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        mast3r_module(),
        magsac_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)  

def pipeline34():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")        
    pipeline = [
        dust3r_module(),
        magsac_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)          

def pipeline35():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    pipeline = [
        image_muxer_module(pair_generator=pair_pyramid, pipe_gather=pipe_union, pipeline=[
            pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
                [
                    dog_module(),
                    patch_module(),
                    deep_descriptor_module(),
                    blob_matching_module(),                    
#                     smnn_module(),      
                    show_matches_module(id_more='blob_show', img_prefix='matches_blob_', mask_idx=[1]),               
                ],
                [
                    hz_module(),
                    patch_module(),
                    deep_descriptor_module(),
                    blob_matching_module(),                    
#                     smnn_module(),                    
                    show_matches_module(id_more='hz_show', img_prefix='matches_hz_', mask_idx=[1]),               
                ],
            ]),
            dtm_module(),
            show_matches_module(id_more='pyramid_dtm_show', img_prefix='matches_dtm_', mask_idx=[1]),               
            sampling_module(),
            mop_miho_ncc_module(ncc=False),
            show_matches_module(id_more='pyramid_mop_show', img_prefix='matches_mop_', mask_idx=[1]),               
            magsac_module(),
            show_matches_module(id_more='pyramid_magasac_show', img_prefix='matches_magasac_', mask_idx=[1]),               
            mop_miho_ncc_module(ncc=False),
            show_matches_module(id_more='pyramid_final_show', img_prefix='matches_final_', mask_idx=[1]),               
        ]),
        dtm_module(),            
        sampling_module(),
        mop_miho_ncc_module(ncc=False),
        magsac_module(),
        show_matches_module(id_more='all_show', img_prefix='matches_', mask_idx=[1]),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline36():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    pipeline = [
        hz_module(),
        patch_module(),
        deep_descriptor_module(),
        blob_matching_module(),   
        show_matches_module(id_more='blob', img_prefix='matches_blob_', mask_idx=[1]),
        dtm_module(),
        show_matches_module(id_more='dtm', img_prefix='matches_dtm_', mask_idx=[1]),
        mop_miho_ncc_module(ncc=False),
        show_matches_module(id_more='mop', img_prefix='matches_mop_', mask_idx=[1]),
        magsac_module(),
        show_matches_module(id_more='magsac', img_prefix='matches_magsac_', mask_idx=[1]),
        dtm_module(guided_matching=True),
        show_matches_module(id_more='dtm_guided', img_prefix='matches_dtm_guided_', mask_idx=[1]),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline37():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
                [
                    dog_module(),
                    patch_module(),
                    deep_descriptor_module(),
#                     blob_matching_module(),                    
                    smnn_module(),      
                ],
                [
                    hz_module(),
                    patch_module(),
                    deep_descriptor_module(),
#                     blob_matching_module(),                    
                    smnn_module(),                    
                ],
            ]),
        dtm_module(),
        mop_miho_ncc_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1]),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline38():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                dog_module(),
                patch_module(),
                deep_descriptor_module(),
            ],
            [
                hz_module(),
                patch_module(),
                deep_descriptor_module(),
            ],
        ]),            
        image_muxer_module(pair_generator=pair_pyramid, pipe_gather=pipe_union, pipeline=[
            blob_matching_module(),                    
#             smnn_module(),      
            dtm_module(),
            mop_miho_ncc_module(ncc=False),
            show_matches_module(id_more='pyramid_show', img_prefix='pyramid_matches_', mask_idx=[1]),                
        ]),
        dtm_module(),
        mop_miho_ncc_module(ncc=False),
        magsac_module(),
        show_matches_module(id_more='all_show', img_prefix='all_matches_', mask_idx=[1]),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline39():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        romav2_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),
    ]    
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)  


def pipeline40(imgs='../data/ET'):
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    for pt_dir, chunk_idx in [('../data/ET_pt1', 0), ('../data/ET_pt2', 1)]:
        os.makedirs(pt_dir, exist_ok=True)
        for src in split_images('../data/ET', n_chunks=2, chunk_idx=chunk_idx):
            shutil.copy(src, os.path.join(pt_dir, os.path.basename(src)))

    start_time = time.time()

    imgs='../data/ET'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'ET_full.db')),
    ]
    run_pairs(pipeline, imgs, db_name=os.path.join(OUTPUT_DIR, 'database_ET_full.hdf5'))

    end_time = time.time()

    imgs='../data/ET_pt1'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'ET_pt1.db')),
    ]
    run_pairs(pipeline, imgs, db_name=os.path.join(OUTPUT_DIR, 'database_ET_pt1.hdf5'))

    start_time2 = time.time()

    imgs='../data/ET_pt2'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'ET_pt2.db')),
    ]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


    merge_colmap_db([os.path.join(OUTPUT_DIR, 'ET_pt1.db'), os.path.join(OUTPUT_DIR, 'ET_pt2.db')], os.path.join(OUTPUT_DIR, 'Merged_ET.db'), img_folder='../data/ET')

    end_time2 = time.time()

    print(f"Execution time full dataset: {end_time - start_time} seconds")
    print(f"Execution time incremental dataset: {end_time2 - start_time2} seconds")


def pipeline41():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        loftr_module(device='cpu'),
        show_kpts_module(id_more='first', prepend_pair=False),
        magsac_module(device='cuda'),
        show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline42():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    colmap_ab_db = os.path.join(OUTPUT_DIR, name_example + '_colmap_ab.db')
    pipeline_a = [
        image_muxer_module(pair_generator=pair_rot4, pipe_gather=pipe_max_matches,
            pipeline=[
                pipeline_muxer_module(
                    pipe_gather=pipe_union,
                    pipeline=[
                        [
                            loftr_module(),
                            show_kpts_module(id_more='1st', img_prefix='a_', prepend_pair=False),
                        ],
                        [
                            deep_joined_module(),
                            show_kpts_module(id_more='2nd', img_prefix='b_', prepend_pair=False),
                            lightglue_module(),
                        ],
                    ],
                ),
                magsac_module(),
                show_matches_module(id_more='1st', img_prefix='union_matches_', mask_idx=[1, 0], prepend_pair=False),
            ],
        ),
        show_kpts_module(id_more='3th', img_prefix='union_', prepend_pair=False),
        show_matches_module(id_more='2nd', img_prefix='best_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=colmap_ab_db),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline_a, imgs, db_name=os.path.join(OUTPUT_DIR, name_example + '_a.hdf5'))

    pipeline_b = [
        roma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=colmap_ab_db),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline_b, imgs, db_name=os.path.join(OUTPUT_DIR, name_example + '_b.hdf5'), colmap_db_or_list=colmap_ab_db, mode='include')


def pipeline43():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        loma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=os.path.join(OUTPUT_DIR, 'colmap.db')),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline44():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    name_db_chunk0 = os.path.join(OUTPUT_DIR, f'database_{name_example}_chunk0.hdf5')
    name_db_chunk1 = os.path.join(OUTPUT_DIR, f'database_{name_example}_chunk1.hdf5')
    name_db_merged = os.path.join(OUTPUT_DIR, f'database_{name_example}_merged.hdf5')

    for name_db in [name_db_chunk0, name_db_chunk1, name_db_merged]:
        if os.path.exists(name_db):
            os.remove(name_db)

    chunk0 = split_images(imgs_dir, n_chunks=2, chunk_idx=0)
    chunk1 = split_images(imgs_dir, n_chunks=2, chunk_idx=1)
    chunk0_names = {os.path.basename(p) for p in chunk0}
    chunk1_names = {os.path.basename(p) for p in chunk1}
    all_names = chunk0_names | chunk1_names

    print(f"chunk0 ({len(chunk0)} imgs): {sorted(chunk0_names)}")
    print(f"chunk1 ({len(chunk1)} imgs): {sorted(chunk1_names)}")

    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        magsac_module(),
    ]
    run_pairs(pipeline, chunk0, db_name=name_db_chunk0)

    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        magsac_module(),
    ]
    run_pairs(pipeline, chunk1, db_name=name_db_chunk1)

    n_intra0 = len(chunk0) * (len(chunk0) - 1) // 2
    n_intra1 = len(chunk1) * (len(chunk1) - 1) // 2

    with h5py.File(name_db_chunk0, 'r') as f:
        root = f['pickled']
        pairs0 = {(im0, im1) for im0 in root if im0 in all_names
                  for im1 in root[im0] if im1 in all_names and hasattr(root[im0][im1], 'keys')}
    assert len(pairs0) == n_intra0, f"Expected {n_intra0} pairs in chunk0 db, got {len(pairs0)}"
    assert all(im0 in chunk0_names and im1 in chunk0_names for im0, im1 in pairs0), \
        "Cross-chunk pair found in chunk0 db"

    with h5py.File(name_db_chunk1, 'r') as f:
        root = f['pickled']
        pairs1 = {(im0, im1) for im0 in root if im0 in all_names
                  for im1 in root[im0] if im1 in all_names and hasattr(root[im0][im1], 'keys')}
    assert len(pairs1) == n_intra1, f"Expected {n_intra1} pairs in chunk1 db, got {len(pairs1)}"
    assert all(im0 in chunk1_names and im1 in chunk1_names for im0, im1 in pairs1), \
        "Cross-chunk pair found in chunk1 db"
    print("  [OK] chunk runs produced only intra-chunk pairs")

    merge_hdf5([name_db_chunk0, name_db_chunk1], name_db_merged)

    with h5py.File(name_db_merged, 'r') as f:
        root = f['pickled']
        pairs_before = {(im0, im1) for im0 in root if im0 in all_names
                        for im1 in root[im0] if im1 in all_names and hasattr(root[im0][im1], 'keys')}
    assert len(pairs_before) == n_intra0 + n_intra1, \
        f"Expected {n_intra0 + n_intra1} pairs after merge, got {len(pairs_before)}"
    print(f"  [OK] merge produced {len(pairs_before)} intra-chunk pairs")

    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        magsac_module(),
    ]
    run_pairs(pipeline, imgs_dir, db_name=name_db_merged)

    with h5py.File(name_db_merged, 'r') as f:
        root = f['pickled']
        pairs_after = {(im0, im1) for im0 in root if im0 in all_names
                       for im1 in root[im0] if im1 in all_names and hasattr(root[im0][im1], 'keys')}
    n_total = len(all_names) * (len(all_names) - 1) // 2
    n_cross = len(chunk0) * len(chunk1)
    assert len(pairs_after) == n_total, \
        f"Expected {n_total} total pairs after full run, got {len(pairs_after)}"
    new_pairs = pairs_after - pairs_before
    assert len(new_pairs) == n_cross, \
        f"Expected {n_cross} new cross-chunk pairs, got {len(new_pairs)}"
    assert all(
        (im0 in chunk0_names and im1 in chunk1_names) or (im0 in chunk1_names and im1 in chunk0_names)
        for im0, im1 in new_pairs
    ), "Non-cross-chunk pair found in newly computed pairs"
    print(f"  [OK] {n_cross} cross-chunk pairs computed, {len(pairs_before)} intra-chunk pairs skipped")

    for name_db in [name_db_chunk0, name_db_chunk1, name_db_merged]:
        if os.path.exists(name_db):
            os.remove(name_db)

    print("pipeline44: ALL ASSERTIONS PASSED")


def pipeline45():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs = '../data/ET'
    colmap_db = os.path.join(OUTPUT_DIR, f'{name_example}_colmap.db')
    name_db = os.path.join(OUTPUT_DIR, f'database_{name_example}.hdf5')

    for f in [colmap_db, name_db]:
        if os.path.exists(f):
            os.remove(f)

    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        magsac_module(),
        to_colmap_module(db=colmap_db),
    ]
    run_pairs(pipeline, imgs, db_name=name_db)

    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        magsac_module(),
        to_colmap_module(db=colmap_db),
    ]

    start_incremental = time.time()
    run_pairs(pipeline, imgs, db_name=name_db, force=True)
    end_incremental = time.time()

    print(f"Execution time re-running on a fully-known dataset (all pairs skipped): {end_incremental - start_incremental} seconds")

    for f in [colmap_db, name_db]:
        if os.path.exists(f):
            os.remove(f)


def pipeline46():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    img_names = {f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

    pipeline = [salad_module()]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    if os.path.exists(name_db):
        os.remove(name_db)

    run_pairs(pipeline, imgs_dir, db_name=name_db)

    shapes = set()
    with h5py.File(name_db, 'r') as f:
        root = f['pickled']
        for name in img_names:
            assert name in root, f"No cached entry for {name}"
            assert 'salad' in root[name], f"No salad output for {name}"
            data = pickled_hdf5.pickled_hdf5.from_numpy(root[name]['salad']['data'][()])
            assert 'global_desc' in data, f"'global_desc' missing for {name}"
            shapes.add(tuple(data['global_desc'].shape))

    assert len(shapes) == 1, f"Inconsistent global_desc shapes across images: {shapes}"

    if os.path.exists(name_db):
        os.remove(name_db)

    print("pipeline46: ALL ASSERTIONS PASSED")


def pipeline47():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    img_names = sorted(f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    pipeline = [salad_module(), cosine_similarity_module()]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    if os.path.exists(name_db):
        os.remove(name_db)

    run_pairs(pipeline, imgs_dir, db_name=name_db)

    n_pairs = 0
    with h5py.File(name_db, 'r') as f:
        root = f['pickled']
        for i in range(len(img_names)):
            for j in range(i + 1, len(img_names)):
                im0, im1 = img_names[i], img_names[j]
                assert im0 in root and im1 in root[im0], f"No cached pair entry for ({im0}, {im1})"
                assert 'cosine_similarity' in root[im0][im1]['salad'], f"No cosine_similarity output for ({im0}, {im1})"
                data = pickled_hdf5.pickled_hdf5.from_numpy(root[im0][im1]['salad']['cosine_similarity']['data'][()])
                assert 'pair_sim' in data, f"'pair_sim' missing for ({im0}, {im1})"
                sim = data['pair_sim']
                assert -1.0 - 1e-4 <= sim <= 1.0 + 1e-4, f"pair_sim {sim} out of [-1, 1] for ({im0}, {im1})"
                n_pairs += 1

    n_expected = len(img_names) * (len(img_names) - 1) // 2
    assert n_pairs == n_expected, f"Expected {n_expected} pairs, checked {n_pairs}"

    if os.path.exists(name_db):
        os.remove(name_db)

    print("pipeline47: ALL ASSERTIONS PASSED")


def pipeline48():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    img_names = sorted(f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    pipeline = [salad_module(), l2_similarity_module()]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    if os.path.exists(name_db):
        os.remove(name_db)

    run_pairs(pipeline, imgs_dir, db_name=name_db)

    n_pairs = 0
    with h5py.File(name_db, 'r') as f:
        root = f['pickled']
        for i in range(len(img_names)):
            for j in range(i + 1, len(img_names)):
                im0, im1 = img_names[i], img_names[j]
                assert im0 in root and im1 in root[im0], f"No cached pair entry for ({im0}, {im1})"
                assert 'l2_similarity' in root[im0][im1]['salad'], f"No l2_similarity output for ({im0}, {im1})"
                data = pickled_hdf5.pickled_hdf5.from_numpy(root[im0][im1]['salad']['l2_similarity']['data'][()])
                assert 'pair_sim' in data, f"'pair_sim' missing for ({im0}, {im1})"
                sim = data['pair_sim']
                assert sim <= 1e-4, f"pair_sim {sim} should be <= 0 (negative L2 distance) for ({im0}, {im1})"
                n_pairs += 1

    n_expected = len(img_names) * (len(img_names) - 1) // 2
    assert n_pairs == n_expected, f"Expected {n_expected} pairs, checked {n_pairs}"

    if os.path.exists(name_db):
        os.remove(name_db)

    print("pipeline48: ALL ASSERTIONS PASSED")


def pipeline49():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    threshold = 0.5
    pairs_path = os.path.join(OUTPUT_DIR, f"{name_example}_pairs.hdf5")
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")

    for f in [pairs_path, name_db]:
        if os.path.exists(f):
            os.remove(f)

    pipeline = [salad_module(), cosine_similarity_module(), conf_module(threshold=threshold, out_path=pairs_path)]
    run_pairs(pipeline, imgs_dir, db_name=name_db)

    img_names = sorted(f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    expected = set()
    with h5py.File(name_db, 'r') as f:
        root = f['pickled']
        for i in range(len(img_names)):
            for j in range(i + 1, len(img_names)):
                im0, im1 = img_names[i], img_names[j]
                data = pickled_hdf5.pickled_hdf5.from_numpy(root[im0][im1]['salad']['cosine_similarity']['data'][()])
                sim = data['pair_sim']
                if sim > threshold:
                    expected.add((im0, im1))

    saved_pairs = conf_module.load_pairs(pairs_path)
    saved_names = {tuple(sorted((os.path.basename(a), os.path.basename(b)))) for a, b in saved_pairs}
    expected_names = {tuple(sorted(p)) for p in expected}

    assert saved_names == expected_names, \
        f"conf_module's saved pairs don't match threshold={threshold} applied to pair_sim"

    for f in [pairs_path, name_db]:
        if os.path.exists(f):
            os.remove(f)

    print("pipeline49: ALL ASSERTIONS PASSED")


def pipeline50():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    pairs_path_high = os.path.join(OUTPUT_DIR, f"{name_example}_pairs_high.hdf5")
    pairs_path_low = os.path.join(OUTPUT_DIR, f"{name_example}_pairs_low.hdf5")

    for f in [name_db, pairs_path_high, pairs_path_low]:
        if os.path.exists(f):
            os.remove(f)

    pipeline_high = [salad_module(), cosine_similarity_module(), conf_module(threshold=0.5, out_path=pairs_path_high)]
    run_pairs(pipeline_high, imgs_dir, db_name=name_db)
    pairs_high = conf_module.load_pairs(pairs_path_high)

    pipeline_low = [salad_module(), cosine_similarity_module(), conf_module(threshold=-1.0, out_path=pairs_path_low)]
    run_pairs(pipeline_low, imgs_dir, db_name=name_db)
    pairs_low = conf_module.load_pairs(pairs_path_low)

    img_names = sorted(f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    n_total = len(img_names) * (len(img_names) - 1) // 2

    assert len(pairs_low) == n_total, \
        f"threshold=-1.0 should accept every pair (cosine similarity >= -1), got {len(pairs_low)} of {n_total}"
    assert len(pairs_high) <= len(pairs_low), \
        "a stricter threshold must not accept more pairs than a looser one on a re-run"

    names_high = {tuple(sorted((os.path.basename(a), os.path.basename(b)))) for a, b in pairs_high}
    names_low = {tuple(sorted((os.path.basename(a), os.path.basename(b)))) for a, b in pairs_low}
    assert names_high <= names_low, \
        "pairs accepted at the high threshold must still be accepted at the low one"

    for f in [name_db, pairs_path_high, pairs_path_low]:
        if os.path.exists(f):
            os.remove(f)

    print("pipeline50: ALL ASSERTIONS PASSED")


def pipeline51():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    seg_mt = segformer_module(stage='matches')

    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        magsac_module(),
        seg_mt,
    ]
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    if os.path.exists(name_db):
        os.remove(name_db)

    run_pairs(pipeline, imgs_dir, db_name=name_db)

    db = pickled_hdf5.pickled_hdf5(name_db, mode='r')
    keys = db.get_keys()

    mt_keys = [k for k in keys if f'/{seg_mt.get_id()}/data' in k]
    assert len(mt_keys) > 0, "No cached entries for the matches-stage segformer module"

    for k in mt_keys:
        data, found = db.get(k)
        assert found, f"Missing cached entry for {k}"
        assert 'm_mask' in data, f"'m_mask' missing at {k}"
        m_mask = data['m_mask']
        assert m_mask.ndim == 2 and m_mask.shape[1] == 2, \
            f"'m_mask' expected shape [M, 2] (non-destructive, extra segmentation column), got {tuple(m_mask.shape)} at {k}"
        assert m_mask.dtype == torch.bool, f"'m_mask' should be boolean at {k}"

    db.close()

    if os.path.exists(name_db):
        os.remove(name_db)

    print("pipeline51: ALL ASSERTIONS PASSED")


def pipeline52(output_folder=OUTPUT_DIR):
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    imgs_dir = '../data/ET'
    kfc_pairs_path = str(output_path / f'{name_example}_kfc_pairs.hdf5')
    transitive_pairs_path = str(output_path / f'{name_example}_transitive_pairs.hdf5')
    name_db = str(output_path / f'database_{name_example}.hdf5')

    for f in [kfc_pairs_path, transitive_pairs_path, name_db]:
        if os.path.exists(f):
            os.remove(f)

    imgs = sorted(resolve_image_folder(imgs_dir))

    # pass 1: global similarity + KFC seed selection
    current_pairs = []
    global_pipeline = [
        salad_module(),
        cosine_similarity_module(),
        kfc_module(pairs=current_pairs, sim_db=name_db, out_path=kfc_pairs_path, n_mst=2),
    ]
    run_pairs(global_pipeline, imgs_dir, db_name=name_db)

    # pass 2: transitive closure over the seed pairs
    transitive = transitive_module(pairs=current_pairs, out_path=transitive_pairs_path)
    live = live_pair_graph(imgs, save_to=str(output_path / f'{name_example}.html'), worklist=transitive)

    match_pipeline = [salad_module(), cosine_similarity_module(), transitive, live]

    while current_pairs:
        run_pairs(match_pipeline, current_pairs, db_name=name_db)

    for f in [kfc_pairs_path, transitive_pairs_path, name_db]:
        if os.path.exists(f):
            os.remove(f)

    print(f"{name_example}: {transitive._graph.number_of_edges()} pairs confirmed "
          f"over {transitive._graph.number_of_nodes()} images")


def pipeline53(output_folder=OUTPUT_DIR):
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    imgs_dir = '../data/ET'
    name_db = str(output_path / f'database_{name_example}.hdf5')
    colmap_db = str(output_path / f'{name_example}.db')
    model_dir = str(output_path / f'{name_example}_model')

    for f in [name_db, colmap_db]:
        if os.path.exists(f):
            os.remove(f)

    pipeline = [
        dog_module(),
        sift_module(),
        smnn_module(),
        magsac_module(),
        to_colmap_module(db=colmap_db, only_matched=True),
        reconstruct_module(db=colmap_db, images=imgs_dir, output=model_dir),
    ]
    run_pairs(pipeline, imgs_dir, db_name=name_db)

    model = pycolmap.Reconstruction(model_dir)
    assert model.num_reg_images() >= 3, f"only {model.num_reg_images()} images registered"
    assert model.num_points3D() > 0

    print(f"{name_example}: {model.num_reg_images()} images registered, {model.num_points3D()} points")


def pipeline54(output_folder=OUTPUT_DIR):
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    imgs_dir = '../data/ET'
    kfc_pairs_path = str(output_path / f'{name_example}_kfc_pairs.hdf5')
    match_db = str(output_path / f'{name_example}.db')
    global_db = str(output_path / f'{name_example}_global.hdf5')
    model_dir = str(output_path / f'{name_example}_model')

    for f in [kfc_pairs_path, match_db]:
        if os.path.exists(f):
            os.remove(f)

    current_pairs = []
    global_pipeline = [
        salad_module(),
        cosine_similarity_module(),
        kfc_module(pairs=current_pairs, sim_db=global_db, out_path=kfc_pairs_path, n_mst=2),
    ]
    run_pairs(global_pipeline, imgs_dir, db_name=global_db)

    transitive = transitive_module(pairs=current_pairs, threshold=0, max_iterations=3)
    hierarchical = hierarchical_reconstruct_module(
        db=match_db, images=imgs_dir, output=model_dir, worklist=current_pairs,
        overlap_threshold=1.1, align_max_error=1.0, align_min_inlier_ratio=0.5,
    )

    match_pipeline = [
        dog_module(),
        sift_module(),
        smnn_module(),
        magsac_module(),
        transitive,
        to_colmap_module(db=match_db, worklist=current_pairs, no_unmatched=False, only_matched=True),
        hierarchical,
    ]

    while current_pairs:
        run_pairs(match_pipeline, current_pairs, db_name=str(output_path / f'{name_example}_match.hdf5'))

    models = [pycolmap.Reconstruction(str(Path(model_dir) / d)) for d in sorted(os.listdir(model_dir))]
    best = max(models, key=lambda m: m.num_reg_images())
    assert best.num_reg_images() >= 3, f"only {best.num_reg_images()} images registered"
    assert best.num_points3D() > 0

    print(f"{name_example}: {len(models)} model(s), best has {best.num_reg_images()} images, {best.num_points3D()} points")
    print(f"{name_example}: {hierarchical.n_models_built} models built, {hierarchical.n_merges} merges performed")


def pipeline55():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    colmap_db = os.path.join(OUTPUT_DIR, f"colmap_{name_example}.db")
    if os.path.exists(colmap_db):
        os.remove(colmap_db)
    pipeline = [
        vggt_omega_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=colmap_db),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline56():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    colmap_db = os.path.join(OUTPUT_DIR, f"colmap_{name_example}.db")
    if os.path.exists(colmap_db):
        os.remove(colmap_db)
    pipeline = [
        mapanything_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=colmap_db),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline57(output_folder=OUTPUT_DIR, n_datasets=None):
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    output_path = Path(output_folder) / name_example
    shutil.rmtree(output_path, ignore_errors=True)
    imgs_dir = output_path / 'imgs'
    imgs_dir.mkdir(parents=True)

    sources = {}
    for d in ['kermits/another_kermit', 'ETs/dioscuri', 'ETs/another_ET']:
        src_dir = Path('../data/imc2025/kaggle_raw_data') / d / 'images'
        for img in os.listdir(src_dir):
            os.symlink((src_dir / img).resolve(), imgs_dir / img)
            sources[img] = d

    global_db = str(output_path / 'global_sim.hdf5')

    datasets = []
    allowed = set()
    global_pipeline = [
        salad_module(),
        cosine_similarity_module(),
        kfc_module(datasets=datasets, allowed=allowed, sim_db=global_db, out_path=None, n_mst=2, multi_dataset=True, cut_ratio=1.0, n_datasets=n_datasets, sim_quantile=0.75),
    ]
    run_pairs(global_pipeline, str(imgs_dir), db_name=global_db)

    for i, current_pairs in enumerate(datasets):
        match_db = str(output_path / f'dataset_{i}.db')

        transitive = transitive_module(
            pairs=current_pairs,
            allowed=allowed,
            threshold=3,
            max_iterations=3,
            out_path=str(output_path / f'dataset_{i}_transitive_pairs.hdf5'),
        )

        hierarchical = hierarchical_reconstruct_module(
            db=match_db,
            images=str(imgs_dir),
            output=str(output_path / f'dataset_{i}_model'),
            worklist=current_pairs,
        )

        match_pipeline = [
            pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
                [
                    deep_joined_module(what='aliked'),
                    lightglue_module(what='aliked'),
                ],
                [
                    deep_joined_module(what='superpoint'),
                    lightglue_module(what='superpoint'),
                ],
                [
                    dog_module(),
                    patch_module(),
                    deep_descriptor_module(),
                    smnn_module(),
                ],
            ]),
            magsac_module(),
            transitive,
            to_colmap_module(db=match_db, worklist=current_pairs, no_unmatched=False, only_matched=True),
            hierarchical,
        ]

        while current_pairs:
            run_pairs(match_pipeline, current_pairs, db_name=str(output_path / 'match.hdf5'))

    model_dirs = sorted(output_path.glob('dataset_*_model'))
    assert len(model_dirs) == 3, f"{len(model_dirs)} datasets found instead of 3"

    found = []
    for model_dir in model_dirs:
        names = set()
        for d in os.listdir(model_dir):
            model = pycolmap.Reconstruction(str(model_dir / d))
            names |= {im.name for im in model.images.values() if im.has_pose}
        dataset_sources = {sources[n] for n in names}
        assert len(dataset_sources) == 1, f"{model_dir.name} mixes {dataset_sources}"
        found.append(dataset_sources.pop())
        print(f"{name_example}: {model_dir.name} -> {found[-1]}, {len(names)} images registered")

    assert len(set(found)) == 3, f"datasets not separated: {found}"


def pipeline58(output_folder=OUTPUT_DIR):
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    output_path = Path(output_folder) / name_example
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True)

    imgs = '../data/ET_random_rotated'
    colmap_db = str(output_path / 'colmap.db')
    fast_model = str(output_path / 'fast_model')
    full_model = str(output_path / 'full_model')

    fast_pipeline = [
        deep_joined_module(what='superpoint', num_features=512, resize=512),
        lightglue_module(what='superpoint'),
        magsac_module(),
        to_colmap_module(db=colmap_db),
        reconstruct_module(db=colmap_db, images=imgs, output=fast_model),
    ]
    run_pairs(fast_pipeline, imgs, db_name=str(output_path / 'fast.hdf5'))

    fast = pycolmap.Reconstruction(fast_model)
    print(f"{name_example}: {fast.num_reg_images()} images registered after the fast pass")

    robust_pipeline = [
        image_muxer_module(pair_generator=pair_rot4, pipe_gather=pipe_max_matches, pipeline=[
            romav2_module(),
            magsac_module(),
        ]),
        to_colmap_module(db=colmap_db),
        reconstruct_module(db=colmap_db, images=imgs, output=full_model),
    ]
    run_pairs(robust_pipeline, imgs, db_name=str(output_path / 'robust.hdf5'), colmap_db_or_list=fast_model, mode='include', colmap_has_pose=False)

    full = pycolmap.Reconstruction(full_model)
    print(f"{name_example}: {full.num_reg_images()} images registered after the robust pass")
    assert full.num_reg_images() >= fast.num_reg_images()


def pipeline59():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    matcher = smnn_extended_module()
    pipeline = [
        dog_module(),
        pipeline_muxer_module(pipe_gather=partial(pipe_sm_product, th=[None, None, 0.5]), add_to_cache=False, pipeline=[
            [
                sift_module(),
                similarity_matrix_module(metric='l2'),
            ],
            [
                dino_descriptor_module(),
                similarity_matrix_module(metric='cosine'),
            ],
            [
                segformer_module(stage='descriptor'),
                similarity_matrix_module(metric='equal'),
            ],
        ]),
        matcher,
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    if os.path.exists(name_db):
        os.remove(name_db)

    run_pairs(pipeline, imgs, db_name=name_db)

    db = pickled_hdf5.pickled_hdf5(name_db, mode='r')
    keys = [k for k in db.get_keys() if f'/{matcher.get_id()}/data' in k]
    assert len(keys) > 0, "No cached entries for smnn_extended_module"
    n_matches = [db.get(k)[0]['m_idx'].shape[0] for k in keys]
    db.close()

    assert max(n_matches) > 0, "smnn_extended_module found no matches on any pair"
    print(f"{name_example}: {sum(n_matches)} matches over {len(keys)} pairs")
    print(f"{name_example}: ALL ASSERTIONS PASSED")


def pipeline60():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    descriptor = dino_descriptor_module()
    pipeline = [
        dog_module(),
        descriptor,
        smnn_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    if os.path.exists(name_db):
        os.remove(name_db)

    run_pairs(pipeline, imgs, db_name=name_db)

    db = pickled_hdf5.pickled_hdf5(name_db, mode='r')
    keys = [k for k in db.get_keys() if f'/{descriptor.get_id()}/data' in k]
    assert len(keys) > 0, "No cached entries for dino_descriptor_module"

    for k in keys:
        desc = db.get(k)[0]['desc']
        assert desc.ndim == 2, f"'desc' should be [N, C], got {tuple(desc.shape)} at {k}"
        assert torch.allclose(desc.norm(dim=1), torch.ones(desc.shape[0], device=desc.device), atol=1e-3), \
            f"'desc' should be L2-normalized at {k}"

    db.close()
    print(f"{name_example}: ALL ASSERTIONS PASSED")


def pipeline61():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    dino_filter = dino_filter_module()
    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        magsac_module(),
        dino_filter,
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    if os.path.exists(name_db):
        os.remove(name_db)

    run_pairs(pipeline, imgs, db_name=name_db)

    db = pickled_hdf5.pickled_hdf5(name_db, mode='r')
    keys = [k for k in db.get_keys() if f'/{dino_filter.get_id()}/data' in k]
    assert len(keys) > 0, "No cached entries for dino_filter_module"

    for k in keys:
        m_mask = db.get(k)[0]['m_mask']
        assert m_mask.ndim == 2 and m_mask.shape[1] == 2, \
            f"'m_mask' expected shape [M, 2] (non-destructive, extra DINO column), got {tuple(m_mask.shape)} at {k}"
        assert m_mask.dtype == torch.bool, f"'m_mask' should be boolean at {k}"

    db.close()
    print(f"{name_example}: ALL ASSERTIONS PASSED")

def pipeline62(output_folder=OUTPUT_DIR):
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    name_db = os.path.join(output_folder, f"database_{name_example}.hdf5")

    for cache_size in [0, 1, 1000]:
        if os.path.exists(name_db):
            os.remove(name_db)

        db = pickled_hdf5.pickled_hdf5(name_db, cache_size=cache_size)
        assert db.get_keys() == []

        db.add('/a/x', 1)
        db.add('/a/y', 2)
        db.add('/b', 3)
        assert sorted(db.get_keys()) == ['/a/x', '/a/y', '/b'], f"cache_size={cache_size}: {db.get_keys()}"

        db.add('/b', 4)
        assert sorted(db.get_keys()) == ['/a/x', '/a/y', '/b'], f"cache_size={cache_size}: {db.get_keys()}"

        db.flush()
        db.add('/c', 5)
        db.remove('/a/x')
        db.remove('/c')
        assert sorted(db.get_keys()) == ['/a/y', '/b'], f"cache_size={cache_size}: {db.get_keys()}"
        assert db.get('/b') == (4, True)

        db.add('/p', 6, force_no_cache=True)
        assert 'pickled/p' in db.get_hdf5(), f"cache_size={cache_size}: force_no_cache entry not written to the file"
        assert 'pickled/p' not in db._cache, f"cache_size={cache_size}: force_no_cache entry kept in the cache"
        assert db.get('/p', force_no_cache=True) == (6, True)
        assert 'pickled/p' not in db._cache, f"cache_size={cache_size}: force_no_cache read kept in the cache"
        assert sorted(db.get_keys()) == ['/a/y', '/b', '/p'], f"cache_size={cache_size}: {db.get_keys()}"
        db.remove('/p')

        db.add('/d', [1, 2])
        data, _ = db.get('/d')
        data.append(3)
        assert db.get('/d') == ([1, 2], True), f"cache_size={cache_size}: modifying a returned object altered the cache"
        db.remove('/d')
        db.close()

        db = pickled_hdf5.pickled_hdf5(name_db, mode='r', cache_size=cache_size)
        assert sorted(db.get_keys()) == ['/a/y', '/b'], f"cache_size={cache_size}: {db.get_keys()}"
        assert db.get('/a/x') == (None, False)
        db.close()

    os.remove(name_db)
    print(f"{name_example}: ALL ASSERTIONS PASSED")


def pipeline63():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs = '../data/ET'
    results = {}
    for parallel in [True, False]:
        muxer = pipeline_muxer_module(pipe_gather=pipe_union, parallel=parallel, pipeline=[
            [
                deep_joined_module(what='aliked'),
                lightglue_module(what='aliked'),
            ],
            [
                deep_joined_module(what='superpoint'),
                lightglue_module(what='superpoint'),
            ],
            [
                dog_module(),
                patch_module(),
                deep_descriptor_module(),
                smnn_module(),
            ],
        ])
        name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}_{'parallel' if parallel else 'sequential'}.hdf5")
        if os.path.exists(name_db):
            os.remove(name_db)

        start_time = time.time()
        run_pairs([muxer], imgs, db_name=name_db)
        print(f"{name_example}: parallel={parallel} took {time.time() - start_time:.1f} s")

        db = pickled_hdf5.pickled_hdf5(name_db, mode='r')
        keys = [k for k in db.get_keys() if f'/{muxer.get_id()}/data' in k]
        results[parallel] = {}
        for k in keys:
            data = db.get(k)[0]
            m_idx = data['m_idx'].cpu().long()
            pts = torch.cat([data['kp'][0].cpu()[m_idx[:, 0]], data['kp'][1].cpu()[m_idx[:, 1]]], dim=1)
            results[parallel][k] = {tuple(p) for p in torch.round(pts * 2).int().tolist()}
        db.close()
        os.remove(name_db)

    assert len(results[True]) > 0, "No cached entries for the muxer"
    assert results[True].keys() == results[False].keys(), "parallel and sequential runs processed different pairs"

    overlaps = {}
    for k in results[True]:
        a, b = results[True][k], results[False][k]
        overlaps[k] = len(a & b) / max(len(a | b), 1)
    worst = min(overlaps, key=overlaps.get)
    print(f"{name_example}: matches shared by the parallel and sequential runs: "
          f"mean {sum(overlaps.values()) / len(overlaps):.4f}, min {overlaps[worst]:.4f} at {worst}")
    assert overlaps[worst] >= 0.99, f"parallel and sequential matches differ at {worst} ({overlaps[worst]:.3f} shared)"

    print(f"{name_example}: ALL ASSERTIONS PASSED")


def pipeline64():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    torch.manual_seed(0)
    centers = torch.nn.functional.normalize(torch.randn(3, 32), dim=1)
    descs = {}
    for c in range(3):
        for i in range(10):
            descs[f'/x/scene{c}_{i:02d}.jpg'] = centers[c] + 0.1 * torch.randn(32)

    imgs = sorted(descs)
    table = {}
    for i, a in enumerate(imgs):
        for b in imgs[i + 1:]:
            table[(a, b)] = torch.nn.functional.cosine_similarity(descs[a], descs[b], dim=0).item()

    def scenes(groups):
        return sorted(sorted({os.path.basename(img).split('_')[0] for img in g}) for g in groups)

    expected = [['scene0'], ['scene1'], ['scene2']]

    groups = kfc_module.split_datasets(table, cut_ratio=1.0, sep_ratio=0.8, min_size=3)
    assert scenes(groups) == expected, f"automatic split: {scenes(groups)}"

    groups = kfc_module.split_datasets(table, min_size=3, n_datasets=3)
    assert scenes(groups) == expected, f"split with n_datasets=3: {scenes(groups)}"

    datasets = []
    pairs = []
    kfc = kfc_module(pairs=pairs, datasets=datasets, sim_db=None, out_path=None, n_mst=2, multi_dataset=True, cut_ratio=1.0)
    for (a, b), sim in table.items():
        kfc.run(img=[a, b], pair_sim=sim)
    kfc.finalize()

    assert len(datasets) == 3, f"{len(datasets)} datasets instead of 3"
    for d in datasets:
        assert len(d) > 0, "empty seed pair list"
        assert len({os.path.basename(img).split('_')[0] for pair in d for img in pair}) == 1, "seed pairs cross datasets"
    assert len(pairs) == sum(len(d) for d in datasets), "'pairs' should hold all the datasets' pairs"

    print(f"{name_example}: ALL ASSERTIONS PASSED")


def pipeline65():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    torch.manual_seed(0)
    d0 = torch.randn(50, 16)
    perm = torch.randperm(50)
    d1 = d0[perm] + 0.01 * torch.randn(50, 16)
    inv = torch.argsort(perm)

    for metric in ['cosine', 'l2', 'l1']:
        sm = similarity_matrix_module(metric=metric).run(desc=[d0, d1])['sm'].cpu()
        assert sm.shape == (50, 50), f"{metric}: shape {tuple(sm.shape)}"
        assert sm.min() >= -1e-6, f"{metric}: negative values"
        assert torch.equal(sm.argmin(dim=1), inv), f"{metric}: the closest descriptor is not the matching one"

    sm = similarity_matrix_module(metric='cosine').run(desc=[d0, d1])['sm']
    assert sm.max() <= 1 + 1e-6, "cosine: values above 1"

    c0 = torch.randint(0, 3, (50, 1)).float()
    c1 = torch.randint(0, 3, (40, 1)).float()
    sm_eq = similarity_matrix_module(metric='equal').run(desc=[c0, c1])['sm'].cpu()
    assert torch.equal(sm_eq, (c0 != c1.T).float()), "'equal' should be 0 for equal descriptors and 1 otherwise"

    a = torch.rand(5, 7)
    b = torch.rand(5, 7)
    assert torch.allclose(pipe_sm_product([{'sm': a}, {'sm': b}])['sm'], a * b), "pipe_sm_product should multiply the matrices"
    assert torch.equal(partial(pipe_sm_product, th=[0.5, None])([{'sm': a}, {'sm': b}])['sm'], b.masked_fill(a >= 0.5, torch.inf)), \
        "pipe_sm_product should keep the pairs below the threshold and set the others to infinity"

    m_idx = smnn_extended_module().run(sm=sm)['m_idx'].cpu().long()
    assert m_idx.shape[0] >= 45, f"only {m_idx.shape[0]} of 50 matches found"
    assert torch.equal(inv[m_idx[:, 0]], m_idx[:, 1]), "smnn_extended_module returned wrong matches"

    c0 = torch.randint(0, 3, (50, 1)).float()
    c0[0] = 99
    c1 = c0[perm]
    c1[inv[0]] = 98
    sm_eq = similarity_matrix_module(metric='equal').run(desc=[c0, c1])['sm']
    sm_masked = partial(pipe_sm_product, th=[None, 0.5])([{'sm': sm}, {'sm': sm_eq}])['sm']
    for th in [0.95, 1.0]:
        m_idx = smnn_extended_module(th=th).run(sm=sm_masked)['m_idx'].cpu().long()
        assert m_idx.shape[0] > 0, f"th={th}: no matches found on the masked matrix"
        assert torch.isfinite(sm_masked.cpu()[m_idx[:, 0], m_idx[:, 1]]).all(), f"th={th}: a discarded pair was matched"
        assert 0 not in m_idx[:, 0], f"th={th}: a keypoint with no allowed candidate was matched"
        assert torch.equal(inv[m_idx[:, 0]], m_idx[:, 1]), f"th={th}: wrong matches on the masked matrix"

    print(f"{name_example}: ALL ASSERTIONS PASSED")


def pipeline66():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0]),
        oanet_module(),
        show_matches_module(id_more='second', img_prefix='matches_after_filter_', mask_idx=[1, 0]),
        magsac_module(),
        show_matches_module(id_more='third', img_prefix='matches_final_', mask_idx=[1, 0]),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline67():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    dog_matcher = pipeline_muxer_module(pipe_gather=pipe_identity, pipeline=[
        [
            dog_module(),
            patch_module(),
            deep_descriptor_module(),
            smnn_module(),
            magsac_module(),
        ],
    ])
    pipeline = [
        dog_matcher,
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0]),
    ]
    imgs = '../data/ET'
    name_db = os.path.join(OUTPUT_DIR, f"database_{name_example}.hdf5")
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline_ssma_transitive(
    images_folder='/home/colombo/Shared/test_kornia',
    output_folder='.',
    n_mst=2,
    max_iterations=10,
    min_matches=0,
    sim_quantile=0.7,
    sim_min=None,
):
    """
    Full split-SSMA run, SIFT global-similarity variant: two passes over
    `images_folder`, back to back.

    Pass 1 - global similarity. Every pair is compared: both images
    downscaled so their longest side is <= 255 px (image_muxer +
    pair_resize's default `max_len`), SIFT + MAGSAC, inlier count stored as
    'pair_sim'. This is
    exhaustive (N*(N-1)/2 pairs) - accurate but heavy on large sets; use
    pipeline_ssma_transitive_salad for a cheap global-descriptor pass 1
    instead. kfc_module records the whole similarity table and, at finalize,
    picks the seed pairs as `n_mst` successive maximum spanning trees.

    Pass 2 - transitive closure, inlined below. `sim_quantile`
    (0..1) / `sim_min` make kfc_module keep only the strongest pairs in
    `allowed`, so weak transitive candidates are never matched.
    """
    print("\n \n")
    print("=" * 50)
    print("Running: pipeline_ssma_transitive")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    kfc_pairs_path = str(output_path / 'ssma_kfc_pairs.hdf5')

    # --- pass 1: SIFT global similarity + KFC seed selection ---
    # Only the per-image detection (dog/sift on the downscaled copies) and the
    # final per-pair 'pair_sim' scalar are cached. The per-pair match arrays
    # (smnn/magsac) are NOT: on a large set they blow the cache up to tens of
    # GB for data that is only used to produce one number per pair. The run
    # still resumes at pair granularity from the cached 'pair_sim'.
    current_pairs = []
    allowed = set()
    global_db = str(output_path / 'ssma_global_sim.hdf5')
    global_pipeline = [
        image_muxer_module(
            # to override pair_resize's defaults (e.g. max_len), wrap it with
            # functools.partial instead, since image_muxer_module always calls
            # pair_generator with just (pair, cache_path, force, pipe_data):
            # pair_generator=partial(pair_resize, max_len=128),
            pair_generator=pair_resize,
            pipe_gather=pipe_max_matches,
            cache_path=str(output_path / 'ssma_resized'),
            add_to_cache=False,
            pipeline=[
                dog_module(),
                sift_module(),
                smnn_module(add_to_cache=False),
                magsac_module(add_to_cache=False),
            ],
        ),
        n_matches_similarity_module(),
        kfc_module(pairs=current_pairs, allowed=allowed, sim_db=global_db, out_path=kfc_pairs_path, n_mst=n_mst, sim_quantile=sim_quantile, sim_min=sim_min),
    ]
    run_pairs(global_pipeline, images_folder, db_name=global_db)
    # current_pairs is already filled in place by kfc_module - no file read needed

    # --- pass 2: transitive closure ---
    # Real SIFT + MAGSAC + COLMAP matching pipeline, one run_pairs() call per
    # round over current_pairs; transitive.finalize() then drops the pairs
    # just matched and appends the next round's candidates, emptying it once
    # `max_iterations` is reached, which ends the while loop below.
    # `sim_min` / `sim_quantile` gate which transitive candidates are queued
    # at all: kfc_module fills `allowed` with the pairs above the cut. `min_matches`
    # gates confirmation on the pass-2 match result.
    transitive_pairs_path = str(output_path / 'ssma_transitive_pairs.hdf5')
    match_db = str(output_path / 'ssma_transitive.db')

    imgs = sorted(resolve_image_folder(images_folder))

    print(f"pipeline_ssma_transitive: {len(current_pairs)} seed pairs selected")

    if not current_pairs:
        print("pipeline_ssma_transitive: no seed pairs, nothing to match")
        return

    transitive = transitive_module(
        pairs=current_pairs,
        allowed=allowed,
        threshold=min_matches,
        max_iterations=max_iterations,
        out_path=transitive_pairs_path,
    )
    live = live_pair_graph(imgs, save_to=str(output_path / 'ssma_transitive_graph.html'), worklist=transitive)

    match_pipeline = [
        dog_module(),
        sift_module(),
        smnn_module(),
        magsac_module(),
        transitive,
        to_colmap_module(db=match_db, worklist=current_pairs, no_unmatched=False),
        reconstruct_module(db=match_db, images=images_folder, output=str(output_path / 'ssma_model'), worklist=current_pairs),
        live,
    ]

    while current_pairs:
        run_pairs(match_pipeline, current_pairs, db_name=str(output_path / 'ssma_transitive.hdf5'))

    print(f"pipeline_ssma_transitive: {transitive._graph.number_of_edges()} pairs matched via transitive closure")


def pipeline_ssma_transitive_salad(
    images_folder='/home/colombo/Shared/subset_ssma',
    output_folder='.',
    n_mst=2,
    max_iterations=3,
    min_matches=3,
    sim_quantile=0.0,
    sim_min=None,
):
    """
    Full split-SSMA run, SALAD global-similarity variant.

    Same as pipeline_ssma_transitive but pass 1 scores every pair with a
    single SALAD global descriptor per image + cosine similarity, instead of
    a downscaled SIFT + MAGSAC match. Still exhaustive (N*(N-1)/2 pairs) but
    only one descriptor per image and a dot product per pair, so it scales to
    large image sets. kfc_module then builds the same table + `n_mst`-MST
    seed selection.

    Pass 2 - transitive closure, inlined below. `sim_quantile`
    (0..1) / `sim_min` make kfc_module keep only the strongest pairs in
    `allowed`, so weak transitive candidates are never matched.
    """
    print("\n \n")
    print("=" * 50)
    print("Running: pipeline_ssma_transitive_salad")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    kfc_pairs_path = str(output_path / 'ssma_kfc_pairs.hdf5')

    # --- pass 1: SALAD global-descriptor similarity + KFC seed selection ---
    current_pairs = []
    allowed = set()
    global_db = str(output_path / 'ssma_global_sim.hdf5')
    global_pipeline = [
        salad_module(),
        cosine_similarity_module(),
        kfc_module(pairs=current_pairs, allowed=allowed, sim_db=global_db, out_path=kfc_pairs_path, n_mst=n_mst, sim_quantile=sim_quantile, sim_min=sim_min),
    ]
    run_pairs(global_pipeline, images_folder, db_name=global_db)
    # current_pairs is already filled in place by kfc_module - no file read needed

    # --- pass 2: transitive closure ---
    # Real SIFT + MAGSAC + COLMAP matching pipeline, one run_pairs() call per
    # round over current_pairs; transitive.finalize() then drops the pairs
    # just matched and appends the next round's candidates, emptying it once
    # `max_iterations` is reached, which ends the while loop below.
    # `sim_min` / `sim_quantile` gate which transitive candidates are queued
    # at all: kfc_module fills `allowed` with the pairs above the cut. `min_matches`
    # gates confirmation on the pass-2 match result.
    transitive_pairs_path = str(output_path / 'ssma_transitive_pairs.hdf5')
    match_db = str(output_path / 'ssma_transitive.db')

    imgs = sorted(resolve_image_folder(images_folder))

    print(f"pipeline_ssma_transitive_salad: {len(current_pairs)} seed pairs selected")

    if not current_pairs:
        print("pipeline_ssma_transitive_salad: no seed pairs, nothing to match")
        return

    transitive = transitive_module(
        pairs=current_pairs,
        allowed=allowed,
        threshold=min_matches,
        max_iterations=max_iterations,
        out_path=transitive_pairs_path,
    )
    live = live_pair_graph(imgs, save_to=str(output_path / 'ssma_transitive_graph.html'), worklist=transitive)

    match_pipeline = [
        dog_module(),
        sift_module(),
        smnn_module(),
        magsac_module(),
        transitive,
        to_colmap_module(db=match_db, worklist=current_pairs, no_unmatched=False),
        reconstruct_module(db=match_db, images=images_folder, output=str(output_path / 'ssma_model'), worklist=current_pairs),
        live,
    ]

    while current_pairs:
        run_pairs(match_pipeline, current_pairs, db_name=str(output_path / 'ssma_transitive.hdf5'))

    print(f"pipeline_ssma_transitive_salad: {transitive._graph.number_of_edges()} pairs matched via transitive closure")


def full_pipeline_ssma(
    images_folder='/home/colombo/Shared/imgs',
    output_folder='/media/dati/full_ssma',
    n_mst=2,
    max_iterations=3,
    min_matches=3,
    sim_quantile=0.75,
    sim_min=None,
):
    print("\n \n")
    print("=" * 50)
    print("Running: full_pipeline_ssma")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    transitive_pairs_path = str(output_path / 'full_ssma_transitive_pairs.hdf5')
    match_db = str(output_path / 'full_ssma_transitive.db')

    current_pairs = []
    allowed = set()
    global_db = str(output_path / 'full_ssma_global_sim.hdf5')
    global_pipeline = [
        salad_module(),
        cosine_similarity_module(),
        kfc_module(pairs=current_pairs, allowed=allowed, sim_db=global_db, out_path=None, n_mst=n_mst, sim_quantile=sim_quantile, sim_min=sim_min),
    ]
    run_pairs(global_pipeline, images_folder, db_name=global_db)

    print(f"full_pipeline_ssma: {len(current_pairs)} seed pairs selected")

    if not current_pairs:
        print("full_pipeline_ssma: no seed pairs, nothing to match")
        return

    transitive = transitive_module(
        pairs=current_pairs,
        allowed=allowed,
        threshold=min_matches,
        max_iterations=max_iterations,
        out_path=transitive_pairs_path,
    )

    match_pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                deep_joined_module(what='aliked'),
                lightglue_module(what='aliked'),
            ],
            [
                deep_joined_module(what='superpoint'),
                lightglue_module(what='superpoint'),
            ],
            [
                dog_module(),
                patch_module(),
                deep_descriptor_module(),
                smnn_module(),
            ],
        ]),
        magsac_module(),
        transitive,
        segformer_module(stage='matches'),
        to_colmap_module(db=match_db, worklist=current_pairs, no_unmatched=False, only_matched=True),
        reconstruct_module(db=match_db, images=images_folder, output=str(output_path / 'full_ssma_model'), worklist=current_pairs),
    ]

    while current_pairs:
        run_pairs(match_pipeline, current_pairs, db_name=str(output_path / 'full_ssma_transitive.hdf5'))

    print(f"full_pipeline_ssma: {transitive._graph.number_of_edges()} pairs matched via transitive closure")


def full_pipeline_ssma_hierarchical(
    images_folder='/home/colombo/Shared/imgs',
    output_folder='/media/dati/full_ssma_hierarchical',
    n_mst=2,
    max_iterations=3,
    min_matches=3,
    sim_quantile=0.75,
    sim_min=None,
):
    print("\n \n")
    print("=" * 50)
    print("Running: full_pipeline_ssma_hierarchical")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    transitive_pairs_path = str(output_path / 'full_ssma_hierarchical_transitive_pairs.hdf5')
    match_db = str(output_path / 'full_ssma_hierarchical_transitive.db')

    current_pairs = []
    allowed = set()
    global_db = str(output_path / 'full_ssma_hierarchical_global_sim.hdf5')
    global_pipeline = [
        salad_module(),
        cosine_similarity_module(),
        kfc_module(pairs=current_pairs, allowed=allowed, sim_db=global_db, out_path=None, n_mst=n_mst, sim_quantile=sim_quantile, sim_min=sim_min),
    ]
    run_pairs(global_pipeline, images_folder, db_name=global_db)

    print(f"full_pipeline_ssma_hierarchical: {len(current_pairs)} seed pairs selected")

    if not current_pairs:
        print("full_pipeline_ssma_hierarchical: no seed pairs, nothing to match")
        return

    transitive = transitive_module(
        pairs=current_pairs,
        allowed=allowed,
        threshold=min_matches,
        max_iterations=max_iterations,
        out_path=transitive_pairs_path,
    )

    hierarchical = hierarchical_reconstruct_module(
        db=match_db,
        images=images_folder,
        output=str(output_path / 'full_ssma_hierarchical_model'),
        worklist=current_pairs,
    )

    match_pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                deep_joined_module(what='aliked'),
                lightglue_module(what='aliked'),
            ],
            [
                deep_joined_module(what='superpoint'),
                lightglue_module(what='superpoint'),
            ],
            [
                dog_module(),
                patch_module(),
                deep_descriptor_module(),
                smnn_module(),
            ],
        ]),
        segformer_module(stage='matches'),
        magsac_module(),
        transitive,
        to_colmap_module(db=match_db, worklist=current_pairs, no_unmatched=False, only_matched=True),
        hierarchical,
    ]

    while current_pairs:
        run_pairs(match_pipeline, current_pairs, db_name=str(output_path / 'full_ssma_hierarchical_transitive.hdf5'))

    print(f"full_pipeline_ssma_hierarchical: {transitive._graph.number_of_edges()} pairs matched via transitive closure")
    print(f"full_pipeline_ssma_hierarchical: {hierarchical.n_models_built} models built, {hierarchical.n_merges} merges performed")


def full_pipeline_ssma_hierarchical_dino(
    images_folder='/home/colombo/Shared/imgs',
    output_folder='/media/dati/full_ssma_hierarchical_dino',
    n_mst=2,
    max_iterations=3,
    min_matches=3,
    sim_quantile=0.75,
    sim_min=None,
    global_db=None,
):
    # global_db: path to an existing global-similarity hdf5 (e.g. the
    # full_ssma_global_sim.hdf5 written by full_pipeline_ssma) to reuse its
    # cached SALAD descriptors; defaults to a fresh file in output_folder.
    print("\n \n")
    print("=" * 50)
    print("Running: full_pipeline_ssma_hierarchical_dino")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    transitive_pairs_path = str(output_path / 'full_ssma_hierarchical_dino_transitive_pairs.hdf5')
    match_db = str(output_path / 'full_ssma_hierarchical_dino_transitive.db')

    current_pairs = []
    allowed = set()
    if global_db is None:
        global_db = str(output_path / 'full_ssma_hierarchical_dino_global_sim.hdf5')
    global_pipeline = [
        salad_module(),
        cosine_similarity_module(),
        kfc_module(pairs=current_pairs, allowed=allowed, sim_db=global_db, out_path=None, n_mst=n_mst, sim_quantile=sim_quantile, sim_min=sim_min),
    ]
    run_pairs(global_pipeline, images_folder, db_name=global_db)

    print(f"full_pipeline_ssma_hierarchical_dino: {len(current_pairs)} seed pairs selected")

    if not current_pairs:
        print("full_pipeline_ssma_hierarchical_dino: no seed pairs, nothing to match")
        return

    transitive = transitive_module(
        pairs=current_pairs,
        allowed=allowed,
        threshold=min_matches,
        max_iterations=max_iterations,
        out_path=transitive_pairs_path,
    )

    hierarchical = hierarchical_reconstruct_module(
        db=match_db,
        images=images_folder,
        output=str(output_path / 'full_ssma_hierarchical_dino_model'),
        worklist=current_pairs,
    )

    match_pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                deep_joined_module(what='aliked'),
                lightglue_module(what='aliked'),
            ],
            [
                deep_joined_module(what='superpoint'),
                lightglue_module(what='superpoint'),
            ],
            [
                dog_module(),
                patch_module(),
                deep_descriptor_module(),
                smnn_module(),
            ],
            [
                loma_module(),
            ],
        ]),
        segformer_module(stage='matches'),
        dino_filter_module(),
        magsac_module(),
        transitive,
        to_colmap_module(db=match_db, worklist=current_pairs, no_unmatched=False, only_matched=True),
        hierarchical,
    ]

    while current_pairs:
        run_pairs(match_pipeline, current_pairs, db_name=str(output_path / 'full_ssma_hierarchical_dino_transitive.hdf5'))

    print(f"full_pipeline_ssma_hierarchical_dino: {transitive._graph.number_of_edges()} pairs matched via transitive closure")
    print(f"full_pipeline_ssma_hierarchical_dino: {hierarchical.n_models_built} models built, {hierarchical.n_merges} merges performed")

