import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
import inspect

import h5py
import pycolmap
import torch

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from core import enable_quadtree, run_pairs, run_close_pairs, split_images, merge_hdf5

project_root = Path(__file__).parent.resolve()

extra_paths = [
    project_root / "r2d2",
    project_root / "mast3r",
    project_root / "matchformer",
    project_root / "aspanformer" / "src",
    project_root / "miho" / "src",
    project_root / "gsm" 
]



for p in extra_paths:
    if p.exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


from descriptors import deep_descriptor_module, patch_module
from global_descriptors import salad_module, standard_descriptor_module
from similarity import cosine_similarity_module, l2_similarity_module, standard_similarity_module
from confidence import conf_module
from image_pairs import image_pairs
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
    roma_module,
    romav2_module,
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
    pair_rot4,
    pipe_max_matches,
    pipe_union,
    pipeline_muxer_module,
    sampling_module,
)
from filters import acne_module, dtm_module, magsac_module, mop_miho_ncc_module
from segmentators import segformer_module
from visualization import (
    show_homography_module,
    show_kpts_module,
    show_matches_module,
    show_patches_module,
)


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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    name_db = f"database_{name_example}.hdf5"
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
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    name_db = f"database_{name_example}.hdf5"
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
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    name_db = f"database_{name_example}.hdf5"
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
        to_colmap_module(),
    ]    
    imgs = '../data/ET'
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    name_db = f"database_{name_example}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)

def pipeline16():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        from_colmap_module(),
        show_kpts_module(img_prefix='sift_', prepend_pair=False),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = f"database_{name_example}.hdf5"
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
    run_pairs(pipeline, imgs, add_path=to_add_path_megadepth)      

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
    run_pairs(pipeline, imgs, add_path=to_add_path_scannet)

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
    run_pairs(pipeline, imgs, add_path=to_add_path_imc)

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
        to_colmap_module(db=f'{name_example}_aliked.db'),            
    ]         
    name_db = f"database_{name_example}_aliked.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    #
    pipeline = [
        deep_joined_module(what='superpoint'),
        lightglue_module(what='superpoint'),
        magsac_module(),
        show_matches_module(img_prefix='superpoint_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=f'{name_example}_superpoint.db'),            
    ]         
    name_db = f"database_{name_example}_superpoint.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    #
    device = torch.device('cpu')
    merge_colmap_db([f'{name_example}_aliked.db', f'{name_example}_superpoint.db'], f'{name_example}_aliked_superpoint.db', img_folder='../data/ET')

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
        to_colmap_module(db=f'database_{name_example}_aliked.db'),            
    ]         
    imgs = '../data/ET'
    name_db = f"database_{name_example}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    os.makedirs('aliked_colmap_models', exist_ok=True)          
    pycolmap.incremental_mapping(database_path=f'database_{name_example}_aliked.db', image_path=imgs, output_path='aliked_colmap_models')            
    filter_colmap_reconstruction(input_model_path='aliked_colmap_models/0', db_path=f'database_{name_example}_aliked.db', img_path=imgs, output_model_path='aliked_colmap_models/filtered_model', to_filter=['et002.jpg', 'et005.jpg'], how_filter='exclude', only_cameras=False, add_3D_points=True)


def pipeline21bis():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    import os
    from pathlib import Path
    
    base_dir = Path(__file__).parent
    print(f"__file__: {__file__}")
    print(f"base_dir: {base_dir}")
    print(f"CWD: {os.getcwd()}")
    print(f"aliked_colmap_models exists: {(base_dir / 'aliked_colmap_models').exists()}")
    print(f"aliked_colmap_models/0 exists: {(base_dir / 'aliked_colmap_models' / '0').exists()}")
    print(f"Contents of aliked_colmap_models: {list((base_dir / 'aliked_colmap_models').iterdir()) if (base_dir / 'aliked_colmap_models').exists() else 'DIR NOT FOUND'}")
    base_dir = Path(__file__).parent

    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=str(base_dir / f"{name_example}_aliked.db")),            
    ]         
    imgs = str(base_dir.parent / 'data' / 'ET')
    
    name_db = f"database_{name_example}.hdf5"
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
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='aliked.db'),            
    ]         
    name_db = f"database_{name_example}_aliked.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    os.makedirs('aliked_colmap_models', exist_ok=True)          
    pycolmap.incremental_mapping(database_path='aliked.db', image_path=imgs, output_path='aliked_colmap_models')            
    filter_colmap_reconstruction(input_model_path='aliked_colmap_models/0', db_path='aliked.db', img_path=imgs, output_model_path='aliked_colmap_models/filtered_model', to_filter=['et002.jpg', 'et005.jpg'], how_filter='exclude', only_cameras=False, add_3D_points=True)
    #
    pipeline = [
        deep_joined_module(what='superpoint'),
        lightglue_module(what='superpoint'),
        magsac_module(),
        show_matches_module(img_prefix='superpoint_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='superpoint.db'),            
    ]         
    name_db = f"database_{name_example}_superpoint.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    os.makedirs('superpoint_colmap_models', exist_ok=True)          
    pycolmap.incremental_mapping(database_path='superpoint.db', image_path=imgs, output_path='superpoint_colmap_models')            
    filter_colmap_reconstruction(input_model_path='superpoint_colmap_models/0', db_path='superpoint.db', img_path=imgs, output_model_path='superpoint_colmap_models/filtered_model', to_filter=['et001.jpg', 'et002.jpg', 'et003.jpg', 'et004.jpg', 'et005.jpg'], how_filter='include', only_cameras=False, add_3D_points=True)
    #
    device = torch.device('cpu')
    align_colmap_models(model_path1='aliked_colmap_models/filtered_model', model_path2='superpoint_colmap_models/filtered_model', imgs_path=imgs, db_path0='aliked.db', db_path1='superpoint.db', output_db='aliked_superpoint.db', output_model='merged_model', th=None)

def pipeline23():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        deep_joined_module(),
        lightglue_module(),
        magsac_module(),
        to_colmap_module(),            
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
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(id_more='1st', img_prefix='aliked_matches_1st_', mask_idx=[1], prepend_pair=False),
        to_colmap_module(db='aliked.db'),            
    ]         
    # imgs = '../data/ET'
    # run_pairs(pipeline, imgs, colmap_db_or_list=['et000.jpg', 'et001.jpg', 'et003.jpg', 'et006.jpg', 'et007.jpg', 'et008.jpg'], mode='exclude')
    imgs = ['et000.jpg', 'et001.jpg', 'et003.jpg', 'et006.jpg', 'et007.jpg', 'et008.jpg']
    name_db = f"database_{name_example}_aliked1.hdf5"
    run_pairs(pipeline, imgs, add_path='../data/ET', db_name= name_db)
    # now the remaining mathing pairs only
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(id_more='2nd', img_prefix='aliked_matches_2nd_', mask_idx=[1], prepend_pair=False),
        to_colmap_module(db='aliked.db'),            
    ]         
    imgs = '../data/ET'
    name_db = f"database_{name_example}_aliked2.hdf5"
    run_pairs(pipeline, imgs, colmap_db_or_list='aliked.db', mode='exclude', colmap_req='matches', db_name=name_db)

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
        to_colmap_module(),                       
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
        to_colmap_module(),
    ]    
    imgs = '../data/ET'
    name_db = f"database_{name_example}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)  


def pipeline40(imgs='../data/ET'):
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    start_time = time.time()

    imgs='../data/ET'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='ET_full.db'),            
    ]         
    run_pairs(pipeline, imgs, db_name='database_ET_full.hdf5')

    end_time = time.time()

    imgs='../data/ET_pt1'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='ET_pt1.db'),            
    ]         
    run_pairs(pipeline, imgs, db_name='database_ET_pt1.hdf5')

    start_time2 = time.time()
    
    imgs='../data/ET_pt2'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='ET_pt2.db'),            
    ]         
    name_db = f"database_{name_example}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    

    merge_colmap_db(['ET_pt1.db', 'ET_pt2.db'], 'Merged_ET.db', img_folder='../data/ET')

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
    name_db = f"database_{name_example}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline42():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
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
        to_colmap_module(db=name_example + '_colmap_ab.db'),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline_a, imgs, db_name=name_example + '_a.hdf5')
    
    pipeline_b = [
        roma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=name_example + '_colmap_ab.db'),
    ]    
    imgs = '../data/ET'
    run_pairs(pipeline_b, imgs, db_name=name_example + '_b.hdf5', colmap_db_or_list=name_example + '_colmap_ab.db', mode='include')


def pipeline43():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")
    pipeline = [
        loma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    name_db = f"database_{name_example}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline_ssma(
    n_chunks=1,
    chunk_idx=0,
    images_folder='/home/colombo/Documenti/newest/IMPED/data/imgs_orig/',
    output_folder='.',
    n_close=10,
):
    print("\n \n")
    print("=" * 50)
    print(f"Running: pipeline_ssma  [chunk {chunk_idx} of {n_chunks}]")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_db = str(output_path / f'ssma_chunk_{chunk_idx}.db')

    pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                deep_joined_module(what='aliked'),
                segformer_module(),
                lightglue_module(what='aliked'),
            ],
            [
                deep_joined_module(what='superpoint'),
                segformer_module(),
                lightglue_module(what='superpoint'),
            ],
            [
                dog_module(),
                patch_module(),
                deep_descriptor_module(),
                segformer_module(),
                smnn_module(),
            ],
        ]),
        magsac_module(),
        segformer_module(stage='matches'),
        to_colmap_module(db=chunk_db),
    ]

    run_close_pairs(
        pipeline,
        images_folder,
        n=n_close,
        db_name=None,
        colmap_db_or_list=chunk_db,
        n_chunks=n_chunks,
        chunk_idx=chunk_idx,
        salad_cache=str(output_path / 'salad_descriptors.pt'),
    )


def pipeline_ssma_mst(
    n_chunks=1,
    chunk_idx=0,
    images_folder='/home/colombo/Documenti/newest/IMPED/data/imgs_orig/',
    output_folder='.',
    threshold=0.99,
):
    print("\n \n")
    print("=" * 50)
    print(f"Running: pipeline_ssma_mst  [chunk {chunk_idx} of {n_chunks}]")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_db = str(output_path / f'ssma_mst_chunk_{chunk_idx}.db')
    pairs_path = str(output_path / f'ssma_mst_pairs_{chunk_idx}.pt')


    coarse_pipeline = [
        salad_module(),
        cosine_similarity_module(),
        conf_module(threshold=threshold, out_path=pairs_path),
    ]
    run_pairs(
        coarse_pipeline,
        images_folder,
        db_name=str(output_path / f'ssma_mst_global_desc_{chunk_idx}.hdf5'),
    )

    pairs = torch.load(pairs_path)

    if n_chunks > 1:
        pairs = list(image_pairs(pairs, check_img=False, chunk_id=chunk_idx, n_chunk=n_chunks))

    pipeline = [
        pipeline_muxer_module(pipe_gather=pipe_union, pipeline=[
            [
                deep_joined_module(what='aliked'),
                segformer_module(),
                lightglue_module(what='aliked'),
            ],
            [
                deep_joined_module(what='superpoint'),
                segformer_module(),
                lightglue_module(what='superpoint'),
            ],
            [
                dog_module(),
                patch_module(),
                deep_descriptor_module(),
                segformer_module(),
                smnn_module(),
            ],
        ]),
        magsac_module(),
        segformer_module(stage='matches'),
        to_colmap_module(db=chunk_db),
    ]

    run_pairs(pipeline, pairs, db_name=str(output_path / f'ssma_mst_matches_{chunk_idx}.hdf5'))


def pipeline44():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    name_db_chunk0 = f'database_{name_example}_chunk0.hdf5'
    name_db_chunk1 = f'database_{name_example}_chunk1.hdf5'
    name_db_merged = f'database_{name_example}_merged.hdf5'

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

    # Run on all images — each image in chunk0 has its intra-chunk pairs done
    # but not its cross-chunk pairs; this verifies partial-pair images get
    # their remaining pairs computed and are not skipped entirely
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
    colmap_db = f'{name_example}_colmap.db'
    name_db = f'database_{name_example}.hdf5'

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

    # A fresh pipeline (and to_colmap_module instance) for the second run:
    # finalize() closes the underlying sqlite connection, so the same
    # to_colmap_module instance can't be finalized twice.
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    name_db = f"database_{name_example}.hdf5"
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
    pairs_path = f"{name_example}_pairs.pt"
    name_db = f"database_{name_example}.hdf5"

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
                # conf_module doesn't cache its own output (see conf_module docstring),
                # so re-derive the expected decision straight from the cached pair_sim.
                if sim > threshold:
                    expected.add((im0, im1))

    saved_pairs = torch.load(pairs_path)
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
    name_db = f"database_{name_example}.hdf5"
    pairs_path_high = f"{name_example}_pairs_high.pt"
    pairs_path_low = f"{name_example}_pairs_low.pt"

    for f in [name_db, pairs_path_high, pairs_path_low]:
        if os.path.exists(f):
            os.remove(f)

    # First pass with a strict threshold, reusing the same hdf5 (so global_desc
    # and pair_sim get cached). This is the scenario conf_module's
    # add_to_cache=False guards: a second pass with a looser threshold must
    # NOT reuse the first pass's pair_conf decisions.
    pipeline_high = [salad_module(), cosine_similarity_module(), conf_module(threshold=0.5, out_path=pairs_path_high)]
    run_pairs(pipeline_high, imgs_dir, db_name=name_db)
    pairs_high = torch.load(pairs_path_high)

    pipeline_low = [salad_module(), cosine_similarity_module(), conf_module(threshold=-1.0, out_path=pairs_path_low)]
    run_pairs(pipeline_low, imgs_dir, db_name=name_db)
    pairs_low = torch.load(pairs_path_low)

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
    seg_kp = segformer_module()
    seg_mt = segformer_module(stage='matches')

    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        seg_kp,
        smnn_module(),
        magsac_module(),
        seg_mt,
    ]
    name_db = f"database_{name_example}.hdf5"
    if os.path.exists(name_db):
        os.remove(name_db)

    run_pairs(pipeline, imgs_dir, db_name=name_db)

    db = pickled_hdf5.pickled_hdf5(name_db, mode='r')
    keys = db.get_keys()

    kp_keys = [k for k in keys if f'/{seg_kp.get_id()}/data' in k]
    assert len(kp_keys) > 0, "No cached entries for the keypoints-stage segformer module"

    for k in kp_keys:
        data, found = db.get(k)
        assert found, f"Missing cached entry for {k}"
        assert 'seg_mask' in data, f"'seg_mask' missing at {k}"
        assert data['seg_mask'].dtype == torch.bool, f"'seg_mask' should be boolean at {k}"
        if 'keypt_mask' in data:
            assert data['keypt_mask'].dtype == torch.bool, f"'keypt_mask' should be boolean at {k}"

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


def pipeline52():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    img_names = {f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

    pipeline = [standard_descriptor_module()]
    name_db = f"database_{name_example}.hdf5"
    if os.path.exists(name_db):
        os.remove(name_db)

    run_pairs(pipeline, imgs_dir, db_name=name_db)

    with h5py.File(name_db, 'r') as f:
        root = f['pickled']
        for name in img_names:
            assert name in root, f"No cached entry for {name}"
            assert 'standard' in root[name], f"No standard_descriptor output for {name}"
            data = pickled_hdf5.pickled_hdf5.from_numpy(root[name]['standard']['data'][()])
            assert 'global_desc' in data, f"'global_desc' missing for {name}"

            global_desc = data['global_desc']
            assert 'kp' in global_desc and 'desc' in global_desc, \
                f"'global_desc' should have 'kp' and 'desc' for {name}"

            kp, desc = global_desc['kp'], global_desc['desc']
            assert kp.ndim == 2 and kp.shape[1] == 2, f"'kp' expected shape [N, 2], got {tuple(kp.shape)} for {name}"
            assert desc.ndim == 2 and desc.shape[1] == 128, \
                f"'desc' expected shape [N, 128], got {tuple(desc.shape)} for {name}"
            assert kp.shape[0] == desc.shape[0], \
                f"'kp' and 'desc' should have the same N, got {kp.shape[0]} vs {desc.shape[0]} for {name}"

    if os.path.exists(name_db):
        os.remove(name_db)

    print("pipeline52: ALL ASSERTIONS PASSED")


def pipeline53():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    img_names = sorted(f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    pipeline = [standard_descriptor_module(), standard_similarity_module()]
    name_db = f"database_{name_example}.hdf5"
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
                assert 'standard_similarity' in root[im0][im1]['standard'], \
                    f"No standard_similarity output for ({im0}, {im1})"
                data = pickled_hdf5.pickled_hdf5.from_numpy(root[im0][im1]['standard']['standard_similarity']['data'][()])
                assert 'pair_sim' in data, f"'pair_sim' missing for ({im0}, {im1})"
                sim = data['pair_sim']
                assert sim >= 0.0, f"pair_sim {sim} should be a non-negative match count for ({im0}, {im1})"
                n_pairs += 1

    n_expected = len(img_names) * (len(img_names) - 1) // 2
    assert n_pairs == n_expected, f"Expected {n_expected} pairs, checked {n_pairs}"

    if os.path.exists(name_db):
        os.remove(name_db)

    print("pipeline53: ALL ASSERTIONS PASSED")


def pipeline54():
    name_example = inspect.currentframe().f_code.co_name
    print("\n \n")
    print("=" * 50)
    print(f"Running: {name_example}")

    imgs_dir = '../data/ET'
    threshold = 5.0
    pairs_path = f"{name_example}_pairs.pt"
    name_db = f"database_{name_example}.hdf5"

    for f in [pairs_path, name_db]:
        if os.path.exists(f):
            os.remove(f)

    # Same swap the todo describes: standard_descriptor_module +
    # standard_similarity_module in place of salad_module +
    # cosine_similarity_module, feeding the same conf_module unchanged.
    pipeline = [standard_descriptor_module(), standard_similarity_module(), conf_module(threshold=threshold, out_path=pairs_path)]
    run_pairs(pipeline, imgs_dir, db_name=name_db)

    img_names = sorted(f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    expected = set()
    with h5py.File(name_db, 'r') as f:
        root = f['pickled']
        for i in range(len(img_names)):
            for j in range(i + 1, len(img_names)):
                im0, im1 = img_names[i], img_names[j]
                data = pickled_hdf5.pickled_hdf5.from_numpy(root[im0][im1]['standard']['standard_similarity']['data'][()])
                sim = data['pair_sim']
                # conf_module doesn't cache its own output (see conf_module docstring),
                # so re-derive the expected decision straight from the cached pair_sim.
                if sim > threshold:
                    expected.add((im0, im1))

    saved_pairs = torch.load(pairs_path)
    saved_names = {tuple(sorted((os.path.basename(a), os.path.basename(b)))) for a, b in saved_pairs}
    expected_names = {tuple(sorted(p)) for p in expected}

    assert saved_names == expected_names, \
        f"conf_module's saved pairs don't match threshold={threshold} applied to pair_sim"

    for f in [pairs_path, name_db]:
        if os.path.exists(f):
            os.remove(f)

    print("pipeline54: ALL ASSERTIONS PASSED")