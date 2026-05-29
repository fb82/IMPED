import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
import inspect

import pycolmap
import torch

from core import enable_quadtree, run_pairs

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
from visualization import (
    show_homography_module,
    show_kpts_module,
    show_matches_module,
    show_patches_module,
)


def pipeline1():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline2():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        loftr_module(),
        show_kpts_module(id_more='first', prepend_pair=False),
        magsac_module(),
        show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline3():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        deep_joined_module(),
        show_kpts_module(id_more='first', prepend_pair=False),
        lightglue_module(),
        magsac_module(),
        show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)

def pipeline4():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)

def pipeline5():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)

def pipeline6():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline7():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")   
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)    

def pipeline8():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        loftr_module(),
        magsac_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        sampling_module(sampling_mode='avg_inlier_matches', overlapping_cells=True, sampling_scale=20),
        show_matches_module(id_more='second', img_prefix='matches_sampled_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)   

def pipeline9():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        loftr_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)       


def pipeline10():   
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)  

def pipeline11(): 
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        loftr_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)  


def pipeline12():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        roma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]    
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)  
 
def pipeline13():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        r2d2_module(),
        # smnn_module(),
        lightglue_module(what='sift', desc_cf=255),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)   

def pipeline14():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)  

def pipeline15():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        aspanformer_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)

def pipeline16():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        from_colmap_module(),
        show_kpts_module(img_prefix='sift_', prepend_pair=False),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
 
def pipeline17(): 
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    imgs = '../data/ET'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=f'{inspect.currentframe().f_code.co_name}_aliked.db'),            
    ]         
    name_db = f"database_{inspect.currentframe().f_code.co_name}_aliked.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    #
    pipeline = [
        deep_joined_module(what='superpoint'),
        lightglue_module(what='superpoint'),
        magsac_module(),
        show_matches_module(img_prefix='superpoint_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=f'{inspect.currentframe().f_code.co_name}_superpoint.db'),            
    ]         
    name_db = f"database_{inspect.currentframe().f_code.co_name}_superpoint.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    #
    device = torch.device('cpu')
    merge_colmap_db([f'{inspect.currentframe().f_code.co_name}_aliked.db', f'{inspect.currentframe().f_code.co_name}_superpoint.db'], f'{inspect.currentframe().f_code.co_name}_aliked_superpoint.db', img_folder='../data/ET')

def pipeline21():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=f'database_{inspect.currentframe().f_code.co_name}_aliked.db'),            
    ]         
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    os.makedirs('aliked_colmap_models', exist_ok=True)          
    pycolmap.incremental_mapping(database_path=f'database_{inspect.currentframe().f_code.co_name}_aliked.db', image_path=imgs, output_path='aliked_colmap_models')            
    filter_colmap_reconstruction(input_model_path='aliked_colmap_models/0', db_path=f'database_{inspect.currentframe().f_code.co_name}_aliked.db', img_path=imgs, output_model_path='aliked_colmap_models/filtered_model', to_filter=['et002.jpg', 'et005.jpg'], how_filter='exclude', only_cameras=False, add_3D_points=True)


def pipeline21bis():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
        to_colmap_module(db=str(base_dir / f"{inspect.currentframe().f_code.co_name}_aliked.db")),            
    ]         
    imgs = str(base_dir.parent / 'data' / 'ET')
    
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    os.makedirs(base_dir / 'aliked_colmap_models', exist_ok=True)    
      
    pycolmap.incremental_mapping(
        database_path=str(base_dir / f"{inspect.currentframe().f_code.co_name}_aliked.db"),
        image_path=imgs,
        output_path=str(base_dir / 'aliked_colmap_models')
    )            
    
    filter_colmap_reconstruction(
        input_model_path=str(base_dir / 'aliked_colmap_models' / '0'),
        db_path=str(base_dir / f"{inspect.currentframe().f_code.co_name}_aliked.db"),
        img_path=imgs,
        output_model_path=str(base_dir / 'aliked_colmap_models' / 'filtered_model'),
        to_filter=['et002.jpg', 'et005.jpg'],
        how_filter='exclude',
        only_cameras=False,
        add_3D_points=True
    )

def pipeline22():

    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    imgs = '../data/ET'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='aliked.db'),            
    ]         
    name_db = f"database_{inspect.currentframe().f_code.co_name}_aliked.hdf5"
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}_superpoint.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    os.makedirs('superpoint_colmap_models', exist_ok=True)          
    pycolmap.incremental_mapping(database_path='superpoint.db', image_path=imgs, output_path='superpoint_colmap_models')            
    filter_colmap_reconstruction(input_model_path='superpoint_colmap_models/0', db_path='superpoint.db', img_path=imgs, output_model_path='superpoint_colmap_models/filtered_model', to_filter=['et001.jpg', 'et002.jpg', 'et003.jpg', 'et004.jpg', 'et005.jpg'], how_filter='include', only_cameras=False, add_3D_points=True)
    #
    device = torch.device('cpu')
    align_colmap_models(model_path1='aliked_colmap_models/filtered_model', model_path2='superpoint_colmap_models/filtered_model', imgs_path=imgs, db_path0='aliked.db', db_path1='superpoint.db', output_db='aliked_superpoint.db', output_model='merged_model', th=None)

def pipeline23():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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

    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}_aliked1.hdf5"
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}_aliked2.hdf5"
    run_pairs(pipeline, imgs, colmap_db_or_list='aliked.db', mode='exclude', colmap_req='matches', db_name=name_db)

def pipeline25():

    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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

    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    imgs_planar, gt_planar, to_add_path_planar = benchmark_setup(bench_path='../bench_data', dataset='planar')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(mode='homography', id_more='H_mode'),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_planar, to_add_path=to_add_path_planar, mode='homography'),
    ]         
    imgs = [imgs_planar[i] for i in range(20)]
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, add_path=to_add_path_planar, force=True, db_name=name_db)   

def pipeline28():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    imgs_imc, gt_imc, to_add_path_imc = benchmark_setup(bench_path='../bench_data', dataset='imc')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_imc, to_add_path=to_add_path_imc, mode='epipolar'),
    ]         
    imgs = [imgs_imc[i] for i in range(10)]
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, add_path=to_add_path_imc, db_name=name_db)   

def pipeline29():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    imgs_megadepth, gt_megadepth, to_add_path_megadepth = benchmark_setup(bench_path='../bench_data', dataset='megadepth')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_megadepth, to_add_path=to_add_path_megadepth, mode='epipolar'),
    ]         
    imgs = [imgs_megadepth[i] for i in range(10)]
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, add_path=to_add_path_megadepth, db_name = name_db)   

def pipeline30():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    imgs_scannet, gt_scannet, to_add_path_scannet = benchmark_setup(bench_path='../bench_data', dataset='scannet')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_scannet, to_add_path=to_add_path_scannet, mode='epipolar'),
    ]         
    imgs = [imgs_scannet[i] for i in range(10)]
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, add_path=to_add_path_scannet, db_name = name_db)   

def pipeline31():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline32():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline33():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        mast3r_module(),
        magsac_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)  

def pipeline34():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")        
    pipeline = [
        dust3r_module(),
        magsac_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)          

def pipeline35():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")

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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline36():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")

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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline37():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline38():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")

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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db) 

def pipeline39():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        romav2_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]    
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)  


def pipeline40(imgs='../data/ET'):
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")

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
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)
    

    merge_colmap_db(['ET_pt1.db', 'ET_pt2.db'], 'Merged_ET.db', img_folder='../data/ET')

    end_time2 = time.time()


    print(f"Execution time full dataset: {end_time - start_time} seconds")
    print(f"Execution time incremental dataset: {end_time2 - start_time2} seconds")


def pipeline41():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        loftr_module(device='cpu'),
        show_kpts_module(id_more='first', prepend_pair=False),
        magsac_module(device='cuda'),
        show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)


def pipeline42():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline_a = [
        image_muxer_module(pair_generator=pair_rot4, pipe_gather=pipe_max_matches, pipeline=[
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
        ]),   
        show_kpts_module(id_more='third', img_prefix='union_', prepend_pair=False),
        show_matches_module(id_more='fourth', img_prefix='union_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='custom_colmap.db'),             
    ]
    imgs = '../data/ET'
    run_pairs(pipeline_a, imgs, db_name='database_custom_a.hdf5')

    pipeline_b = [
        roma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='custom_colmap.db'),
    ]    
    imgs = '../data/ET'
    run_pairs(pipeline_b, imgs, db_name='database_custom_b.hdf5')

def pipeline43():
    print("\n \n")
    print("=" * 50)
    print(f"Running: {inspect.currentframe().f_code.co_name}")
    pipeline = [
        loma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]    
    imgs = '../data/ET'
    name_db = f"database_{inspect.currentframe().f_code.co_name}.hdf5"
    run_pairs(pipeline, imgs, db_name=name_db)  
