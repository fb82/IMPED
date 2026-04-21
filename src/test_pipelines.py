import os
import warnings
import torch
import sys
from pathlib import Path
import pycolmap

from core import device, pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args, enable_quadtree
from image_pairs import image_pairs



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


from detectors import dog_module, hz_module, keynet_module, r2d2_module


from descriptors import deep_descriptor_module, patch_module, sift_module

from matchers import aspanformer_module, blob_matching_module, dust3r_module, lightglue_module, loftr_module, mast3r_module, matchformer_module, roma_module, smnn_module, deep_joined_module

if enable_quadtree:
    from matchers import quadtreeattention_module

from filters import dtm_module, acne_module, adalam_module, fcgnn_module, gms_module, lpm_module, magsac_module, mop_miho_ncc_module, oanet_module, poselib_module
import dtm.src.dtm as dtm

from colmap_fun import coldb_ext, SIMPLE_RADIAL, UNDEFINED, DEGENERATE, CALIBRATED, UNCALIBRATED, PLANAR, PANORAMIC, PLANAR_OR_PANORAMIC, WATERMARK, MULTIPLE,  from_colmap_module, kpts_from_colmap, to_colmap_module, kpts_as_colmap, merge_colmap_db, filter_colmap_reconstruction, align_colmap_models

from ensemble import pipe_union, sampling, sortrows, sampling_module, image_muxer_module, pipeline_muxer_module, pair_rot4, pipe_max_matches, to_pyramid, pair_pyramid


from benchmark import  megadepth_1500_list, scannet_1500_list, resize_megadepth, resize_scannet, setup_images_megadepth, setup_images_scannet, benchmark_setup, megadepth_scannet_setup, imc_phototourism_setup, planar_setup, download_megadepth, download_scannet, download_planar, relative_pose_error_angular, relative_pose_error_metric, estimate_pose, error_auc, invalid_matches, homography_error_heat_map, epipolar_error_heat_map, register_by_Horn, evaluate_rec, pairwise_benchmark_module
from visualization import show_kpts_module, visualize_LAF, show_matches_module, show_homography_module, show_patches_module, colorize_plane

def pipeline1():
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
    run_pairs(pipeline, imgs)


def pipeline2():
    pipeline = [
        loftr_module(),
        show_kpts_module(id_more='first', prepend_pair=False),
        magsac_module(),
        show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)


def pipeline3():
    pipeline = [
        deep_joined_module(),
        show_kpts_module(id_more='first', prepend_pair=False),
        lightglue_module(),
        magsac_module(),
        show_matches_module(id_more='second', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)

def pipeline4():
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
    run_pairs(pipeline, imgs)

def pipeline5():
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
    run_pairs(pipeline, imgs)

def pipeline6():
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
    run_pairs(pipeline, imgs)


def pipeline7():   
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
    run_pairs(pipeline, imgs)    

def pipeline8():
    pipeline = [
        loftr_module(),
        magsac_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        sampling_module(sampling_mode='avg_inlier_matches', overlapping_cells=True, sampling_scale=20),
        show_matches_module(id_more='second', img_prefix='matches_sampled_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)   

def pipeline9():
    pipeline = [
        loftr_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)       


def pipeline10():   
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)  

def pipeline11(): 
    pipeline = [
        loftr_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)  


def pipeline12():
## Not working: manca modulo local_corr probabilmente errore mancanza scheda nvidia
    pipeline = [
        roma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(),
    ]    
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)  
 
def pipeline13():
    pipeline = [
        r2d2_module(),
        # smnn_module(),
        lightglue_module(what='sift', desc_cf=255),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)   

def pipeline14():
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
    run_pairs(pipeline, imgs)  

def pipeline15():
## Not working without NVIDIA GPU
      pipeline = [
          aspanformer_module(),
          magsac_module(),
          show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
          to_colmap_module(),
      ]
      imgs = '../data/ET'
      run_pairs(pipeline, imgs)

def pipeline16():
    pipeline = [
        from_colmap_module(),
        show_kpts_module(img_prefix='sift_', prepend_pair=False),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)
 
def pipeline17(): 
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
## Working
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
## Working
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
## Working
    imgs = '../data/ET'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='aliked.db'),            
    ]         
    run_pairs(pipeline, imgs)
    #
    pipeline = [
        deep_joined_module(what='superpoint'),
        lightglue_module(what='superpoint'),
        magsac_module(),
        show_matches_module(img_prefix='superpoint_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='superpoint.db'),            
    ]         
    run_pairs(pipeline, imgs)
    #
    device = torch.device('cpu')
    merge_colmap_db(['aliked.db', 'superpoint.db'], 'aliked_superpoint.db', img_folder='../data/ET')

def pipeline21():
# Not working
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='aliked.db'),            
    ]         
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)
    os.makedirs('aliked_colmap_models', exist_ok=True)          
    pycolmap.incremental_mapping(database_path='aliked.db', image_path=imgs, output_path='aliked_colmap_models')            
    filter_colmap_reconstruction(input_model_path='aliked_colmap_models/0', db_path='aliked.db', img_path=imgs, output_model_path='aliked_colmap_models/filtered_model', to_filter=['et002.jpg', 'et005.jpg'], how_filter='exclude', only_cameras=False, add_3D_points=True)

from pathlib import Path

def pipeline21bis():
    base_dir = Path(__file__).parent

    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db=str(base_dir / 'aliked.db')),            
    ]         
    imgs = str(base_dir.parent / 'data' / 'ET')
    run_pairs(pipeline, imgs)
    os.makedirs(base_dir / 'aliked_colmap_models', exist_ok=True)          
    pycolmap.incremental_mapping(
        database_path=str(base_dir / 'aliked.db'),
        image_path=imgs,
        output_path=str(base_dir / 'aliked_colmap_models')
    )            
    filter_colmap_reconstruction(
        input_model_path=str(base_dir / 'aliked_colmap_models' / '0'),
        db_path=str(base_dir / 'aliked.db'),
        img_path=imgs,
        output_model_path=str(base_dir / 'aliked_colmap_models' / 'filtered_model'),
        to_filter=['et002.jpg', 'et005.jpg'],
        how_filter='exclude',
        only_cameras=False,
        add_3D_points=True
    )

def pipeline22():
# Not working
    imgs = '../data/ET'
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='aliked_matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='aliked.db'),            
    ]         
    run_pairs(pipeline, imgs)
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
    run_pairs(pipeline, imgs)
    os.makedirs('superpoint_colmap_models', exist_ok=True)          
    pycolmap.incremental_mapping(database_path='superpoint.db', image_path=imgs, output_path='superpoint_colmap_models')            
    filter_colmap_reconstruction(input_model_path='superpoint_colmap_models/0', db_path='superpoint.db', img_path=imgs, output_model_path='superpoint_colmap_models/filtered_model', to_filter=['et001.jpg', 'et002.jpg', 'et003.jpg', 'et004.jpg', 'et005.jpg'], how_filter='include', only_cameras=False, add_3D_points=True)
    #
    device = torch.device('cpu')
    align_colmap_models(model_path1='aliked_colmap_models/filtered_model', model_path2='superpoint_colmap_models/filtered_model', imgs_path=imgs, db_path0='aliked.db', db_path1='superpoint.db', output_db='aliked_superpoint.db', output_model='merged_model', th=None)

def pipeline23():
# Working
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
## Working
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
    run_pairs(pipeline, imgs, add_path='../data/ET')
    # now the remaining mathing pairs only
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(id_more='2nd', img_prefix='aliked_matches_2nd_', mask_idx=[1], prepend_pair=False),
        to_colmap_module(db='aliked.db'),            
    ]         
    imgs = '../data/ET'
    run_pairs(pipeline, imgs, colmap_db_or_list='aliked.db', mode='exclude', colmap_req='matches')

def pipeline25():
## Working
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
# Working
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
## Working
    imgs_planar, gt_planar, to_add_path_planar = benchmark_setup(bench_path='../bench_data', dataset='planar')
    

    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_planar, to_add_path=to_add_path_planar, mode='homography'),
    ]         
    imgs = [imgs_planar[i] for i in range(20)]
    run_pairs(pipeline, imgs, add_path=to_add_path_planar)   


def debug_module(name="debug"):
    class DebugModule:
        def __init__(self, name="debug"):
            self.name = name
            self.add_to_cache = False   # ← important, prevents cache errors

        def get_id(self):
            return self.name

        def run(self, **kwargs):
            # Try to extract the pair object - the framework seems to pass it in different ways
            pair = None
            if 'pair' in kwargs:
                pair = kwargs['pair']
            elif 'data' in kwargs and hasattr(kwargs['data'], 'results'):
                pair = kwargs['data']
            elif len(kwargs) == 1:
                # last resort: take the only value if it looks like a pair
                val = next(iter(kwargs.values()))
                if hasattr(val, 'results'):
                    pair = val

            if pair is None:
                print(f"[{self.name}] Warning: could not find pair in kwargs. Keys: {list(kwargs.keys())}")
                return kwargs  # return input unchanged

            print(f"\n=== DEBUG {self.name} - Pair {getattr(pair, 'idx', 'unknown')} ===")
            
            if hasattr(pair, 'results') and pair.results:
                last = pair.results[-1]
                keys = list(last.keys())
                print(f"  Keys ({len(keys)}): {sorted(keys)}")
                print(f"  Inliers / matches: {last.get('inliers', last.get('num_inliers', last.get('matches', 'N/A')))}")
                print(f"  'H_error_1' present? → { 'H_error_1' in last }")
                print(f"  'H' present? → { 'H' in last }")
                print(f"  Epipolar/F keys: {[k for k in keys if any(x in k.lower() for x in ['epipolar', 'fundamental', 'f_error', 'f_matrix'])]}")
            else:
                print("  No results stored yet on this pair.")

            # Return exactly what was passed in (safest for the pipeline)
            return kwargs

        def finalize(self, **kwargs):
            return kwargs   # dummy

    return DebugModule(name)

def safe_debug_module(name="after_magsac"):
    class SafeDebug:
        def __init__(self, name="debug"):
            self.name = name
            self.add_to_cache = False

        def get_id(self):
            return self.name

        def run(self, **kwargs):
            print(f"\n=== {self.name} ===")
            print("  Keys received:", list(kwargs.keys()))
            
            # Look for the last result on the pair (common pattern)
            if 'pair' in kwargs and hasattr(kwargs['pair'], 'results') and kwargs['pair'].results:
                last = kwargs['pair'].results[-1]
                print("  Last result keys:", sorted(last.keys()))
                print("  'H' present?", 'H' in last)
                print("  'F' present?", 'F' in last)
                print("  Inliers / mask sum:", last.get('m_mask', torch.tensor([])).sum().item() if 'm_mask' in last else 'N/A')
            elif kwargs:
                # fallback: print top-level keys
                for k, v in list(kwargs.items())[:10]:   # limit output
                    print(f"    {k}: {type(v)}")
            
            return kwargs   # pass through unchanged

        def finalize(self, **kwargs):
            return kwargs

    return SafeDebug(name)

def pipeline27bis():
## Working
    imgs_planar, gt_planar, to_add_path_planar = benchmark_setup(bench_path='../bench_data', dataset='planar')

    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(mode='homography'),
        #debug_module(name="debug"),
        #safe_debug_module(name="after_magsac"),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_planar, to_add_path=to_add_path_planar, mode='homography'),
    ]         
    imgs = [imgs_planar[i] for i in range(20)]
    run_pairs(pipeline, imgs, add_path=to_add_path_planar)   

def pipeline28():
## Working
    imgs_imc, gt_imc, to_add_path_imc = benchmark_setup(bench_path='../bench_data', dataset='imc')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_imc, to_add_path=to_add_path_imc, mode='epipolar'),
    ]         
    imgs = [imgs_imc[i] for i in range(10)]
    run_pairs(pipeline, imgs, add_path=to_add_path_imc)   

def pipeline29():
# Working
    imgs_megadepth, gt_megadepth, to_add_path_megadepth = benchmark_setup(bench_path='../bench_data', dataset='megadepth')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_megadepth, to_add_path=to_add_path_megadepth, mode='epipolar'),
    ]         
    imgs = [imgs_megadepth[i] for i in range(10)]
    run_pairs(pipeline, imgs, add_path=to_add_path_megadepth)   

def pipeline30():
# Working
    imgs_scannet, gt_scannet, to_add_path_scannet = benchmark_setup(bench_path='../bench_data', dataset='scannet')
    pipeline = [
        deep_joined_module(what='aliked'),
        lightglue_module(what='aliked'),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        pairwise_benchmark_module(gt=gt_scannet, to_add_path=to_add_path_scannet, mode='epipolar'),
    ]         
    imgs = [imgs_scannet[i] for i in range(10)]
    run_pairs(pipeline, imgs, add_path=to_add_path_scannet)   

def pipeline31():
## Working
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
    run_pairs(pipeline, imgs) 

def pipeline32():
# Working
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
    run_pairs(pipeline, imgs) 

def pipeline33():
# Working
    pipeline = [
        mast3r_module(),
        magsac_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)  

def pipeline34():        
# Working
    pipeline = [
        dust3r_module(),
        magsac_module(),
        show_matches_module(id_more='first', img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs)          

def pipeline35():
## Working
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
    run_pairs(pipeline, imgs) 

def pipeline36():
## Working
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
    run_pairs(pipeline, imgs) 

def pipeline37():
# Working
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
    run_pairs(pipeline, imgs) 

def pipeline38():
## Working
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
    run_pairs(pipeline, imgs) 