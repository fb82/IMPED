import os
import warnings
import torch
import sys
from pathlib import Path

from core import device, pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args, enable_quadtree

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
from image_pairs import image_pairs

import test_pipelines


import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':       
    with torch.inference_mode():  
        # Pipelines go from 1 to 38
        # 12 can't be run without NVIDIA
        # 15 not working
        # 21 not working
        # 22 not working
        # 27 not working index error
        # 28 not working index error
        # 29 not working index error
        # 30 not working index error
        # 31 not working strange min() error
        # 32 not working index error
        # 35 not working index error
        # 37 not working index error
        # 38 not working index error

        # run_pipeline and apply homo needed in muxers


        print('Running pipeline 1')
        test_pipelines.pipeline1()
        print('Running pipeline 2')
        test_pipelines.pipeline2()  
        print('Running pipeline 3')
        test_pipelines.pipeline3()
        print('Running pipeline 4')
        test_pipelines.pipeline4()
        print('Running pipeline 5')
        test_pipelines.pipeline5()
        print('Running pipeline 6')
        test_pipelines.pipeline6()
        print('Running pipeline 7')
        test_pipelines.pipeline7()
        print('Running pipeline 8')
        test_pipelines.pipeline8()
        print('Running pipeline 9')
        test_pipelines.pipeline9()
        print('Running pipeline 10')
        test_pipelines.pipeline10()

        # print('Running pipeline 11')
        # test_pipelines.pipeline11()
        # print('Running pipeline 12')
        # test_pipelines.pipeline12()
        # print('Running pipeline 13')
        # test_pipelines.pipeline13()
        # print('Running pipeline 14')
        # test_pipelines.pipeline14()
        # print('Running pipeline 15')
        # test_pipelines.pipeline15()
        # print('Running pipeline 16')
        # test_pipelines.pipeline16()
        # print('Running pipeline 17')
        # test_pipelines.pipeline17()
        # print('Running pipeline 18')
        # test_pipelines.pipeline18()
        # print('Running pipeline 19')
        # test_pipelines.pipeline19()
        # print('Running pipeline 20')
        # test_pipelines.pipeline20()

        # print('Running pipeline 23')
        # test_pipelines.pipeline23()
        # print('Running pipeline 24')
        # test_pipelines.pipeline24()
        # print('Running pipeline 25')
        # test_pipelines.pipeline25()
        # print('Running pipeline 26')
        # test_pipelines.pipeline26()
        # print('Running pipeline 27')
        # test_pipelines.pipeline27bis()
        # print('Running pipeline 28')
        # test_pipelines.pipeline28()
        # print('Running pipeline 29')
        # test_pipelines.pipeline29()
        # print('Running pipeline 30')
        # test_pipelines.pipeline30()

        # print('Running pipeline 31')
        # test_pipelines.pipeline31()
        # print('Running pipeline 32')
        # test_pipelines.pipeline32()
        # print('Running pipeline 33')
        # test_pipelines.pipeline33()
        # print('Running pipeline 34')
        # test_pipelines.pipeline34()
        # print('Running pipeline 35')
        # test_pipelines.pipeline35()
        # print('Running pipeline 36')
        # test_pipelines.pipeline36()
        # print('Running pipeline 37')
        # test_pipelines.pipeline37()
        # print('Running pipeline 38')
        # test_pipelines.pipeline38()

    print('doh!')


