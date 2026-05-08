# IMPED

## TODO list

|  Task   | DONE  | WIP |
|-----|---|---|
| Create requirements file | ✅ | |
| Test all existing pipelines | ✅ | |
| Fix broken pipelines | ✅ | |
| Move test pipelines to test files | ✅ | |
| Split codebase | ✅ | |
| Comment codebase | ✅ | |
| Add latest updates | ✅ | |
| Delete & refactor repetitions  |  | 📝 |
| Add contracts |  | 📝 |
| Add Roma v2 | ✅ |  |
| Add LoMa |  | 📝 |
| Optimize code* | ✅ |  |

*for example merge database very slow with colmap


## File structure 
```
src/
├── core/
│   ├── __init__.py
│   ├── device.py              # device setup, global flags (show_progress, pipe_color)
│   ├── pipeline.py            # run_pipeline, run_pairs, finalize_pipeline, go_iter
│   ├── geometry.py            # laf2homo, homo2laf, apply_homo, change_patch_homo
│   │                          # decompose_H, decompose_H_other
│   └── utils.py               # set_args, compressed_pickle, decompress_pickle
│                              # qvec2rotmat, vector_norm, quaternion_matrix
│                              # affine_matrix_from_points
│
├── detectors/
│   ├── __init__.py
│   ├── dog_module.py          # dog_module
│   ├── keynet_module.py       # keynet_module
│   ├── hz_module.py           # hz_module
│   └── r2d2_module.py         # r2d2_module (incl. NonMaxSuppression, extract_multiscale)
│
├── descriptors/
│   ├── __init__.py
│   ├── patch_module.py        # patch_module
│   ├── deep_descriptor.py     # deep_descriptor_module
│   └── sift_module.py         # sift_module
│
├── matchers/
│   ├── __init__.py
│   ├── smnn_module.py         # smnn_module
│   ├── lightglue_module.py    # lightglue_module, deep_joined_module
│   ├── loftr_module.py        # loftr_module
│   ├── roma_module.py         # roma_module
│   ├── mast3r_module.py       # mast3r_module
│   ├── dust3r_module.py       # dust3r_module + dust3r_* helper functions
│   ├── matchformer_module.py  # matchformer_module
│   ├── aspanformer_module.py  # aspanformer_module
│   ├── quadtreeattention.py   # quadtreeattention_module (conditional on enable_quadtree)
│   └── blob_matching.py       # blob_matching_module
│
├── filters/
│   ├── __init__.py
│   ├── magsac_module.py       # magsac_module
│   ├── poselib_module.py      # poselib_module
│   ├── lpm_module.py          # lpm_module
│   ├── gms_module.py          # gms_module, gms_matcher_custom
│   ├── adalam_module.py       # adalam_module, adalamfilter_custom
│   ├── fcgnn_module.py        # fcgnn_module, fcgnn_custom, download_fcgnn
│   ├── oanet_module.py        # oanet_module, download_oanet
│   ├── acne_module.py         # acne_module, download_acne
│   ├── dtm_module.py          # dtm_module
│   └── mop_miho_ncc_module.py # mop_miho_ncc_module
│
├── ensemble/
│   ├── __init__.py
│   ├── sampling.py            # pipe_union, sampling, sortrows, sampling_module
│   ├── muxers.py              # image_muxer_module, pipeline_muxer_module
│   │                          # pair_rot4, pipe_max_matches
│   └── pyramid.py             # to_pyramid, pair_pyramid
│
├── colmap/
│   ├── __init__.py
│   ├── colmap_ext.py          # coldb_ext class + constants (SIMPLE_RADIAL, etc.)
│   ├── to_colmap_module.py    # to_colmap_module, kpts_as_colmap
│   ├── from_colmap_module.py  # from_colmap_module, kpts_from_colmap
│   └── merge_colmap.py        # merge_colmap_db, filter_colmap_reconstruction
│                              # align_colmap_models
│
├── benchmark/
│   ├── __init__.py
│   ├── datasets.py            # megadepth_1500_list, scannet_1500_list
│   │                          # resize_megadepth, resize_scannet
│   │                          # setup_images_megadepth, setup_images_scannet
│   │                          # benchmark_setup, megadepth_scannet_setup
│   │                          # imc_phototourism_setup, planar_setup
│   │                          # download_* functions
│   ├── metrics.py             # relative_pose_error_angular, relative_pose_error_metric
│   │                          # estimate_pose, error_auc
│   │                          # invalid_matches, homography_error_heat_map
│   │                          # epipolar_error_heat_map
│   │                          # register_by_Horn, evaluate_rec
│   └── benchmark_module.py    # pairwise_benchmark_module (all run_* and finalize_* methods)
│
├── visualization/
│   ├── __init__.py
│   ├── show_kpts.py           # show_kpts_module, visualize_LAF
│   ├── show_matches.py        # show_matches_module
│   ├── show_homography.py     # show_homography_module
│   ├── show_patches.py        # show_patches_module
│   └── colorize.py            # colorize_plane
│
└────── image_pairs.py         # image_pairs class (iterator)

```
## Environment setup

```linux
python -m venv imped
source imped/bin/activate
pip install -r src/requirements.txt
```


