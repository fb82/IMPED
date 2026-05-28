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
| Optimize incremental adding of images | ✅ |  |
| Optimize hdf5 database |  | 📝 |
| Device as optional parameter | ✅ |  |



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

## Usage

This repository uses `src/imped.py` as the main entry point. The easiest way to run a pipeline is to edit `src/imped.py` and point it to a pipeline defined in `src/test_pipelines.py` or to define your own pipeline directly.

### Running an existing pipeline

In `src/imped.py`, the current entrypoint looks like this:

```python
if __name__ == '__main__':
    with torch.inference_mode():
        print('Running pipeline 38')
        test_pipelines.pipeline38()
```

Change the pipeline number to any existing function from `src/test_pipelines.py`, for example:

```python
if __name__ == '__main__':
    with torch.inference_mode():
        print('Running pipeline 1')
        test_pipelines.pipeline1()
```

Then run:

```linux
python src/imped.py
```

### Defining a custom pipeline

A pipeline is a Python list of module instances. Each element can be a detector, descriptor, matcher, filter, visualization module, or ensemble helper.

Example custom pipeline in `src/imped.py`:

```python
from test_pipelines import (
    dog_module,
    patch_module,
    deep_descriptor_module,
    smnn_module,
    magsac_module,
    show_matches_module,
)
from core import run_pairs


def custom_pipeline():
    pipeline = [
        dog_module(),
        patch_module(),
        deep_descriptor_module(),
        smnn_module(),
        magsac_module(),
        show_matches_module(
            id_more='only',
            img_prefix='matches_',
            mask_idx=[1, 0],
            prepend_pair=False,
        ),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline, imgs, db_name='database_custom.hdf5')


if __name__ == '__main__':
    with torch.inference_mode():
        custom_pipeline()
```

### Device control

Most pipeline modules support an optional `device` argument or module attribute. `core.pipeline.run_pipeline()` detects a module's target `device` and moves intermediate tensors there automatically.

Example:

```python
from test_pipelines import loftr_module, magsac_module

pipeline = [
    loftr_module(device='cpu'),
    magsac_module(device='cuda'),
]
```

This makes it easy to mix CPU and GPU computation for different modules when needed.

### `run_pairs()` options

The pipeline is executed through `core.pipeline.run_pairs()`. Common options include:

- `pipeline`: list of modules to execute.
- `imgs`: folder path or list of image paths.
- `db_name`: output HDF5 database filename (default `database.hdf5`).
- `db_mode`: database mode, usually `'a'` to append.
- `force`: set to `True` to rerun modules even when cached results exist.
- `add_path`: prefix to apply to image paths if you pass relative pairs.
- `colmap_db_or_list`: optional COLMAP database or list for pair selection.
- `mode`: pairing mode for `image_pairs` (default `'exclude'`).
- `colmap_req`: required COLMAP data type (default `'geometry'`).
- `colmap_min_matches`: minimum matches for COLMAP-based pairing.

### Common pipeline components

The repository includes many modules in `src/`:

- `detectors`: `dog_module`, `hz_module`, `r2d2_module`
- `descriptors`: `patch_module`, `deep_descriptor_module`, `sift_module`
- `matchers`: `smnn_module`, `lightglue_module`, `loftr_module`, `roma_module`, `mast3r_module`, `dust3r_module`, `matchformer_module`, `aspanformer_module`
- `filters`: `magsac_module`, `acne_module`, `dtm_module`, `gms_module`, `lpm_module`, `oanet_module`, `fcgnn_module`, `mop_miho_ncc_module`
- `visualization`: `show_kpts_module`, `show_matches_module`, `show_patches_module`, `show_homography_module`
- `ensemble`: `image_muxer_module`, `pipeline_muxer_module`, `pair_pyramid`, `pair_rot4`, `pipe_union`, `pipe_max_matches`, `sampling_module`

### Notes

- The `imgs` argument may be a directory path or an explicit list of image file paths.
- `src/test_pipelines.py` contains many working examples of complete pipelines.
- If you want to add new pipelines, define them in `src/test_pipelines.py` and call them from `src/imped.py`.
- Pipelines can be combined and refined using ensemble utilities such as `sampling_module`, `pipe_union`, `image_muxer_module`, and `pipeline_muxer_module`.
- The `sampling` functionality is useful for merging matching outputs, removing duplicates, and producing a consolidated set of matches from multiple sub-pipelines.
- COLMAP integration is exposed through `src/colmap_fun`: export features and matches to COLMAP, import COLMAP keypoints back into the pipeline, use COLMAP databases to select pairs, and merge results smoothly across different computation paths.
- `run_pairs()` can use a COLMAP database or list via `colmap_db_or_list` and supports incremental processing with `colmap_req` and `colmap_min_matches`.


