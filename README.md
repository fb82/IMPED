# IMPED (Image Matching PipelinED)

**IMPED** is a modular image matching and feature pipeline toolkit. It provides a unified, composable interface for building, testing, and benchmarking image matching pipelines — from keypoint detection and description to geometric filtering, COLMAP integration, and ensemble methods.

---

## Overview

IMPED is designed around a simple principle: a pipeline is a list of modules. Each module — detector, descriptor, matcher, filter, ensemble helper, or visualization tool — is a self-contained unit that can be freely combined, swapped, and benchmarked. This makes it easy to prototype new pipelines, reproduce existing methods, and evaluate combinations systematically.

Key features:
- **Modular by design** — mix and match detectors, descriptors, matchers, and filters in any combination.
- **Broad method coverage** — includes SIFT, R2D2, KeyNet, HardNet, LightGlue, LoFTR, LoMa, RoMa, MASt3R, DUSt3R, MatchFormer, ASpanFormer, and more.
- **Ensemble support** — union, muxing, pyramid, and sampling utilities for combining multiple pipelines.
- **COLMAP integration** — export/import features and matches, use COLMAP databases for pair selection, and merge reconstructions.
- **Benchmarking tools** — built-in support for MegaDepth-1500, ScanNet-1500, IMC PhotoTourism, and planar datasets with standard pose and homography metrics.
- **Incremental processing** — HDF5-backed caching avoids redundant computation across runs.
- **Device-aware execution** — per-module CPU/GPU assignment with automatic tensor routing.
- **Automatic pair selection** — a transitive pipeline (a `transitive` package module inside a regular `run_pairs()` pipeline) avoids exhaustive pairwise matching on large datasets by scoring pairs with a cheap global descriptor and growing the match graph transitively.
- **Live graph visualization** — watch pairs get confirmed in real time on an interactive, auto-refreshing graph view.

For a quick tour of what is possible, browse `src/test_pipelines.py`; it contains many ready-to-run examples covering a wide range of pipeline combinations.

---

## Installation

```bash
python -m venv imped
source imped/bin/activate
pip install -r src/requirements.txt
```

---

## Usage

The main entry point is `src/imped.py`. The quickest way to get started is to point it to one of the predefined pipelines in `src/test_pipelines.py`, or define your own directly.

### Running a predefined pipeline

Edit `src/imped.py` to select a pipeline:

```python
if __name__ == '__main__':
    with torch.inference_mode():
        test_pipelines.pipeline15()
```

Then run:

```bash
python src/imped.py
```

### Defining a custom pipeline

A pipeline is a Python list of module instances. The following example runs a classic detect-describe-match-filter pipeline with match visualization:

```python

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

### Advanced example: ensemble pipelines with COLMAP export

The following example showcases capabilities that go beyond what prior frameworks offered: rotation-robust matching via `image_muxer_module`, multi-pipeline fusion via `pipeline_muxer_module`, and incremental COLMAP export from independent runs that share a single database making it straightforward to merge results from different matchers in a subsequent reconstruction step.

`pipeline_a` fuses LoFTR and LightGlue (via `deep_joined_module`) under a `pipeline_muxer_module` that takes the union of their matches. That fused pipeline is then wrapped in an `image_muxer_module` with `pair_rot4`, which evaluates four 90° rotations of each image pair and keeps the orientation that yields the most matches, useful for datasets with high degree of rotations. Besides saving the computation in a HDF5 database, all results are exported to a shared COLMAP database.

`pipeline_b` runs RoMa independently saving data in another HDF5 database, but directly merging matches in the same COLMAP database of `pipeline_a`.

```python

def advanced_ensemble_pipeline():
    pipeline_a = [
        image_muxer_module(
            pair_generator=pair_rot4,
            pipe_gather=pipe_max_matches,
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
        to_colmap_module(db='custom_colmap_ab.db'),
    ]
    imgs = '../data/ET'
    run_pairs(pipeline_a, imgs, db_name='database_custom_a.hdf5')

    pipeline_b = [
        roma_module(),
        magsac_module(),
        show_matches_module(img_prefix='matches_', mask_idx=[1, 0], prepend_pair=False),
        to_colmap_module(db='custom_colmap_ab.db'),
    ]
    run_pairs(pipeline_b, imgs, db_name='database_custom_b.hdf5', colmap_db_or_list='custom_colmap_ab.db', mode='include')


if __name__ == '__main__':
    with torch.inference_mode():
        advanced_ensemble_pipeline()
```
This pipeline is already implemented in `src/test_pipelines.py` as `pipeline42()`.

### Module options

Each module exposes a number of configuration options. These are not yet fully documented; please refer to the source code of each module for the available arguments.

### Adding custom modules

Writing a new module is intentionally simple: implement the required interface and drop the file into the appropriate subdirectory. The module can then be composed into any pipeline just like a built-in one. Contributions and new integrations are welcome feel free to open a pull request.

### Device control

Each module accepts an optional `device` argument. `run_pipeline()` detects the target device per module and routes tensors automatically, making it straightforward to mix CPU and GPU stages:

```python
pipeline = [
    loftr_module(device='cpu'),
    magsac_module(device='cuda'),
]
```

### `run_pairs()` options

| Argument | Description |
|---|---|
| `pipeline` | List of modules to execute |
| `imgs` | Directory path or list of image file paths |
| `db_name` | Output HDF5 database filename (default: `database.hdf5`) |
| `db_mode` | Database open mode, typically `'a'` to append |
| `force` | If `True`, rerun modules even when cached results exist |
| `add_path` | Prefix applied to image paths when passing relative pairs |
| `colmap_db_or_list` | Optional COLMAP database or pair list for pair selection |
| `mode` | Pairing mode for `image_pairs` (default: `'exclude'`) |
| `colmap_req` | Required COLMAP data type (default: `'geometry'`) |
| `colmap_min_matches` | Minimum match count for COLMAP-based pairing |

### Automatic pair selection with a transitive pipeline

For datasets too large to match exhaustively, a **transitive pipeline** builds the pair list for a real matching pipeline automatically, using a cheap global-descriptor pass instead of brute-forcing every combination. It's just a regular pipeline passed to `run_pairs()`:

1. A **global descriptor** module (`salad_module`, `standard_descriptor_module`) computes one embedding per image.
2. A **similarity** module (`cosine_similarity_module`, `l2_similarity_module`, `standard_similarity_module`) scores every candidate pair from those embeddings.
3. An **initial-selection** module (`transitive.percentage_module`, `.max_uses_module`, `.kfc_module`) — this is what marks the pipeline as transitive. Since there's no confirmed graph yet to expand from, it decides which pairs to try in the first round.
4. `conf_module` keeps the pairs scoring above a `threshold` and accumulates them — except with `kfc_module` (see below), which needs no `conf_module` at all.
5. Every later round grows the graph **transitively** — if A-B and B-C are confirmed, A-C is tried next — so most of the dataset never needs to be scored pairwise. This only ever has new pairs to find with `percentage_module`/`max_uses_module`, though: both deliberately hold back some already-above-threshold pairs in the seed round (a percentage cap, a per-image cap) for later rounds to pick up. `kfc_module` never holds anything back — it always converges in exactly one round (see below).

```python
coarse_pipeline = [
    salad_module(),
    l2_similarity_module(),
    percentage_module(percentage=0.1),
    conf_module(threshold=-1.0, out_path='pairs.pt'),
]

run_pairs(coarse_pipeline, '../data/ET')
```

`run_pairs()` detects the `percentage_module`/`max_uses_module` in `coarse_pipeline` and drives the round-by-round transitive-closure logic for you (see `transitive.transitive_step.TransitiveRounds` for the algorithm) instead of running once over every possible pair. `max_rounds`, `on_round` and `on_candidates` are extra `run_pairs()` keyword arguments that only apply to a transitive pipeline.

`kfc_module` adapts Keypoint Filtering by Coverage (Bellavia et al., 2022) to seed selection: it keeps every pair above `threshold`, plus whatever extra (even below-threshold) pairs two overlapping maximum-spanning-tree passes over the similarity graph need to guarantee every image stays connected. Unlike `percentage_module`/`max_uses_module`, those extra pairs score *below* `threshold` on purpose — so a separate `conf_module` re-checking that same threshold would just silently discard them again, undoing the whole point. `kfc_module` therefore confirms every pair it selects itself (it keeps its own `_table`, exactly like `conf_module`), so a `kfc_module`-driven pipeline needs no `conf_module` at all:

```python
coarse_pipeline = [
    salad_module(),
    cosine_similarity_module(),
    kfc_module(threshold=0.99, out_path='pairs.pt'),
]

run_pairs(coarse_pipeline, '../data/ET')
```

The confirmed pairs end up in `pairs.pt`, ready to feed a real matching pipeline via `run_pairs(real_pipeline, imgs, colmap_db_or_list=torch.load('pairs.pt'), mode='include')`.

**Watching it live**: `visualization.live_pair_graph` renders the pair graph as an interactive, auto-refreshing HTML page (via pyvis/vis.js) while a transitive `run_pairs()` call — or even a plain one — runs, using its `on_pair`/`on_candidates` hooks. Confirmed pairs appear as thick blue (first round) or green (transitive) edges labeled with their score; everything else shows as a thin dashed red edge. See `pipeline_et_transitive_live()` and `pipeline_et_run_pairs_live()` in `src/test_pipelines.py` for complete examples.

### Nearest-neighbor pair selection with `run_close_pairs()`

A simpler alternative to a transitive pipeline when you just want each image matched against its `n` most similar neighbours, no threshold or rounds involved. It ranks every image against every other using DINOv2 SALAD embeddings, keeps each image's top-`n` closest matches, and runs `pipeline` on the resulting pair list:

```python
run_close_pairs(pipeline, '../data/ET', n=10)
```

`salad_cache` optionally saves the computed embeddings to a `.pt` file so repeated runs (including different chunks in a distributed run, see below) don't recompute them.

### Distributed / multi-machine processing

Two independent ways to spread a run across machines, both used by copying the same script and images to each machine and giving every worker its own `chunk_idx`:

- **Split the pair list** (`run_close_pairs(..., n_chunks=N, chunk_idx=i)`): every worker generates the same full, deterministic pair list, then only actually runs the pairs assigned to it round-robin — full coverage, no coordination needed. Point `colmap_db_or_list` at a COLMAP database shared between workers (e.g. on shared storage) to skip pairs another worker already computed.
- **Split the image set** (`split_images(imgs, n_chunks, chunk_idx)`): divides the images themselves into `n_chunks` disjoint groups; run each group through `run_pairs()` independently (each worker only ever sees pairs within its own group) and merge the resulting per-worker HDF5 databases afterwards with `merge_hdf5([db_chunk0, db_chunk1, ...], merged_db)`.

See `pipeline44()` in `src/test_pipelines.py` for a worked example of the split/merge path.

---

## Module Reference

### Detectors
`dog_module` · `hz_module` · `r2d2_module` · `keynet_module`

### Descriptors
`patch_module` · `deep_descriptor_module` · `sift_module`

### Matchers
`smnn_module` · `lightglue_module` · `loftr_module` · `roma_module` ·  `romav2_module` ·  `loma_module` · `mast3r_module` · `dust3r_module` · `matchformer_module` · `aspanformer_module`

### Filters
`magsac_module` · `poselib_module` · `adalam_module` · `gms_module` · `lpm_module` · `dtm_module` · `fcgnn_module` · `oanet_module` · `acne_module` · `mop_miho_ncc_module`

### Segmentation
`segformer_module` — SegFormer semantic segmentation (Cityscapes labels), flags keypoints/matches falling on specific classes

### Ensemble
`image_muxer_module` · `pipeline_muxer_module` · `pipe_union` · `pipe_max_matches` · `pair_rot4` · `pair_pyramid` · `sampling_module`

### Global Descriptors & Similarity
`salad_module` · `standard_descriptor_module` · `cosine_similarity_module` · `l2_similarity_module` · `standard_similarity_module`

### Pair Selection
`conf_module` · `transitive.percentage_module` · `transitive.max_uses_module` · `transitive.kfc_module`

### Visualization
`show_kpts_module` · `show_matches_module` · `show_patches_module` · `show_homography_module` · `live_pair_graph` 

### COLMAP
`to_colmap_module` · `from_colmap_module` · `merge_colmap_db` · `filter_colmap_reconstruction` · `align_colmap_models`

---

## Repository Structure

```
src/
├── core/
│   ├── device.py              # Device setup, global flags
│   ├── pipeline.py            # run_pipeline, run_pairs (incl. transitive pipelines), finalize_pipeline
│   ├── geometry.py            # Homography and LAF utilities
│   └── utils.py               # Argument handling, serialization, math utils
│
├── detectors/
│   ├── dog_module.py
│   ├── keynet_module.py
│   ├── hz_module.py
│   └── r2d2_module.py
│
├── descriptors/
│   ├── patch_module.py
│   ├── deep_descriptor.py
│   └── sift_module.py
│
├── matchers/
│   ├── smnn_module.py
│   ├── lightglue_module.py
│   ├── loftr_module.py
│   ├── roma_module.py
│   ├── romav2_module.py
│   ├── loma_module.py
│   ├── mast3r_module.py
│   ├── dust3r_module.py
│   ├── matchformer_module.py
│   ├── aspanformer_module.py
│   ├── quadtreeattention.py
│   └── blob_matching.py
│
├── filters/
│   ├── magsac_module.py
│   ├── poselib_module.py
│   ├── lpm_module.py
│   ├── gms_module.py
│   ├── adalam_module.py
│   ├── fcgnn_module.py
│   ├── oanet_module.py
│   ├── acne_module.py
│   ├── dtm_module.py
│   └── mop_miho_ncc_module.py
│
├── segmentators/
│   └── segformer_module.py
│
├── ensemble/
│   ├── sampling.py
│   ├── muxers.py
│   └── pyramid.py
│
├── global_descriptors/
│   ├── salad_module.py
│   └── standard_descriptor.py
│
├── similarity/
│   ├── cosine_similarity_module.py
│   ├── l2_similarity_module.py
│   └── standard_similarity_module.py
│
├── confidence/
│   └── conf_module.py
│
├── transitive/
│   ├── percentage_module.py
│   ├── max_uses_module.py
│   ├── kfc_module.py          # Keypoint Filtering by Coverage (Bellavia et al., 2022), adapted for seed selection
│   └── transitive_step.py     # round-by-round transitive-closure logic, invoked by run_pairs
│
├── colmap/
│   ├── colmap_ext.py
│   ├── to_colmap_module.py
│   ├── from_colmap_module.py
│   └── merge_colmap.py
│
├── benchmark/
│   ├── datasets.py            # MegaDepth, ScanNet, IMC, planar dataset setup
│   ├── metrics.py             # Pose error, AUC, epipolar/homography metrics
│   └── benchmark_module.py    # Pairwise benchmark runner
│
├── visualization/
│   ├── show_kpts.py
│   ├── show_matches.py
│   ├── show_homography.py
│   ├── show_patches.py
│   ├── colorize.py
│   └── live_pair_graph.py     # interactive live pair-graph viewer
│
└── image_pairs.py             # image_pairs iterator
```

---

## Notes

- `src/test_pipelines.py` contains many complete working examples. It is the recommended starting point for understanding how pipelines are composed.
- Ensemble utilities (`pipe_union`, `sampling_module`, `image_muxer_module`, `pipeline_muxer_module`) are useful for combining outputs from multiple sub-pipelines, deduplicating matches, and consolidating results.
- COLMAP integration supports exporting features and matches, importing COLMAP keypoints back into the pipeline, using COLMAP databases for pair selection, and merging results across computation paths.
- The `imgs` argument to `run_pairs()` accepts either a directory path or an explicit list of image paths.
- Results are cached in HDF5 format; set `force=True` to reprocess from scratch.
- The `on_pair(pair, pipe_data)` hook on `run_pairs()`, and its extra `on_candidates(scored, threshold)` hook for transitive pipelines, aren't limited to `live_pair_graph` — any callback with a matching signature works, so custom progress tracking or logging can be wired in the same way.
- `segformer_module` never removes keypoints or matches — it only annotates them (`keypt_mask`, an extra column on `m_mask`) as falling on an specific semantic class, so the original, unfiltered data stays available to any module placed after it.

---

## Roadmap

- [ ] Delete & refactor repeated code across modules
- [ ] Optimize HDF5 database read/write performance
- [ ] Improve documentation
- [ ] Add support for triplet or more images matching
