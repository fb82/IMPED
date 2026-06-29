__version__ = "0.1.0"

from .device import device, pipe_color, show_progress, enable_quadtree

from .pipeline import (
    run_pipeline,
    run_pairs,
    finalize_pipeline,
    go_iter,
    split_images,
    merge_hdf5,
)
from .geometry import (
    laf2homo, 
    homo2laf,
    apply_homo,
    change_patch_homo,
    decompose_H,
    decompose_H_other
)
from .utils import (
    set_args,
    check_data,
    DONTCARE,
    NKEYPOINTS,
    NMATCHES,
    compressed_pickle,
    decompress_pickle,
    qvec2rotmat,
    vector_norm,
    quaternion_matrix,
    affine_matrix_from_points
)

#from ..image_pairs import image_pairs

#__all__ = ["main_function", "helper1", "helper2"]