import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
os.chdir(project_root)
external_root = project_root.parent / "external"

if str(external_root) not in sys.path:
    sys.path.insert(0, str(external_root))

modified_root = external_root / "modified"
if str(modified_root) not in sys.path:
    sys.path.insert(0, str(modified_root))

import warnings

import torch

from core import enable_quadtree

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
if enable_quadtree:
    pass


from imped_tests import test_pipelines

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

if __name__ == '__main__':

    with torch.inference_mode():
        test_pipelines.pipeline62()
        test_pipelines.pipeline64()
        test_pipelines.pipeline65()

        test_pipelines.pipeline14()
        test_pipelines.pipeline66()
        test_pipelines.pipeline15()
        test_pipelines.pipeline16()
        test_pipelines.pipeline20()
        test_pipelines.pipeline21()
        test_pipelines.pipeline21bis()
        test_pipelines.pipeline42()
        test_pipelines.pipeline44()

        test_pipelines.pipeline51()
        test_pipelines.pipeline60()
        test_pipelines.pipeline61()
        test_pipelines.pipeline59()
        test_pipelines.pipeline63()

        test_pipelines.pipeline49()
        test_pipelines.pipeline50()
        test_pipelines.pipeline52()
        test_pipelines.pipeline53()
        test_pipelines.pipeline54()
        test_pipelines.pipeline58()
        test_pipelines.pipeline57(n_datasets=3)
        test_pipelines.pipeline57()

    print('done!')
