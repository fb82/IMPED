import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
external_root = project_root.parent / "external"

if str(external_root) not in sys.path:
    sys.path.insert(0, str(external_root))

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
    external_root / "gsm"
]


for p in extra_paths:
    if p.exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
if enable_quadtree:
    pass


import test_pipelines

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

if __name__ == '__main__':

    with torch.inference_mode():
        test_pipelines.pipeline_ssma_transitive()

    print('done!')


