import os
import sys
import time
from datetime import timedelta
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

# before any import: TensorFlow (ACNe) reads its log level when it is
# loaded, and the deprecation warnings of timm and dust3r are emitted at
# import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

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

# torch.backends.cudnn.enabled = False

if __name__ == '__main__':

    with torch.inference_mode():
        test_pipelines.pipeline57()
        start = time.perf_counter()
        test_pipelines.ssma_fast(
            images_folder='/home/colombo/Shared/imgs',
            output_folder='/media/dati/Modelli_ssma_annunziata/ssma_fast',
        )
        print(f"ssma_fast: {timedelta(seconds=round(time.perf_counter() - start))} (h:mm:ss)")

        


    print('done!')
