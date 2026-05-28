import sys
import warnings
from pathlib import Path

import torch

from core import enable_quadtree

project_root = Path(__file__).parent.resolve()

extra_paths = [
    project_root / "r2d2",
    project_root / "mast3r",
    project_root / "matchformer",
    project_root / "aspanformer" / "src",
    project_root / "miho" / "src",
    project_root / "romav2" / "src",
    project_root / "loma" / "src",
    project_root / "gsm"
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
        # Pipelines go from 1 to 38
        
        test_pipelines.pipeline1()

    print('doh!')


