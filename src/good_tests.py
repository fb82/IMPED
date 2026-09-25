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
        # # PASSED
        test_pipelines.pipeline1()
        test_pipelines.pipeline2()  
        test_pipelines.pipeline3()
        test_pipelines.pipeline4()
        test_pipelines.pipeline5()
        test_pipelines.pipeline6()
        test_pipelines.pipeline7()
        test_pipelines.pipeline8()
        test_pipelines.pipeline9()
        test_pipelines.pipeline10()


        test_pipelines.pipeline11()
        test_pipelines.pipeline12()
        test_pipelines.pipeline13()
        test_pipelines.pipeline14()
        test_pipelines.pipeline15()
        test_pipelines.pipeline16()
        test_pipelines.pipeline17()
        test_pipelines.pipeline18()
        test_pipelines.pipeline19()
        test_pipelines.pipeline20()

        test_pipelines.pipeline21()
        test_pipelines.pipeline21bis()
        test_pipelines.pipeline22()  
        test_pipelines.pipeline23()
        test_pipelines.pipeline24()
        test_pipelines.pipeline25()
        test_pipelines.pipeline26()
        test_pipelines.pipeline27()
        test_pipelines.pipeline28()
        test_pipelines.pipeline29()
        test_pipelines.pipeline30()

        test_pipelines.pipeline31()
        test_pipelines.pipeline32()
        test_pipelines.pipeline33()
        test_pipelines.pipeline34()
        test_pipelines.pipeline35()
        test_pipelines.pipeline36()
        test_pipelines.pipeline37()
        test_pipelines.pipeline38()
        test_pipelines.pipeline39()
        test_pipelines.pipeline40()

        test_pipelines.pipeline41()
        test_pipelines.pipeline42()
        test_pipelines.pipeline43()
        test_pipelines.pipeline44()
        test_pipelines.pipeline45()
        test_pipelines.pipeline46()
        test_pipelines.pipeline47()
        test_pipelines.pipeline48()
        test_pipelines.pipeline49()
        test_pipelines.pipeline50()

        test_pipelines.pipeline51()
        test_pipelines.pipeline52()
        test_pipelines.pipeline53()
        test_pipelines.pipeline54()

        print('doh!')


