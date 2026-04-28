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


        # PASSED
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


        print('Running pipeline 11')
        test_pipelines.pipeline11()
        print('Running pipeline 12')
        test_pipelines.pipeline12()
        print('Running pipeline 13')
        test_pipelines.pipeline13()
        print('Running pipeline 14')
        test_pipelines.pipeline14()
        print('Running pipeline 15')
        test_pipelines.pipeline15()
        print('Running pipeline 16')
        test_pipelines.pipeline16()
        print('Running pipeline 17')
        test_pipelines.pipeline17()
        print('Running pipeline 18')
        test_pipelines.pipeline18()
        print('Running pipeline 19')
        test_pipelines.pipeline19()
        print('Running pipeline 20')
        test_pipelines.pipeline20()

        print('Running pipeline 21')
        test_pipelines.pipeline21()
        print('Running pipeline 22')
        test_pipelines.pipeline22()       
        print('Running pipeline 23')
        test_pipelines.pipeline23()
        print('Running pipeline 24')
        test_pipelines.pipeline24()
        print('Running pipeline 25')
        test_pipelines.pipeline25()
        print('Running pipeline 26')
        test_pipelines.pipeline26()
        print('Running pipeline 27')
        test_pipelines.pipeline27()
        print('Running pipeline 28')
        test_pipelines.pipeline28()
        print('Running pipeline 29')
        test_pipelines.pipeline29()
        print('Running pipeline 30')
        test_pipelines.pipeline30()

        print('Running pipeline 31')
        test_pipelines.pipeline31()
        print('Running pipeline 32')
        test_pipelines.pipeline32()
        print('Running pipeline 33')
        test_pipelines.pipeline33()
        print('Running pipeline 34')
        test_pipelines.pipeline34()
        print('Running pipeline 35')
        test_pipelines.pipeline35()
        print('Running pipeline 36')
        test_pipelines.pipeline36()
        print('Running pipeline 37')
        test_pipelines.pipeline37()

        print('doh!')


