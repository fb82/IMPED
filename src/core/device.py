
import threading

import matplotlib
import torch

matplotlib.use('Agg')

import miho.src.miho as mop_miho
import miho.src.miho_other as mop
import miho.src.ncc as ncc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

mop_miho.device = device
mop.device = device
ncc.device = device

# device = torch.device('cpu')
pipe_color = ['red', 'blue', 'lime', 'fuchsia', 'yellow']
show_progress = True

# matplotlib.pyplot keeps global state: modules drawing with it hold this lock,
# so they can run in parallel sub-pipelines of a pipeline_muxer_module.
plot_lock = threading.Lock()

enable_quadtree = False