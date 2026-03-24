
import torch
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

enable_quadtree = False