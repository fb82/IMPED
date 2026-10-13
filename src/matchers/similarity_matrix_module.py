import torch
import torch.nn.functional as F

from core import device as global_device
from core import set_args, check_data


class similarity_matrix_module:
    """
    A pair module comparing the descriptors of the two images, to be
    matched afterwards by smnn_extended_module.

    The [N0, N1] matrix 'sm' is returned as 1 - similarity, i.e. as a
    distance: lower means more similar, as kornia's matchers expect.
    `metric` sets how it is built:
    - 'cosine': 1 - cosine similarity (clamped at 0), in [0, 1];
    - 'l2' / 'l1': the L2 / L1 distance;
    - 'equal': 0 where the descriptors are equal, 1 otherwise (e.g. for
      the per-keypoint semantic class given by segformer_module).

    The matrix is not cached by default, since it grows with the product
    of the two keypoint counts.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = False
        self.device = torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])

        self.args = {
            'id_more': '',
            'metric': 'cosine',
            }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('similarity_matrix', args, self.args)
        self.id_string += '_' + self.args['metric']

        self.required_input = {'desc': 2}


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        check_data(args, self.required_input)

        d0 = args['desc'][0].to(self.device).float()
        d1 = args['desc'][1].to(self.device).float()

        if self.args['metric'] == 'cosine':
            sm = 1 - (F.normalize(d0, dim=1) @ F.normalize(d1, dim=1).T).clamp(min=0)
        elif self.args['metric'] == 'l2':
            sm = torch.cdist(d0, d1)
        elif self.args['metric'] == 'l1':
            sm = torch.cdist(d0, d1, p=1)
        elif self.args['metric'] == 'equal':
            sm = 1 - (d0[:, None, :] == d1[None, :, :]).all(dim=2).float()
        else:
            raise ValueError(f"unknown metric {self.args['metric']!r}, use 'cosine', 'l2', 'l1' or 'equal'")

        return {'sm': sm}
