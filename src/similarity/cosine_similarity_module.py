import numpy as np
import torch

from core import device as global_device, set_args, check_data


class cosine_similarity_module:
    """
    A pair module computing the cosine similarity between two images' global
    descriptors (e.g. 'global_desc' from salad_module).

    Stores the resulting scalar under 'pair_sim' in the pair-level cache, so
    it can be reused by pair-selection/confidence modules downstream without
    recomputing it.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True

        table = args.pop('table', None)

        self.args = {
            'id_more': '',
            'mode': 'pair',
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('cosine_similarity', args, self.args)

        if self.args['mode'] == 'table':
            self.add_to_cache = False

        self._table = table if table is not None else {}
        self._descs = {}

        self.required_input = {'global_desc': 2}


    def get_id(self):
        return self.id_string


    def finalize(self):
        if self.args['mode'] == 'table':
            self._table.update(cosine_similarity_table(self._descs))


    def run(self, **args):
        check_data(args, self.required_input)

        d0 = args['global_desc'][0]
        d1 = args['global_desc'][1]

        if self.args['mode'] == 'table':
            self._descs[args['img'][0]] = d0
            self._descs[args['img'][1]] = d1

        sim = torch.nn.functional.cosine_similarity(d0.unsqueeze(0), d1.unsqueeze(0)).item()

        return {'pair_sim': sim}


def cosine_similarity_table(descs):
    imgs = sorted(descs)

    stacked = torch.stack([descs[img] for img in imgs]).to(global_device)
    stacked = torch.nn.functional.normalize(stacked, dim=1)
    sim = (stacked @ stacked.T).cpu().numpy()

    rows, cols = np.triu_indices(len(imgs), k=1)
    values = sim[rows, cols].tolist()

    return {(imgs[i], imgs[j]): v for i, j, v in zip(rows.tolist(), cols.tolist(), values)}
