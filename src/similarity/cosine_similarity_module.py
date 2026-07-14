import torch

from core import set_args, check_data


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

        self.args = {
            'id_more': '',
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('cosine_similarity', args, self.args)

        self.required_input = {'global_desc': 2}


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        check_data(args, self.required_input)

        d0 = args['global_desc'][0]
        d1 = args['global_desc'][1]

        sim = torch.nn.functional.cosine_similarity(d0.unsqueeze(0), d1.unsqueeze(0)).item()

        return {'pair_sim': sim}
