import torch

from core import set_args, check_data


class l2_similarity_module:
    """
    A pair module computing a similarity score between two images' global
    descriptors (e.g. 'global_desc' from salad_module) based on Euclidean
    (L2) distance.

    Stores the result under 'pair_sim' as the *negative* L2 distance, so
    that — like cosine_similarity_module — higher values mean more similar.
    This keeps the sign convention of 'pair_sim' consistent no matter which
    similarity module produced it, so a downstream threshold-based module
    (e.g. conf_module) doesn't need to know which metric was used.
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

        self.id_string, self.args = set_args('l2_similarity', args, self.args)

        self.required_input = {'global_desc': 2}


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        check_data(args, self.required_input)

        d0 = args['global_desc'][0]
        d1 = args['global_desc'][1]

        sim = -torch.norm(d0 - d1, p=2).item()

        return {'pair_sim': sim}
