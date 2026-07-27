import cv2
import numpy as np

from core import set_args, check_data


class standard_similarity_module:
    """
    A pair module computing the "standard" global descriptor similarity: the
    number of good matches found between two images' SIFT descriptor sets
    (e.g. 'global_desc' from standard_descriptor_module), matched with a
    brute-force ratio test. Alternative to cosine_similarity_module /
    l2_similarity_module, meant to be paired with standard_descriptor_module
    instead of salad_module.

    Stores the resulting match count under 'pair_sim' — same key as
    cosine_similarity_module/l2_similarity_module — so it plugs into the
    same [-> conf_module] cascade unchanged.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'ratio': 0.75,
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('standard_similarity', args, self.args)

        self.required_input = {'global_desc': 2}

        self.matcher = cv2.BFMatcher()


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        check_data(args, self.required_input)

        d0 = args['global_desc'][0]['desc'].cpu().numpy()
        d1 = args['global_desc'][1]['desc'].cpu().numpy()

        if d0.shape[0] < 2 or d1.shape[0] < 2:
            return {'pair_sim': 0.0}

        knn = self.matcher.knnMatch(d0.astype(np.float32), d1.astype(np.float32), k=2)

        n_good = sum(1 for m, n in knn if m.distance < self.args['ratio'] * n.distance)

        return {'pair_sim': float(n_good)}
