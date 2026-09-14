import torch

from core import set_args


class n_matches_similarity_module:
    """
    A pair module that turns a match count into a global similarity score.

    It reads the current match mask ('m_mask' - e.g. the RANSAC inlier mask
    left by magsac_module) and stores the number of surviving matches under
    'pair_sim'.

    Same output key as cosine_similarity_module / l2_similarity_module, so it
    plugs into the same [-> conf_module / kfc_module] cascade unchanged.
    Intended to sit at the end of a cheap low-resolution SIFT + RANSAC pass,
    so that "how many geometrically consistent matches survive" becomes the
    pair similarity.
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

        self.id_string, self.args = set_args('n_matches_similarity', args, self.args)

        self.required_input = {'m_mask': None}


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        m_mask = args.get('m_mask')

        if m_mask is None:
            return {'pair_sim': 0.0}

        if torch.is_tensor(m_mask):
            n_good = int(m_mask.sum().item())
        else:
            n_good = int(sum(bool(x) for x in m_mask))

        return {'pair_sim': float(n_good)}
