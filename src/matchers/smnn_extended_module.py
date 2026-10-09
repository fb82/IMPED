import kornia as K
import torch

from core import device as global_device
from core import set_args, check_data, NMATCHES


class smnn_extended_module:
    """
    Symmetric Mutual Nearest Neighbors matching on a given similarity matrix.

    Same matching as smnn_module, but instead of computing the distances
    from 'desc' it takes the [N0, N1] similarity matrix 'sm' (values in
    [0, 1]) already in the pipeline data, e.g. from similarity_matrix_module
    or the combination of several of them through a pipeline_muxer_module
    with pipe_sm_product. It is turned into the distance matrix 1 - sm
    expected by kornia right before matching.

    Attributes:
        th (float): The distance ratio threshold, as in smnn_module.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True
        self.device = torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])

        self.args = {
            'id_more': '',
            'th': 0.95,
            }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('smnn_extended', args, self.args)

        self.required_output = {
            'm_idx':  [NMATCHES, 2],
            'm_val':  [NMATCHES],
            'm_mask': [NMATCHES],
        }


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        dm = 1 - args['sm'].to(self.device)

        val, idxs = K.feature.match_smnn(dm.new_zeros((dm.shape[0], 1)), dm.new_zeros((dm.shape[1], 1)), self.args['th'], dm=dm)

        result = {'m_idx': idxs, 'm_val': val.squeeze(1), 'm_mask': torch.ones(idxs.shape[0], device=self.device, dtype=torch.bool)}
        check_data(result, self.required_output)
        return result
