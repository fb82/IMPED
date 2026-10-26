import warnings

import numpy as np
import torch
from PIL import Image

from core import device as global_device
from core import set_args


class superansac_module:
    """
    A geometric verification module using SupeRANSAC (Barath, "SupeRANSAC:
    One RANSAC to Rule them All", 2025), from the external/superansac
    submodule: a drop-in alternative to magsac_module.

    SupeRANSAC combines MAGSAC++ scoring with a stronger pipeline (SPRT
    early rejection, nested RANSAC local optimization, iterated least
    squares final fitting), giving more accurate models than plain MAGSAC++
    at a similar cost.

    Attributes:
        mode (str): 'fundamental_matrix' for 3D scenes or 'homography'
            for planes/rotations.
        px_th (float): Inlier threshold in pixels.
        conf (float): Confidence level (0 to 1).
        min_iters / max_iters (int): Bounds on the RANSAC iterations.
        sampler (str): SupeRANSAC sampler. 'Uniform' by default, since the
            matches reaching this module aren't sorted by quality; 'PROSAC'
            (SupeRANSAC's own default) only makes sense if they are.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'mode': 'fundamental_matrix',
            'conf': 0.9999,
            'min_iters': 1000,
            'max_iters': 5000,
            'px_th': 3,
            'sampler': 'Uniform',
            }
        self.device = torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('superansac', args, self.args)

        # imported here, so that the other filters still load when the
        # SupeRANSAC bindings aren't built
        import pysuperansac
        self.sac = pysuperansac


    def get_config(self):
        # a fresh config for each call: SupeRANSAC writes the normalized
        # threshold (and other refinement settings) back into the config it
        # gets, so reusing it shrinks the threshold at every pair
        config = self.sac.RANSACSettings()
        config.inlier_threshold = self.args['px_th']
        config.confidence = self.args['conf']
        config.min_iterations = self.args['min_iters']
        config.max_iterations = self.args['max_iters']
        config.sampler = getattr(self.sac.SamplerType, self.args['sampler'])
        return config


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        pt1_ = args['kp'][0]
        pt2_ = args['kp'][1]
        mi = args['m_idx']
        mm = args['m_mask']
        if mm.dim() > 1: mm = mm.all(dim=1)

        pt1 = pt1_[mi[mm][:, 0]]
        pt2 = pt2_[mi[mm][:, 1]]

        if torch.is_tensor(pt1):
            pt1 = pt1.detach().cpu().numpy()
            pt2 = pt2.detach().cpu().numpy()

        if self.args['mode'] == 'fundamental_matrix':
            sac_to_run = self.sac.estimateFundamentalMatrix
            sac_min = 7
        else:
            sac_to_run = self.sac.estimateHomography
            sac_min = 4

        F = None
        mask = torch.zeros(pt1.shape[0], device=self.device, dtype=torch.bool)

        if pt1.shape[0] >= sac_min:
            w1, h1 = Image.open(args['img'][0]).size
            w2, h2 = Image.open(args['img'][1]).size
            correspondences = np.ascontiguousarray(np.concatenate([pt1, pt2], axis=1), dtype=np.float64)
            try:
                F, inliers, _, _ = sac_to_run(correspondences, [w1, h1, w2, h2], None, config=self.get_config())
                mask[torch.as_tensor(np.asarray(inliers, dtype=np.int64), device=self.device)] = True
            except Exception as e:
                warnings.warn(f"SupeRANSAC failed: {e}")
                F = None

        aux = mm.clone()
        mm[aux] = mask

        if F is not None:
            F = torch.tensor(np.asarray(F), device=self.device)

        if self.args['mode'] == 'fundamental_matrix':
            return {'m_mask': mm, 'F': F}
        else:
            return {'m_mask': mm, 'H': F}
