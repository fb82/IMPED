import kornia as K
import torch

from core import device as global_device
from core import laf2homo, set_args


class dog_v2_module:
    """
    A Difference of Gaussians (DoG) detector computed by Kornia on the GPU.

    It is the same detector as dog_module (the one of the SIFT framework),
    but instead of OpenCV it uses the scale-space detector of Kornia's
    SIFTFeatureScaleSpace: a ScaleSpaceDetector with BlobDoG responses,
    sub-pixel ConvQuadInterp3d refinement and dominant gradient orientation.
    Since Kornia 0.9.0 its extrema refinement and orientation histograms are
    batched (kornia PR #4254), which makes it faster than OpenCV's on the GPU.

    It outputs the same 'kp', 'kH', 'kr' as dog_module, so it can replace it
    in front of any descriptor (sift_module, patch_module,
    deep_descriptor_module...).

    Attributes:
        upright (bool): If True, all keypoints have the same orientation
            (no orientation estimation).
        num_features (int): Number of strongest DoG extrema kept (default:
            8000, as dog_module, which also has the thresholds disabled).
    """
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True

        self.device =  torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])

        self.args = {
            'id_more': '',
            'upright': False,
            'num_features': 8000,
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('dog_v2', args, self.args)
        self.detector = K.feature.SIFTFeatureScaleSpace(
            num_features=self.args['num_features'],
            upright=self.args['upright'],
            device=self.device,
        ).detector.eval()


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        img = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=self.device).unsqueeze(0)
        kp, kr = self.detector(img)
        kp = kp.detach().to(self.device).squeeze(0)
        kr = kr.detach().to(self.device).squeeze(0)

        # since Kornia 0.9.0 the output is padded to num_features with zero
        # LAFs, which cannot be inverted by laf2homo
        filled = torch.linalg.det(kp[:, :, :2]).abs() > 0
        kp, kH = laf2homo(kp[filled])

        return {'kp': kp, 'kH': kH, 'kr': kr[filled]}
