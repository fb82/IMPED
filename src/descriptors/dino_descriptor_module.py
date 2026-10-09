import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core import device as global_device
from core import set_args, check_data, DONTCARE, NKEYPOINTS


class dino_descriptor_module:
    """
    A descriptor extraction module using DINOv2 patch features.

    Like sift_module, it takes pre-existing keypoints and only describes
    them: the image is passed once through DINOv2, and each keypoint gets
    the L2-normalized patch feature bilinearly sampled at its position.
    Being normalized, the descriptors are meant to be compared with the
    cosine similarity.
    """
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True
        self.device = torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])

        self.args = {
            'id_more': '',
            'model': 'dinov2_vits14',
            'resize': 518,
            }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('dino_descriptor', args, self.args)

        self.model = torch.hub.load('facebookresearch/dinov2', self.args['model']).to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.required_input = {
            'idx': None,
            'img': DONTCARE,
            'kp':  DONTCARE,
        }
        self.required_output = {'desc': [NKEYPOINTS, DONTCARE]}

    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        check_data(args, self.required_input)

        im = Image.open(args['img'][args['idx']]).convert('RGB')
        W, H = im.size
        s = self.args['resize'] / max(W, H)
        w = max(14, round(W * s / 14) * 14)
        h = max(14, round(H * s / 14) * 14)

        x = torch.from_numpy(np.array(im.resize((w, h), Image.Resampling.BICUBIC))).to(self.device).float()
        x = x.permute(2, 0, 1)[None] / 255
        x = (x - self.mean) / self.std

        with torch.no_grad():
            tokens = self.model.forward_features(x)['x_norm_patchtokens']
        feat = tokens.reshape(1, h // 14, w // 14, -1).permute(0, 3, 1, 2)

        kp = args['kp'][args['idx']].to(self.device)
        grid = torch.stack([(kp[:, 0] + 0.5) / W * 2 - 1, (kp[:, 1] + 0.5) / H * 2 - 1], dim=1).view(1, 1, -1, 2)
        desc = F.grid_sample(feat, grid.float(), mode='bilinear', align_corners=False)[0, :, 0].T
        desc = F.normalize(desc, dim=1)

        result = {'desc': desc}
        check_data(result, self.required_output)
        return result
