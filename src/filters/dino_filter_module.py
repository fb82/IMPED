from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core import device as global_device
from core import set_args


class dino_filter_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'model': 'dinov2_vits14',
            'resize': 518,
            'sim_th': 0.5,
            'cache_max': 64,
            }
        self.device = torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])

        if 'add_to_cache' in args.keys():
            self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('dino_filter', args, self.args)

        self.model = torch.hub.load('facebookresearch/dinov2', self.args['model']).to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.features = OrderedDict()


    def get_id(self):
        return self.id_string


    def finalize(self):
        self.features.clear()


    def get_features(self, img_path):
        if img_path in self.features:
            self.features.move_to_end(img_path)
            return self.features[img_path]

        im = Image.open(img_path).convert('RGB')
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
        feat = F.normalize(feat, dim=1)

        self.features[img_path] = (feat, W, H)
        if len(self.features) > self.args['cache_max']:
            self.features.popitem(last=False)

        return feat, W, H


    def sample(self, feat, kp, W, H):
        grid = torch.stack([(kp[:, 0] + 0.5) / W * 2 - 1, (kp[:, 1] + 0.5) / H * 2 - 1], dim=1).view(1, 1, -1, 2)
        f = F.grid_sample(feat, grid.float(), mode='bilinear', align_corners=False)
        return F.normalize(f[0, :, 0].T, dim=1)


    def run(self, **args):
        m_idx = args['m_idx']
        m_mask = args['m_mask']

        feat0, W0, H0 = self.get_features(args['img'][0])
        feat1, W1, H1 = self.get_features(args['img'][1])

        kp0 = args['kp'][0][m_idx[:, 0]].to(self.device)
        kp1 = args['kp'][1][m_idx[:, 1]].to(self.device)

        f0 = self.sample(feat0, kp0, W0, H0)
        f1 = self.sample(feat1, kp1, W1, H1)

        sim = (f0 * f1).sum(dim=1)
        keep = (sim >= self.args['sim_th']).to(m_mask.device)

        if m_mask.dim() == 1:
            m_mask = m_mask[:, None]

        return {'m_mask': torch.cat([m_mask, keep[:, None]], dim=1)}
