import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

from core import device as global_device
from core import set_args


class mapanything_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'model': 'facebook/map-anything',
            'resolution_set': 518,
            'max_keypoints': 2000,
            'stride': 2,
            'depth_tol': 0.05,
            'patch_radius': 16,
            }
        self.device = torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])

        if 'add_to_cache' in args.keys():
            self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('mapanything', args, self.args)

        self.model = MapAnything.from_pretrained(self.args['model']).to(self.device).eval()


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def to_original(self, img_path, kps, H, W):
        with Image.open(img_path) as im:
            W0, H0 = exif_transpose(im).size

        s = max(W / W0, H / H0) + 1e-8
        rW = int(W0 * s)
        rH = int(H0 * s)

        kps = kps.clone()
        kps[:, 0] = (kps[:, 0] + (rW - W) // 2) * W0 / rW
        kps[:, 1] = (kps[:, 1] + (rH - H) // 2) * H0 / rH
        return kps


    def run(self, **args):
        image0 = args['img'][0]
        image1 = args['img'][1]

        views = load_images([image0, image1], resolution_set=self.args['resolution_set'])
        predictions = self.model.infer(views, memory_efficient_inference=False, use_amp=True, amp_dtype='bf16', apply_mask=True, mask_edges=True)

        pred0 = predictions[0]
        pred1 = predictions[1]

        pts0 = pred0['pts3d'][0].float().to(self.device)
        mask0 = pred0['mask'][0, ..., 0].bool().to(self.device)
        conf0 = pred0['conf'][0].float().to(self.device)

        depth1 = pred1['depth_z'][0, ..., 0].float().to(self.device)
        mask1 = pred1['mask'][0, ..., 0].bool().to(self.device)
        conf1 = pred1['conf'][0].float().to(self.device)
        K1 = pred1['intrinsics'][0].float().to(self.device)
        pose1 = pred1['camera_poses'][0].float().to(self.device)

        H, W = depth1.shape

        s = self.args['stride']
        y, x = torch.meshgrid(
            torch.arange(0, H, s, device=self.device),
            torch.arange(0, W, s, device=self.device),
            indexing='ij')
        x = x.flatten()
        y = y.flatten()

        valid = mask0[y, x]
        x, y = x[valid], y[valid]
        p_world = pts0[y, x]
        c0 = conf0[y, x]

        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        p_cam1 = (p_world - t1) @ R1

        z1 = p_cam1[:, 2]
        x1 = K1[0, 0] * p_cam1[:, 0] / z1 + K1[0, 2]
        y1 = K1[1, 1] * p_cam1[:, 1] / z1 + K1[1, 2]

        valid = (z1 > 0) & (x1 >= 0) & (x1 <= W - 1) & (y1 >= 0) & (y1 <= H - 1)
        x, y, c0, x1, y1, z1 = x[valid], y[valid], c0[valid], x1[valid], y1[valid], z1[valid]

        xi = x1.round().long()
        yi = y1.round().long()
        d1 = depth1[yi, xi]
        c1 = conf1[yi, xi]

        consistent = mask1[yi, xi] & ((z1 - d1).abs() < self.args['depth_tol'] * d1)
        x, y, x1, y1 = x[consistent], y[consistent], x1[consistent], y1[consistent]
        score = (c0[consistent] * c1[consistent]).sqrt()

        score, order = torch.sort(score, descending=True)
        if self.args['max_keypoints'] is not None:
            order = order[:self.args['max_keypoints']]
            score = score[:self.args['max_keypoints']]

        kps1 = torch.stack([x[order], y[order]], dim=1).float()
        kps2 = torch.stack([x1[order], y1[order]], dim=1)

        kps1 = self.to_original(image0, kps1, H, W).detach().to(self.device)
        kps2 = self.to_original(image1, kps2, H, W).detach().to(self.device)

        kp = [kps1, kps2]

        kH = [
            torch.zeros((kp[0].shape[0], 3, 3), device=self.device),
            torch.zeros((kp[0].shape[0], 3, 3), device=self.device),
        ]

        kH[0][:, [0, 1], 2] = -kp[0] / self.args['patch_radius']
        kH[0][:, 0, 0] = 1 / self.args['patch_radius']
        kH[0][:, 1, 1] = 1 / self.args['patch_radius']
        kH[0][:, 2, 2] = 1

        kH[1][:, [0, 1], 2] = -kp[1] / self.args['patch_radius']
        kH[1][:, 0, 0] = 1 / self.args['patch_radius']
        kH[1][:, 1, 1] = 1 / self.args['patch_radius']
        kH[1][:, 2, 2] = 1

        kr = [
            torch.full((kp[0].shape[0],), torch.nan, device=self.device),
            torch.full((kp[0].shape[0],), torch.nan, device=self.device)
        ]

        m_idx = torch.zeros((kp[0].shape[0], 2), device=self.device, dtype=torch.int)
        m_idx[:, 0] = torch.arange(kp[0].shape[0])
        m_idx[:, 1] = torch.arange(kp[0].shape[0])

        m_mask = torch.ones(m_idx.shape[0], device=self.device, dtype=torch.bool)

        m_val = score.detach().to(self.device)

        return {
            'kp': kp,
            'kH': kH,
            'kr': kr,
            'm_idx': m_idx,
            'm_val': m_val,
            'm_mask': m_mask
        }
