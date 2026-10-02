import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images, _crop_to_supported_aspect_ratio
from vggt_omega.utils.pose_enc import encoding_to_camera

from core import device as global_device
from core import set_args


class vggt_omega_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'checkpoint': 'vggt_omega_1b_512.pt',
            'resize': 512,
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

        self.id_string, self.args = set_args('vggt_omega', args, self.args)

        checkpoint_path = hf_hub_download('facebook/VGGT-Omega', self.args['checkpoint'])
        self.model = VGGTOmega().eval()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = self.model.to(self.device)


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def to_original(self, img_path, kps, H, W):
        h, w = load_and_preprocess_images([img_path], image_resolution=self.args['resize']).shape[-2:]
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            W0, H0 = im.size
            cW, cH = _crop_to_supported_aspect_ratio(im).size

        kps = kps.clone()
        kps[:, 0] = (kps[:, 0] - (W - w) // 2) * cW / w + (W0 - cW) // 2
        kps[:, 1] = (kps[:, 1] - (H - h) // 2) * cH / h + (H0 - cH) // 2
        return kps


    def run(self, **args):
        image0 = args['img'][0]
        image1 = args['img'][1]

        images = load_and_preprocess_images([image0, image1], image_resolution=self.args['resize']).to(self.device)
        predictions = self.model(images)

        H, W = predictions['images'].shape[-2:]
        extrinsics, intrinsics = encoding_to_camera(predictions['pose_enc'], (H, W))
        extrinsics = extrinsics[0].float()
        intrinsics = intrinsics[0].float()
        depth = predictions['depth'][0, ..., 0].float()
        conf = predictions['depth_conf'][0].float()

        s = self.args['stride']
        y, x = torch.meshgrid(
            torch.arange(0, H, s, device=self.device, dtype=torch.float),
            torch.arange(0, W, s, device=self.device, dtype=torch.float),
            indexing='ij')
        x = x.flatten()
        y = y.flatten()
        d0 = depth[0][y.long(), x.long()]
        c0 = conf[0][y.long(), x.long()]

        K0 = intrinsics[0]
        p_cam0 = torch.stack([(x - K0[0, 2]) / K0[0, 0] * d0, (y - K0[1, 2]) / K0[1, 1] * d0, d0], dim=1)

        R0 = extrinsics[0, :, :3]
        t0 = extrinsics[0, :, 3]
        p_world = (p_cam0 - t0) @ R0

        R1 = extrinsics[1, :, :3]
        t1 = extrinsics[1, :, 3]
        p_cam1 = p_world @ R1.T + t1

        K1 = intrinsics[1]
        z1 = p_cam1[:, 2]
        x1 = K1[0, 0] * p_cam1[:, 0] / z1 + K1[0, 2]
        y1 = K1[1, 1] * p_cam1[:, 1] / z1 + K1[1, 2]

        valid = (d0 > 0) & (z1 > 0) & (x1 >= 0) & (x1 <= W - 1) & (y1 >= 0) & (y1 <= H - 1)
        x, y, c0, x1, y1, z1 = x[valid], y[valid], c0[valid], x1[valid], y1[valid], z1[valid]

        xi = x1.round().long()
        yi = y1.round().long()
        d1 = depth[1][yi, xi]
        c1 = conf[1][yi, xi]

        consistent = (z1 - d1).abs() < self.args['depth_tol'] * d1
        x, y, x1, y1 = x[consistent], y[consistent], x1[consistent], y1[consistent]
        score = (c0[consistent] * c1[consistent]).sqrt()

        score, order = torch.sort(score, descending=True)
        if self.args['max_keypoints'] is not None:
            order = order[:self.args['max_keypoints']]
            score = score[:self.args['max_keypoints']]

        kps1 = torch.stack([x[order], y[order]], dim=1)
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
