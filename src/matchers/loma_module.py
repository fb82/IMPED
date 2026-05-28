import torch
from loma import LoMa, LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR

from core import device as global_device
from core import set_args


_BACKBONE_MAP = {
    'loma-b':    LoMaB,
    'loma-b128': LoMaB128,
    'loma-l':    LoMaL,
    'loma-g':    LoMaG,
    'loma-r':    LoMaR,
}


class loma_module:
    """
    A local feature matching module using the LoMa family of matchers.

    LoMa works similarly to LightGlue but with significantly improved
    robustness and accuracy. It leverages local keypoint descriptions and
    is a drop-in replacement in SfM and Visual Localization pipelines.
    Multiple backbone sizes are available: B (base), B128, L (large),
    G (giant), and R (rotation-invariant).

    Attributes:
        backbone (str): Which LoMa backbone to load. One of:
            'loma-b', 'loma-b128', 'loma-l', 'loma-g', 'loma-r'.
        max_keypoints (int or None): If set, caps the number of returned
            matches by retaining only the top-scoring ones.
        patch_radius (int): Radius used to build the per-keypoint local
            homography that other pipeline stages may consume.
    """

    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True

        self.args = {
            'id_more': '',
            'backbone': 'loma-b',   # 'loma-b' | 'loma-b128' | 'loma-l' | 'loma-g' | 'loma-r'
            'max_keypoints': 2000,
            'patch_radius': 16,
        }

        self.device = torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])

        if 'add_to_cache' in args:
            self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('loma', args, self.args)

        backbone_key = self.args['backbone']
        if backbone_key not in _BACKBONE_MAP:
            raise ValueError(
                f"Unknown LoMa backbone '{backbone_key}'. "
                f"Choose one of: {list(_BACKBONE_MAP.keys())}"
            )

        backbone_cls = _BACKBONE_MAP[backbone_key]
        self.loma_model = LoMa(backbone_cls())
        # Move the underlying network to the requested device if the
        # LoMa API exposes a .to() method (it inherits from nn.Module).
        if hasattr(self.loma_model, 'to'):
            self.loma_model = self.loma_model.to(self.device)


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        img0_path = args['img'][0]
        img1_path = args['img'][1]

        # LoMa returns numpy arrays in image-pixel coordinates.
        kptsA, kptsB = self.loma_model.match(img0_path, img1_path)

        # Convert to torch tensors on the target device.
        kps1 = torch.tensor(kptsA, dtype=torch.float32, device=self.device)
        kps2 = torch.tensor(kptsB, dtype=torch.float32, device=self.device)

        # Optionally cap the number of matches (keep highest-index ones;
        # LoMa already sorts by confidence descending).
        if self.args['max_keypoints'] is not None:
            n = min(self.args['max_keypoints'], kps1.shape[0])
            kps1 = kps1[:n]
            kps2 = kps2[:n]

        n_matches = kps1.shape[0]
        kp = [kps1, kps2]

        # Build per-keypoint local homographies (same convention as
        # roma_module: a similarity centred on each keypoint scaled by
        # patch_radius).
        r = self.args['patch_radius']
        kH = [
            torch.zeros((n_matches, 3, 3), device=self.device),
            torch.zeros((n_matches, 3, 3), device=self.device),
        ]
        for i in range(2):
            kH[i][:, [0, 1], 2] = -kp[i] / r
            kH[i][:, 0, 0] = 1.0 / r
            kH[i][:, 1, 1] = 1.0 / r
            kH[i][:, 2, 2] = 1.0

        # LoMa does not expose per-match confidence scores through its
        # public API, so we fill kr (orientation) with NaN and m_val
        # (match score) with ones.
        kr = [
            torch.full((n_matches,), float('nan'), device=self.device),
            torch.full((n_matches,), float('nan'), device=self.device),
        ]

        m_idx = torch.zeros((n_matches, 2), device=self.device, dtype=torch.int)
        m_idx[:, 0] = torch.arange(n_matches)
        m_idx[:, 1] = torch.arange(n_matches)

        m_mask = torch.ones(n_matches, device=self.device, dtype=torch.bool)
        m_val = torch.ones(n_matches, device=self.device)

        return {
            'kp': kp,
            'kH': kH,
            'kr': kr,
            'm_idx': m_idx,
            'm_val': m_val,
            'm_mask': m_mask,
        }