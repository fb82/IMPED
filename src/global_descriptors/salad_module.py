import os

from PIL import Image
import torch
from torchvision import transforms

from core import device as global_device, set_args


_SALAD_TRANSFORM = transforms.Compose([
    transforms.Resize((322, 322), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_salad_model = None


_SALAD_CONFLICTING_MODULES = ('models', 'vpr_model', 'utils')


def _get_salad_model(salad_path, salad_device):
    global _salad_model
    if _salad_model is None:
        import sys

        # Other submodules (e.g. mast3r/dust3r) may have already cached a 'models'
        # or 'utils' package in sys.modules, which shadows SALAD's own modules.
        # We temporarily evict those entries, do the import, then restore them.
        saved = {}
        for key in list(sys.modules):
            if key in _SALAD_CONFLICTING_MODULES or any(
                key.startswith(m + '.') for m in _SALAD_CONFLICTING_MODULES
            ):
                saved[key] = sys.modules.pop(key)

        sys.path.insert(0, salad_path)
        try:
            from vpr_model import VPRModel
            from models.backbones.dinov2 import DINOV2_ARCHS
            import utils as salad_utils

            # get_loss / get_miner require pytorch_metric_learning which is a
            # training-only dependency. Stub them out — they are never called
            # during inference (forward() only uses backbone + aggregator).
            _orig_get_loss = salad_utils.get_loss
            _orig_get_miner = salad_utils.get_miner
            salad_utils.get_loss = lambda *a, **kw: None
            salad_utils.get_miner = lambda *a, **kw: None

            backbone = 'dinov2_vitb14'
            try:
                model = VPRModel(
                    backbone_arch=backbone,
                    backbone_config={
                        'num_trainable_blocks': 4,
                        'return_token': True,
                        'norm_layer': True,
                    },
                    agg_arch='SALAD',
                    agg_config={
                        'num_channels': DINOV2_ARCHS[backbone],
                        'num_clusters': 64,
                        'cluster_dim': 128,
                        'token_dim': 256,
                    },
                )
            finally:
                salad_utils.get_loss = _orig_get_loss
                salad_utils.get_miner = _orig_get_miner
            model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    'https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt',
                    map_location=torch.device('cpu'),
                )
            )
        finally:
            sys.path.remove(salad_path)
            # Evict the SALAD-specific entries we just imported, then restore the originals.
            for key in list(sys.modules):
                if key in _SALAD_CONFLICTING_MODULES or any(
                    key.startswith(m + '.') for m in _SALAD_CONFLICTING_MODULES
                ):
                    del sys.modules[key]
            sys.modules.update(saved)

        _salad_model = model
        _salad_model.eval()
        _salad_model.to(salad_device)
    return _salad_model


class salad_module:
    """
    A single-image module computing a DINOv2-SALAD global descriptor per image.

    Stores the descriptor under 'global_desc' in the per-image cache, so it
    can be reused by similarity/confidence modules downstream without
    recomputing it, and without needing to be wired up outside the pipeline.
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
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('salad', args, self.args)

        salad_path = os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'salad')
        self.salad_path = os.path.normpath(salad_path)


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        model = _get_salad_model(self.salad_path, self.device)

        img = Image.open(args['img'][args['idx']]).convert('RGB')
        x = _SALAD_TRANSFORM(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            desc = model(x)

        return {'global_desc': desc.squeeze(0).cpu()}
