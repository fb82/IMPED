import os

from PIL import Image
import torch

from core import device as global_device, set_args
from core.pipeline import _get_salad_model, _SALAD_TRANSFORM


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

        salad_path = os.path.join(os.path.dirname(__file__), '..', 'salad')
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
