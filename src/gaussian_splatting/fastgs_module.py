import os
import shutil
import subprocess
import sys

import pycolmap

from core import set_args
from reconstruct.reconstruct_module import largest_model

FASTGS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'fastgs'))

# FastGS's own presets (train_base.sh / train_big.sh): 'big' densifies more
# often and with a lower gradient threshold, for better quality at a
# slightly longer training time
VARIANTS = {
    'base': {'densification_interval': 500, 'grad_abs_thresh': 0.0012},
    'big': {'densification_interval': 100, 'grad_abs_thresh': 0.0005},
}


class fastgs_module:
    """
    Trains a 3D Gaussian Splatting model of a COLMAP reconstruction with
    FastGS (Ren et al., "FastGS: Training 3D Gaussian Splatting in 100
    Seconds", CVPR 2026), from the external/fastgs submodule.

    Like reconstruct_module it works in finalize(), so it goes at the end of
    the pipeline that builds or repairs the model, or is called directly;
    with `worklist` it waits until that list is empty. `model` is a
    reconstruction module or a model folder, of which the model with the
    most registered images is used. finalize():
      * undistorts the model and its images with COLMAP into
        `output/scene` (images/ and sparse/0/, PINHOLE cameras, the format
        FastGS reads);
      * runs FastGS's train.py on it, in a separate process, writing to
        `output/model`; the Gaussians end up in
        `output/model/point_cloud/iteration_<iterations>/point_cloud.ply`
        (kept in `self.ply`), in the standard 3DGS format, so any 3DGS
        viewer opens them.

    `variant` is 'base' (fastest) or 'big' (better quality); `resolution`
    is passed as FastGS's -r (-1: images wider than 1600 px are scaled
    down to 1600); `data_device='cpu'` keeps the images off the GPU, for
    large datasets; `train_args` adds any other train.py argument, e.g.
    ['--mult', '0.7'].
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False

        self.args = {
            'id_more': '',
            'model': None,
            'images': '.',
            'output': 'fastgs',
            'variant': 'base',
            'iterations': 30000,
            'resolution': -1,
            'data_device': 'cuda',
            'train_args': [],
            'worklist': None,
        }

        self.id_string, self.args = set_args('fastgs', args, self.args)
        self.ply = None

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        return {}

    def _scene(self, model_path):
        scene = os.path.join(self.args['output'], 'scene')
        shutil.rmtree(scene, ignore_errors=True)
        os.makedirs(self.args['output'], exist_ok=True)
        pycolmap.undistort_images(output_path=scene, input_path=model_path, image_path=self.args['images'])

        sparse = os.path.join(scene, 'sparse')
        os.makedirs(os.path.join(sparse, '0'))
        for name in os.listdir(sparse):
            if name != '0':
                shutil.move(os.path.join(sparse, name), os.path.join(sparse, '0', name))
        return scene

    def finalize(self):
        if self.args['worklist']:
            return

        model = self.args['model']
        if isinstance(model, str) and not os.path.isdir(model):
            print(f"fastgs_module: no model in {model}")
            return
        model_path, rec = largest_model(model)
        if rec is None:
            print("fastgs_module: no model to train on")
            return

        scene = self._scene(model_path)
        model_out = os.path.abspath(os.path.join(self.args['output'], 'model'))
        iterations = self.args['iterations']
        variant = VARIANTS[self.args['variant']]

        command = [
            sys.executable, 'train.py',
            '-s', os.path.abspath(scene),
            '-m', model_out,
            '-r', str(self.args['resolution']),
            '--data_device', self.args['data_device'],
            '--optimizer_type', 'default',
            '--iterations', str(iterations),
            '--position_lr_max_steps', str(iterations),
            '--densification_interval', str(variant['densification_interval']),
            '--grad_abs_thresh', str(variant['grad_abs_thresh']),
            '--test_iterations', '-1',
            '--checkpoint_iterations', '-1',
            *map(str, self.args['train_args']),
        ]

        env = dict(os.environ)
        device = str(self.args.get('device', ''))
        if device.startswith('cuda:'):
            env['CUDA_VISIBLE_DEVICES'] = device.split(':')[1]

        print(f"fastgs_module: training on {model_path} ({rec.num_reg_images()} images, "
              f"{rec.num_points3D()} points), variant {self.args['variant']}")
        subprocess.run(command, cwd=FASTGS, env=env, check=True)

        self.ply = os.path.join(model_out, 'point_cloud', f'iteration_{iterations}', 'point_cloud.ply')
        print(f"fastgs_module: Gaussians written to {self.ply}")
