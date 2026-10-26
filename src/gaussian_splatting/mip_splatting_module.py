import os
import shutil
import subprocess
import sys

import pycolmap

from core import set_args
from reconstruct.reconstruct_module import largest_model

MIP_SPLATTING = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'mip-splatting'))

# fuses the 3D smoothing filter into the Gaussians (create_fused_ply.py, which
# only reads iteration 30000), so standard 3DGS viewers render them as
# Mip-Splatting does
FUSE = (
    "import sys\n"
    "from scene.gaussian_model import GaussianModel\n"
    "gaussians = GaussianModel(3)\n"
    "gaussians.load_ply(sys.argv[1])\n"
    "gaussians.save_fused_ply(sys.argv[2])\n"
)


class mip_splatting_module:
    """
    Trains a 3D Gaussian Splatting model of a COLMAP reconstruction with
    Mip-Splatting (Yu et al., "Mip-Splatting: Alias-free 3D Gaussian
    Splatting", CVPR 2024), from the external/mip-splatting submodule: an
    alternative to fastgs_module, slower but alias-free, i.e. without the
    artifacts of 3DGS when the scene is rendered at a different resolution
    or distance than the training images.

    Like reconstruct_module it works in finalize(), so it goes at the end of
    the pipeline that builds or repairs the model, or is called directly;
    with `worklist` it waits until that list is empty. `model` is a
    reconstruction module or a model folder, of which the model with the
    most registered images is used. finalize():
      * undistorts the model and its images with COLMAP into
        `output/scene` (images/ and sparse/0/, PINHOLE cameras, the format
        Mip-Splatting reads);
      * runs Mip-Splatting's train.py on it, in a separate process, writing
        to `output/model`;
      * fuses into the Gaussians the 3D smoothing filter they carry, which
        only Mip-Splatting's renderer applies: `self.ply` is
        `output/model/fused.ply`, for any 3DGS viewer, and `self.raw_ply`
        the training output, for Mip-Splatting's own render.py.

    `resolution` is passed as Mip-Splatting's -r (-1: images wider than
    1600 px are scaled down to 1600); `data_device='cpu'` keeps the images
    off the GPU, for large datasets; `train_args` adds any other train.py
    argument.

    `kernel_size` is the size of the 2D Mip filter (Mip-Splatting's
    default, 0.1); `port` is the one of its network viewer, which train.py
    always opens, so two trainings running at the same time need different
    ports.
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
            'output': 'mip_splatting',
            'iterations': 30000,
            'resolution': -1,
            'data_device': 'cuda',
            'kernel_size': 0.1,
            'port': 6009,
            'train_args': [],
            'worklist': None,
        }

        self.id_string, self.args = set_args('mip_splatting', args, self.args)
        self.ply = None
        self.raw_ply = None

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
            print(f"mip_splatting_module: no model in {model}")
            return
        model_path, rec = largest_model(model)
        if rec is None:
            print("mip_splatting_module: no model to train on")
            return

        scene = self._scene(model_path)
        model_out = os.path.abspath(os.path.join(self.args['output'], 'model'))
        iterations = self.args['iterations']

        command = [
            sys.executable, 'train.py',
            '-s', os.path.abspath(scene),
            '-m', model_out,
            '-r', str(self.args['resolution']),
            '--data_device', self.args['data_device'],
            '--kernel_size', str(self.args['kernel_size']),
            '--iterations', str(iterations),
            '--position_lr_max_steps', str(iterations),
            '--test_iterations', '-1',
            '--save_iterations', str(iterations),
            '--port', str(self.args['port']),
            *map(str, self.args['train_args']),
        ]

        env = dict(os.environ)
        device = str(self.args.get('device', ''))
        if device.startswith('cuda:'):
            env['CUDA_VISIBLE_DEVICES'] = device.split(':')[1]

        print(f"mip_splatting_module: training on {model_path} ({rec.num_reg_images()} images, "
              f"{rec.num_points3D()} points)")
        subprocess.run(command, cwd=MIP_SPLATTING, env=env, check=True)

        self.raw_ply = os.path.join(model_out, 'point_cloud', f'iteration_{iterations}', 'point_cloud.ply')
        self.ply = os.path.join(model_out, 'fused.ply')
        subprocess.run([sys.executable, '-c', FUSE, self.raw_ply, self.ply], cwd=MIP_SPLATTING, env=env, check=True)
        print(f"mip_splatting_module: Gaussians written to {self.ply}")
