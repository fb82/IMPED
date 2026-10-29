import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pycolmap
from PIL import Image

from core import set_args
from reconstruct.reconstruct_module import largest_model
from segmentators.segformer_module import DEFAULT_EXCLUDE, segformer_module

FASTGS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'fastgs'))
MASKED_TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'modified', 'fastgs-splatting', 'fastgs_masked_train.py'))

# what doesn't belong to a scene: people and vehicles, the sky kept
MASK_CLASSES = [c for c in DEFAULT_EXCLUDE if c != 'sky']

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

    With `masks=True` what segformer_module finds of `mask_classes` (by
    default people and vehicles, not the sky) is left out of the training:
      * the SegFormer class maps cached in `seg_cache_dir` (the
        segformer_module default, so a run with segformer_module already
        made them; the missing ones are computed) give a mask of each image,
        undistorted like the images into `output/scene/masks`;
      * the 3D points seen mostly on masked pixels are removed from the
        model, so no Gaussian starts on them;
      * train.py runs through fastgs_masked_train.py, which leaves the
        masked pixels out of the loss.
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
            'masks': False,
            'mask_classes': MASK_CLASSES,
            'seg_cache_dir': 'aux/seg_cache',
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

    def _masks(self, model_path, scene):
        # the SegFormer keep mask of each registered image, written with the
        # image's own name and undistorted by COLMAP exactly like the images,
        # then saved as <image_name>.png, the name FastGS gives the camera
        rec = pycolmap.Reconstruction(model_path)
        names = [image.name for image in rec.images.values() if image.has_pose]
        seg = segformer_module(exclude_classes=self.args['mask_classes'], seg_cache_dir=self.args['seg_cache_dir'])

        masks = os.path.join(scene, 'masks')
        os.makedirs(masks)
        with tempfile.TemporaryDirectory() as tmp:
            for name in names:
                path = os.path.join(self.args['images'], name)
                width, height = Image.open(path).size
                keep = seg._get_keep_mask(path, width, height).cpu().numpy()
                os.makedirs(os.path.dirname(os.path.join(tmp, 'masks', name)), exist_ok=True)
                Image.fromarray(keep.astype(np.uint8) * 255).save(os.path.join(tmp, 'masks', name), quality=100)

            undistorted = os.path.join(tmp, 'undistorted')
            pycolmap.undistort_images(output_path=undistorted, input_path=model_path, image_path=os.path.join(tmp, 'masks'))
            for name in names:
                keep = np.array(Image.open(os.path.join(undistorted, 'images', name)).convert('L')) > 127
                Image.fromarray(keep.astype(np.uint8) * 255).save(os.path.join(masks, os.path.basename(name).split('.')[0] + '.png'))
        seg.finalize()
        return masks

    @staticmethod
    def _drop_masked_points(sparse, masks):
        # removes the 3D points seen on masked pixels in most of their views,
        # one mask in memory at a time
        rec = pycolmap.Reconstruction(sparse)
        views = defaultdict(list)
        for point_id, point in rec.points3D.items():
            for element in point.track.elements:
                views[element.image_id].append((point_id, element.point2D_idx))

        masked = defaultdict(int)
        for image_id, observations in views.items():
            image = rec.images[image_id]
            keep = np.array(Image.open(os.path.join(masks, os.path.basename(image.name).split('.')[0] + '.png'))) > 127
            for point_id, idx in observations:
                x, y = image.points2D[idx].xy
                x = min(max(int(x), 0), keep.shape[1] - 1)
                y = min(max(int(y), 0), keep.shape[0] - 1)
                if not keep[y, x]:
                    masked[point_id] += 1

        drop = [point_id for point_id, n in masked.items() if 2 * n > rec.points3D[point_id].track.length()]
        for point_id in drop:
            rec.delete_point3D(point_id)
        rec.write(sparse)
        print(f"fastgs_module: {len(drop)} of {len(drop) + rec.num_points3D()} points seen mostly on masked pixels removed")

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
        train = ['train.py']
        if self.args['masks']:
            masks = self._masks(model_path, scene)
            self._drop_masked_points(os.path.join(scene, 'sparse', '0'), masks)
            train = [MASKED_TRAIN, '--masks', os.path.abspath(masks)]

        model_out = os.path.abspath(os.path.join(self.args['output'], 'model'))
        iterations = self.args['iterations']
        variant = VARIANTS[self.args['variant']]

        command = [
            sys.executable, *train,
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
              f"{rec.num_points3D()} points), variant {self.args['variant']}"
              + (f", masking {', '.join(self.args['mask_classes'])}" if self.args['masks'] else ''))
        subprocess.run(command, cwd=FASTGS, env=env, check=True)

        self.ply = os.path.join(model_out, 'point_cloud', f'iteration_{iterations}', 'point_cloud.ply')
        print(f"fastgs_module: Gaussians written to {self.ply}")
