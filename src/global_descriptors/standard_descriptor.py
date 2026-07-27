import concurrent.futures
import os

import cv2
import numpy as np
import torch

from core import device as global_device, set_args


def _compute_one(img_path, size):
    detector = cv2.SIFT_create()

    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)

    kp, desc = detector.detectAndCompute(im, None)

    if desc is None:
        desc = np.zeros((0, 128), dtype=np.float32)
        pts = np.zeros((0, 2), dtype=np.float32)
    else:
        pts = np.array([k.pt for k in kp], dtype=np.float32)

    return {
        'kp': torch.tensor(pts, dtype=torch.float),
        'desc': torch.tensor(desc, dtype=torch.float),
    }


def compute_standard_descriptors(imgs, size=128, n_jobs=None):
    """
    Bulk-computes standard (SIFT) global descriptors for a list of image
    paths, keyed by path. n_jobs=None (default) runs sequentially, matching
    standard_descriptor_module.run() one image at a time; n_jobs=-1 or an
    int > 1 parallelizes across a process pool (one worker per CPU when -1)

    """
    if n_jobs is None or n_jobs == 1:
        return {img: _compute_one(img, size) for img in imgs}

    workers = os.cpu_count() if n_jobs == -1 else n_jobs
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_compute_one, imgs, [size] * len(imgs)))

    return dict(zip(imgs, results))


class standard_descriptor_module:
    """
    A single-image module computing a classical (SIFT-based) global
    descriptor per image.
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
            'size': 128,
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('standard', args, self.args)


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        desc = _compute_one(args['img'][args['idx']], self.args['size'])

        global_desc = {
            'kp': desc['kp'].to(self.device),
            'desc': desc['desc'].to(self.device),
        }

        return {'global_desc': global_desc}
