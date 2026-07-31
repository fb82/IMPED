import os
import tempfile

import cv2
import torch

from core import device as global_device, set_args
from descriptors.sift_module import sift_module
from detectors.dog_module import dog_module


def compute_standard_descriptors(imgs):
    """
    Bulk-computes standard (SIFT) global descriptors for a list of image
    paths, keyed by path.
    """
    module = standard_descriptor_module(add_to_cache=False)
    return {img: module.run(idx=0, img=[img])['global_desc'] for img in imgs}


class standard_descriptor_module:
    """
    A single-image module computing a classical (SIFT-based) global
    descriptor per image, by chaining the DoG detector and SIFT descriptor
    modules and packaging their output under 'global_desc'.
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

        self.detector = dog_module(device=self.device, add_to_cache=False)
        self.descriptor = sift_module(device=self.device, add_to_cache=False)


    def get_id(self):
        return self.id_string


    def finalize(self):
        return


    def run(self, **args):
        img = args['img'][args['idx']]
        size = self.args['size']

        im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)

        fd, resized_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        try:
            cv2.imwrite(resized_path, im)

            det = self.detector.run(idx=0, img=[resized_path])
            desc = self.descriptor.run(idx=0, img=[resized_path], kp=[det['kp']], kH=[det['kH']])
        finally:
            os.remove(resized_path)

        global_desc = {
            'kp': det['kp'].to(self.device),
            'desc': desc['desc'].to(self.device),
        }

        return {'global_desc': global_desc}
