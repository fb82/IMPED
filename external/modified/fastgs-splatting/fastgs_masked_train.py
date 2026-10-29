"""
Runs FastGS's train.py ignoring the pixels outside a mask, without changing
FastGS: started by fastgs_module from the FastGS folder, as

    python fastgs_masked_train.py --masks <folder> <train.py arguments>

`<folder>` holds one <image name without extension>.png per image (the
image_name FastGS gives to its cameras), white where the image is kept and
black on what is removed (people, cars...). Each masked pixel of the
reference images is set to NaN when the cameras are loaded, and every
function comparing a render with a reference image replaces it with the
rendered value: the masked pixels give no loss and no gradient, and don't
drive the densification either. A camera without a mask file is kept whole.
"""
import os
import runpy
import sys

import numpy as np
import torch
from PIL import Image

masks = sys.argv[sys.argv.index('--masks') + 1]
del sys.argv[sys.argv.index('--masks'):sys.argv.index('--masks') + 2]
sys.argv[0] = 'train.py'
sys.path[0] = os.getcwd()

import fused_ssim
import utils.camera_utils as camera_utils
import utils.fast_utils as fast_utils
import utils.loss_utils as loss_utils


def fill(gt, image):
    return torch.where(torch.isnan(gt), image.detach(), gt)


load_cam = camera_utils.loadCam
l1_loss = loss_utils.l1_loss
ssim = fused_ssim.fused_ssim
get_loss = fast_utils.get_loss


def masked_load_cam(args, id, cam_info, resolution_scale):
    cam = load_cam(args, id, cam_info, resolution_scale)
    path = os.path.join(masks, cam_info.image_name + '.png')
    if os.path.isfile(path):
        keep = Image.open(path).convert('L').resize((cam.image_width, cam.image_height), Image.NEAREST)
        keep = torch.from_numpy(np.array(keep) > 127).to(cam.original_image.device)
        cam.original_image[:, ~keep] = float('nan')
    return cam


def masked_l1_loss(network_output, gt):
    return l1_loss(network_output, fill(gt, network_output))


def masked_ssim(img1, img2, *args, **kwargs):
    return ssim(img1, fill(img2, img1), *args, **kwargs)


def masked_get_loss(reconstructed_image, original_image):
    return get_loss(reconstructed_image, fill(original_image, reconstructed_image))


# train.py imports these names when it is run below; fast_utils has
# already bound its own copies
camera_utils.loadCam = masked_load_cam
loss_utils.l1_loss = masked_l1_loss
fused_ssim.fused_ssim = masked_ssim
fast_utils.l1_loss = masked_l1_loss
fast_utils.fast_ssim = masked_ssim
fast_utils.get_loss = masked_get_loss

runpy.run_path('train.py', run_name='__main__')
