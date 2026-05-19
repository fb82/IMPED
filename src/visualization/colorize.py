
import cv2
import numpy as np
import torch
from matplotlib import colormaps

from core import device as global_device

def colorize_plane(ims, heat, cmap_name='viridis', max_val=45, cf=0.7, save_to='plane_acc.png', device=None):
    """
    Applies a colored heatmap overlay to a grayscale image to visualize planar regions.

    This function takes a 2D 'heat' tensor (where values typically represent 
    plane IDs or confidence scores) and maps them to colors using a specified 
    colormap. The resulting heatmap is then blended with a grayscale version 
    of the original image.

    Args:
        ims (str): Path to the source background image file.
        heat (torch.Tensor): A 2D tensor of shape (H, W). Values should be 
            integers where -1 represents the background (no plane) and 
            non-negative values represent specific regions/planes.
        cmap_name (str): The name of the Matplotlib colormap to use 
            (default: 'viridis').
        max_val (int): The maximum value used to normalize the colormap scale.
        cf (float): The transparency/alpha factor (0.0 to 1.0). Higher values 
            make the heatmap more opaque.
        save_to (str): The file path where the final blended image will be saved.

    Returns:
        None: The function writes the output image directly to disk.

    Note:
        The function handles the conversion from standard RGB colormaps to BGR 
        to ensure compatibility with OpenCV's `imwrite` format.
    """
    device = device if device is not None else global_device
    im_gray = cv2.imread(ims, cv2.IMREAD_GRAYSCALE)
    im_gray = torch.tensor(im_gray, device=device).unsqueeze(0).repeat(3,1,1).permute(1,2,0)
    heat_mask = heat != -1
    heat_ = heat.clone()
    cmap = (colormaps[cmap_name](np.arange(0,(max_val + 1)) / max_val))[:, [2, 1, 0]]
    heat_[heat_ > max_val - 1] = max_val - 1
    heat_[heat_ == -1] = max_val
    cmap = torch.tensor(cmap, device=device)
    heat_im = cmap[heat_.type(torch.long)]
    heat_im = heat_im.type(torch.float) * 255
    blend_mask = heat_mask.unsqueeze(-1).type(torch.float) * cf
    imm = heat_im * blend_mask + im_gray.type(torch.float) * (1 - blend_mask)                    
    cv2.imwrite(save_to, imm.type(torch.uint8).detach().cpu().numpy())   
 
