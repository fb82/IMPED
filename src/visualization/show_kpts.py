import os

import cv2
import kornia as K
import matplotlib.pyplot as plt

from core import homo2laf, set_args


class show_kpts_module:
    """
    A module for visualizing Local Affine Frames (LAFs) on a per-image basis.

    It renders keypoints not just as dots, but as ellipses or oriented frames 
    that represent the local geometry (scale and orientation) detected by 
    modules like 'patch_module' or 'sift_module'. 

    It can operate in 'single_image' mode (one file per image) or 
    'matching' mode (filtering points based on whether they were 
    successfully matched).

    Attributes:
        mask_idx: If None, shows all detected points. If a list (e.g., [1]), 
            it only shows points that were verified as inliers in a match.
        params (list): A list of dictionaries containing plotting 
            styles (color, linewidth, and whether to draw the orientation vector).
        cache_path (str): The directory where the resulting JPG/PNG files are stored.
    """
    def __init__(self, **args):
        self.single_image = True
        self.pipeliner = False        
        self.pass_through = True
        self.add_to_cache = True
        
        self.args = {
            'id_more': '',
            'img_prefix': '',
            'img_suffix': '',
            'cache_path': 'show_imgs',
            'prepend_pair': False,
            'ext': '.jpg',
            'force': False,
            'mask_idx': None, # None: all single image, -1: all both images, list: filtered both images
            'params': [{'color': 'r', 'linewidth': 1, 'draw_ori': True}, {'color': 'g', 'linewidth': 1, 'draw_ori': True}],
        }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
                
        self.id_string, self.args = set_args('show_kpts' , args, self.args)
        if self.args['mask_idx'] is not None: self.single_image = False

                
    def get_id(self): 
        return self.id_string

    
    def finalize(self):
        return

    
    def run(self, **args): 
        if not self.single_image:
            idxs = [0, 1]
        else:
            idxs = [args['idx']]

        for idx in idxs:
            im = args['img'][idx]
            cache_path = self.args['cache_path']
            img = os.path.split(im)[1]
            img_name, _ = os.path.splitext(img)
            if self.args['prepend_pair']:
                img0 = os.path.splitext(os.path.split(args['img'][0])[1])[0]
                img1 = os.path.splitext(os.path.split(args['img'][1])[1])[0]
                cache_path = os.path.join(cache_path, img0 + '_' + img1)
                
            new_img = os.path.join(cache_path, self.args['img_prefix'] + img_name + self.args['img_suffix'] + self.args['ext'])
    
            if not os.path.isfile(new_img) or self.args['force']:
                os.makedirs(cache_path, exist_ok=True)
                img = cv2.cvtColor(cv2.imread(args['img'][idx]), cv2.COLOR_BGR2RGB)    
                lafs = homo2laf(args['kp'][idx], args['kH'][idx])
    
                if (self.args['mask_idx'] is None) or (self.args['mask_idx'] == -1) or ('m_idx' not in args):
                    mask_idx = -1
                    params = self.args['params'][-1]
                else:
                    if not isinstance(self.args['mask_idx'], list): self.args['mask_idx'] = [self.args['mask_idx']]
                    mask_idx = self.args['mask_idx']
                    params = self.args['params']
                                    
                fig = plt.figure()
                ax = None
                img = K.image_to_tensor(img, False)
    
                if mask_idx == -1: 
                    fig, ax = visualize_LAF(img, lafs, 0, fig=fig, ax=ax, return_fig_ax=True, **params)
    
                else:
                    for i in mask_idx:                
                        m_idx = args['m_idx'][:, idx]
                        m_mask = args['m_mask']
                        m_idx = m_idx[m_mask == i]
                        if m_idx.shape[0] < 1: continue
                        lafs_ = lafs[:, m_idx]
                        
                        fig, ax = visualize_LAF(img, lafs_, 0, fig=fig, ax=ax, return_fig_ax=True, **params[i])
                        img = None
    
                plt.axis('off')
                plt.savefig(new_img, dpi=150, bbox_inches='tight')
                plt.close(fig)

        return {}



def visualize_LAF(img, LAF, img_idx = 0, color='r', linewidth=1, draw_ori = True, fig=None, ax = None, return_fig_ax = False, **kwargs):
    """
    Renders Local Affine Frames (LAFs) onto an image for visual inspection.

    An LAF is more than a point; it's a 2x3 matrix representing a local 
    coordinate system. This function draws the boundary of the 'patch' 
    (usually an ellipse) and optionally a line indicating the dominant 
    orientation.

    Args:
        img (torch.Tensor): The image tensor (CxHxW).
        LAF (torch.Tensor): The Local Affine Frames tensor (Nx2x3).
        draw_ori (bool): If True, draws a line from the center to the 
                         edge to show the rotation of the feature.
        return_fig_ax (bool): If True, returns the Matplotlib figure 
                              and axis for further plotting.
    """
    from kornia_moons.feature import to_numpy_image

    x, y = K.feature.laf.get_laf_pts_to_draw(K.feature.laf.scale_laf(LAF, 0.5), img_idx)

    if not draw_ori:
        x= x[1:]
        y= y[1:]

    if (fig is None and ax is None):
        fig, ax = plt.subplots(1,1, **kwargs)

    if (fig is not None and ax is None):
        ax = fig.add_axes([0, 0, 1, 1])
    
    if img is not None:
        ax.imshow(to_numpy_image(img[img_idx]))

    ax.plot(x, y, color, linewidth=linewidth)
    if return_fig_ax : return fig, ax

    return
