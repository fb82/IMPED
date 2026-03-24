import os
import warnings
import pickled_hdf5.pickled_hdf5 as pickled_hdf5
import time
from tqdm import tqdm
import torchvision.transforms as transforms

import torch
import kornia as K
from kornia_moons.feature import opencv_kpts_from_laf, laf_from_opencv_kpts
import cv2
import numpy as np
from PIL import Image
import poselib
import gdown
import zipfile
import tarfile
import csv
import shutil
import bz2
import _pickle as cPickle
import argparse
import math
import copy
import wget
import pycolmap
import scipy
import miho.src.miho as mop_miho
import miho.src.miho_other as mop
import miho.src.ncc as ncc

import matplotlib.pyplot as plt
from matplotlib import colormaps
import plot.viz2d as viz
import plot.utils as viz_utils
import sys
from pathlib import Path

from core import device, pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, image_pairs, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args


conf_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(conf_path, 'mast3r/dust3r'))

from dust3r.inference import inference as dust3r_inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images as dust3r_load_images
from dust3r.image_pairs import make_pairs as dust3r_make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


def dust3r_add_cameras(viz, poses, focals=None, images=None, imsizes=None, colors=None, **kw):
    """
    Batch processes and adds multiple camera frustums to a 3D scene.

    This utility iterates through a sequence of camera poses and associated 
    metadata (focals, images, colors), calling `dust3r_add_camera` for each 
    instance. It uses a safe indexing helper to handle cases where some 
    attribute lists might be None.

    Args:
        viz (SceneViz): The visualization object containing the 3D scene.
        poses (Iterable[Tensor/Array]): A collection of 4x4 Camera-to-World 
            transformation matrices.
        focals (Iterable, optional): A collection of focal lengths or 
            3x3 intrinsic matrices corresponding to each pose.
        images (Iterable, optional): A collection of images captured by 
            each camera.
        imsizes (Iterable, optional): A collection of (Width, Height) tuples 
            for each camera.
        colors (Iterable, optional): A collection of RGB tuples for 
            each camera frustum's edges.
        **kw: Additional keyword arguments passed down to the 
            `dust3r_add_camera` and `dust3r_add_scene_cam` functions 
            (e.g., cam_size).

    Returns:
        None: The function updates the 'viz' object in-place.
    """
    get = lambda arr,idx: None if arr is None else arr[idx]
    for i, pose_c2w in enumerate(poses):
        dust3r_add_camera(viz, pose_c2w, get(focals,i), image=get(images,i), color=get(colors,i), imsize=get(imsizes,i), **kw)


def dust3r_add_camera(viz, pose_c2w, focal=None, color=(0, 0, 0), image=None, imsize=None, cam_size=0.03):
    """
    Higher-level wrapper to add a single camera frustum to a 3D visualizer.

    This function prepares data for visualization by converting PyTorch tensors 
    to NumPy arrays and automatically extracting focal length and image 
    dimensions from an intrinsic matrix if provided.

    Args:
        viz (SceneViz): The visualization object containing the 3D scene.
        pose_c2w (Tensor/Array): 4x4 Camera-to-World transformation matrix.
        focal (float/Tensor/Array, optional): The camera focal length. Can be 
            a single value or a 3x3 intrinsic matrix.
        color (tuple, optional): RGB color for the camera wireframe (0-255). 
            Defaults to black (0, 0, 0).
        image (Tensor/Array, optional): The image captured by the camera.
        imsize (tuple, optional): (Width, Height) of the image. If None and 
            an intrinsic matrix is provided, it is estimated from the 
            principal point.
        cam_size (float): The scale/width of the camera frustum in 3D units.

    Returns:
        SceneViz: The updated visualizer instance.

    Note:
        If a 3x3 intrinsic matrix is passed as 'focal', the function calculates 
        the geometric mean of the focal lengths ($f = \sqrt{f_x \cdot f_y}$) and 
        estimates the image size by doubling the principal point coordinates 
        ($c_x, c_y$).
    """
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import img_to_arr

    pose_c2w, focal, color, image = to_numpy((pose_c2w, focal, color, image))
    image = img_to_arr(image)
    if isinstance(focal, np.ndarray) and focal.shape == (3,3):
        intrinsics = focal
        focal = (intrinsics[0,0] * intrinsics[1,1]) ** 0.5
        if imsize is None:
            imsize = (2*intrinsics[0,2], 2*intrinsics[1,2])
    
    dust3r_add_scene_cam(viz.scene, pose_c2w, color, image, focal, imsize=imsize, screen_width=cam_size, marker=None)
    return viz


def dust3r_add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, 
                  screen_width=0.03, marker=None):
    """
    Adds a 3D mesh representation of a camera to a trimesh scene.

    This function constructs a visual camera frustum based on camera intrinsics 
    (focal length, image size) and extrinsics (pose_c2w). It can also map the 
    actual captured image onto the 3D "screen" of the camera frustum.

    Args:
        scene (trimesh.Scene): The scene object to which the camera geometry is added.
        pose_c2w (np.ndarray): A 4x4 Camera-to-World transformation matrix.
        edge_color (list/tuple): RGB color for the camera wireframe/edges.
        image (np.ndarray, optional): The RGB image captured by the camera to be 
            textured onto the frustum face.
        focal (float, optional): The focal length of the camera. If None, it is 
            estimated based on image dimensions.
        imsize (tuple, optional): (Width, Height) of the image.
        screen_width (float): The physical size of the camera representation 
            in the 3D world coordinates.
        marker (str, optional): If set to 'o', adds a small sphere at the 
            optical center of the camera.

    Returns:
        None: The function modifies the input 'scene' object in-place.

    Notes:
        - The function uses an 'OPENGL' coordinate system conversion (flipping 
          Y and Z axes) to align typical computer vision camera conventions 
          with the 3D renderer.
        - The camera body is generated as a 4-section cone (pyramid) rotated 
          45 degrees to align with the image axes.
    """

    from dust3r.utils.geometry import geotrf

    import PIL.Image
    from scipy.spatial.transform import Rotation

    try:
        import trimesh
    except ImportError:
        print('/!\\ module trimesh is not installed, cannot visualize results /!\\')

    OPENGL = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal.flatten()[0]
    if not focal:
        focal = min(H,W) * 1.1 # default value

    # create fake camera
    height = max( screen_width/10, focal * screen_width / H )
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == 'o':
        marker = trimesh.creation.icosphere(3, radius=screen_width/4)
        marker.vertices += pose_c2w[:3,3]
        marker.visual.face_colors[:,:3] = edge_color
        scene.add_geometry(marker)


def dust3r_show(scene, show_pw_cams=False, show_pw_pts3d=False, cam_size=None, **kw):
    """
    Visualizes the 3D reconstruction scene, including point clouds and camera poses.

    This function populates a 3D visualizer with the geometric data stored in a 
    DUSt3R 'scene' object. It renders the dense point clouds (colored by image 
    pixels) and represents the cameras as 3D frustums.

    Args:
        scene (GlobalAligner): An aligned DUSt3R scene object containing images, 
            3D points, confidence masks, and camera poses.
        show_pw_cams (bool, optional): If True, visualizes "Pair-Wise" camera 
            poses (raw predictions before global alignment). Defaults to False.
        show_pw_pts3d (bool, optional): If True, visualizes the raw pair-wise 
            3D point clouds alongside the aligned ones. Defaults to False.
        cam_size (float, optional): The scale/size of the camera frustums in 
            the 3D plot. If None, it is calculated automatically.
        **kw: Additional keyword arguments passed to the visualizer's `show()` method 
            (e.g., window title, background color).

    Returns:
        SceneViz: The visualizer object instance containing the rendered scene.

    Note:
        The function handles two scenarios for point cloud coloring:
        1. If 'scene.imgs' is available, points are colored using the actual 
           image textures.
        2. If 'scene.imgs' is None, each view is assigned a random solid color.
    """
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.device import to_numpy
    from dust3r.cloud_opt.commons import edge_str
    from dust3r.utils.geometry import geotrf
    import numpy as np

    viz = SceneViz()
    if scene.imgs is None:
        colors = np.random.randint(0, 256, size=(scene.n_imgs, 3))
        colors = list(map(tuple, colors.tolist()))
        for n in range(scene.n_imgs):
            viz.add_pointcloud(scene.get_pts3d()[n], colors[n], scene.get_masks()[n])
    else:
        viz.add_pointcloud(scene.get_pts3d(), scene.imgs, scene.get_masks())
        colors = np.random.randint(256, size=(scene.n_imgs, 3))

    # camera poses
    im_poses = to_numpy(scene.get_im_poses())
    if cam_size is None:
        cam_size = auto_cam_size(im_poses)
    dust3r_add_cameras(viz, im_poses, scene.get_focals(), colors=colors,
        images=scene.imgs, imsizes=scene.imsizes, cam_size=cam_size)
    if show_pw_cams:
        pw_poses = scene.get_pw_poses()
        dust3r_add_cameras(viz, pw_poses, color=(192, 0, 192), cam_size=cam_size)

        if show_pw_pts3d:
            pts = [geotrf(pw_poses[e], scene.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(scene.edges)]
            viz.add_pointcloud(pts, (128, 0, 128))

    viz.show(**kw)
    return viz

  
class dust3r_module:
    """
    A pipeline module that leverages the DUSt3R model for 3D-informed feature matching.

    Instead of traditional descriptor matching, this module performs a dense 3D 
    reconstruction of the image pair, aligns them in a global coordinate space, 
    and extracts reciprocal 2D matches based on 3D point proximity. It can 
    optionally perform 3D pose refinement (Point Cloud Optimization).

    Attributes:
        args (dict): Configuration parameters including:
            - model (str): Pretrained model path or HuggingFace ID.
            - max_matches (int): Upper limit of matches to return.
            - resize (int): Image resolution for inference (default: 512).
            - patch_radius (int): Radius used for patch-based homography (kH).
            - 3D_pose_refinement (bool): If True, runs global point cloud optimization.
        model (AsymmetricCroCo3DStereo): The loaded DUSt3R/MASt3R neural network.

    Args:
        **args: Keyword arguments to override default settings.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
                                
        self.args = {
            'id_more': '',
            'model': 'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt',
            'max_matches': 2048,
            'schedule': 'cosine',
            'lr': 0.01,
            'niter': 300,
            'resize': 512, 
            'patch_radius': 16,
            '3D_pose_refinement': False,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('mast3r', args, self.args)        

        # you can put the path to a local checkpoint in model_name if needed
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.args['model']).to(device)
        
        
    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        warnings.simplefilter(action='always', category=FutureWarning)        
        
        return


    def run(self, **args):  
        """
        Executes the DUSt3R inference and matching pipeline.

        The process follows these steps:
        1. Load and resize image pairs.
        2. Run DUSt3R inference to get raw 3D point predictions and confidence maps.
        3. Align views using a Global Aligner (PairViewer or PointCloudOptimizer).
        4. Extract reciprocal matches by finding 3D points that correspond in both views.
        5. Scale 2D keypoints back to original image dimensions.
        6. Compute local patch homographies (kH) for downstream tasks.

        Args:
            **args: Dictionary containing the input data:
                - img (list[str]): Paths to the two images to be matched.

        Returns:
            dict: A dictionary containing:
                - kp (list[Tensor]): Keypoint coordinates for both images.
                - kH (list[Tensor]): Patch-based homography matrices for each keypoint.
                - kr (list[Tensor]): Rotation values (currently placeholders).
                - m_idx (Tensor): Indices mapping matches between image 0 and 1.
                - m_val (Tensor): Confidence values (initially set to 1/True).
                - m_mask (Tensor): Boolean mask indicating valid matches.
        """      
        warnings.simplefilter(action='ignore', category=FutureWarning)        
        
        image0 = args['img'][0]
        image1 = args['img'][1]

        with torch.inference_mode(mode=(not self.args['3D_pose_refinement'])):        
            images = dust3r_load_images([image0, image1], size=self.args['resize'], verbose=False)
            pairs = dust3r_make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = dust3r_inference(pairs, self.model, device, batch_size=1, verbose=False)
    
            # at this stage, you have the raw dust3r predictions
            # view1, pred1 = output['view1'], output['pred1']
            # view2, pred2 = output['view2'], output['pred2']
            # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
            #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
            # in each view you have:
            # an integer image identifier: view1['idx'] and view2['idx']
            # the img: view1['img'] and view2['img']
            # the image shape: view1['true_shape'] and view2['true_shape']
            # an instance string output by the dataloader: view1['instance'] and view2['instance']
            # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
            # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
            # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']
        
            # next we'll use the global_aligner to align the predictions
            # depending on your task, you may be fine with the raw output and not need it
            # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
            # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
        
            if not self.args['3D_pose_refinement']:
                scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=False)      
            else:
                scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)

            loss = scene.compute_global_alignment(init="mst", niter=self.args['niter'], schedule=self.args['schedule'], lr=self.args['lr'])
    
        # retrieve useful values from scene:
        imgs = scene.imgs
        # focals = scene.get_focals()
        # poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()
    
        # visualize reconstruction
        # dust3r_show(scene)
    
        # find 2D-2D matches between the two images
        from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
        
        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        # print(f'found {num_matches} matches')
        matches_im1 = pts2d_list[1][reciprocal_in_P2]
        matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]
    
        # # visualize a few matches
        # from matplotlib import pyplot as pl
        # n_viz = 10
        # match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
        # viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    
        # H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        # img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        # img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        # img = np.concatenate((img0, img1), axis=1)
        # pl.figure()
        # pl.imshow(img)
        # cmap = pl.get_cmap('jet')
        # for i in range(n_viz):
        #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        #     pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        # pl.show(block=True)

        kps1 = matches_im0
        kps2 = matches_im1
        
        max_m = self.args['max_matches']
        n_m = kps1.shape[0]
        if np.isfinite(max_m) and (n_m > max_m):
            idx = np.linspace(0, n_m - 1, max_m).astype(int)
            kps1 = kps1[idx]
            kps2 = kps2[idx]

        s1 = max(Image.open(image0).size)
        s2 = max(Image.open(image1).size)
        
        kps1 = torch.tensor(kps1 * s1 / self.args['resize'], device=device, dtype=torch.float)
        kps2 = torch.tensor(kps2 * s2 / self.args['resize'], device=device, dtype=torch.float)
        
        kp = [kps1, kps2]
        kH = [
            torch.zeros((kp[0].shape[0], 3, 3), device=device),
            torch.zeros((kp[0].shape[0], 3, 3), device=device),
            ]
        
        kH[0][:, [0, 1], 2] = -kp[0] / self.args['patch_radius']
        kH[0][:, 0, 0] = 1 / self.args['patch_radius']
        kH[0][:, 1, 1] = 1 / self.args['patch_radius']
        kH[0][:, 2, 2] = 1

        kH[1][:, [0, 1], 2] = -kp[1] / self.args['patch_radius']
        kH[1][:, 0, 0] = 1 / self.args['patch_radius']
        kH[1][:, 1, 1] = 1 / self.args['patch_radius']
        kH[1][:, 2, 2] = 1

        kr = [torch.full((kp[0].shape[0],), torch.nan, device=device), torch.full((kp[0].shape[0],), torch.nan, device=device)]        

        m_mask = torch.full((kps1.shape[0], ), 1, device=device, dtype=torch.bool)
        m_val = torch.full((kps1.shape[0], ), 1, device=device, dtype=torch.bool)

        m_idx = torch.zeros((kp[0].shape[0], 2), device=device, dtype=torch.int)
        m_idx[:, 0] = torch.arange(kp[0].shape[0])
        m_idx[:, 1] = torch.arange(kp[0].shape[0])

        return {'kp': kp, 'kH': kH, 'kr': kr, 'm_idx': m_idx, 'm_val': m_val, 'm_mask': m_mask}
