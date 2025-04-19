from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import numpy as np


def add_cameras(viz, poses, focals=None, images=None, imsizes=None, colors=None, **kw):
    get = lambda arr,idx: None if arr is None else arr[idx]
    for i, pose_c2w in enumerate(poses):
        add_camera(viz, pose_c2w, get(focals,i), image=get(images,i), color=get(colors,i), imsize=get(imsizes,i), **kw)


def add_camera(viz, pose_c2w, focal=None, color=(0, 0, 0), image=None, imsize=None, cam_size=0.03):

    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import img_to_arr

    pose_c2w, focal, color, image = to_numpy((pose_c2w, focal, color, image))
    image = img_to_arr(image)
    if isinstance(focal, np.ndarray) and focal.shape == (3,3):
        intrinsics = focal
        focal = (intrinsics[0,0] * intrinsics[1,1]) ** 0.5
        if imsize is None:
            imsize = (2*intrinsics[0,2], 2*intrinsics[1,2])
    
    add_scene_cam(viz.scene, pose_c2w, color, image, focal, imsize=imsize, screen_width=cam_size, marker=None)
    return viz


def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, 
                  screen_width=0.03, marker=None):

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


def show(scene, show_pw_cams=False, show_pw_pts3d=False, cam_size=None, **kw):

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
    add_cameras(viz, im_poses, scene.get_focals(), colors=colors,
        images=scene.imgs, imsizes=scene.imsizes, cam_size=cam_size)
    if show_pw_cams:
        pw_poses = scene.get_pw_poses()
        add_cameras(viz, pw_poses, color=(192, 0, 192), cam_size=cam_size)

        if show_pw_pts3d:
            pts = [geotrf(pw_poses[e], scene.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(scene.edges)]
            viz.add_pointcloud(pts, (128, 0, 128))

    viz.show(**kw)
    return viz


if __name__ == '__main__':
    the_images = [
        'et004.jpg',
        'et008.jpg',
        ]
    n_viz = 10    
    
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = 'checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(the_images, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
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
    
  # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # visualize reconstruction
    show(scene)

    # find 2D-2D matches between the two images
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # visualize a few matches
    from matplotlib import pyplot as pl
    match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)
