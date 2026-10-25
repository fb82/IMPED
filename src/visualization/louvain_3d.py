import base64
import colorsys
import json
import os
import webbrowser
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from core import set_args
from reconstruct.louvain_module import conductance, is_weak, louvain_communities
from reconstruct.reconstruct_module import largest_model

GOLDEN = 0.618033988749895
TEMPLATE = os.path.join(os.path.dirname(__file__), 'louvain_3d.html')


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode('ascii')


def _hex(hue, saturation, lightness):
    r, g, b = colorsys.hls_to_rgb(hue % 1, lightness, saturation)
    return '#%02x%02x%02x' % (round(r * 255), round(g * 255), round(b * 255))


def louvain_colors(weak):
    """One color per community: strong hues for the weak ones, pale for the others."""
    colors, n_weak = [], 0
    for k, is_weak_k in enumerate(weak):
        if is_weak_k:
            colors.append(_hex(n_weak * GOLDEN, 0.9, 0.55))
            n_weak += 1
        else:
            colors.append(_hex(k * GOLDEN, 0.2, 0.6))
    return colors


def _names(clusters):
    names = set()
    for cluster in clusters or ():
        names |= {cluster} if isinstance(cluster, str) else set(cluster)
    return names


def _frustum(image, scale, aspect):
    # apex and the four far corners of the camera, in world coordinates
    rotation = image.cam_from_world().rotation.matrix()
    center = image.projection_center()
    half_w, half_h = 0.5 * aspect[0], 0.5 * aspect[1]
    corners = np.array([[-half_w, -half_h, 1], [half_w, -half_h, 1], [half_w, half_h, 1], [-half_w, half_h, 1]])
    return center, center + scale * corners @ rotation


def _aspect(rec, image):
    camera = rec.cameras[image.camera_id]
    side = max(camera.width, camera.height)
    return camera.width / side, camera.height / side


def louvain_3d(model, save_to='louvain_3d.html', compare=None, highlight=(), max_points=300000, open_browser=False):
    """
    Draws the largest model of `model` (a reconstruction module or a model
    folder) as an interactive three.js page: cameras as frustums and 3D
    points, both colored by the Louvain community of the covisibility graph
    (the one louvain_module uses), weak communities in strong colors and the
    others in pale ones. A point takes the community of most of the cameras
    that see it, so the points of a weak community are the suspect ones.

    `compare` is a previous model in the same frame (e.g. the one a
    repair_reconstruction_module started from): the cameras named in
    `highlight` (image names, or clusters of image names) are drawn in
    white, joined by a line to their position in `compare`, and the ones
    missing from `model` are drawn as grey ghosts at their old position.
    At most `max_points` points are drawn, sampled at random.
    """
    _, rec = largest_model(model)
    if rec is None:
        print("louvain_3d: no model to draw")
        return None

    graph, communities = louvain_communities(rec)
    communities = sorted(communities, key=len, reverse=True)
    community_of = {image_id: k for k, community in enumerate(communities) for image_id in community}
    weak = [is_weak(graph, community) for community in communities]

    colors = louvain_colors(weak)
    n_weak = sum(weak)

    images = [image for image in rec.images.values() if image.has_pose and image.image_id in community_of]
    centers = np.array([image.projection_center() for image in images])
    if len(centers) > 1:
        scale = 0.5 * float(np.median(cKDTree(centers).query(centers, k=2)[0][:, 1]))
    else:
        scale = 1.0
    scale = scale or 1.0
    up = -np.mean([image.cam_from_world().rotation.matrix()[1] for image in images], axis=0)
    up /= np.linalg.norm(up) or 1

    old = {}
    if compare is not None:
        _, previous = largest_model(compare)
        if previous is not None:
            old = {image.name: (image, _aspect(previous, image)) for image in previous.images.values() if image.has_pose}
    highlight = _names(highlight)

    points = list(rec.points3D.values())
    n_points = len(points)
    if n_points > max_points:
        keep = np.random.default_rng(0).choice(n_points, max_points, replace=False)
        points = [points[k] for k in sorted(keep)]

    origin = np.median(np.array([p.xyz for p in points]), axis=0) if points else centers.mean(axis=0)

    xyz = [[] for _ in communities]
    rgb = [[] for _ in communities]
    for point in points:
        votes = Counter(community_of[e.image_id] for e in point.track.elements if e.image_id in community_of)
        if not votes:
            continue
        k = votes.most_common(1)[0][0]
        xyz[k].append(point.xyz - origin)
        rgb[k].append(point.color)

    cameras = [[] for _ in communities]
    moves = []
    for image in images:
        center, corners = _frustum(image, scale, _aspect(rec, image))
        entry = {'n': image.name, 'c': (center - origin).tolist(), 'f': (corners - origin).ravel().tolist()}
        if image.name in highlight:
            entry['h'] = 1
            if image.name in old:
                entry['o'] = (old[image.name][0].projection_center() - origin).tolist()
                moves.append(float(np.linalg.norm(center - old[image.name][0].projection_center())))
        cameras[community_of[image.image_id]].append(entry)

    registered = {image.name for image in images}
    lost = []
    for name in sorted(highlight - registered):
        if name in old:
            image, aspect = old[name]
            center, corners = _frustum(image, scale, aspect)
            lost.append({'n': name, 'c': (center - origin).tolist(), 'f': (corners - origin).ravel().tolist()})

    data = {
        'title': os.path.basename(os.path.normpath(model.args['output'] if hasattr(model, 'args') else model)),
        'scale': scale,
        'up': up.tolist(),
        'n_images': len(images),
        'n_points': n_points,
        'n_drawn': sum(map(len, xyz)),
        'compare': os.path.basename(os.path.normpath(compare)) if isinstance(compare, str) else None,
        'n_highlight': len(highlight),
        'n_back': len(highlight & registered),
        'median_move': float(np.median(moves)) if moves else None,
        'lost': lost,
        'communities': [{
            'size': len(community),
            'conductance': round(conductance(graph, community), 4),
            'weak': bool(weak[k]),
            'color': colors[k],
            'cameras': cameras[k],
            'points': _b64(np.array(xyz[k], dtype=np.float32).reshape(-1, 3)),
            'rgb': _b64(np.array(rgb[k], dtype=np.uint8).reshape(-1, 3)),
        } for k, community in enumerate(communities)],
    }

    with open(TEMPLATE) as f:
        page = f.read().replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
    os.makedirs(os.path.dirname(os.path.abspath(save_to)), exist_ok=True)
    with open(save_to, 'w') as f:
        f.write(page)

    print(f"louvain_3d: {len(communities)} communities, {n_weak} weak, {len(images)} cameras, "
          f"{data['n_drawn']} of {n_points} points -> {save_to}")
    if open_browser:
        webbrowser.open('file://' + os.path.abspath(save_to))
    return save_to


class louvain_3d_module:
    """
    Pipeline wrapper of louvain_3d: finalize() draws `model` to `save_to`
    once `worklist`, if given, is empty, so it can be placed at the end of
    a pipeline, after the module that builds or repairs the model. Skipped
    when `model` is a folder that doesn't exist.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False

        self.args = {
            'id_more': '',
            'model': None,
            'save_to': 'louvain_3d.html',
            'compare': None,
            'highlight': (),
            'max_points': 300000,
            'open_browser': False,
            'worklist': None,
        }

        self.id_string, self.args = set_args('louvain_3d', args, self.args)

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        return {}

    def finalize(self):
        if self.args['worklist']:
            return

        model, compare = self.args['model'], self.args['compare']
        if isinstance(model, str) and not os.path.isdir(model):
            print(f"louvain_3d_module: no model in {model}")
            return
        if isinstance(compare, str) and not os.path.isdir(compare):
            compare = None

        louvain_3d(model, save_to=self.args['save_to'], compare=compare, highlight=self.args['highlight'],
                   max_points=self.args['max_points'], open_browser=self.args['open_browser'])
