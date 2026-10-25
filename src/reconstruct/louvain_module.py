import os

import networkx as nx
import numpy as np
import pycolmap
import scipy.sparse as sp

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from core import set_args
from .reconstruct_module import largest_model


def db_graph(db):
    database = pycolmap.Database.open(db)
    graph = nx.Graph()
    for image in database.read_all_images():
        graph.add_node(image.image_id, name=image.name)
    for pair_id, geometry in zip(*database.read_two_view_geometries()):
        if len(geometry.inlier_matches):
            graph.add_edge(*pycolmap.pair_id_to_image_pair(pair_id), weight=len(geometry.inlier_matches))
    database.close()
    return graph


def covisibility_graph(rec):
    ids = sorted(rec.reg_image_ids())
    col = {image_id: k for k, image_id in enumerate(ids)}

    rows, cols = [], []
    for k, point in enumerate(rec.points3D.values()):
        for element in point.track.elements:
            rows.append(k)
            cols.append(col[element.image_id])

    seen = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(rec.num_points3D(), len(ids)))
    seen.data[:] = 1
    shared = (seen.T @ seen).tocoo()

    graph = nx.Graph()
    graph.add_nodes_from(ids)
    graph.add_weighted_edges_from((ids[a], ids[b], w) for a, b, w in zip(shared.row, shared.col, shared.data) if a < b)
    return graph


def conductance(graph, cluster):
    volume = nx.volume(graph, cluster, weight='weight')
    return nx.cut_size(graph, cluster, weight='weight') / volume if volume else 0


def louvain_communities(rec, resolution=5):
    graph = covisibility_graph(rec)
    return graph, nx.community.louvain_communities(graph, weight='weight', resolution=resolution, seed=0)


def is_weak(graph, cluster, max_conductance=0.1, max_size=100):
    return len(cluster) <= min(max_size, len(graph) // 2) and conductance(graph, cluster) < max_conductance


def weak_clusters(rec, resolution=5, max_conductance=0.1, max_size=100):
    graph, clusters = louvain_communities(rec, resolution)
    return [cluster for cluster in clusters if is_weak(graph, cluster, max_conductance, max_size)]


def unregistered_clusters(rec, graph):
    registered = set(rec.reg_image_ids())
    outside = graph.subgraph(image_id for image_id in graph if image_id not in registered)
    return [set(cluster) for cluster in nx.connected_components(outside)]


def repair_pairs(rec, clusters, graph, images, sim_table=None, n_near=50, per_image=10, max_angle=60, min_dist=0, skip_matched=True):
    registered = [im for im in rec.images.values() if im.has_pose]
    ids = np.array([im.image_id for im in registered])
    row = {image_id: k for k, image_id in enumerate(ids)}
    center = np.array([im.projection_center() for im in registered])
    axis = np.array([im.cam_from_world().rotation.matrix()[2] for im in registered])

    def path(image_id):
        return os.path.join(images, graph.nodes[image_id]['name'])

    def sim(a, b):
        if sim_table is None:
            return 0
        return sim_table.get((path(a), path(b)), sim_table.get((path(b), path(a)), 0))

    def nearest(ref, outside):
        dist = np.linalg.norm(center - center[row[ref]], axis=1)
        dist[~outside | (axis @ axis[row[ref]] < np.cos(np.radians(max_angle))) | (dist < min_dist)] = np.inf
        return [ids[k] for k in np.argsort(dist)[:n_near] if np.isfinite(dist[k])]

    pairs = set()
    for cluster in clusters:
        outside = np.array([image_id not in cluster for image_id in ids])
        anchors = {other for image_id in cluster for other in graph[image_id] if other in row and other not in cluster}
        for image_id in cluster:
            refs = [image_id] if image_id in row else sorted(anchors)
            if refs:
                near = list(dict.fromkeys(other for ref in refs for other in nearest(ref, outside)))
            elif sim_table is not None:
                near = list(ids[outside])
            else:
                continue
            near = [other for other in near if not (skip_matched and graph.has_edge(image_id, other))]
            near.sort(key=lambda other: sim(image_id, other), reverse=True)
            pairs |= {tuple(sorted((path(image_id), path(other)))) for other in near[:per_image]}
    return sorted(pairs)


class louvain_module:
    """
    Finds the parts of a COLMAP reconstruction to repair and the pairs to
    match for them, like kfc_module selects the seed pairs: it doesn't match
    anything, the pairs are matched by any pipeline, and
    repair_reconstruction_module then builds the repaired model.

    `model` is either a reconstruction module (its models are read once it
    is done) or the folder of an existing reconstruction; the model with the
    most registered images is analyzed. finalize():
      * picks the clusters to repair: with `auto`, the Louvain communities
        of the covisibility graph weakly tied to the rest of the model and
        the images of `db` outside the model, plus the images named in
        `images_to_fix`;
      * pairs each of their images with the registered images closest to
        it in space, `per_image` per image, chosen by global similarity
        (`sim_table`);
      * fills `output` (a dict, filled in place like kfc_module's `pairs`,
        also available as `self.output`), and saves to `out_path`, with
        {'model': path of the analyzed model,
         'clusters': [set of image names, ...],
         'pairs': [(img0, img1), ...]}.

    Placed at the end of the pipeline that builds the model, with the same
    `worklist`, it runs once the model is done; the next pipeline matches
    the pairs and repairs the model:

        repair = {}
        run_pairs([..., reconstruct_module(output='model', worklist=pp, ...),
                   louvain_module(model='model', db='colmap.db', output=repair, worklist=pp, ...)], pp)
        run_pairs(matchers + [
            to_colmap_module(db='colmap.db', no_unmatched=False, only_matched=True),
            repair_reconstruction_module(db='colmap.db', model=repair['model'],
                                         clusters=repair['clusters'], output='model_fixed', ...),
        ], repair['pairs'])

    Calling finalize() directly runs it outside a pipeline; read the
    dictionary back with `louvain_module.load(out_path)`.
    """
    def __init__(self, **args):
        output = args.pop('output', None)

        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False

        self.args = {
            'id_more': '',
            'db': 'colmap.db',
            'images': '.',
            'model': None,
            'out_path': 'louvain.hdf5',
            'sim_table': None,
            'per_image': 10,
            'images_to_fix': [],
            'auto': True,
            'min_dist': 0,
            'skip_matched': True,
            'worklist': None,
        }

        self.id_string, self.args = set_args('louvain', args, self.args)
        self.output = output if output is not None else {}
        self.output.update({'model': None, 'clusters': [], 'pairs': []})

    @staticmethod
    def load(out_path, id_string='louvain'):
        """Read back the {'model', 'clusters', 'pairs'} dictionary."""
        aux_hdf5 = pickled_hdf5.pickled_hdf5(out_path, mode='r', label_prefix='pickled/' + id_string)
        output = {'model': aux_hdf5.get('/model')[0], 'clusters': [], 'pairs': []}
        for key in aux_hdf5.get_keys():
            if key.startswith('/clusters/'):
                output['clusters'].append(set(aux_hdf5.get(key)[0]))
            elif key.startswith('/pairs/'):
                output['pairs'].append(tuple(aux_hdf5.get(key)[0]))
        aux_hdf5.close()
        return output

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        return {}

    def finalize(self):
        if self.args['worklist']:
            return

        model_path, rec = largest_model(self.args['model'])
        if rec is None:
            print("louvain_module: no model to repair")
            return

        graph = db_graph(self.args['db'])
        weak = weak_clusters(rec) if self.args['auto'] else []
        unregistered = unregistered_clusters(rec, graph) if self.args['auto'] else []
        forced = {image_id for image_id in graph if graph.nodes[image_id]['name'] in self.args['images_to_fix']}
        print(f"louvain_module: {len(weak)} weak clusters, sizes {sorted(map(len, weak), reverse=True)}, "
              f"{sum(map(len, unregistered))} images outside the model in {len(unregistered)} clusters, "
              f"{len(forced)} images to fix given")
        clusters = weak + unregistered + ([forced] if forced else [])

        pairs = repair_pairs(
            rec, clusters, graph, self.args['images'], self.args['sim_table'],
            per_image=self.args['per_image'],
            min_dist=self.args['min_dist'],
            skip_matched=self.args['skip_matched'],
        ) if clusters else []

        self.output.update({
            'model': model_path,
            'clusters': [{graph.nodes[image_id]['name'] for image_id in cluster} for cluster in clusters],
            'pairs': pairs,
        })

        if self.args['out_path'] is not None:
            if os.path.exists(self.args['out_path']):
                os.remove(self.args['out_path'])

            aux_hdf5 = pickled_hdf5.pickled_hdf5(self.args['out_path'], mode='a', label_prefix='pickled/' + self.id_string)
            aux_hdf5.add('/model', model_path)
            for i, cluster in enumerate(self.output['clusters']):
                aux_hdf5.add(f'/clusters/{i}', sorted(cluster))
            for img0, img1 in pairs:
                aux_hdf5.add('/pairs/' + os.path.split(img0)[-1] + '/' + os.path.split(img1)[-1], (img0, img1))
            aux_hdf5.close()

        print(f"louvain_module: {len(pairs)} pairs to match for {len(clusters)} clusters of {model_path}, "
              f"saved to {self.args['out_path']}")
