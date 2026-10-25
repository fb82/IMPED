import os
import shutil

import networkx as nx
import pycolmap

from core import set_args
from .louvain_module import louvain_communities
from .reconstruct_module import model_paths, snapshot_db


def split_groups(rec, min_tie=0.5, min_size=20):
    """
    Splits a model into groups of images weakly tied to each other, e.g.
    different scenes glued together by a few wrong matches: the Louvain
    communities of the covisibility graph (as weak_clusters) are merged,
    the most tied pair first, while they are tied by at least `min_tie`
    cameras' worth of covisibility, i.e. the weight between the two groups
    divided by the mean weighted degree of the cameras of the smaller one.
    Unlike the conductance, this doesn't drop as groups grow, so long
    scenes are merged back whole, while a few wrong matches between two
    scenes are worth a small fraction of a camera. The groups with fewer
    than `min_size` images are not scenes but parts of one to repair, so
    they join the group they are most tied to. Returns the list of image id
    sets, largest first.
    """
    graph, communities = louvain_communities(rec)
    groups = [set(community) for community in communities]
    volume = [nx.volume(graph, group, weight='weight') for group in groups]
    group_of = {image_id: k for k, group in enumerate(groups) for image_id in group}
    link = {}
    for a, b, w in graph.edges(data='weight'):
        ka, kb = group_of[a], group_of[b]
        if ka != kb:
            key = (min(ka, kb), max(ka, kb))
            link[key] = link.get(key, 0) + w

    def merge(ka, kb):
        groups[ka] |= groups[kb]
        volume[ka] += volume[kb]
        groups[kb], volume[kb] = None, 0
        for (x, y), w in list(link.items()):
            if kb in (x, y):
                del link[(x, y)]
                other = x if y == kb else y
                if other != ka:
                    key = (min(ka, other), max(ka, other))
                    link[key] = link.get(key, 0) + w

    def tie(key):
        smaller = min(key, key=lambda k: volume[k])
        return link[key] * len(groups[smaller]) / volume[smaller] if volume[smaller] else 0

    while link:
        key = max(link, key=tie)
        if tie(key) < min_tie:
            break
        merge(*key)

    alive = [k for k, group in enumerate(groups) if group is not None]
    big = [k for k in alive if len(groups[k]) >= min_size] or [max(alive, key=lambda k: len(groups[k]))]
    for k in sorted(set(alive) - set(big), key=lambda k: len(groups[k])):
        ties = {(x if y == k else y): w for (x, y), w in link.items() if k in (x, y)}
        target = max(ties, key=ties.get) if ties else max(big, key=lambda k: len(groups[k]))
        merge(target, k)

    return sorted((group for group in groups if group is not None), key=len, reverse=True)


def reconstruct_clusters(db, images, output, clusters, min_size=3):
    """
    Runs a separate COLMAP incremental reconstruction for each cluster of
    images (e.g. from transitive.pair_clusters), all reading the same
    database. The models of the i-th largest cluster are written to
    `output/<i>/`, next to the list of its images in `output/<i>/images.txt`.
    Clusters with fewer than `min_size` images are skipped.

    Returns the list of {model_id: pycolmap.Reconstruction} dicts, one per
    reconstructed cluster.
    """
    results = []
    for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)):
        if len(cluster) < min_size:
            continue

        path = os.path.join(output, str(i))
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'images.txt'), 'w') as f:
            f.write('\n'.join(sorted(cluster)) + '\n')

        options = pycolmap.IncrementalPipelineOptions()
        options.image_names = sorted(cluster)
        models = pycolmap.incremental_mapping(database_path=db, image_path=images, output_path=path, options=options)
        results.append(models)

        registered = max((m.num_reg_images() for m in models.values()), default=0)
        print(f"reconstruct_clusters: cluster {i}, {len(cluster)} images, {len(models)} model(s), "
              f"{registered} images registered in the largest")

    return results


class split_module:
    """
    Separates the different scenes of a reconstruction built from a mixed
    image set: at the end of the pipeline that builds the model, with the
    same `worklist`, finalize() splits every model of `model` (a
    reconstruction module or a model folder) into groups of images weakly
    tied to each other (split_groups: Louvain communities of the
    covisibility graph merged while tied by at least `min_tie` cameras'
    worth of covisibility, groups smaller than `min_size` kept with their
    scene), then reconstructs each group on its own from `db`, so the wrong
    matches between scenes no longer distort them.

    The i-th largest group is written to `output/<i>/` (its models and the
    list of its images in `images.txt`, see reconstruct_clusters);
    `datasets`, a list filled in place like kfc_module's `pairs`, gets the
    image names of each group.
    """
    def __init__(self, **args):
        datasets = args.pop('datasets', None)

        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False

        self.args = {
            'id_more': '',
            'db': 'colmap.db',
            'images': '.',
            'model': None,
            'output': 'datasets',
            'min_tie': 0.5,
            'min_size': 20,
            'worklist': None,
        }

        self.id_string, self.args = set_args('split', args, self.args)
        self.datasets = datasets if datasets is not None else []

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        return {}

    def finalize(self):
        if self.args['worklist']:
            return

        model = self.args['model']
        paths = model._results() if hasattr(model, '_results') else model_paths(model)

        # largest model first; an image already in a group of a larger
        # model is not assigned again
        models = sorted((pycolmap.Reconstruction(path) for path in paths), key=lambda m: m.num_reg_images(), reverse=True)
        groups, assigned = [], set()
        for rec in models:
            split = split_groups(rec, min_tie=self.args['min_tie'], min_size=self.args['min_size'])
            print(f"split_module: model with {rec.num_reg_images()} images -> {len(split)} group(s), "
                  f"sizes {[len(group) for group in split]}")
            for group in split:
                names = {rec.images[image_id].name for image_id in group} - assigned
                if names:
                    groups.append(names)
                    assigned |= names

        self.datasets[:] = sorted(groups, key=len, reverse=True)

        database_path = snapshot_db(self.args['db'])
        try:
            reconstruct_clusters(database_path, self.args['images'], self.args['output'], self.datasets, min_size=1)
        finally:
            shutil.rmtree(os.path.dirname(database_path), ignore_errors=True)
