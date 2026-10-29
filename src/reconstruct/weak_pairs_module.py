import os

from core import set_args
from .louvain_module import covisibility_graph, db_graph
from .reconstruct_module import largest_model


def weak_pairs(rec, graph, min_shared=50, max_pairs=None):
    """The pairs matched in the database (edges of `graph`, from db_graph)
    with both images in `rec` but fewer than `min_shared` 3D points seen by
    both, as (image_id, image_id, shared points), weakest first; at most
    `max_pairs` of them, and never more than twice the images of `rec`."""
    covisibility = covisibility_graph(rec)
    registered = set(rec.reg_image_ids())

    pairs = []
    for a, b in graph.edges:
        if a in registered and b in registered:
            shared = covisibility[a][b]['weight'] if covisibility.has_edge(a, b) else 0
            if shared < min_shared:
                pairs.append((a, b, int(shared)))

    limit = 2 * rec.num_reg_images()
    if max_pairs is not None:
        limit = min(limit, max_pairs)
    pairs.sort(key=lambda pair: pair[2])
    return pairs[:limit]


class weak_pairs_module:
    """
    Adds to the repair found by louvain_module the pairs of the model that
    see few points in common: the pairs matched in `db`, with both images
    registered in the largest model of `model`, sharing fewer than
    `min_shared` 3D points; only the weakest ones are kept, at most
    `max_pairs` and never more than twice the images of the model. Each of
    them becomes a cluster of its two
    images, so repair_reconstruction_module removes and registers them
    again, and the pair itself is added to the pairs to match.

    `output` is the same dict given to louvain_module, extended in place
    (also available as `self.output`); the module goes right after
    louvain_module, with the same `model`, `db`, `images` and `worklist`,
    since finalize() runs in pipeline order:

        repair = {}
        run_pairs([..., louvain_module(model='model', db='colmap.db', output=repair, ...),
                   weak_pairs_module(model='model', db='colmap.db', output=repair, ...)], pp)
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
            'min_shared': 50,
            'max_pairs': None,
            'worklist': None,
        }

        self.id_string, self.args = set_args('weak_pairs', args, self.args)
        self.output = output if output is not None else {}
        self.output.setdefault('model', None)
        self.output.setdefault('clusters', [])
        self.output.setdefault('pairs', [])

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        return {}

    def finalize(self):
        if self.args['worklist']:
            return

        model_path, rec = largest_model(self.args['model'])
        if rec is None:
            print("weak_pairs_module: no model to look at")
            return

        graph = db_graph(self.args['db'])
        weak = weak_pairs(rec, graph, self.args['min_shared'], self.args['max_pairs'])

        def name(image_id):
            return graph.nodes[image_id]['name']

        clusters = [{name(a), name(b)} for a, b, _ in weak]
        pairs = {tuple(sorted((os.path.join(self.args['images'], name(a)), os.path.join(self.args['images'], name(b))))) for a, b, _ in weak}

        if self.output['model'] is None:
            self.output['model'] = model_path
        self.output['clusters'] = self.output['clusters'] + clusters
        self.output['pairs'] = sorted(set(self.output['pairs']) | pairs)

        images = set().union(*clusters)
        print(f"weak_pairs_module: {len(weak)} weakest pairs with fewer than {self.args['min_shared']} shared points, "
              f"{len(images)} of {rec.num_reg_images()} images of {model_path} to register again, "
              f"{len(self.output['pairs'])} pairs to match in all")
