import os

import networkx as nx
import numpy as np

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from core import set_args
from core.pipeline import go_iter


class kfc_module:
    """
    Keypoint Filtering by Coverage (KFC) pair selector, adapted from Bellavia
    et al., 2022, "Image orientation with a hybrid pipeline robust to
    rotations and wide-baselines" (ISPRS Archives XLVI-2/W1-2022), run here as
    a pass-through recorder.

    Placed at the end of a global-similarity pipeline, it observes every
    pair's 'pair_sim' without touching pipe_data. `__init__` creates an empty
    similarity table and an empty (to be filled) selected-pairs list; each
    run() records `pair_sim` for the current (img0, img1) in the table.

    finalize():
      * merges in the table already saved under '/sim_table' in `sim_db`
        by a previous run, if any, and saves the result back there as a
        single entry;
      * builds a graph weighted by 'pair_sim';
      * extracts `n_mst` successive maximum spanning trees, removing every
        edge picked by the earlier trees before computing the next one
        (`n_mst` defaults to 2);
      * concatenates their edges into the selected-pairs list;
      * saves the selected-pairs list to the HDF5 store at `out_path`.

    Read the selection back with `kfc_module.load_pairs(out_path)` to drive a
    second matching pipeline, e.g.
    `run_pairs(real_pipeline, imgs, colmap_db_or_list=kfc_module.load_pairs(out_path), mode='include')`.

    For a caller in the same process that already holds a list before
    run_pairs() starts, pass it in as `pairs`: finalize() fills that same
    list in place (never reassigns it), so the caller's variable has the
    selection right after run_pairs() returns.

        pp = []
        run_pairs([..., kfc_module(pairs=pp, sim_db=..., n_mst=2)], imgs, db_name=...)
        # pp now holds the selected (img0, img1) pairs

    Passing a set as `allowed` makes finalize() fill it, in place, with the
    pairs (as frozensets of basenames) whose similarity is at least
    `sim_min`, or at least the `sim_quantile` quantile (0..1) of the table
    when `sim_min` is not set, or all pairs when neither is set; give the
    same set to transitive_module so that it only proposes those pairs.

    With `multi_dataset=True`, finalize() first splits the images into
    separate datasets by cutting the first maximum spanning tree (see
    `split_datasets`), then runs the KFC selection on each one. `datasets`
    (a list passed by reference) is filled with one seed-pair list per
    dataset; `pairs` gets all of them concatenated.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False

        pairs = args.pop('pairs', None)
        datasets = args.pop('datasets', None)
        allowed = args.pop('allowed', None)

        self.args = {
            'id_more': '',
            'out_path': 'kfc_pairs.hdf5',
            'sim_db': None,
            'n_mst': 2,
            'multi_dataset': False,
            'cut_ratio': 0.5,
            'sep_ratio': 0.8,
            'min_dataset_size': 3,
            'n_datasets': None,
            'sim_min': None,
            'sim_quantile': 0.0,
        }

        self.id_string, self.args = set_args('kfc', args, self.args)

        self._table = {}
        self.selected = pairs if pairs is not None else []
        self.datasets = datasets if datasets is not None else []
        self.allowed = allowed
        self._n_seen = 0

    @staticmethod
    def load_pairs(out_path, id_string='kfc'):
        """Read back the flat list of selected (img0, img1) pairs."""
        aux_hdf5 = pickled_hdf5.pickled_hdf5(out_path, mode='r', label_prefix='pickled/' + id_string)
        pairs = []
        for key in aux_hdf5.get_keys():
            if not key.startswith('/pairs/'):
                continue
            val, is_found = aux_hdf5.get(key)
            if is_found:
                pairs.append(tuple(val))
        aux_hdf5.close()
        return pairs

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        assert 'pair_sim' in pipe_data, \
            "missing required key 'pair_sim' — add a similarity module before this one"

        img0, img1 = pipe_data['img'][0], pipe_data['img'][1]
        self._table[(img0, img1)] = pipe_data['pair_sim']
        self._n_seen += 1

        return {}

    @staticmethod
    def select_pairs(table, n_mst=2):
        """
        KFC selection over an arbitrary {(img0, img1): pair_sim} table:
        `n_mst` successive maximum spanning trees, each computed after the
        edges chosen by the previous ones have been removed. Returns the
        concatenated list of (img0, img1) pairs, oriented as in `table`.

        Exposed as a staticmethod so a second pipeline can build its seed
        list straight from the table this module cached, without an instance.
        """
        if not table:
            return []

        graph = nx.Graph()
        for (a, b), sim in go_iter(list(table.items()), msg='kfc: building graph'):
            graph.add_edge(a, b, weight=sim)

        orientation = {frozenset((a, b)): (a, b) for a, b in table}

        selected = []
        for i in range(n_mst):
            if graph.number_of_edges() == 0:
                break
            tree_edges = list(nx.maximum_spanning_tree(graph, weight='weight').edges())
            selected.extend(orientation.get(frozenset((a, b)), (a, b)) for a, b in tree_edges)
            graph.remove_edges_from(tree_edges)
            print(f"kfc_module: MST {i + 1}/{n_mst} -> {len(tree_edges)} edges")

        return selected

    @staticmethod
    def split_datasets(table, cut_ratio=0.5, sep_ratio=0.8, min_size=3, n_datasets=None):
        """
        Splits the images of `table` into separate datasets by cutting edges
        of its maximum spanning tree. An edge is a candidate cut when its
        similarity is below `cut_ratio` times the mean similarity of the tree
        edges touching its two endpoints. Candidates are tried from the
        weakest; a cut is kept only if both sides have at least `min_size`
        images and the mean similarity between the two sides is below
        `sep_ratio` times the lower of the two within-side means. With
        `n_datasets` set, `cut_ratio`/`sep_ratio` are ignored: edges are cut
        by increasing similarity relative to their neighbours (still keeping
        `min_size`) until there are `n_datasets` sets. Returns the list of
        image sets.
        """
        graph = nx.Graph()
        for (a, b), sim in go_iter(list(table.items()), msg='kfc: building graph'):
            graph.add_edge(a, b, weight=sim)
        tree = nx.maximum_spanning_tree(graph, weight='weight')

        scored = []
        for u, v, sim in tree.edges(data='weight'):
            near = [tree[x][y]['weight'] for x in (u, v) for y in tree[x] if {x, y} != {u, v}]
            if near:
                scored.append((sim / np.mean(near), sim, u, v))

        if n_datasets is None:
            candidates = sorted((sim, u, v) for ratio, sim, u, v in scored if ratio < cut_ratio)

            idx = {img: i for i, img in enumerate(graph.nodes)}
            sim_m = np.full((len(idx), len(idx)), np.nan, dtype=np.float32)
            for (a, b), sim in table.items():
                sim_m[idx[a], idx[b]] = sim
                sim_m[idx[b], idx[a]] = sim

            def mean_sim(g0, g1):
                return np.nanmean(sim_m[np.ix_([idx[x] for x in g0], [idx[x] for x in g1])])
        else:
            candidates = [(sim, u, v) for ratio, sim, u, v in sorted(scored)]

        n_groups = nx.number_connected_components(tree)
        for sim, u, v in candidates:
            if n_datasets is not None and n_groups >= n_datasets:
                break
            tree.remove_edge(u, v)
            g0 = nx.node_connected_component(tree, u)
            g1 = nx.node_connected_component(tree, v)
            keep = len(g0) >= min_size and len(g1) >= min_size
            msg = ''
            if keep and n_datasets is None:
                sep = mean_sim(g0, g1) / min(mean_sim(g0, g0), mean_sim(g1, g1))
                keep = sep < sep_ratio
                msg = f", between/within similarity {sep:.3g} (sep_ratio {sep_ratio})"
            if keep:
                print(f"kfc_module: cut at similarity {sim:.4g} -> {len(g0)} / {len(g1)} images{msg}")
                n_groups += 1
            else:
                tree.add_edge(u, v, weight=sim)

        return [set(g) for g in nx.connected_components(tree)]

    def _select(self):
        if not self.args['multi_dataset']:
            return self.select_pairs(self._table, self.args['n_mst'])

        groups = self.split_datasets(self._table, self.args['cut_ratio'], self.args['sep_ratio'], self.args['min_dataset_size'], self.args['n_datasets'])
        group_of = {img: i for i, g in enumerate(groups) for img in g}
        tables = [{} for _ in groups]
        for (a, b), sim in self._table.items():
            if group_of[a] == group_of[b]:
                tables[group_of[a]][(a, b)] = sim

        self.datasets[:] = [self.select_pairs(t, self.args['n_mst']) for t in tables]
        print(f"kfc_module: {len(groups)} dataset(s) found, sizes {[len(g) for g in groups]}")
        return [p for d in self.datasets for p in d]

    def _fill_allowed(self):
        cut = self.args['sim_min']
        q = self.args['sim_quantile']
        if cut is None and q and q > 0.0 and self._table:
            vals = sorted(self._table.values())
            cut = vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))]
        self.allowed.clear()
        self.allowed.update(frozenset((os.path.basename(a), os.path.basename(b)))
                            for (a, b), sim in self._table.items() if cut is None or sim >= cut)
        cut_msg = 'no cut' if cut is None else f'similarity cut {cut:.4g}'
        print(f"kfc_module: {cut_msg}, {len(self.allowed)} of {len(self._table)} pairs allowed")

    def finalize(self):
        if self.args['sim_db'] is not None:
            db = pickled_hdf5.pickled_hdf5(self.args['sim_db'])
            table, found = db.get('/sim_table')
            if found:
                table.update(self._table)
                self._table = table
            db.add('/sim_table', self._table)
            db.close()

        if self.allowed is not None:
            self._fill_allowed()

        # mutate the list in place (never reassign it) so that if `pairs` was
        # passed in at construction, the caller's own list object also ends
        # up holding the selection - not just self.selected.
        self.selected[:] = self._select()

        if self.args['out_path'] is not None:
            if os.path.exists(self.args['out_path']):
                os.remove(self.args['out_path'])

            aux_hdf5 = pickled_hdf5.pickled_hdf5(self.args['out_path'], mode='a', label_prefix='pickled/' + self.id_string)
            for img0, img1 in self.selected:
                data_key = '/pairs/' + os.path.split(img0)[-1] + '/' + os.path.split(img1)[-1]
                aux_hdf5.add(data_key, (img0, img1))
            aux_hdf5.close()

        print(f"kfc_module: {len(self._table)} pairs recorded (of {self._n_seen} seen), "
              f"{len(self.selected)} selected over {self.args['n_mst']} MST(s), "
              f"saved to {self.args['out_path']}")
