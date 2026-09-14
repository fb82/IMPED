import os

import networkx as nx

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
      * builds a graph weighted by 'pair_sim';
      * extracts `n_mst` successive maximum spanning trees, removing every
        edge picked by the earlier trees before computing the next one
        (`n_mst` defaults to 2);
      * concatenates their edges into the selected-pairs list;
      * saves both the full similarity table and the selected-pairs list to
        the HDF5 store at `out_path`.

    Read the selection back with `kfc_module.load_pairs(out_path)` (and the
    full table with `kfc_module.load_table(out_path)`) to drive a second
    matching pipeline, e.g.
    `run_pairs(real_pipeline, imgs, colmap_db_or_list=kfc_module.load_pairs(out_path), mode='include')`.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False

        self.args = {
            'id_more': '',
            'out_path': 'kfc_pairs.hdf5',
            'n_mst': 2,
        }

        self.id_string, self.args = set_args('kfc', args, self.args)

        self._table = {}
        self._selected = []
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

    @staticmethod
    def load_table(out_path, id_string='kfc'):
        """Read back the full {(img0, img1): pair_sim} similarity table."""
        aux_hdf5 = pickled_hdf5.pickled_hdf5(out_path, mode='r', label_prefix='pickled/' + id_string)
        table = {}
        for key in aux_hdf5.get_keys():
            if not key.startswith('/table/'):
                continue
            val, is_found = aux_hdf5.get(key)
            if is_found:
                img0, img1, sim = val
                table[(img0, img1)] = sim
        aux_hdf5.close()
        return table

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

    def _select(self):
        return self.select_pairs(self._table, self.args['n_mst'])

    def finalize(self):
        self._selected = self._select()

        if os.path.exists(self.args['out_path']):
            os.remove(self.args['out_path'])

        aux_hdf5 = pickled_hdf5.pickled_hdf5(self.args['out_path'], mode='a', label_prefix='pickled/' + self.id_string)
        for (img0, img1), sim in self._table.items():
            data_key = '/table/' + os.path.split(img0)[-1] + '/' + os.path.split(img1)[-1]
            aux_hdf5.add(data_key, (img0, img1, sim))
        for img0, img1 in self._selected:
            data_key = '/pairs/' + os.path.split(img0)[-1] + '/' + os.path.split(img1)[-1]
            aux_hdf5.add(data_key, (img0, img1))
        aux_hdf5.close()

        print(f"kfc_module: {len(self._table)} pairs recorded (of {self._n_seen} seen), "
              f"{len(self._selected)} selected over {self.args['n_mst']} MST(s), "
              f"saved to {self.args['out_path']}")
