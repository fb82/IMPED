import os

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from core import set_args


class conf_module:
    """
    A pair module selecting which pairs are worth running the full matching
    pipeline on, based on an upstream similarity score ('pair_sim' — e.g.
    from cosine_similarity_module or l2_similarity_module).

    For each pair, stores a confidence value under 'pair_conf': 0 if the
    pair is below `threshold` (discard), otherwise the similarity score
    itself (kept, degree of confidence to take it) — matching the
    dic scheme described for pair selection.

    Every kept pair is also accumulated in an in-memory table as the
    pipeline runs; at finalize() that list of (img0, img1) pairs is saved to
    the HDF5 store at `out_path` (one entry per pair, keyed by the two image
    basenames), in the same way pairwise_benchmark_module logs its stats.
    Read it back with `conf_module.load_pairs(out_path)`. This is the piece
    that replaces run_mst_pairs: running [salad_module, a similarity module,
    conf_module] through run_pairs over the full image set builds this store,
    and a second, real matching pipeline can then be run only on those pairs
    via
    `run_pairs(real_pipeline, imgs, colmap_db_or_list=conf_module.load_pairs(out_path), mode='include')`.

    'pair_conf' is not cached by default (`add_to_cache=False`), even though
    the pipeline's normal per-pair caching would happily do so: the decision
    depends on `threshold`, which isn't part of the cache key, so re-running
    the coarse pass with a lower threshold (to progressively grow the set of
    pairs matched, without recomputing 'pair_sim' — that one IS safely
    cacheable, since it doesn't depend on threshold) would otherwise silently
    return the stale pair_conf from the first run instead of recomputing it.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = False

        self.args = {
            'id_more': '',
            'threshold': 0.5,
            'out_path': 'conf_pairs.hdf5',
        }

        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('conf', args, self.args)

        self._table = []
        self._n_seen = 0


    @staticmethod
    def load_pairs(out_path, id_string='conf'):
        """Read back the list of (img0, img1) pairs stored by conf_module."""
        aux_hdf5 = pickled_hdf5.pickled_hdf5(out_path, mode='r', label_prefix='pickled/' + id_string)
        pairs = []
        for key in aux_hdf5.get_keys():
            val, is_found = aux_hdf5.get(key)
            if is_found: pairs.append(tuple(val))
        aux_hdf5.close()
        return pairs


    def get_id(self):
        return self.id_string


    def run(self, **args):
        assert 'pair_sim' in args, "missing required key 'pair_sim' — add a similarity module before this one"

        sim = args['pair_sim']
        self._n_seen += 1

        conf = sim if sim > self.args['threshold'] else 0.0

        if conf > 0.0:
            self._table.append((args['img'][0], args['img'][1]))

        return {'pair_conf': conf}


    def finalize(self):
        aux_hdf5 = pickled_hdf5.pickled_hdf5(self.args['out_path'], mode='a', label_prefix='pickled/' + self.id_string)
        for img0, img1 in self._table:
            data_key = '/' + os.path.split(img0)[-1] + '/' + os.path.split(img1)[-1]
            aux_hdf5.add(data_key, (img0, img1))
        aux_hdf5.close()
        print(f"conf_module: {len(self._table)} pairs above threshold={self.args['threshold']} "
              f"(of {self._n_seen} seen), saved to {self.args['out_path']}")
