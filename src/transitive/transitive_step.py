import itertools

import cv2
import networkx as nx

import pickled_hdf5.pickled_hdf5 as pickled_hdf5

from core.pipeline import _cached_single_image, _cached_pair_similarity, _resolve_and_sort_imgs, go_iter


class TransitiveRounds:
    """
    Builds the pair list for one round of a transitive pipeline, and tracks
    the confirmed-pair graph across rounds. Used by core.run_pairs(): once it
    finds a transitive-selection module in `pipeline`, it constructs
    one of these and calls `next_pairs()` / `round_done()` from its main loop
    instead of generating every possible pair up front — see run_pairs()'s
    docstring for how the two fit together.

    """

    def __init__(self, pipeline, imgs, selection_module, add_path, db_name, db_mode, on_candidates=None):
        cv2.setNumThreads(20)

        self.selection_module = selection_module
        self.db_name = db_name
        self.db_mode = db_mode
        self.on_candidates = on_candidates

        self.imgs = _resolve_and_sort_imgs(imgs, add_path)

        self.descriptor = next((m for m in pipeline if getattr(m, 'single_image', False)), None)
        assert self.descriptor is not None, \
            "pipeline must include a single-image global descriptor module (e.g. salad_module, standard_descriptor_module)"

        self.conf = next((m for m in pipeline if hasattr(m, '_table')), None)
        assert self.conf is not None, \
            "pipeline must include a conf_module (or compatible module exposing '_table')"

        self.similarity = next(
            (m for m in pipeline if m is not self.descriptor and m is not self.conf and m is not selection_module),
            None,
        )
        assert self.similarity is not None, \
            "pipeline must include a pair similarity module consuming 'global_desc' (e.g. cosine_similarity_module, standard_similarity_module)"

        desc_db = pickled_hdf5.pickled_hdf5(db_name, mode=db_mode)

        print(f"run_pairs (transitive): computing global descriptors with '{self.descriptor.get_id()}'")
        self.global_desc = {
            img: _cached_single_image(desc_db, self.descriptor, img)['global_desc']
            for img in go_iter(self.imgs, msg='computing global descriptors')
        }

        # run_pairs() below opens its own h5py.File handle on db_name each
        # round; keeping this one open concurrently (same file, no SWMR)
        # risks the other handle seeing stale/partial writes, so release it.
        desc_db.close()

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.imgs)

        self.threshold = self.conf.args['threshold']
        self.tried = set()
        self._n_before = 0

    def _rank(self, candidates):
        sim_db = pickled_hdf5.pickled_hdf5(self.db_name, mode=self.db_mode)
        scored = [
            (
                _cached_pair_similarity(sim_db, self.descriptor, self.similarity, a, b, self.global_desc[a], self.global_desc[b]),
                (a, b),
            )
            for a, b in go_iter(candidates, msg='ranking candidate pairs')
        ]
        # released before run_pairs() opens its own handle on db_name below
        sim_db.close()

        scored.sort(key=lambda x: x[0], reverse=True)
        if self.on_candidates is not None:
            self.on_candidates(scored, self.threshold)
        return scored

    def _candidates_first_round(self):
        imgs = self.imgs
        candidates = [
            (imgs[i], imgs[j])
            for i in range(len(imgs))
            for j in range(i + 1, len(imgs))
            if (imgs[i], imgs[j]) not in self.tried
        ]
        return self.selection_module.select(self._rank(candidates), self.threshold)

    def _candidates_transitive(self):
        candidates = set()
        for b in self.graph.nodes:
            for a, c in itertools.combinations(self.graph.neighbors(b), 2):
                pair = (min(a, c), max(a, c))
                if pair not in self.tried:
                    candidates.add(pair)

        return [pair for sim, pair in self._rank(list(candidates)) if sim > self.threshold]

    def next_pairs(self):
        """Builds the pair list for the next round: the initial selection if
        no pair has been confirmed yet, transitive closure otherwise. Returns
        [] once there's nothing left worth trying."""
        pairs = self._candidates_first_round() if self.graph.number_of_edges() == 0 else self._candidates_transitive()
        if pairs:
            self.tried.update(pairs)
            self._n_before = len(self.conf._table)
            self._n_pairs_tried = len(pairs)
        return pairs

    def round_done(self, n_round):
        """Call once `pipeline` has been run on the pairs `next_pairs()` just
        returned: folds newly confirmed pairs into `graph` and returns how
        many were newly confirmed this round."""
        n_new = len(self.conf._table) - self._n_before
        self.graph.add_edges_from(self.conf._table[self._n_before:])
        print(f"run_pairs (transitive): round {n_round}, {self._n_pairs_tried} pairs tried, {n_new} newly confirmed")
        return n_new
