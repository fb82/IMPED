import itertools
import os

import networkx as nx

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from core import set_args


class transitive_module:
    """
    Drives the transitive-closure loop of a run_pairs() pipeline.

    Like kfc_module, it takes its list by reference: pass the same `pairs`
    list kfc_module was given (already filled with the seed pairs by the
    time pass 1's run_pairs() returns) and finalize() keeps mutating that
    same list in place - here called `pp`. Put it in the match pipeline like
    any other module, and drive the loop from outside:

        transitive = transitive_module(pairs=seed_pairs, ...)
        pipeline = [..., transitive, ...]
        while transitive.pp:
            run_pairs(pipeline, list(transitive.pp), db_name=...)

    run_pairs() calls finalize_pipeline(pipeline) at the end of every call,
    which calls transitive.finalize() same as any other module: it drops the
    pairs computed in that iteration from `pp` and appends the not-yet-tried
    transitive-closure candidates (a-c for each confirmed a-b, b-c) - unless
    `iter` has reached `max_iterations`, in which case it empties `pp`
    instead, which ends the caller's while loop.

    finalize() also sets `self.args['continue']` to whether `pp` is still
    non-empty after this round. Other modules placed after `transitive` in
    the pipeline (e.g. to_colmap_module, live_pair_graph) can be given a
    `worklist=transitive` reference and check `worklist.args['continue']` in
    their own finalize() to skip their real teardown (closing a db, stopping
    a live view) until the closure is actually done.

    Each run() records the pair just matched and, when its match count
    clears `threshold`, adds it to the confirmed graph.

    Two independent gates keep the closure from growing to the full graph:

    * `threshold` - on the *matching* result of pass 2: a pair only becomes a
      confirmed edge (and can therefore spawn transitive candidates) when its
      match count exceeds this. Natural for SIFT (inlier count); pass an
      appropriate value for whatever 'pair_sim' the pipeline produces.

    * `sim_min` / `sim_quantile` - on the *global similarity* of pass 1
      (`sim_table`, the table kfc_module cached). A transitive candidate a-c
      is only queued if its pass-1 similarity is high enough, so weak pairs
      are never matched at all. `sim_quantile` (0..1) is scale-independent -
      it cuts at that quantile of the table's values - so the same setting
      works whether pass 1 produced SIFT match counts or SALAD cosine
      similarities. `sim_min` sets an absolute cut instead and takes
      precedence when given. With neither set, every candidate is queued
      (previous behaviour).
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False

        pairs = args.pop('pairs', None)
        sim_table = args.pop('sim_table', None)

        self.args = {
            'id_more': '',
            'threshold': 0.0,
            'sim_min': None,
            'sim_quantile': 0.0,
            'max_iterations': 10,
            'out_path': 'transitive_pairs.hdf5',
        }

        self.id_string, self.args = set_args('transitive', args, self.args)

        # start clean: the snapshot file is a fresh output, not an accumulator
        if os.path.exists(self.args['out_path']):
            os.remove(self.args['out_path'])

        # pass-1 similarity, keyed by the unordered pair of basenames
        self._sim = {}
        if sim_table:
            for (a, b), v in sim_table.items():
                self._sim[frozenset((os.path.basename(a), os.path.basename(b)))] = v

        self._sim_cut = self._compute_sim_cut()

        self.pp = pairs if pairs is not None else []
        self._graph = nx.Graph()
        self._tried = {frozenset(p) for p in self.pp}
        self._computed_this_round = set()
        self.pair_round = {}
        self.iter = 0
        self._n_skipped_low_sim = 0

    def _compute_sim_cut(self):
        if self.args['sim_min'] is not None:
            return float(self.args['sim_min'])

        q = self.args['sim_quantile']
        if not self._sim or not q or q <= 0.0:
            return None

        vals = sorted(self._sim.values())
        idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        cut = vals[idx]
        print(f"transitive_module: pass-1 similarity cut at quantile {q} -> {cut:.4g} "
              f"({len(vals)} scored pairs)")
        return cut

    def get_id(self):
        return self.id_string

    def _sim_ok(self, a, c):
        """Whether the pass-1 global similarity of pair (a, c) is high enough
        to bother matching it. True when no similarity gate is configured."""
        if self._sim_cut is None:
            return True
        v = self._sim.get(frozenset((os.path.basename(a), os.path.basename(c))))
        return v is not None and v >= self._sim_cut

    @staticmethod
    def _match_count(pipe_data):
        """How many matches the pipeline confirmed for this pair: an upstream
        'pair_sim' if present, otherwise the number of set entries in the
        RANSAC inlier mask 'm_mask'."""
        sim = pipe_data.get('pair_sim')
        if sim is not None:
            return sim

        m_mask = pipe_data.get('m_mask')
        if m_mask is None:
            return 0
        try:
            return int(m_mask.sum())
        except AttributeError:
            return int(sum(bool(x) for x in m_mask))

    def run(self, **pipe_data):
        a, b = pipe_data['img'][0], pipe_data['img'][1]

        self._computed_this_round.add(frozenset((a, b)))
        self._tried.add(frozenset((a, b)))

        n_matches = self._match_count(pipe_data)
        if n_matches > self.args['threshold']:
            self._graph.add_edge(a, b, weight=n_matches)

        return {'pair_sim': float(n_matches)}

    def finalize(self):
        # drop the pairs we just computed from pp, in place
        self.pp[:] = [
            (a, b) for a, b in self.pp
            if frozenset((a, b)) not in self._computed_this_round
        ]
        self.iter += 1

        for p in self._computed_this_round:
            self.pair_round[frozenset(p)] = self.iter

        if self.iter >= self.args['max_iterations']:
            self.pp[:] = []
            new_pairs = []
        else:
            # append the transitive-closure candidates worth trying next
            new_pairs = []
            for b in list(self._graph.nodes):
                for a, c in itertools.combinations(sorted(self._graph.neighbors(b)), 2):
                    key = frozenset((a, c))
                    if key in self._tried:
                        continue
                    self._tried.add(key)
                    if not self._sim_ok(a, c):
                        self._n_skipped_low_sim += 1
                        continue
                    new_pairs.append((a, c))
            self.pp.extend(new_pairs)

        self._computed_this_round = set()
        self.args['continue'] = bool(self.pp)

        self._save()
        print(f"transitive_module: iteration {self.iter}, "
              f"+{len(new_pairs)} transitive pairs "
              f"({self._n_skipped_low_sim} skipped so far for low pass-1 similarity), "
              f"{len(self.pp)} left to do, "
              f"{self._graph.number_of_edges()} pairs confirmed so far")

    def _save(self):
        aux = pickled_hdf5.pickled_hdf5(self.args['out_path'], mode='a', label_prefix='pickled/' + self.id_string)
        aux.add('/todo/round_' + str(self.iter), [tuple(p) for p in self.pp])
        aux.add('/confirmed', [tuple(sorted(e)) for e in self._graph.edges()])
        aux.close()

    @staticmethod
    def load_pairs(out_path, id_string='transitive'):
        """Read back the confirmed (img0, img1) pairs of the transitive
        closure. Returns [] if nothing was ever written (e.g. no seed pairs)."""
        if not os.path.exists(out_path):
            return []
        aux = pickled_hdf5.pickled_hdf5(out_path, mode='r', label_prefix='pickled/' + id_string)
        val, ok = aux.get('/confirmed')
        aux.close()
        return [tuple(p) for p in val] if ok else []
