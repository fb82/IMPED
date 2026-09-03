import time

import networkx as nx
import torch

from core import set_args
from core.pipeline import go_iter


class kfc_module:
    """
    Seed-selection module for a transitive run_pairs() pipeline, adapting
    Keypoint Filtering by Coverage (KFC) from Bellavia et al., 2022, "Image
    orientation with a hybrid pipeline robust to rotations and
    wide-baselines" (ISPRS Archives XLVI-2/W1-2022) to pick seed pairs
    instead of pruning a Bundle Adjustment.
    """
    is_transitive_initial_selection = True

    def __init__(self, **args):
        self.args = {
            'id_more': '',
            'threshold': 0.5,
            'out_path': 'kfc_pairs.pt',
        }

        self.id_string, self.args = set_args('kfc', args, self.args)
        self.add_to_cache = False

        self._table = []
        self._n_seen = 0

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        self._n_seen += 1
        self._table.append((pipe_data['img'][0], pipe_data['img'][1]))
        return {'continue': True, 'pair_conf': 1.0}

    def finalize(self):
        torch.save(self._table, self.args['out_path'])
        print(f"kfc_module: {len(self._table)} pairs kept (of {self._n_seen} tried), "
              f"saved to {self.args['out_path']}")

    def select(self, scored, threshold):
        if not scored:
            return []

        print(f"kfc_module: selecting from {len(scored)} scored pairs")

        start_time = time.time()
        graph = nx.Graph()
        for sim, (a, b) in go_iter(scored, msg='kfc: building main graph'):
            graph.add_edge(a, b, weight=sim)
        print(f"kfc_module: built main graph ({graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges) in {time.time() - start_time:.1f}s")

        max_sim = max(sim for sim, _ in scored)

        start_time = time.time()
        complementary = nx.Graph()
        for sim, (a, b) in go_iter(scored, msg='kfc: building complementary graph'):
            complementary.add_edge(a, b, weight=max_sim - sim)
        print(f"kfc_module: built complementary graph in {time.time() - start_time:.1f}s")

        start_time = time.time()
        best_tree_edges = {
            frozenset(e) for e in nx.minimum_spanning_tree(complementary, weight='weight').edges()
        }
        print(f"kfc_module: computed primary MST ({len(best_tree_edges)} edges) "
              f"in {time.time() - start_time:.1f}s")

        start_time = time.time()
        remainder = complementary.copy()
        remainder.remove_edges_from(tuple(e) for e in best_tree_edges)
        print(f"kfc_module: built remainder graph in {time.time() - start_time:.1f}s")

        start_time = time.time()
        backup_tree_edges = {
            frozenset(e) for e in nx.minimum_spanning_tree(remainder, weight='weight').edges()
        }
        print(f"kfc_module: computed backup MST ({len(backup_tree_edges)} edges) "
              f"in {time.time() - start_time:.1f}s")

        selected = [
            (a, b) for sim, (a, b) in scored
            if frozenset((a, b)) in best_tree_edges or frozenset((a, b)) in backup_tree_edges
        ]
        print(f"kfc_module: selected {len(selected)} pairs")

        return selected