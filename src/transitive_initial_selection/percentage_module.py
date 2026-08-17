from core import set_args


class percentage_module:
    """
    First-round selection strategy for run_transitive_pairs: out of every
    possible pair, ranked by similarity, keeps at most the top `percentage`
    fraction — e.g. percentage=0.1 keeps the best 10% of all pairs. Pairs
    at or below `threshold` (conf_module's threshold, passed in by
    run_transitive_pairs) are never kept, even if they'd fall within that
    fraction.
    """
    def __init__(self, **args):
        self.args = {
            'id_more': '',
            'percentage': 0.1,
        }

        self.id_string, self.args = set_args('percentage', args, self.args)

    def get_id(self):
        return self.id_string

    def select(self, scored, threshold):
        above = [(sim, pair) for sim, pair in scored if sim > threshold]
        n = int(len(scored) * self.args['percentage'])
        return [pair for _, pair in above[:n]]
