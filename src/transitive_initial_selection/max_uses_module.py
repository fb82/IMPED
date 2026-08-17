from core import set_args


class max_uses_module:
    """
    First-round selection strategy for run_transitive_pairs: walks pairs
    ranked by similarity, highest first, keeping a pair only if neither of
    its two images has already been kept in `max_uses` pairs this round.
    Pairs at or below `threshold` (conf_module's threshold, passed in by
    run_transitive_pairs) are never kept.
    """
    def __init__(self, **args):
        self.args = {
            'id_more': '',
            'max_uses': 3,
        }

        self.id_string, self.args = set_args('maxuses', args, self.args)

    def get_id(self):
        return self.id_string

    def select(self, scored, threshold):
        uses = {}
        selected = []

        for sim, (a, b) in scored:
            if sim <= threshold:
                break

            if uses.get(a, 0) >= self.args['max_uses'] or uses.get(b, 0) >= self.args['max_uses']:
                continue

            uses[a] = uses.get(a, 0) + 1
            uses[b] = uses.get(b, 0) + 1
            selected.append((a, b))

        return selected
