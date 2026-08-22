from core import set_args


class max_uses_module:
    """
    First-round selection strategy for a transitive run_pairs() pipeline:
    walks pairs ranked by similarity, highest first, keeping a pair only if
    neither of its two images has already been kept in `max_uses` pairs this
    round. Pairs at or below `threshold` (conf_module's threshold) are never
    kept.
    """
    is_transitive_initial_selection = True

    def __init__(self, **args):
        self.args = {
            'id_more': '',
            'max_uses': 3,
        }

        self.id_string, self.args = set_args('maxuses', args, self.args)
        self.add_to_cache = False

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        """
        Lets this module sit inside a regular pipeline passed to run_pairs():
        it doesn't touch the pair itself, it just marks the run as a
        transitive one, so run_pairs (via transitive.transitive_step.TransitiveRounds)
        knows to keep going with further rounds — see that module's docstring.
        """
        return {'continue': True}

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
