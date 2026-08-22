from core import set_args


class percentage_module:
    """
    First-round selection strategy for a transitive run_pairs() pipeline: out
    of every possible pair, ranked by similarity, keeps at most the top
    `percentage` fraction — e.g. percentage=0.1 keeps the best 10% of all
    pairs. Pairs at or below `threshold` (conf_module's threshold) are never
    kept, even if they'd fall within that fraction.
    """
    is_transitive_initial_selection = True

    def __init__(self, **args):
        self.args = {
            'id_more': '',
            'percentage': 0.1,
        }

        self.id_string, self.args = set_args('percentage', args, self.args)
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
        above = [(sim, pair) for sim, pair in scored if sim > threshold]
        n = int(len(scored) * self.args['percentage'])
        return [pair for _, pair in above[:n]]
