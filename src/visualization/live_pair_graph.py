import os
import webbrowser

import networkx as nx
from pyvis.network import Network

MAX_VISUALIZED_CANDIDATES = 5000


class live_pair_graph:
    """
    Interactive, incrementally-updated visualisation of the pair graph built
    by a transitive run_pairs() pipeline, using pyvis (vis.js):
    draggable/zoomable nodes, hover tooltips, click-to-highlight neighbours.

    Drop the instance straight into the match pipeline, after `transitive`
    (see transitive_module). Its run() records every pair the pipeline
    processes into `self.graph`/`self.rejected`/`self.pending`, and always
    redraws (recomputes the layout and rewrites the HTML) once per round, in
    finalize() - called automatically by finalize_pipeline() right after
    `transitive.finalize()`, so it always sees that round's final state.
    Edges are coloured using `transitive.pair_round`, which round
    confirmed each pair.

    `redraw_every` additionally redraws every N confirmed/rejected pairs
    within a round, for finer-grained feedback on long rounds; `None`
    (default when a `worklist` is given) disables it, leaving only the
    once-per-round redraw. Without a `worklist` there is no notion of
    "round" at all, so `redraw_every` defaults to 1 (redraw on every pair).
    """
    def __init__(self, imgs, save_to='live_pair_graph.html', open_browser=True, worklist=None, redraw_every=None):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False
        self.id_string = 'live_pair_graph'

        self.graph = nx.Graph()
        self.graph.add_nodes_from(imgs)
        self.rejected = {}
        self.pending = {}
        self.save_to = save_to
        self.worklist = worklist
        self.redraw_every = redraw_every if redraw_every is not None else (None if worklist is not None else 1)
        self._pairs_since_redraw = 0

        self.pos = {n: (x * 1000, y * 1000) for n, (x, y) in nx.spring_layout(self.graph).items()}

        self._do_redraw()
        if open_browser:
            webbrowser.open('file://' + os.path.abspath(save_to))

    def get_id(self):
        return self.id_string

    def _known(self, img_a, img_b):
        return img_a in self.graph and img_b in self.graph

    def _maybe_redraw(self, relayout):
        if self.redraw_every is None:
            return
        self._pairs_since_redraw += 1
        if self._pairs_since_redraw < self.redraw_every:
            return
        self._pairs_since_redraw = 0
        if relayout:
            self._recompute_layout()
        self._do_redraw()

    def add_pair(self, img_a, img_b, conf=None):
        if not self._known(img_a, img_b):
            return
        self.graph.add_edge(img_a, img_b, conf=conf)
        self.rejected.pop((min(img_a, img_b), max(img_a, img_b)), None)
        self.pending.pop((min(img_a, img_b), max(img_a, img_b)), None)
        self._maybe_redraw(relayout=True)

    def add_rejected_pair(self, img_a, img_b, conf=None):
        if not self._known(img_a, img_b):
            return
        key = (min(img_a, img_b), max(img_a, img_b))
        if self.graph.has_edge(img_a, img_b):
            return
        if key in self.rejected and self.rejected[key] == conf:
            return
        self.rejected[key] = conf
        self.pending.pop(key, None)
        self._maybe_redraw(relayout=False)

    def sync_worklist(self):
        if self.worklist is None:
            return
        pending = {}
        for a, b in getattr(self.worklist, 'pp', []):
            if not self._known(a, b):
                continue
            key = (min(a, b), max(a, b))
            if self.graph.has_edge(a, b) or key in self.rejected:
                continue
            pending[key] = None
        self.pending = pending

    def on_pair(self, pair, pipe_data):
        img_a, img_b = pair
        conf = pipe_data.get('pair_sim')

        keep = pipe_data.get('pair_conf', True)
        if self.worklist is not None:
            th = self.worklist.args.get('threshold') if hasattr(self.worklist, 'args') else None
            if 'pair_conf' not in pipe_data and conf is not None and th is not None:
                keep = conf > th

        if keep:
            self.add_pair(img_a, img_b, conf=conf)
        else:
            self.add_rejected_pair(img_a, img_b, conf=conf)

    def on_candidates(self, scored, threshold):
        if len(scored) > MAX_VISUALIZED_CANDIDATES:
            print(f"live_pair_graph: skipping visualization of {len(scored)} candidates "
                  f"(over the {MAX_VISUALIZED_CANDIDATES} cap) - only confirmed pairs will be shown")
            return
        for sim, pair in scored:
            self.add_rejected_pair(*pair, conf=sim)

    def on_round(self, graph, n_round, n_new):
        self.sync_worklist()

    def run(self, **pipe_data):
        self.on_pair((pipe_data['img'][0], pipe_data['img'][1]), pipe_data)
        self.sync_worklist()
        return {}

    def finalize(self):
        if self.worklist is not None:
            self._recompute_layout()
            self._do_redraw()
            self._pairs_since_redraw = 0
            if self.worklist.args.get('continue', False):
                return
        self.stop()

    def stop(self):
        self._do_redraw()

    def _recompute_layout(self):
        seed = {n: (x / 1000, y / 1000) for n, (x, y) in self.pos.items()}
        self.pos = {n: (x * 1000, y * 1000) for n, (x, y) in nx.spring_layout(self.graph, pos=seed).items()}

    def _do_redraw(self):
        net = Network(height='90vh', width='100%', notebook=False)
        net.from_nx(self.graph)
        for node in net.nodes:
            node['label'] = os.path.basename(node['id'])
            node['title'] = node['id']
            x, y = self.pos[node['id']]
            node['x'], node['y'] = x, y
            node['fixed'] = False

        for edge in net.edges:
            iteration = None
            if self.worklist is not None:
                iteration = self.worklist.pair_round.get(frozenset((edge['from'], edge['to'])))
            edge['color'] = '#2ca02c' if iteration and iteration > 1 else '#1f77b4'  # green vs blue
            edge['width'] = 4
            conf = edge.get('conf')
            if conf is not None:
                edge['label'] = f'{conf:.2f}'
                edge['title'] = f'conf: {conf:.4f}' + (f' — round {iteration}' if iteration else '')

        for (img_a, img_b), conf in self.pending.items():
            if not self._known(img_a, img_b):
                continue
            net.add_edge(
                img_a, img_b,
                color='#ff7f0e', dashes=True, width=2,  # orange: queued on the worklist
                title='queued for this iteration',
            )

        for (img_a, img_b), conf in self.rejected.items():
            if not self._known(img_a, img_b):
                continue
            net.add_edge(
                img_a, img_b,
                color='red', dashes=True, width=1,
                label=f'{conf:.2f}' if conf is not None else None,
                title=f'rejected — sim: {conf:.4f}' if conf is not None else 'rejected',
            )


        net.set_options("""
        var options = {
          "physics": {
            "enabled": false,
            "stabilization": false
          },
          "interaction": {
            "dragNodes": true,
            "zoomView": true,
            "dragView": true
          }
        }
        """)
        net.write_html(self.save_to, notebook=False, open_browser=False)
