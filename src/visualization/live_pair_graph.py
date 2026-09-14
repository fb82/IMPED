import os
import time
import webbrowser

import networkx as nx
from pyvis.network import Network

MAX_VISUALIZED_CANDIDATES = 5000


class live_pair_graph:
    """
    Interactive, incrementally-updated visualisation of the pair graph built
    by a transitive run_pairs() pipeline, using pyvis (vis.js):
    draggable/zoomable nodes, hover tooltips, click-to-highlight neighbours.

    Two ways to drive it:

    * as callbacks - pass `on_pair` / `on_candidates` / `on_round` to
      run_pairs(); or
    * as a pass-through pipeline module - drop the instance straight into the
      pipeline list. Its run() records every pair the pipeline processes, and
      when it is given a `worklist` (the transitive_module instance) it also
      redraws the pairs still on that module's worklist after each pair, so
      the picture tracks the modified list at every transitive iteration.
    """
    def __init__(self, imgs, save_to='live_pair_graph.html', refresh_seconds=2, open_browser=True, worklist=None):
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
        self.refresh_seconds = refresh_seconds
        self.stopped = False
        self.first_round_done = False
        self.worklist = worklist
        self._last_redraw_time = 0.0

        self.pos = {n: (x * 1000, y * 1000) for n, (x, y) in nx.spring_layout(self.graph).items()}

        self._redraw()
        if open_browser:
            webbrowser.open('file://' + os.path.abspath(save_to))

    def get_id(self):
        return self.id_string

    def _known(self, img_a, img_b):
        return img_a in self.graph and img_b in self.graph

    def add_pair(self, img_a, img_b, conf=None, transitive=False):
        if not self._known(img_a, img_b):
            return
        self.graph.add_edge(img_a, img_b, conf=conf, transitive=transitive)
        self.rejected.pop((min(img_a, img_b), max(img_a, img_b)), None)
        self.pending.pop((min(img_a, img_b), max(img_a, img_b)), None)
        seed = {n: (x / 1000, y / 1000) for n, (x, y) in self.pos.items()}
        self.pos = {n: (x * 1000, y * 1000) for n, (x, y) in nx.spring_layout(self.graph, pos=seed).items()}
        self._redraw()

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
        self._redraw()

    def sync_worklist(self):
        """Refresh the 'pending' edges from the worklist module's current
        list of pairs still to do (everything not already confirmed or
        rejected), so the drawing reflects the list after the last
        transitive_module.finalize()."""
        if self.worklist is None:
            return
        pending = {}
        for a, b in getattr(self.worklist, '_pairs', []):
            if not self._known(a, b):
                continue
            key = (min(a, b), max(a, b))
            if self.graph.has_edge(a, b) or key in self.rejected:
                continue
            pending[key] = None
        self.pending = pending
        self.first_round_done = getattr(self.worklist, '_round', 0) > 0
        self._redraw()

    def on_pair(self, pair, pipe_data):
        img_a, img_b = pair
        conf = pipe_data.get('pair_sim')

        transitive = self.first_round_done
        keep = pipe_data.get('pair_conf', True)
        if self.worklist is not None:
            transitive = getattr(self.worklist, '_round', 0) > 0
            th = self.worklist.args.get('threshold') if hasattr(self.worklist, 'args') else None
            if 'pair_conf' not in pipe_data and conf is not None and th is not None:
                keep = conf > th

        if keep:
            self.add_pair(img_a, img_b, conf=conf, transitive=transitive)
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
        self.first_round_done = True
        self.sync_worklist()

    def run(self, **pipe_data):
        self.on_pair((pipe_data['img'][0], pipe_data['img'][1]), pipe_data)
        self.sync_worklist()
        return {}

    def finalize(self):
        self.stop()

    def stop(self):
        self.stopped = True
        self._redraw(force=True)

    def _redraw(self, force=False):
        if not force and time.time() - self._last_redraw_time < self.refresh_seconds:
            return
        self._last_redraw_time = time.time()
        self._do_redraw()

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
            edge['color'] = '#2ca02c' if edge.get('transitive') else '#1f77b4'  # green vs blue
            edge['width'] = 4
            conf = edge.get('conf')
            if conf is not None:
                edge['label'] = f'{conf:.2f}'
                edge['title'] = f'conf: {conf:.4f}'

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

        if not self.stopped:
            with open(self.save_to, 'r') as f:
                html = f.read()
            refresh_tag = f'<meta http-equiv="refresh" content="{self.refresh_seconds}">'
            html = html.replace('<head>', f'<head>\n    {refresh_tag}', 1)
            with open(self.save_to, 'w') as f:
                f.write(html)
