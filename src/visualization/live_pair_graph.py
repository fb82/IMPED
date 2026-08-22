import os
import webbrowser

import networkx as nx
from pyvis.network import Network


class live_pair_graph:
    """
    Interactive, incrementally-updated visualisation of the pair graph built
    by core.run_transitive_pairs, using pyvis (vis.js): draggable/zoomable
    nodes, hover tooltips, click-to-highlight neighbours.
    """
    def __init__(self, imgs, save_to='live_pair_graph.html', refresh_seconds=2, open_browser=True):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(imgs)
        self.rejected = {}
        self.save_to = save_to
        self.refresh_seconds = refresh_seconds
        self.stopped = False
        self.first_round_done = False

        self.pos = {n: (x * 1000, y * 1000) for n, (x, y) in nx.spring_layout(self.graph).items()}

        self._redraw()
        if open_browser:
            webbrowser.open('file://' + os.path.abspath(save_to))

    def add_pair(self, img_a, img_b, conf=None, transitive=False):
        self.graph.add_edge(img_a, img_b, conf=conf, transitive=transitive)
        self.rejected.pop((min(img_a, img_b), max(img_a, img_b)), None)
        seed = {n: (x / 1000, y / 1000) for n, (x, y) in self.pos.items()}
        self.pos = {n: (x * 1000, y * 1000) for n, (x, y) in nx.spring_layout(self.graph, pos=seed).items()}
        self._redraw()

    def add_rejected_pair(self, img_a, img_b, conf=None):
        key = (min(img_a, img_b), max(img_a, img_b))
        if self.graph.has_edge(img_a, img_b):
            return  
        if key in self.rejected and self.rejected[key] == conf:
            return  
        self.rejected[key] = conf
        self._redraw()

    def on_pair(self, pair, pipe_data):
        img_a, img_b = pair
        conf = pipe_data.get('pair_sim')
        if pipe_data.get('pair_conf', True):
            self.add_pair(img_a, img_b, conf=conf, transitive=self.first_round_done)
        else:
            self.add_rejected_pair(img_a, img_b, conf=conf)

    def on_candidates(self, scored, threshold):
        for sim, pair in scored:
            self.add_rejected_pair(*pair, conf=sim)

    def on_round(self, graph, n_round, n_new):
        self.first_round_done = True

    def stop(self):
        self.stopped = True
        self._redraw()

    def _redraw(self):
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

        for (img_a, img_b), conf in self.rejected.items():
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
