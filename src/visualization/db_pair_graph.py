import os
import webbrowser

import networkx as nx

import colmap_db.database as coldb
from colmap_fun import coldb_ext

from .live_pair_graph import live_pair_graph


def read_db_pairs(db):
    if not os.path.isfile(db):
        raise FileNotFoundError(db)

    con = coldb_ext(db)
    names = dict(con.get_images())
    rows = {}
    for i, table in enumerate(('matches', 'two_view_geometries')):
        for pair_id, n in con.execute(f"SELECT pair_id, rows FROM {table}"):
            rows.setdefault(pair_id, [None, None])[i] = n
    con.close()

    pairs = {}
    for pair_id, (n_matches, n_inliers) in rows.items():
        id_a, id_b = (int(i) for i in coldb.pair_id_to_image_ids(pair_id))
        if id_a in names and id_b in names:
            pairs[(names[id_a], names[id_b])] = (n_matches, n_inliers)

    return list(names.values()), pairs


def db_pair_graph(db, save_to=None, open_browser=True, min_inliers=1, show_rejected=True, model=None):
    """
    Static counterpart of live_pair_graph: draws the pair graph stored in a
    COLMAP database. Pairs with at least `min_inliers` verified matches are
    confirmed edges labelled with the inlier count (the raw match count when
    the database has no two_view_geometries row), the others are rejected.
    With `model`, the page can also group the nodes by the Louvain
    communities of that reconstruction (see live_pair_graph).
    Returns the networkx graph of the confirmed pairs.
    """
    if save_to is None:
        save_to = os.path.splitext(db)[0] + '_pair_graph.html'

    imgs, pairs = read_db_pairs(db)
    verified = any(n_inliers is not None for _, n_inliers in pairs.values())

    live = live_pair_graph(imgs, save_to=save_to, open_browser=False, model=model)
    for (img_a, img_b), (n_matches, n_inliers) in pairs.items():
        n = (n_inliers if verified else n_matches) or 0
        if n >= min_inliers:
            live.graph.add_edge(img_a, img_b, n_matches=n_matches, n_inliers=n_inliers,
                                label=str(n), title=f'inliers: {n_inliers} — matches: {n_matches}')
        elif show_rejected:
            live.rejected[(img_a, img_b)] = None
    live._recompute_layout()
    live.stop()

    print(f"db_pair_graph: {live.graph.number_of_edges()} pairs ({len(live.rejected)} rejected) "
          f"over {live.graph.number_of_nodes()} images, "
          f"{nx.number_connected_components(live.graph)} connected components -> {save_to}")

    if open_browser:
        webbrowser.open('file://' + os.path.abspath(save_to))

    return live.graph
