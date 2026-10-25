import heapq
from collections import Counter

import networkx as nx


def pair_clusters(graph, seed=0):
    """
    Splits the pair graph into tightly connected clusters (Louvain
    communities), the same groups drawn by the pair graph viewer. Images
    without any pair are left out. Returns a list of image sets.
    """
    isolated = set(nx.isolates(graph))
    return [set(c) for c in nx.community.louvain_communities(graph.subgraph(set(graph) - isolated), seed=seed)]


def join_pairs(graph, clusters, sim_table, tried=(), min_links=3, per_link=5):
    """
    Proposes the new pairs to match so that the clusters join, without
    trying every missing pair.

    Two clusters count as joined when at least `min_links` confirmed pairs of
    `graph` connect them, and joined clusters merge into groups. Two clusters
    of different groups are as close as their most similar pair in
    `sim_table` ({(img0, img1): sim}, the global similarity); a maximum
    spanning tree over the groups picks the fewest links that connect them
    all, each between the two closest clusters of the groups it joins, and
    for each link the `per_link` most similar pairs of those two clusters are
    returned. Pairs in `graph` or in `tried` are never proposed again.

    Call it again after matching the returned pairs, with the updated graph
    and the same clusters: it returns [] once all the clusters are joined or
    no candidate pair is left.
    """
    cluster_of = {img: i for i, cluster in enumerate(clusters) for img in cluster}
    links = Counter(frozenset((cluster_of[a], cluster_of[b])) for a, b in graph.edges()
                    if a in cluster_of and b in cluster_of and cluster_of[a] != cluster_of[b])

    joined = nx.Graph()
    joined.add_nodes_from(range(len(clusters)))
    joined.add_edges_from(tuple(key) for key, n in links.items() if n >= min_links)
    group_of = {i: g for g, component in enumerate(nx.connected_components(joined)) for i in component}

    candidates = {}
    for (a, b), sim in sim_table.items():
        if a not in cluster_of or b not in cluster_of:
            continue
        ca, cb = cluster_of[a], cluster_of[b]
        if group_of[ca] == group_of[cb] or graph.has_edge(a, b) or frozenset((a, b)) in tried:
            continue
        best = candidates.setdefault(frozenset((ca, cb)), [])
        heapq.heappush(best, (sim, a, b))
        if len(best) > per_link:
            heapq.heappop(best)

    groups = nx.Graph()
    for link, best in candidates.items():
        ga, gb = (group_of[c] for c in link)
        sim = max(best)[0]
        if not groups.has_edge(ga, gb) or groups[ga][gb]['weight'] < sim:
            groups.add_edge(ga, gb, weight=sim, link=link)

    new_pairs = []
    for _, _, link in nx.maximum_spanning_tree(groups).edges(data='link'):
        new_pairs += [(a, b) for _, a, b in sorted(candidates[link], reverse=True)]

    print(f"join_pairs: {len(clusters)} clusters in {nx.number_connected_components(joined)} joined group(s), "
          f"{len(new_pairs)} new pairs proposed")
    return new_pairs
