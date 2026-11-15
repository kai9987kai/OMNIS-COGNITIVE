"""Graph topology analysis for SONAPSE connectomes."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Set

import networkx as nx

from sonapse.connectome import SonapseConnectome


def analyze_topology(connectome: SonapseConnectome) -> Dict[str, Any]:
    """Compute structural & topological metrics for a SonapseConnectome G(t)."""
    active_nodes = [nid for nid, n in connectome.neurons.items() if n.active]
    n_count = len(active_nodes)
    active_syns = connectome.active_synapses()
    e_count = len(active_syns)

    if n_count == 0:
        return {
            "num_neurons": 0,
            "num_synapses": 0,
            "density": 0.0,
            "mean_degree": 0.0,
            "degree_distribution": {},
            "avg_path_length": 0.0,
            "clustering_coefficient": 0.0,
            "weakly_connected_components": 0,
            "strongly_connected_components": 0,
            "modularity": 0.0,
            "isolated_neurons": [],
            "redundant_pathways": 0,
            "top_centrality": [],
        }

    # Exact directed density D = |E| / (N(N-1)) without self-connections
    density = (e_count / float(n_count * (n_count - 1))) if n_count > 1 else 0.0

    g_dir = nx.DiGraph()
    for nid in active_nodes:
        g_dir.add_node(nid, cluster=connectome.neurons[nid].cluster_id)
    for syn in active_syns:
        g_dir.add_edge(syn.pre_id, syn.post_id, weight=abs(syn.weight))

    g_undir = g_dir.to_undirected()

    in_deg = dict(g_dir.in_degree())
    out_deg = dict(g_dir.out_degree())
    tot_deg = {nid: in_deg[nid] + out_deg[nid] for nid in active_nodes}
    mean_degree = sum(tot_deg.values()) / float(n_count)
    isolated = [nid for nid, d in tot_deg.items() if d == 0]

    # Degree distribution histogram
    deg_hist: Dict[int, int] = defaultdict(int)
    for d in tot_deg.values():
        deg_hist[int(d)] += 1

    # Average directed shortest path length over reachable pairs
    total_dist = 0
    reachable_pairs = 0
    for src in active_nodes:
        dists = nx.single_source_shortest_path_length(g_dir, src)
        for dst, d in dists.items():
            if dst != src:
                total_dist += d
                reachable_pairs += 1
    avg_path_length = (total_dist / float(reachable_pairs)) if reachable_pairs > 0 else 0.0

    clustering_coeff = float(nx.average_clustering(g_undir)) if n_count >= 3 and e_count > 0 else 0.0
    wcc = nx.number_weakly_connected_components(g_dir)
    scc = nx.number_strongly_connected_components(g_dir)

    # Modularity Q based on neuron cluster_id partitions
    communities_map: Dict[int, Set[str]] = defaultdict(set)
    for nid in active_nodes:
        communities_map[connectome.neurons[nid].cluster_id].add(nid)
    communities = [c for c in communities_map.values() if c]
    if g_undir.number_of_edges() > 0 and len(communities) > 1:
        modularity = float(nx.algorithms.community.modularity(g_undir, communities))
    else:
        modularity = 0.0

    # Centrality (degree & betweenness)
    betw = nx.betweenness_centrality(g_dir, normalized=True) if n_count > 2 else {nid: 0.0 for nid in active_nodes}
    deg_cent = nx.degree_centrality(g_dir) if n_count > 1 else {nid: 0.0 for nid in active_nodes}
    top_centrality = sorted(
        [
            {
                "neuron_id": nid,
                "degree_centrality": round(float(deg_cent[nid]), 4),
                "betweenness_centrality": round(float(betw[nid]), 4),
            }
            for nid in active_nodes
        ],
        key=lambda x: (x["betweenness_centrality"], x["degree_centrality"]),
        reverse=True,
    )[:5]

    # Redundant pathways: (u, v) pairs connected by >= 2 distinct directed paths of length <= 3
    adj: Dict[str, List[str]] = defaultdict(list)
    for syn in active_syns:
        adj[syn.pre_id].append(syn.post_id)

    redundant_count = 0
    for u in active_nodes:
        target_paths: Dict[str, int] = defaultdict(int)
        queue: deque = deque([(u, 0, {u})])
        while queue:
            curr, depth, visited = queue.popleft()
            if depth >= 3:
                continue
            for nxt in adj.get(curr, []):
                if nxt in visited:
                    continue
                target_paths[nxt] += 1
                queue.append((nxt, depth + 1, visited | {nxt}))
        for count in target_paths.values():
            if count >= 2:
                redundant_count += 1

    return {
        "num_neurons": n_count,
        "num_synapses": e_count,
        "density": float(density),
        "mean_degree": float(mean_degree),
        "degree_distribution": {str(k): v for k, v in sorted(deg_hist.items())},
        "avg_path_length": float(avg_path_length),
        "clustering_coefficient": float(clustering_coeff),
        "weakly_connected_components": int(wcc),
        "strongly_connected_components": int(scc),
        "modularity": float(modularity),
        "isolated_neurons": isolated,
        "redundant_pathways": int(redundant_count),
        "top_centrality": top_centrality,
    }
