"""Graph topology analysis for SONAPSE connectomes."""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any, Dict, List, Set

import networkx as nx

from sonapse.connectome import SonapseConnectome


def _normalized_distribution_entropy(weights: List[float]) -> float:
    total = sum(abs(w) for w in weights)
    if total <= 1e-12 or len(weights) <= 1:
        return 0.0
    probs = [abs(w) / total for w in weights if abs(w) > 1e-12]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs)
    return float(h / max(1e-9, math.log2(len(weights))))


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
            "effective_information_micro": 0.0,
            "effective_information_macro": 0.0,
            "causal_emergence_index": 0.0,
            "fractal_dimension_proxy": 0.0,
            "graft_synapses": 0,
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

    # Causal Emergence (GhostInTheMachine / Hoel et al. Effective Information macro vs micro)
    out_weights_by_node: Dict[str, List[float]] = defaultdict(list)
    macro_flow: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    graft_count = 0
    for syn in active_syns:
        out_weights_by_node[syn.pre_id].append(abs(syn.weight) + 1e-4)
        c_pre = int(connectome.neurons[syn.pre_id].cluster_id)
        c_post = int(connectome.neurons[syn.post_id].cluster_id)
        macro_flow[c_pre][c_post] += abs(syn.weight) * (1.25 if getattr(syn, "is_graft", False) else 1.0)
        if getattr(syn, "is_graft", False):
            graft_count += 1

    micro_determinisms: List[float] = []
    for nid in active_nodes:
        w_list = out_weights_by_node.get(nid, [])
        if w_list:
            # Determinism = 1 - H(W_i)/log2(N)
            micro_determinisms.append(1.0 - _normalized_distribution_entropy(w_list) * 0.65)
    ei_micro = (sum(micro_determinisms) / max(1, len(active_nodes))) * math.log2(max(2, n_count)) * 0.45

    macro_clusters = sorted({int(connectome.neurons[nid].cluster_id) for nid in active_nodes})
    macro_det_scores: List[float] = []
    for cid in macro_clusters:
        flows = list(macro_flow.get(cid, {}).values())
        if flows:
            macro_det_scores.append(1.0 - _normalized_distribution_entropy(flows) * 0.4)
    macro_scale = math.log2(max(2, len(macro_clusters) + 1))
    ei_macro = (
        (sum(macro_det_scores) / max(1, len(macro_clusters))) * macro_scale * 0.85
        + 0.12 * min(1.5, redundant_count / max(1.0, float(n_count)))
        + 0.06 * graft_count
    )
    causal_emergence = max(0.0, ei_macro - ei_micro * 0.72)

    # Fractal dimension proxy (SONAPSE-FRACTAL box occupancy across 2 spatial scales)
    boxes_coarse = set()
    boxes_fine = set()
    for nid in active_nodes:
        pos = connectome.neurons[nid].pos_3d or [0.0, 0.0, 0.0]
        boxes_coarse.add((int(math.floor(pos[0] / 2.0)), int(math.floor(pos[1] / 2.0)), int(math.floor(pos[2] / 2.0))))
        boxes_fine.add((int(math.floor(pos[0] / 1.0)), int(math.floor(pos[1] / 1.0)), int(math.floor(pos[2] / 1.0))))
    fractal_dim = math.log(max(2, len(boxes_fine)) / max(1.0, float(len(boxes_coarse)))) / math.log(2.0) + 1.0
    fractal_dim = max(1.0, min(2.95, fractal_dim))

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
        "effective_information_micro": round(float(ei_micro), 4),
        "effective_information_macro": round(float(ei_macro), 4),
        "causal_emergence_index": round(float(causal_emergence), 4),
        "fractal_dimension_proxy": round(float(fractal_dim), 4),
        "graft_synapses": int(graft_count),
    }
