"""Standalone runnable demonstration of SONAPSE dynamic connectome & topology analytics."""

from __future__ import annotations

import json

from sonapse.connectome import SonapseConnectome
from sonapse.interfaces import ConnectomeProtocol, TopologyAnalyzerProtocol
from sonapse.topology import analyze_topology


def main() -> None:
    conn = SonapseConnectome(input_dim=8, hidden_dim=8, output_dim=5, neuron_model="lif", seed=11)
    assert isinstance(conn, ConnectomeProtocol)
    assert isinstance(analyze_topology, TopologyAnalyzerProtocol)

    outs = conn.step([0.8, -0.4, 0.7, 0.0, 0.1, 0.2, -0.2, 0.9])
    delta_norm = conn.update_plasticity(reward_prediction_error=1.1, learning_rate=0.08, chosen_action=1)
    grown = conn.grow_connections(max_new=3)
    topo = analyze_topology(conn)

    print("=== SONAPSE CONNECTOME EXAMPLE ===")
    print(
        json.dumps(
            {
                "neuron_model": conn.neuron_model,
                "outputs": {k: round(v, 4) for k, v in outs.items()},
                "weight_delta_norm": round(delta_norm, 5),
                "grown_synapses": grown,
                "topology": {
                    "density": round(topo["density"], 4),
                    "clustering_coefficient": round(topo["clustering_coefficient"], 4),
                    "modularity": round(topo["modularity"], 4),
                    "redundant_pathways": topo["redundant_pathways"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
