"""Formal typed interface protocols for the SONAPSE neural connectome."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ConnectomeProtocol(Protocol):
    """Contract for dynamic directed neural connectomes G(t) = (V, E(t), W(t))."""

    def step(self, input_values: List[float] | Dict[str, float]) -> Dict[str, float]: ...
    def update_plasticity(
        self,
        reward_prediction_error: float,
        learning_rate: float = 0.08,
        reg_lambda: float = 0.002,
        mode: str = "three_factor",
        ewc_lambda: float = 0.0,
        chosen_action: Optional[int] = None,
    ) -> float: ...
    def prune_weak_connections(self, min_weight: float = 0.02) -> int: ...
    def grow_connections(self, max_new: int = 3, max_synapses: Optional[int] = None, max_density: Optional[float] = None) -> int: ...
    def to_visualization_graph(self) -> Dict[str, Any]: ...
    def active_neuron_count(self) -> int: ...


@runtime_checkable
class TopologyAnalyzerProtocol(Protocol):
    """Contract for graph topology analyzers returning density, clustering, modularity, and redundant paths."""

    def __call__(self, connectome: Any) -> Dict[str, Any]: ...
