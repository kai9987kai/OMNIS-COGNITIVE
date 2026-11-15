"""Formal typed interface protocols for MORPHEUS regenerative fault-tolerance computing."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class RegeneratorProtocol(Protocol):
    """Contract for structural fault injection, recovery measurement, and plasticity rehabilitation."""

    def detect_faults(self, organism: Any) -> Dict[str, Any]: ...
    def introduce_damage(
        self,
        organism: Any,
        neuron_lesion_ratio: float = 0.25,
        synapse_sever_ratio: float = 0.25,
    ) -> Dict[str, Any]: ...
    def evaluate_performance(
        self,
        organism: Any,
        reference_world_state: Dict[str, Any],
        steps: int = 12,
    ) -> Dict[str, float]: ...
    def regenerate_step(
        self,
        organism: Any,
        world: Optional[Any] = None,
        learning_steps: int = 8,
    ) -> Dict[str, Any]: ...
    def run_regeneration_experiment(
        self,
        organism: Any,
        world: Any,
        neuron_lesion_ratio: float = 0.30,
        synapse_sever_ratio: float = 0.30,
        recovery_rounds: int = 3,
    ) -> Dict[str, Any]: ...
