"""Formal typed interface protocols for the GENESIS developmental and evolutionary engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class GenesisEngineProtocol(Protocol):
    """Contract for organism creation, 8-stage progression, selection, recombination, and lineage tracking."""

    generation_counter: int
    lineage: Dict[str, Dict[str, Any]]

    def create_organism(
        self,
        organism_id: Optional[str] = None,
        generation: int = 0,
        parent_ids: Optional[List[str]] = None,
        neuron_model: str = "rate_coded",
        learning_rate: float = 0.08,
        reg_lambda: float = 0.002,
        seed: Optional[int] = None,
        use_deep_rl: bool = False,
    ) -> Any: ...
    def advance_stage(self, organism: Any, target_stage: Optional[str] = None) -> str: ...
    def recombine(
        self,
        parent_a: Any,
        parent_b: Any,
        generation: int,
        crossover_rate: float = 0.65,
    ) -> Any: ...
    def mutate(self, organism: Any, mutation_rate: float = 0.15) -> Any: ...
    def evolve_generation(
        self,
        ecosystem: Any,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.65,
        elite_fraction: float = 0.25,
        target_population: Optional[int] = None,
    ) -> Dict[str, Any]: ...
    def compare_checkpoints(
        self,
        organism: Any,
        idx_a: int = 0,
        idx_b: int = -1,
    ) -> Dict[str, Any]: ...
