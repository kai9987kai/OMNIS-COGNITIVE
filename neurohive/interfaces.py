"""Formal typed interface protocols for the NEUROHIVE artificial-life ecosystem."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class OrganismProtocol(Protocol):
    """Contract for an embodied artificial cognitive organism."""

    organism_id: str
    energy: float
    alive: bool
    fitness: float

    def step(
        self,
        world: Any,
        tick: int,
        fitness_weights: Optional[Dict[str, float]] = None,
        enable_plasticity: bool = True,
        enable_memory_replay: bool = True,
    ) -> Dict[str, Any]: ...
    def compute_fitness(self, fitness_weights: Optional[Dict[str, float]] = None) -> float: ...
    def record_checkpoint(self, label: str) -> None: ...
    def get_summary(self) -> Dict[str, Any]: ...
    def to_dict(self) -> Dict[str, Any]: ...


@runtime_checkable
class EcosystemProtocol(Protocol):
    """Contract for a population-level multi-agent ecosystem simulation."""

    organisms: Dict[str, Any]
    tick: int

    def add_organism(self, organism: Any) -> bool: ...
    def remove_dead_organisms(self) -> List[str]: ...
    def step(self, fitness_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]: ...
    def mean_fitness(self) -> float: ...
    def get_population_summary(self) -> Dict[str, Any]: ...
