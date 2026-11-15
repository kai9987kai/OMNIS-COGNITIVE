"""Formal typed interface protocols for the OMNIS Cognition Pipeline."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from diamond.environment import DiamondWorld
from mnemorph.memory import MnemorphMemory
from sonapse.connectome import SonapseConnectome


@runtime_checkable
class CognitivePipelineProtocol(Protocol):
    """Contract for executing a single organism cognitive-behavioral cycle."""

    def step(
        self,
        organism_id: str,
        tick: int,
        world: DiamondWorld,
        connectome: SonapseConnectome,
        memory: MnemorphMemory,
        learner: Any,
        world_model: Any,
        learning_rate: float = 0.08,
        reg_lambda: float = 0.002,
        plasticity_mode: str = "three_factor",
        enable_plasticity: bool = True,
        enable_structural_adaptation: bool = True,
        enable_memory_ops: bool = True,
        ewc_lambda: float = 0.0,
        neural_weight: float = 0.35,
    ) -> Dict[str, Any]: ...
