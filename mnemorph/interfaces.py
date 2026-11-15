"""Formal typed interface protocols for the MNEMORPH adaptive memory subsystem."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from mnemorph.memory import MemoryClass, MemoryRecord


@runtime_checkable
class MemoryHierarchyProtocol(Protocol):
    """Contract for multi-level adaptive memory encoding, retrieval, consolidation, replay, and forgetting."""

    def encode(
        self,
        memory_class: MemoryClass,
        content: Dict[str, Any],
        embedding: List[float],
        timestamp: float = 0.0,
        reward_assoc: float = 0.0,
        novelty: Optional[float] = None,
        significance: float = 0.5,
        usefulness: float = 0.3,
        provenance: str = "experience",
    ) -> MemoryRecord: ...
    def retrieve(
        self,
        query_embedding: List[float],
        memory_class: Optional[MemoryClass] = None,
        top_k: int = 5,
        current_time: Optional[float] = None,
    ) -> List[MemoryRecord]: ...
    def consolidate(self, importance_threshold: float = 0.52) -> int: ...
    def replay(self, batch_size: int = 8) -> Tuple[List[MemoryRecord], float]: ...
    def decay_and_forget(self, current_time: float, decay_scale: float = 1.0) -> int: ...
    def get_observatory_snapshot(self) -> Dict[str, Any]: ...
