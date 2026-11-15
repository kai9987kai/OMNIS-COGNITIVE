"""Formal typed interface protocols for SUPERMIX-AETHER model and agent orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class SpecialistModelCallableProtocol(Protocol):
    """Contract for synchronous specialist model adapters (Torch neural & external HTTP)."""

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...


@runtime_checkable
class OrchestrationRouterProtocol(Protocol):
    """Contract for model registration, agent registration, task routing, and status telemetry."""

    def register_model(self, model: Any) -> None: ...
    def register_agent(self, agent_id: str, role: str, capabilities: List[str]) -> None: ...
    def route_task(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]: ...
    def get_status(self) -> Dict[str, Any]: ...
