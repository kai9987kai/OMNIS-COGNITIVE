"""SUPERMIX-AETHER flexible local model and specialist agent orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional

from core.events import EventBus


class TorchNeuralSpecialist:
    """Local PyTorch 2-layer neural specialist for policy scoring and confidence estimation."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 5, seed: int = 42) -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        gen = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        with torch.no_grad():
            for param in self.net.parameters():
                param.copy_(torch.randn(param.shape, generator=gen) * 0.25)
        self.net.eval()

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        features = list(payload.get("features", [0.0] * 8))
        while len(features) < 8:
            features.append(0.0)
        x = self.torch.tensor([features[:8]], dtype=self.torch.float32)
        with self.torch.no_grad():
            logits = self.net(x)[0]
            probs = self.torch.softmax(logits, dim=-1)
            scores = [round(float(v), 5) for v in logits.tolist()]
            conf = round(float(probs.max().item()), 4)
        return {
            "action_scores": scores,
            "confidence": conf,
            "backend": "pytorch_mlp",
        }


class ExternalHTTPModelAdapter:
    """Optional HTTP endpoint adapter for external model inference (strictly disabled by default)."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        timeout_s: float = 5.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout_s = float(timeout_s)

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(self.endpoint_url, json=payload, headers=headers)
            resp.raise_for_status()
            return dict(resp.json())


@dataclass
class SpecialistModel:
    """Registered local or optional external specialist backend."""

    model_id: str
    task_types: List[str]
    backend_type: str = "local"  # local | external_optional
    handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    priority: int = 1
    healthy: bool = True
    calls: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SupermixAether:
    """Routes cognitive and research subtasks across specialist local models with optional external hooks."""

    def __init__(self, allow_external_apis: bool = False, event_bus: Optional[EventBus] = None) -> None:
        self.allow_external_apis = bool(allow_external_apis)
        self.event_bus = event_bus
        self.models: Dict[str, SpecialistModel] = {}
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.approved_knowledge_pool: List[Dict[str, Any]] = []
        self._register_default_local_specialists()

    def _register_default_local_specialists(self) -> None:
        self.register_model(
            SpecialistModel(
                model_id="local_foraging_heuristic",
                task_types=["navigation", "foraging", "decision_arbitration"],
                backend_type="local",
                priority=10,
                handler=self._local_foraging_handler,
                metadata={"description": "Local sensory-gradient navigation specialist"},
            )
        )
        self.register_model(
            SpecialistModel(
                model_id="local_topology_critic",
                task_types=["topology_audit", "plasticity_advice"],
                backend_type="local",
                priority=8,
                handler=self._local_topology_critic_handler,
                metadata={"description": "Local graph density & redundancy advisor"},
            )
        )
        self.register_model(
            SpecialistModel(
                model_id="local_memory_synthesizer",
                task_types=["memory_synthesis", "conflict_resolution"],
                backend_type="local",
                priority=7,
                handler=self._local_memory_synthesizer_handler,
                metadata={"description": "Local memory conflict & prototype synthesizer"},
            )
        )
        self.register_model(
            SpecialistModel(
                model_id="local_torch_policy_evaluator",
                task_types=["neural_policy_evaluation", "decision_arbitration", "navigation"],
                backend_type="local",
                priority=9,
                handler=TorchNeuralSpecialist(seed=42),
                metadata={"description": "Local PyTorch MLP specialist policy scorer"},
            )
        )

    @staticmethod
    def _local_foraging_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        features = payload.get("features", [0.0] * 8)
        res_dx = float(features[0]) if len(features) > 0 else 0.0
        res_dy = float(features[1]) if len(features) > 1 else 0.0
        haz_prox = float(features[5]) if len(features) > 5 else 0.0
        scores = [0.0, 0.0, 0.0, 0.0, 0.1]
        if abs(res_dx) >= abs(res_dy):
            scores[1 if res_dx > 0 else 3] += 0.8
        else:
            scores[2 if res_dy > 0 else 0] += 0.8
        if haz_prox > 0.8:
            scores[4] -= 0.4
        return {"action_scores": scores, "confidence": 0.85}

    @staticmethod
    def _local_topology_critic_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        density = float(payload.get("density", 0.15))
        isolated = len(payload.get("isolated_neurons", []))
        recommendation = "stable"
        if isolated > 0 or density < 0.08:
            recommendation = "grow_connections"
        elif density > 0.30:
            recommendation = "prune_weak_connections"
        return {"recommendation": recommendation, "density": density, "isolated_count": isolated}

    @staticmethod
    def _local_memory_synthesizer_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        conflicts = payload.get("conflicts", [])
        return {
            "conflict_count": len(conflicts),
            "resolution_policy": "prefer_higher_confidence_and_recency" if conflicts else "none_needed",
        }

    def register_model(self, model: SpecialistModel) -> None:
        if model.backend_type == "external_optional" and not self.allow_external_apis:
            model.healthy = False
        self.models[model.model_id] = model

    def register_agent(self, agent_id: str, role: str, capabilities: List[str]) -> None:
        self.registered_agents[agent_id] = {
            "agent_id": agent_id,
            "role": role,
            "capabilities": capabilities,
        }

    def share_approved_knowledge(self, source_agent: str, knowledge: Dict[str, Any]) -> None:
        self.approved_knowledge_pool.append({"source": source_agent, "knowledge": knowledge})
        if len(self.approved_knowledge_pool) > 100:
            self.approved_knowledge_pool = self.approved_knowledge_pool[-100:]

    def route_task(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Route a task to highest-priority healthy model with automatic fallback on failure."""
        candidates = [
            m for m in self.models.values()
            if task_type in m.task_types and m.healthy
            and (m.backend_type == "local" or self.allow_external_apis)
        ]
        candidates.sort(key=lambda m: m.priority, reverse=True)

        for model in candidates:
            t0 = time.perf_counter()
            model.calls += 1
            try:
                if model.handler is None:
                    raise RuntimeError("No callable handler configured")
                result = model.handler(payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                model.total_latency_ms += elapsed_ms
                return {
                    "status": "ok",
                    "model_id": model.model_id,
                    "backend_type": model.backend_type,
                    "latency_ms": round(elapsed_ms, 4),
                    "result": result,
                }
            except Exception as exc:
                model.failures += 1
                model.healthy = False
                if self.event_bus:
                    self.event_bus.emit(
                        "supermix.model_failure",
                        source="supermix_aether",
                        payload={"model_id": model.model_id, "error": str(exc)},
                    )

        return {"status": "fallback_empty", "model_id": None, "result": {}}

    def get_status(self) -> Dict[str, Any]:
        return {
            "allow_external_apis": self.allow_external_apis,
            "registered_agents": len(self.registered_agents),
            "shared_knowledge_items": len(self.approved_knowledge_pool),
            "models": [
                {
                    "model_id": m.model_id,
                    "task_types": m.task_types,
                    "backend_type": m.backend_type,
                    "healthy": m.healthy,
                    "calls": m.calls,
                    "failures": m.failures,
                    "avg_latency_ms": round(m.total_latency_ms / max(1, m.calls), 4),
                }
                for m in self.models.values()
            ],
        }
