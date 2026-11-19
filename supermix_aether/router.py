"""SUPERMIX-AETHER flexible local model and specialist agent orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
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
        self.register_model(
            SpecialistModel(
                model_id="local_active_inference_epistemic",
                task_types=["decision_arbitration", "navigation", "foraging"],
                backend_type="local",
                priority=8,
                handler=self._local_active_inference_handler,
                metadata={"description": "Active Inference Expected Free Energy epistemic navigation specialist"},
            )
        )

    @staticmethod
    def _local_active_inference_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        features = list(payload.get("features", [0.0] * 8))
        while len(features) < 8:
            features.append(0.0)
        res_dx, res_dy, res_prox = float(features[0]), float(features[1]), float(features[2])
        haz_dx, haz_dy, haz_prox = float(features[3]), float(features[4]), float(features[5])
        energy_norm = float(features[7])
        epistemic_bonus = 0.25 * (1.0 - res_prox)
        scores = [
            -0.55 * res_dy + 0.45 * haz_dy + epistemic_bonus,
            0.55 * res_dx - 0.45 * haz_dx + epistemic_bonus,
            0.55 * res_dy - 0.45 * haz_dy + epistemic_bonus,
            -0.55 * res_dx + 0.45 * haz_dx + epistemic_bonus,
            0.20 * (1.0 - haz_prox) * (1.0 - energy_norm),
        ]
        return {"action_scores": [round(float(s), 5) for s in scores], "confidence": 0.82}

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

    def tether_arbitrate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Supermix-Tether & FAULTLINE counterfactual intervention control and ensemble correlation audit."""
        base_features = list(payload.get("features", [0.4, -0.2, 0.65, -0.3, 0.25, 0.35, 0.1, 0.75]))
        while len(base_features) < 8:
            base_features.append(0.0)

        relevant_perturbed = list(base_features)
        relevant_perturbed[0] = -relevant_perturbed[0] if abs(relevant_perturbed[0]) > 0.05 else 0.6
        relevant_perturbed[1] = -relevant_perturbed[1] if abs(relevant_perturbed[1]) > 0.05 else -0.6

        irrelevant_perturbed = list(base_features)
        irrelevant_perturbed[6] = min(1.0, irrelevant_perturbed[6] + 0.45)

        decision_models = [
            m for m in self.models.values()
            if "decision_arbitration" in m.task_types and m.healthy and m.handler is not None
            and (m.backend_type == "local" or self.allow_external_apis)
        ]

        donor_reports: List[Dict[str, Any]] = []
        weighted_consensus = [0.0] * 5
        total_weight = 0.0
        top_picks: List[int] = []

        for m in decision_models:
            base_out = m.handler({"features": base_features})
            rel_out = m.handler({"features": relevant_perturbed})
            irr_out = m.handler({"features": irrelevant_perturbed})

            s_base = list(base_out.get("action_scores", [0.0] * 5))[:5]
            s_rel = list(rel_out.get("action_scores", [0.0] * 5))[:5]
            s_irr = list(irr_out.get("action_scores", [0.0] * 5))[:5]

            rel_effect = math.sqrt(sum((a - b) ** 2 for a, b in zip(s_base, s_rel)) / 5.0)
            irr_leakage = math.sqrt(sum((a - b) ** 2 for a, b in zip(s_base, s_irr)) / 5.0)
            tether_score = max(0.05, min(1.0, (0.35 + rel_effect) / (0.35 + rel_effect + 1.5 * irr_leakage)))

            best_act = int(max(range(len(s_base)), key=lambda i: s_base[i]))
            top_picks.append(best_act)
            weight = tether_score * float(base_out.get("confidence", 0.75))
            total_weight += weight
            for i in range(5):
                weighted_consensus[i] += weight * float(s_base[i])

            donor_reports.append(
                {
                    "model_id": m.model_id,
                    "preferred_action": best_act,
                    "relevant_intervention_effect": round(float(rel_effect), 4),
                    "irrelevant_leakage": round(float(irr_leakage), 4),
                    "tether_fidelity": round(float(tether_score), 4),
                    "action_scores": [round(float(x), 4) for x in s_base],
                }
            )

        if total_weight > 1e-9:
            weighted_consensus = [round(v / total_weight, 5) for v in weighted_consensus]
        chosen_action = int(max(range(5), key=lambda i: weighted_consensus[i]))

        # FAULTLINE higher-order correlated disagreement/failure metric
        unique_picks = len(set(top_picks)) if top_picks else 1
        disagreement_rate = (unique_picks - 1) / float(max(1, len(top_picks)))
        mean_leakage = sum(d["irrelevant_leakage"] for d in donor_reports) / float(max(1, len(donor_reports)))
        faultline_risk = round(min(1.0, 0.45 * disagreement_rate + 0.55 * mean_leakage), 4)

        return {
            "protocol": "Supermix-Tether + FAULTLINE Counterfactual Audit",
            "donors_evaluated": len(donor_reports),
            "consensus_action": chosen_action,
            "consensus_scores": weighted_consensus,
            "faultline_correlated_risk": faultline_risk,
            "ensemble_status": "ROBUST" if faultline_risk < 0.35 else "ELEVATED_CORRELATION",
            "donor_reports": donor_reports,
        }

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
