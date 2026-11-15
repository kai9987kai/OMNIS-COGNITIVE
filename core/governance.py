"""Scientific governance, resource quotas, sandboxing, and controlled self-improvement engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import copy
import hashlib
import hmac
import json
from typing import Any, Callable, Dict, List, Optional
import uuid

from core.events import EventBus
from core.persistence import OmnisDatabase
from core.schemas import SCHEMA_VERSION, ResourceQuota, SystemConfig, utc_now_iso


def verify_api_key(
    provided_or_config: Any,
    expected_or_provided: Optional[str] = None,
    required: bool = False,
) -> bool:
    """Constant-time verification of API key credentials when authentication is enabled."""
    if isinstance(provided_or_config, SystemConfig):
        cfg = provided_or_config
        provided_key = expected_or_provided
        expected_key = cfg.api_key
        required = bool(cfg.require_api_auth)
    else:
        provided_key = provided_or_config
        expected_key = expected_or_provided or ""
    if not required:
        return True
    if not provided_key or not expected_key:
        return False
    return hmac.compare_digest(str(provided_key), str(expected_key))


def run_with_timeout(fn: Callable[..., Any], timeout_sec: float, *args: Any, **kwargs: Any) -> Any:
    """Execute a bounded callable inside an isolated worker thread with timeout enforcement."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=max(0.05, float(timeout_sec)))
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Execution exceeded timeout limit of {timeout_sec}s") from exc


def generate_reproducibility_manifest(config: SystemConfig, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate SHA-256 integrity and reproducibility manifest for an experiment or run."""
    safe_cfg = config.model_dump(exclude={"api_key"})
    cfg_bytes = json.dumps(safe_cfg, sort_keys=True).encode("utf-8")
    cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "config_sha256": cfg_sha,
        "seed": config.seed,
        "config_snapshot": safe_cfg,
        "extra": extra or {},
    }


class GovernanceManager:
    """Enforces resource limits, input validation, audit logging, and safe bounds."""

    def __init__(
        self,
        quotas: ResourceQuota,
        db: Optional[OmnisDatabase] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.quotas = quotas
        self.db = db
        self.event_bus = event_bus

    def record_audit(self, actor: str, action: str, details: Dict[str, Any]) -> None:
        ts = utc_now_iso()
        if self.db is not None:
            self.db.append_audit_log(ts, actor, action, details)
        if self.event_bus is not None:
            self.event_bus.emit(
                "governance.audit",
                source="governance",
                payload={"actor": actor, "action": action, "details": details},
            )

    def check_population_cap(self, current_size: int, proposed_additions: int = 1) -> int:
        """Return bounded number of additions permitted without exceeding max_population_size."""
        available = max(0, self.quotas.max_population_size - current_size)
        allowed = min(max(0, proposed_additions), available)
        if allowed < proposed_additions:
            self.record_audit(
                "neurohive",
                "population_cap_enforced",
                {
                    "current_size": current_size,
                    "requested": proposed_additions,
                    "allowed": allowed,
                    "max_population_size": self.quotas.max_population_size,
                },
            )
        return allowed

    def check_connectome_limits(self, num_neurons: int, num_synapses: int) -> Dict[str, bool]:
        """Validate neuron, synapse, and density constraints."""
        density = 0.0
        if num_neurons > 1:
            density = num_synapses / (num_neurons * (num_neurons - 1))
        return {
            "neurons_ok": num_neurons <= self.quotas.max_neurons_per_organism,
            "synapses_ok": num_synapses <= self.quotas.max_synapses_per_organism,
            "density_ok": density <= self.quotas.max_connectome_density + 1e-9,
        }

    def validate_config_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate candidate parameter updates."""
        allowed_keys = {
            "learning_rate",
            "regularization_lambda",
            "discount_gamma",
            "mutation_rate",
            "crossover_rate",
            "elite_fraction",
            "plasticity_mode",
            "default_neuron_model",
        }
        cleaned: Dict[str, Any] = {}
        for k, v in patch.items():
            if k not in allowed_keys:
                raise ValueError(f"Unsupported or restricted config parameter: {k}")
            if k == "learning_rate":
                val = float(v)
                if not (0.001 <= val <= 0.5):
                    raise ValueError("learning_rate must be in [0.001, 0.5]")
                cleaned[k] = val
            elif k == "regularization_lambda":
                val = float(v)
                if not (0.0 <= val <= 0.2):
                    raise ValueError("regularization_lambda must be in [0.0, 0.2]")
                cleaned[k] = val
            elif k == "discount_gamma":
                val = float(v)
                if not (0.1 <= val <= 0.999):
                    raise ValueError("discount_gamma must be in [0.1, 0.999]")
                cleaned[k] = val
            elif k in ("mutation_rate", "crossover_rate", "elite_fraction"):
                val = float(v)
                if not (0.01 <= val <= 1.0):
                    raise ValueError(f"{k} must be in [0.01, 1.0]")
                cleaned[k] = val
            elif k == "plasticity_mode":
                if v not in ("three_factor", "hebbian", "hybrid"):
                    raise ValueError(f"Invalid plasticity_mode: {v}")
                cleaned[k] = v
            elif k == "default_neuron_model":
                if v not in ("conventional", "rate_coded", "threshold", "lif"):
                    raise ValueError(f"Invalid default_neuron_model: {v}")
                cleaned[k] = v
        return cleaned


class SelfImprovementEngine:
    """Controlled self-improvement pipeline with sandboxed evaluation, benchmark gates, rollback, and approval."""

    def __init__(self, governance: GovernanceManager, base_config: SystemConfig) -> None:
        self.governance = governance
        self.active_config = copy.deepcopy(base_config)
        self.config_checkpoints: List[Dict[str, Any]] = [
            {
                "checkpoint_id": "ckpt_initial",
                "timestamp": utc_now_iso(),
                "config": self.active_config.model_dump(),
            }
        ]
        self.proposals: Dict[str, Dict[str, Any]] = {}

    def analyze_bottlenecks(self, benchmark_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect benchmark summary and generate a hypothesis-driven configuration proposal."""
        candidates = benchmark_summary.get("architectures", {})
        full = candidates.get("D_FULL_OMNIS", {})
        retention = float(full.get("retention_mean", 0.0))
        reward = float(full.get("task_performance_mean", 0.0))
        compute_cost = float(full.get("compute_cost_mean", 0.0))

        patch: Dict[str, Any] = {}
        rationale: List[str] = []

        if retention < 0.75:
            patch["regularization_lambda"] = round(min(0.05, self.active_config.regularization_lambda + 0.003), 4)
            rationale.append("Retention below 0.75: slightly increase synaptic regularization to dampen drift.")
        if reward < 1.5:
            patch["learning_rate"] = round(min(0.25, self.active_config.learning_rate * 1.15), 4)
            rationale.append("Task performance below target: modestly increase learning rate.")
        if compute_cost > 0.65:
            patch["mutation_rate"] = round(max(0.05, self.active_config.mutation_rate * 0.9), 4)
            rationale.append("Compute cost elevated: tighten mutation rate to stabilize compact connectomes.")
        if not patch:
            patch["learning_rate"] = round(max(0.02, self.active_config.learning_rate * 0.95), 4)
            rationale.append("Fine-tuning learning rate decay for convergence stability.")

        proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
        validated_patch = self.governance.validate_config_patch(patch)
        proposal = {
            "proposal_id": proposal_id,
            "created_at": utc_now_iso(),
            "status": "proposed",
            "patch": validated_patch,
            "rationale": rationale,
            "gate_passed": False,
            "sandbox_evaluation": None,
        }
        self.proposals[proposal_id] = proposal
        self.governance.record_audit("self_improvement", "proposal_created", proposal)
        return proposal

    def evaluate_proposal_sandbox(
        self,
        proposal_id: str,
        evaluator_fn: Callable[[SystemConfig], Dict[str, float]],
        min_score_delta: float = 0.0,
    ) -> Dict[str, Any]:
        """Run isolated sandbox evaluation comparing current active config vs candidate config."""
        proposal = self.proposals[proposal_id]
        baseline_metrics = evaluator_fn(copy.deepcopy(self.active_config))

        candidate_dict = self.active_config.model_dump()
        candidate_dict.update(proposal["patch"])
        candidate_config = SystemConfig(**candidate_dict)
        candidate_metrics = evaluator_fn(candidate_config)

        base_score = float(baseline_metrics.get("composite_score", 0.0))
        cand_score = float(candidate_metrics.get("composite_score", 0.0))
        delta = cand_score - base_score
        gate_passed = bool(delta >= min_score_delta)

        proposal["status"] = "validated_pending_approval" if gate_passed else "gate_rejected"
        proposal["gate_passed"] = gate_passed
        proposal["sandbox_evaluation"] = {
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "score_delta": delta,
            "min_score_delta": min_score_delta,
        }
        self.governance.record_audit("self_improvement", "proposal_sandbox_evaluated", proposal)
        return proposal

    def approve_and_deploy(self, proposal_id: str, approver: str = "researcher") -> SystemConfig:
        """Deploy a gate-validated proposal only after explicit approval, saving a rollback checkpoint."""
        proposal = self.proposals[proposal_id]
        if not proposal.get("gate_passed"):
            raise RuntimeError(f"Proposal {proposal_id} has not passed benchmark gates.")
        self.config_checkpoints.append(
            {
                "checkpoint_id": f"ckpt_{len(self.config_checkpoints)}_{uuid.uuid4().hex[:6]}",
                "timestamp": utc_now_iso(),
                "config": self.active_config.model_dump(),
            }
        )
        updated = self.active_config.model_dump()
        updated.update(proposal["patch"])
        self.active_config = SystemConfig(**updated)
        proposal["status"] = "deployed"
        proposal["approved_by"] = approver
        proposal["deployed_at"] = utc_now_iso()
        self.governance.record_audit(
            approver,
            "proposal_deployed",
            {"proposal_id": proposal_id, "patch": proposal["patch"]},
        )
        return self.active_config

    def rollback(self, checkpoint_index: int = -1, actor: str = "researcher") -> SystemConfig:
        """Rollback active configuration to an earlier saved checkpoint."""
        ckpt = self.config_checkpoints[checkpoint_index]
        self.active_config = SystemConfig(**ckpt["config"])
        self.governance.record_audit(
            actor,
            "config_rollback",
            {"checkpoint_id": ckpt["checkpoint_id"]},
        )
        return self.active_config
