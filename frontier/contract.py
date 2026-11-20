"""Shared frontier contract, neutral engine hook, capacity-matched harness, and registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class FrontierSpec:
    """Default-off specification for optional live frontier mechanisms."""

    enabled: bool = False
    enable_ctm_synchrony: bool = False
    enable_kuramoto: bool = False
    enable_continuum_memory: bool = False
    enable_btsp: bool = False
    enable_fast_weights: bool = False
    enable_associative_memory: bool = False
    enable_dendritic: bool = False
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrontierSpec":
        valid_fields = {
            "enabled",
            "enable_ctm_synchrony",
            "enable_kuramoto",
            "enable_continuum_memory",
            "enable_btsp",
            "enable_fast_weights",
            "enable_associative_memory",
            "enable_dendritic",
            "seed",
        }
        unknown = set(data.keys()) - valid_fields
        if unknown:
            raise TypeError(f"Unknown FrontierSpec fields: {sorted(unknown)}")
        return cls(**data)

    def any_active(self) -> bool:
        return bool(
            self.enabled
            and (
                self.enable_ctm_synchrony
                or self.enable_kuramoto
                or self.enable_continuum_memory
                or self.enable_btsp
                or self.enable_fast_weights
                or self.enable_associative_memory
                or self.enable_dendritic
            )
        )


class FrontierExtensionHook:
    """Neutral extension hook attached to OmnisCore.

    When spec.enabled is False (the default), all hook calls are pure no-ops that leave
    core simulation trajectories bit-for-bit unchanged.
    """

    def __init__(self, spec: Optional[FrontierSpec] = None) -> None:
        self.spec = spec or FrontierSpec()
        self.observers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.telemetry_history: List[Dict[str, Any]] = []

    def register_observer(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self.observers[name] = fn

    def configure(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.spec.to_dict()
        current.update({k: v for k, v in updates.items() if k in current})
        self.spec = FrontierSpec.from_dict(current)
        return self.spec.to_dict()

    def on_step(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Invoke enabled frontier observers; returns None immediately when disabled."""
        if not self.spec.any_active():
            return None
        snapshot: Dict[str, Any] = {"tick": context.get("tick", 0), "observers": {}}
        for name, obs_fn in self.observers.items():
            flag_key = f"enable_{name}"
            if getattr(self.spec, flag_key, False):
                snapshot["observers"][name] = obs_fn(context)
        self.telemetry_history.append(snapshot)
        if len(self.telemetry_history) > 100:
            self.telemetry_history = self.telemetry_history[-100:]
        return snapshot

    def get_status(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "active": self.spec.any_active(),
            "registered_observers": sorted(self.observers.keys()),
            "telemetry_records": len(self.telemetry_history),
            "latest_telemetry": self.telemetry_history[-1] if self.telemetry_history else None,
        }


def _paired_ci_and_evalue(diffs: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Compute paired mean diff, 95% CI, t-stat, and anytime-valid sequential e-value."""
    n = len(diffs)
    if n == 0:
        return {
            "n": 0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "ci_95_low": 0.0,
            "ci_95_high": 0.0,
            "ci_excludes_zero": False,
            "e_value": 1.0,
            "anytime_valid_reject": False,
        }
    mean_d = sum(diffs) / float(n)
    if n > 1:
        var_d = sum((x - mean_d) ** 2 for x in diffs) / float(n - 1)
        std_d = math.sqrt(max(0.0, var_d))
    else:
        std_d = 0.0
    se = std_d / math.sqrt(float(n)) if n > 0 else 0.0
    # Conservative Student-t critical value lookup for small paired-seed samples
    t_crit_map = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
    t_crit = t_crit_map.get(max(1, n - 1), 2.045)
    margin = t_crit * se
    ci_lo = mean_d - margin
    ci_hi = mean_d + margin
    excludes_zero = bool(ci_lo > 0.0) if mean_d > 0 else bool(ci_hi < 0.0)

    # Anytime-valid exponential martingale e-value under sub-Gaussian scale sigma
    sigma = max(0.05, std_d)
    lam = min(1.5, max(0.2, abs(mean_d) / (sigma * sigma + 1e-6)))
    log_e = 0.0
    for d in diffs:
        log_e += lam * d - 0.5 * (lam ** 2) * (sigma ** 2)
    e_val = math.exp(max(-20.0, min(40.0, log_e)))
    return {
        "n": n,
        "mean_diff": round(float(mean_d), 5),
        "std_diff": round(float(std_d), 5),
        "ci_95_low": round(float(ci_lo), 5),
        "ci_95_high": round(float(ci_hi), 5),
        "ci_excludes_zero": excludes_zero,
        "e_value": round(float(e_val), 4),
        "anytime_valid_reject": bool(e_val >= (1.0 / max(1e-6, alpha))),
    }


class CapacityMatchedHarness:
    """Evaluates frontier modules across paired seeds against baseline and capacity-matched control.

    A module only receives verdict='improved' if its paired 95% confidence interval strictly
    excludes zero above BOTH the baseline AND the capacity-matched control (when applicable).
    """

    @staticmethod
    def evaluate(
        module_id: str,
        category: str,
        metric_name: str,
        seeds: List[int],
        trial_fn: Callable[[int], Dict[str, Any]],
        higher_is_better: bool = True,
        requires_capacity_control: bool = True,
    ) -> Dict[str, Any]:
        seed_list = list(seeds or [42, 101, 202, 303, 404])
        baseline_scores: List[float] = []
        candidate_scores: List[float] = []
        control_scores: List[float] = []
        details_last: Dict[str, Any] = {}

        for s in seed_list:
            out = trial_fn(int(s))
            b = float(out["baseline"])
            c = float(out["candidate"])
            k = float(out.get("capacity_matched_control", b))
            baseline_scores.append(b)
            candidate_scores.append(c)
            control_scores.append(k)
            if "details" in out and isinstance(out["details"], dict):
                details_last = out["details"]

        sign = 1.0 if higher_is_better else -1.0
        diffs_vs_base = [sign * (c - b) for c, b in zip(candidate_scores, baseline_scores)]
        diffs_vs_ctrl = [sign * (c - k) for c, k in zip(candidate_scores, control_scores)]

        stats_base = _paired_ci_and_evalue(diffs_vs_base)
        stats_ctrl = _paired_ci_and_evalue(diffs_vs_ctrl)

        improved_base = bool(stats_base["mean_diff"] > 0 and stats_base["ci_excludes_zero"])
        improved_ctrl = (
            bool(stats_ctrl["mean_diff"] > 0 and stats_ctrl["ci_excludes_zero"])
            if requires_capacity_control
            else True
        )

        if improved_base and improved_ctrl:
            verdict = "improved"
        elif stats_base["mean_diff"] < 0 and stats_base["ci_excludes_zero"]:
            verdict = "regressed"
        else:
            verdict = "inconclusive"

        mean_base = sum(baseline_scores) / max(1, len(baseline_scores))
        mean_cand = sum(candidate_scores) / max(1, len(candidate_scores))
        mean_ctrl = sum(control_scores) / max(1, len(control_scores))

        payload = {
            "module_id": module_id,
            "category": category,
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
            "requires_capacity_control": requires_capacity_control,
            "seeds": seed_list,
            "baseline_mean": round(float(mean_base), 5),
            "candidate_mean": round(float(mean_cand), 5),
            "capacity_matched_control_mean": round(float(mean_ctrl), 5),
            "vs_baseline": stats_base,
            "vs_capacity_control": stats_ctrl,
            "verdict": verdict,
            "details": details_last,
        }
        receipt_bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        payload["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
        return payload


@dataclass
class FrontierModuleEntry:
    """Registry descriptor for one frontier research module."""

    module_id: str
    title: str
    category: str
    description: str
    requires_capacity_control: bool
    runner: Callable[[List[int]], Dict[str, Any]]

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "requires_capacity_control": self.requires_capacity_control,
        }


def build_frontier_registry() -> Dict[str, FrontierModuleEntry]:
    """Assemble the complete registry of all 18 frontier modules and NEUROWEAVE mechanisms."""
    from frontier.analysis import (
        run_causal_emergence_benchmark,
        run_connectivity_benchmark,
        run_topk_sparse_dictionary_benchmark,
    )
    from frontier.biophysics import (
        run_hodgkin_huxley_cable_benchmark,
        run_nir_export_benchmark,
    )
    from frontier.memory_systems import (
        run_associative_memory_benchmark,
        run_btsp_binary_memory_benchmark,
        run_continuum_memory_benchmark,
        run_fast_weights_benchmark,
    )
    from frontier.methodology import (
        run_anytime_valid_stats_benchmark,
        run_decision_sensitive_selection_benchmark,
        run_faultline_ensemble_benchmark,
        run_verifier_gated_qd_benchmark,
    )
    from frontier.models import (
        run_dendritic_vs_mlp_benchmark,
        run_tiny_recursive_sudoku_benchmark,
    )
    from frontier.optimizers import (
        run_muon_vs_adam_benchmark,
        run_rigl_sparse_training_benchmark,
    )
    from frontier.temporal import (
        run_ctm_synchrony_benchmark,
        run_kuramoto_binding_benchmark,
    )

    entries = [
        FrontierModuleEntry(
            module_id="ctm_synchrony",
            title="Mini Continuous Thought Machine & Synchronisation Readout",
            category="Temporal",
            description="Neuron-level temporal history traces with pairwise synchronization readout S_t = Z_t Z_t^T over adaptive thought ticks.",
            requires_capacity_control=True,
            runner=run_ctm_synchrony_benchmark,
        ),
        FrontierModuleEntry(
            module_id="kuramoto_binding",
            title="Kuramoto Coupled-Oscillator Phase Binding",
            category="Temporal",
            description="Phase-coupled oscillators wired to SONAPSE connectome clusters computing order parameter R(t) and phase-locking binding.",
            requires_capacity_control=False,
            runner=run_kuramoto_binding_benchmark,
        ),
        FrontierModuleEntry(
            module_id="continuum_memory",
            title="Continuum Memory (Nested Learning Multi-Timescale)",
            category="Memory",
            description="Three frequency-separated nested consolidation timescales retaining prior mappings under continual shift.",
            requires_capacity_control=True,
            runner=run_continuum_memory_benchmark,
        ),
        FrontierModuleEntry(
            module_id="btsp_binary_memory",
            title="One-Shot BTSP Binary-Synapse Memory",
            category="Memory",
            description="Plateau-potential gated single-shot imprinting onto sparse binary synapses vs capacity-matched random projection.",
            requires_capacity_control=True,
            runner=run_btsp_binary_memory_benchmark,
        ),
        FrontierModuleEntry(
            module_id="associative_memory",
            title="Modern Hopfield Associative Memory (NEUROWEAVE Mechanism)",
            category="NEUROWEAVE Gaps",
            description="Continuous softmax energy Modern Hopfield associative retrieval restoring corrupted sensory patterns in one step.",
            requires_capacity_control=True,
            runner=run_associative_memory_benchmark,
        ),
        FrontierModuleEntry(
            module_id="fast_weights",
            title="Associative Fast Weights Working Memory (NEUROWEAVE Mechanism)",
            category="NEUROWEAVE Gaps",
            description="Outer-product decaying fast-weight matrix A_t = lambda A_{t-1} + eta h_t h_t^T for rapid sequential key-value binding.",
            requires_capacity_control=True,
            runner=run_fast_weights_benchmark,
        ),
        FrontierModuleEntry(
            module_id="tiny_recursive_model",
            title="Tiny Recursive Model (TRM) on 4x4 Sudoku",
            category="Models",
            description="Iterative latent reasoning (z, y) <- f(x, y, z) on real 4x4 Sudoku puzzles vs capacity-matched single-pass solver.",
            requires_capacity_control=True,
            runner=run_tiny_recursive_sudoku_benchmark,
        ),
        FrontierModuleEntry(
            module_id="dendritic_layer",
            title="Active Dendritic Compartments vs Dense MLP",
            category="Models",
            description="Context-gated nonlinear dendritic branch subunits vs parameter-matched dense MLP on context-remapped tasks.",
            requires_capacity_control=True,
            runner=run_dendritic_vs_mlp_benchmark,
        ),
        FrontierModuleEntry(
            module_id="muon_optimizer",
            title="Muon (Newton-Schulz Orthogonalized Momentum) vs Adam",
            category="Optimisers",
            description="5-step quintic Newton-Schulz orthogonalized momentum updates on 2D neural-graft weights vs Adam.",
            requires_capacity_control=False,
            runner=run_muon_vs_adam_benchmark,
        ),
        FrontierModuleEntry(
            module_id="rigl_sparse_training",
            title="RigL Dynamic Sparse Training",
            category="Optimisers",
            description="Fixed-budget 70% sparse training combining magnitude pruning and gradient-magnitude regrowth vs static sparse mask.",
            requires_capacity_control=True,
            runner=run_rigl_sparse_training_benchmark,
        ),
        FrontierModuleEntry(
            module_id="topk_sparse_dictionary",
            title="TopK Sparse Dictionary (SAE) with Causal Feature Ablation",
            category="Analysis",
            description="Exact Top-k overcomplete sparse autoencoder over SONAPSE activations with causal interventional feature ablation.",
            requires_capacity_control=True,
            runner=run_topk_sparse_dictionary_benchmark,
        ),
        FrontierModuleEntry(
            module_id="functional_effective_connectivity",
            title="Functional & Effective Connectivity (FC + Granger/TE)",
            category="Analysis",
            description="Separates zero-lag functional correlation from directed lagged effective connectivity on SONAPSE spike/rate traces.",
            requires_capacity_control=False,
            runner=run_connectivity_benchmark,
        ),
        FrontierModuleEntry(
            module_id="causal_emergence_audit",
            title="Macro-vs-Micro Causal Emergence & Effective Information",
            category="Analysis",
            description="Evaluates macro cluster coarse-graining effective information gains over micro synaptic dynamics.",
            requires_capacity_control=False,
            runner=run_causal_emergence_benchmark,
        ),
        FrontierModuleEntry(
            module_id="anytime_valid_statistics",
            title="Anytime-Valid Sequential Statistics (E-Values)",
            category="Methodology",
            description="Martingale e-values and always-valid confidence sequences preventing optional-stopping Type-I error inflation.",
            requires_capacity_control=False,
            runner=run_anytime_valid_stats_benchmark,
        ),
        FrontierModuleEntry(
            module_id="decision_sensitive_selection",
            title="Decision-Sensitive Experiment Selection (EVSI)",
            category="Methodology",
            description="Selects next research assay by Expected Value of Sample Information vs uniform random experiment scheduling.",
            requires_capacity_control=False,
            runner=run_decision_sensitive_selection_benchmark,
        ),
        FrontierModuleEntry(
            module_id="faultline_joint_failure",
            title="FAULTLINE Joint-Failure Ensemble Audit",
            category="Methodology",
            description="Measures higher-order correlated voting failure risk and evaluates Tether-de-correlated ensemble mitigation.",
            requires_capacity_control=False,
            runner=run_faultline_ensemble_benchmark,
        ),
        FrontierModuleEntry(
            module_id="verifier_gated_qd_search",
            title="Verifier-Gated Quality-Diversity (MAP-Elites) Search",
            category="Discovery",
            description="Populates a 2D [Wiring Economy x Causal Emergence] archive strictly gated by the CROSSWIRE Mealy safety contract.",
            requires_capacity_control=False,
            runner=run_verifier_gated_qd_benchmark,
        ),
        FrontierModuleEntry(
            module_id="hodgkin_huxley_cable",
            title="Hodgkin-Huxley Multi-Compartment Cable Reference",
            category="Biophysics",
            description="Phenomenological 4-variable (V, m, h, n) RK4 multi-compartment axonal cable propagating an action potential.",
            requires_capacity_control=False,
            runner=run_hodgkin_huxley_cable_benchmark,
        ),
        FrontierModuleEntry(
            module_id="nir_interchange_export",
            title="NIR (Neuromorphic Intermediate Representation) Export",
            category="Biophysics",
            description="Exports live SONAPSE connectomes into validated NIR-compliant Input/Affine/LIF/Output interchange graphs.",
            requires_capacity_control=False,
            runner=run_nir_export_benchmark,
        ),
    ]
    return {e.module_id: e for e in entries}
