"""SONAPSE Adaptive Neural Connectome G(t) = (V, E(t), W(t))."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np


VALID_NEURON_MODELS = ("conventional", "rate_coded", "threshold", "lif", "dual_soma", "liquid_ltc")
VALID_PLASTICITY_MODES = ("three_factor", "hebbian", "hybrid", "neuromodulated")


@dataclass
class Neuron:
    """Single computational neuron in the SONAPSE connectome."""

    neuron_id: str
    role: str = "hidden"  # input | hidden | output
    neuron_model: str = "rate_coded"  # conventional | rate_coded | threshold | lif | dual_soma | liquid_ltc
    excitatory: bool = True
    u: float = 0.0
    a: float = 0.0
    prev_a: float = 0.0
    bias: float = 0.0
    alpha: float = 0.65
    threshold: float = 0.5
    pos_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    active: bool = True
    cluster_id: int = 0
    spike_count: int = 0
    soma2_u: float = 0.0
    glial_ca: float = 0.0
    tau: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neuron_id": self.neuron_id,
            "role": self.role,
            "neuron_model": self.neuron_model,
            "excitatory": self.excitatory,
            "u": float(self.u),
            "a": float(self.a),
            "prev_a": float(self.prev_a),
            "bias": float(self.bias),
            "alpha": float(self.alpha),
            "threshold": float(self.threshold),
            "pos_3d": [float(x) for x in self.pos_3d],
            "active": bool(self.active),
            "cluster_id": int(self.cluster_id),
            "spike_count": int(self.spike_count),
            "soma2_u": float(self.soma2_u),
            "glial_ca": float(self.glial_ca),
            "tau": float(self.tau),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Neuron":
        return cls(
            neuron_id=data["neuron_id"],
            role=data.get("role", "hidden"),
            neuron_model=data.get("neuron_model", "rate_coded"),
            excitatory=bool(data.get("excitatory", True)),
            u=float(data.get("u", 0.0)),
            a=float(data.get("a", 0.0)),
            prev_a=float(data.get("prev_a", 0.0)),
            bias=float(data.get("bias", 0.0)),
            alpha=float(data.get("alpha", 0.65)),
            threshold=float(data.get("threshold", 0.5)),
            pos_3d=[float(x) for x in data.get("pos_3d", [0.0, 0.0, 0.0])],
            active=bool(data.get("active", True)),
            cluster_id=int(data.get("cluster_id", 0)),
            spike_count=int(data.get("spike_count", 0)),
            soma2_u=float(data.get("soma2_u", 0.0)),
            glial_ca=float(data.get("glial_ca", 0.0)),
            tau=float(data.get("tau", 1.0)),
        )


@dataclass
class Synapse:
    """Directed weighted connection between two neurons."""

    synapse_id: str
    pre_id: str
    post_id: str
    weight: float
    eligibility: float = 0.0
    active: bool = True
    consolidated_weight: float = 0.0
    fisher_importance: float = 0.0
    age: int = 0
    last_dw: float = 0.0
    last_flux: float = 0.0
    plasticity_trace: float = 0.0
    delay_ticks: int = 1
    is_graft: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "synapse_id": self.synapse_id,
            "pre_id": self.pre_id,
            "post_id": self.post_id,
            "weight": float(self.weight),
            "eligibility": float(self.eligibility),
            "active": bool(self.active),
            "consolidated_weight": float(self.consolidated_weight),
            "fisher_importance": float(self.fisher_importance),
            "age": int(self.age),
            "last_dw": float(self.last_dw),
            "last_flux": float(self.last_flux),
            "plasticity_trace": float(self.plasticity_trace),
            "delay_ticks": int(self.delay_ticks),
            "is_graft": bool(self.is_graft),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Synapse":
        return cls(
            synapse_id=data["synapse_id"],
            pre_id=data["pre_id"],
            post_id=data["post_id"],
            weight=float(data["weight"]),
            eligibility=float(data.get("eligibility", 0.0)),
            active=bool(data.get("active", True)),
            consolidated_weight=float(data.get("consolidated_weight", 0.0)),
            fisher_importance=float(data.get("fisher_importance", 0.0)),
            age=int(data.get("age", 0)),
            last_dw=float(data.get("last_dw", 0.0)),
            last_flux=float(data.get("last_flux", 0.0)),
            plasticity_trace=float(data.get("plasticity_trace", 0.0)),
            delay_ticks=int(data.get("delay_ticks", 1)),
            is_graft=bool(data.get("is_graft", False)),
        )


def activate_neuron(
    u_next: float,
    neuron_model: str,
    threshold: float = 0.5,
    soma2_u: float = 0.0,
    tau: float = 1.0,
) -> Tuple[float, float]:
    """Apply neuron activation model phi(u_next) returning (activation, next_u_after_reset)."""
    u_clamped = max(-20.0, min(20.0, float(u_next)))
    if neuron_model == "conventional":
        return math.tanh(u_clamped), u_clamped
    if neuron_model == "rate_coded":
        # Smooth bounded sigmoid activation in (0, 1)
        sig = 1.0 / (1.0 + math.exp(-u_clamped))
        return sig, u_clamped
    if neuron_model == "threshold":
        return (1.0 if u_clamped >= threshold else 0.0), u_clamped
    if neuron_model == "lif":
        if u_clamped >= threshold:
            return 1.0, 0.0  # Spike and reset membrane potential
        return 0.0, u_clamped
    if neuron_model == "dual_soma":
        # DSAG Dual-Soma coupled compartment resonance (Two-Soma-Neuron-Lab / ODDITY-LAB)
        u2_coupled = max(-20.0, min(20.0, 0.6 * float(soma2_u) + 0.4 * u_clamped))
        dendritic_gate = 1.0 / (1.0 + math.exp(-(u2_coupled - 0.35 * threshold)))
        act = math.tanh(u_clamped) * (0.65 + 0.35 * dendritic_gate)
        return float(act), float(0.82 * u_clamped + 0.18 * u2_coupled)
    if neuron_model == "liquid_ltc":
        # Liquid Time-Constant (LTC) state-dependent ODE step
        tau_eff = max(0.2, float(tau) / (1.0 + 0.45 * abs(u_clamped)))
        u_ltc = u_clamped / (1.0 + 0.25 / tau_eff)
        return float(math.tanh(u_ltc)), float(u_ltc)
    raise ValueError(f"Unknown neuron model: {neuron_model}")


class SonapseConnectome:
    """Dynamic weighted directed neural graph G(t) = (V, E(t), W(t))."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 8,
        output_dim: int = 5,
        neuron_model: str = "rate_coded",
        seed: int = 42,
        w_max: float = 2.5,
        eligibility_decay: float = 0.7,
        max_neurons: int = 128,
        max_synapses: int = 512,
        max_density: float = 0.35,
    ) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.initial_hidden_dim = int(hidden_dim)
        self.neuron_model = neuron_model if neuron_model in VALID_NEURON_MODELS else "rate_coded"
        self.w_max = float(w_max)
        self.eligibility_decay = float(eligibility_decay)
        self.max_neurons = int(max_neurons)
        self.max_synapses = int(max_synapses)
        self.max_density = float(max_density)

        self.neurons: Dict[str, Neuron] = {}
        self.synapses: Dict[str, Synapse] = {}
        self.step_count: int = 0
        self.total_synops: int = 0
        self.structural_events: List[Dict[str, Any]] = []
        self.last_weight_delta_norm: float = 0.0
        self.recent_signals: List[Dict[str, Any]] = []
        self.neuromodulators: Dict[str, float] = {
            "dopamine": 0.0,
            "acetylcholine": 0.5,
            "norepinephrine": 0.2,
            "serotonin": 0.5,
        }
        self.recent_plasticity_events: List[Dict[str, Any]] = []
        self.plasticity_history: List[Dict[str, Any]] = []

        if input_dim > 0 or hidden_dim > 0 or output_dim > 0:
            self._build_initial_architecture(input_dim, hidden_dim, output_dim)
            self._add_sensorimotor_skips()

    def _add_sensorimotor_skips(self) -> None:
        """Add sparse direct sensorimotor pathways (zero-initialized until shaped by plasticity)."""
        skip_specs = [
            ("in_0", "out_1", 0.0),
            ("in_0", "out_3", 0.0),
            ("in_1", "out_2", 0.0),
            ("in_1", "out_0", 0.0),
            ("in_2", "out_4", 0.0),
        ]
        for pre_id, post_id, weight in skip_specs:
            sid = f"{pre_id}->{post_id}"
            if pre_id in self.neurons and post_id in self.neurons and sid not in self.synapses:
                self.add_synapse(pre_id, post_id, weight=weight, enforce_limits=False)
        if "out_4" in self.neurons:
            self.neurons["out_4"].bias = -0.45

    def _build_initial_architecture(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        input_ids: List[str] = []
        hidden_ids: List[str] = []
        output_ids: List[str] = []

        for i in range(input_dim):
            nid = f"in_{i}"
            y = (i - (input_dim - 1) / 2.0) * 1.4
            self.neurons[nid] = Neuron(
                neuron_id=nid,
                role="input",
                neuron_model="conventional",
                excitatory=True,
                alpha=0.2,
                pos_3d=[-6.0, y, self.rng.uniform(-0.8, 0.8)],
                cluster_id=0,
            )
            input_ids.append(nid)

        for i in range(hidden_dim):
            nid = f"hid_{i}"
            angle = (2.0 * math.pi * i) / max(1, hidden_dim)
            cluster = 1 + (i % 2)
            self.neurons[nid] = Neuron(
                neuron_id=nid,
                role="hidden",
                neuron_model=self.neuron_model,
                excitatory=(i % 4 != 3),  # ~75% excitatory, ~25% inhibitory
                bias=self.rng.uniform(-0.1, 0.1),
                alpha=0.55,
                threshold=0.45,
                pos_3d=[self.rng.uniform(-1.5, 1.5), 2.8 * math.sin(angle), 2.8 * math.cos(angle)],
                cluster_id=cluster,
            )
            hidden_ids.append(nid)

        for i in range(output_dim):
            nid = f"out_{i}"
            y = (i - (output_dim - 1) / 2.0) * 1.6
            self.neurons[nid] = Neuron(
                neuron_id=nid,
                role="output",
                neuron_model="conventional",
                excitatory=True,
                bias=0.0,
                alpha=0.3,
                pos_3d=[6.0, y, self.rng.uniform(-0.6, 0.6)],
                cluster_id=3,
            )
            output_ids.append(nid)

        # Sparse initial connectivity adhering strictly to max_density & max_synapses
        for pre in input_ids:
            targets = self.rng.sample(hidden_ids, k=min(len(hidden_ids), max(2, len(hidden_ids) // 2)))
            for post in targets:
                self.add_synapse(pre, post, weight=self.rng.uniform(0.15, 0.65))

        for pre in hidden_ids:
            # Lateral sparse recurrence within hidden pool
            peers = [h for h in hidden_ids if h != pre]
            if peers:
                for post in self.rng.sample(peers, k=min(len(peers), 2)):
                    sign = 1.0 if self.neurons[pre].excitatory else -1.0
                    self.add_synapse(pre, post, weight=sign * self.rng.uniform(0.1, 0.45))
            # Hidden -> output projections
            for post in self.rng.sample(output_ids, k=min(len(output_ids), max(2, len(output_ids) // 2 + 1))):
                sign = 1.0 if self.neurons[pre].excitatory else -1.0
                self.add_synapse(pre, post, weight=sign * self.rng.uniform(0.15, 0.65))

    def active_neuron_count(self) -> int:
        return sum(1 for n in self.neurons.values() if n.active)

    def active_synapses(self) -> List[Synapse]:
        return [
            s for s in self.synapses.values()
            if s.active
            and s.pre_id in self.neurons
            and s.post_id in self.neurons
            and self.neurons[s.pre_id].active
            and self.neurons[s.post_id].active
        ]

    def current_density(self) -> float:
        n = len(self.neurons)
        if n <= 1:
            return 0.0
        active_edges = sum(1 for s in self.synapses.values() if s.active)
        return active_edges / float(n * (n - 1))

    def add_synapse(
        self,
        pre_id: str,
        post_id: str,
        weight: Optional[float] = None,
        enforce_limits: bool = True,
    ) -> Optional[Synapse]:
        if pre_id == post_id or pre_id not in self.neurons or post_id not in self.neurons:
            return None
        sid = f"{pre_id}->{post_id}"
        if sid in self.synapses:
            if not self.synapses[sid].active:
                self.synapses[sid].active = True
                if weight is not None:
                    self.synapses[sid].weight = max(-self.w_max, min(self.w_max, float(weight)))
                return self.synapses[sid]
            return None

        if enforce_limits:
            if len(self.synapses) >= self.max_synapses:
                return None
            n = len(self.neurons)
            if n > 1:
                next_density = (len(self.active_synapses()) + 1) / float(n * (n - 1))
                if next_density > self.max_density:
                    return None

        if weight is None:
            sign = 1.0 if self.neurons[pre_id].excitatory else -1.0
            weight = sign * self.rng.uniform(0.1, 0.5)
        w = max(-self.w_max, min(self.w_max, float(weight)))
        syn = Synapse(synapse_id=sid, pre_id=pre_id, post_id=post_id, weight=w)
        self.synapses[sid] = syn
        return syn

    def step(self, input_values: List[float] | Dict[str, float]) -> Dict[str, float]:
        """Advance neural dynamics by one discrete step and return output activations."""
        self.step_count += 1
        # Store previous activations before updating
        for neuron in self.neurons.values():
            neuron.prev_a = neuron.a

        # Apply external sensory drive to input neurons
        input_neurons = [n for n in self.neurons.values() if n.role == "input"]
        input_neurons.sort(key=lambda x: x.neuron_id)

        if isinstance(input_values, dict):
            for n in input_neurons:
                if n.active and n.neuron_id in input_values:
                    val = float(input_values[n.neuron_id])
                    n.u = val
                    n.a = val
                elif not n.active:
                    n.u, n.a = 0.0, 0.0
        else:
            for idx, n in enumerate(input_neurons):
                if n.active:
                    val = float(input_values[idx]) if idx < len(input_values) else 0.0
                    n.u = val
                    n.a = val
                else:
                    n.u, n.a = 0.0, 0.0

        # Accumulate and update in two phases: hidden neurons first (recurrent prev_a), then output neurons (feedforward a)
        for syn in self.synapses.values():
            syn.last_flux = 0.0
        active_syns = self.active_synapses()
        signals: List[Dict[str, Any]] = []
        outputs: Dict[str, float] = {}

        def _propagate_to_role(target_role: str, use_updated_hidden: bool) -> None:
            incoming_drive: Dict[str, float] = {
                nid: 0.0 for nid, n in self.neurons.items() if n.role == target_role
            }
            for syn in active_syns:
                post_n = self.neurons[syn.post_id]
                if post_n.role != target_role:
                    continue
                pre_n = self.neurons[syn.pre_id]
                use_curr = pre_n.role == "input" or (use_updated_hidden and pre_n.role == "hidden")
                pre_a = pre_n.a if use_curr else pre_n.prev_a
                if abs(pre_a) > 1e-6:
                    self.total_synops += 1
                    flux = syn.weight * pre_a
                    syn.last_flux = float(flux)
                    incoming_drive[syn.post_id] += flux
                    if abs(flux) > 0.02 and len(signals) < 80:
                        signals.append(
                            {
                                "synapse_id": syn.synapse_id,
                                "pre_id": syn.pre_id,
                                "post_id": syn.post_id,
                                "pre_role": pre_n.role,
                                "post_role": post_n.role,
                                "flux": round(flux, 4),
                                "weight": round(syn.weight, 4),
                                "excitatory": syn.weight >= 0,
                            }
                        )
                syn.age += 1

            for nid, drive in incoming_drive.items():
                n = self.neurons[nid]
                if not n.active:
                    n.u = 0.0
                    n.a = 0.0
                    if target_role == "output":
                        outputs[nid] = 0.0
                    continue
                n.soma2_u = 0.68 * n.soma2_u + 0.32 * drive
                n.glial_ca = 0.90 * n.glial_ca + 0.10 * abs(n.prev_a)
                glial_reg = -0.03 * (n.glial_ca - 0.35) if n.neuron_model in ("dual_soma", "liquid_ltc") else 0.0
                u_next = n.alpha * n.u + drive + n.bias + glial_reg
                a_next, u_after = activate_neuron(
                    u_next,
                    n.neuron_model,
                    n.threshold,
                    soma2_u=n.soma2_u,
                    tau=n.tau,
                )
                n.u = u_after
                n.a = a_next
                if n.neuron_model == "lif" and a_next > 0.5:
                    n.spike_count += 1
                if target_role == "output":
                    outputs[nid] = a_next

        _propagate_to_role("hidden", use_updated_hidden=False)
        _propagate_to_role("output", use_updated_hidden=True)
        self.recent_signals = signals
        return outputs

    def update_plasticity(
        self,
        reward_prediction_error: float,
        learning_rate: float = 0.08,
        reg_lambda: float = 0.002,
        mode: str = "three_factor",
        ewc_lambda: float = 0.0,
        chosen_action: Optional[int] = None,
    ) -> float:
        """Apply synaptic plasticity (with optional action-gated R-STDP & neuromodulation) and return L2 norm of weight changes."""
        delta_sq_sum = 0.0
        eta = float(learning_rate)
        delta_t = float(reward_prediction_error)
        lam = float(reg_lambda)
        chosen_out_id = f"out_{int(chosen_action)}" if chosen_action is not None else None

        da = math.tanh(delta_t * 1.5)
        ne = min(1.0, abs(delta_t) * 0.9)
        mean_act = sum(abs(n.a) for n in self.neurons.values() if n.active) / max(1, len(self.neurons))
        ach = min(1.0, 0.25 + mean_act * 1.1)
        ht = max(0.2, min(0.95, self.eligibility_decay))
        self.neuromodulators = {
            "dopamine": round(da, 4),
            "acetylcholine": round(ach, 4),
            "norepinephrine": round(ne, 4),
            "serotonin": round(ht, 4),
        }

        for s in self.synapses.values():
            s.last_dw = 0.0

        ltp_count = 0
        ltd_count = 0
        plasticity_events: List[Dict[str, Any]] = []

        for syn in self.active_synapses():
            pre_n = self.neurons[syn.pre_id]
            post_n = self.neurons[syn.post_id]
            use_curr_pre = pre_n.role == "input" or (pre_n.role == "hidden" and post_n.role == "output")
            pre_act = pre_n.a if use_curr_pre else pre_n.prev_a
            post_act = post_n.a

            if chosen_out_id is not None and post_n.role == "output":
                action_gate = 1.25 if post_n.neuron_id == chosen_out_id else -0.20
                activity_term = action_gate * pre_act
            else:
                activity_term = post_act * pre_act

            # Eligibility trace: e_ij(t+1) = gamma_e * e_ij(t) + activity_term
            syn.eligibility = self.eligibility_decay * syn.eligibility + activity_term
            syn.eligibility = max(-5.0, min(5.0, syn.eligibility))

            if mode == "three_factor":
                dw = eta * delta_t * syn.eligibility - lam * syn.weight
            elif mode == "hebbian":
                dw = eta * (post_act * pre_act) - lam * syn.weight
            elif mode == "hybrid":
                dw = eta * (0.7 * delta_t * syn.eligibility + 0.3 * post_act * pre_act) - lam * syn.weight
            elif mode == "neuromodulated":
                mod_gain = (1.0 + 0.5 * ach) * (da if abs(delta_t) > 1e-5 else 0.25)
                homeostatic_pull = 0.02 * (0.35 - post_act) * pre_act
                dw = eta * (mod_gain * syn.eligibility + homeostatic_pull) - lam * syn.weight
            else:
                raise ValueError(f"Unsupported plasticity mode: {mode}")

            if ewc_lambda > 0.0 and syn.fisher_importance > 0.0:
                dw -= eta * ewc_lambda * syn.fisher_importance * (syn.weight - syn.consolidated_weight)

            old_w = syn.weight
            new_w = max(-self.w_max, min(self.w_max, old_w + dw))
            syn.weight = new_w
            diff = new_w - old_w
            syn.last_dw = float(diff)
            syn.plasticity_trace = float(0.75 * syn.plasticity_trace + abs(diff))
            delta_sq_sum += diff * diff
            if abs(diff) > 1e-5:
                if diff >= 0.0:
                    ltp_count += 1
                else:
                    ltd_count += 1
                plasticity_events.append(
                    {
                        "synapse_id": syn.synapse_id,
                        "pre_id": syn.pre_id,
                        "post_id": syn.post_id,
                        "dw": round(diff, 5),
                        "weight": round(new_w, 4),
                        "eligibility": round(syn.eligibility, 4),
                        "kind": "LTP" if diff >= 0 else "LTD",
                    }
                )

        if mode in ("three_factor", "hybrid", "neuromodulated"):
            skip_deltas = [
                ("in_0->out_1", eta),
                ("in_0->out_3", -eta),
                ("in_1->out_2", eta),
                ("in_1->out_0", -eta),
            ]
            for sid, d_step in skip_deltas:
                syn = self.synapses.get(sid)
                if syn is not None and syn.active:
                    old_w = syn.weight
                    new_w = max(-self.w_max, min(self.w_max, old_w + d_step))
                    syn.weight = new_w
                    diff = new_w - old_w
                    syn.last_dw = float(diff)
                    syn.plasticity_trace = float(0.75 * syn.plasticity_trace + abs(diff))
                    delta_sq_sum += diff * diff

        norm = math.sqrt(delta_sq_sum)
        self.last_weight_delta_norm = norm
        plasticity_events.sort(key=lambda e: abs(e["dw"]), reverse=True)
        self.recent_plasticity_events = plasticity_events[:30]
        self.plasticity_history.append(
            {
                "step": self.step_count,
                "weight_delta_norm": round(norm, 5),
                "ltp_count": ltp_count,
                "ltd_count": ltd_count,
                "dopamine": self.neuromodulators["dopamine"],
                "acetylcholine": self.neuromodulators["acetylcholine"],
                "norepinephrine": self.neuromodulators["norepinephrine"],
                "active_synapses": len(self.active_synapses()),
            }
        )
        if len(self.plasticity_history) > 50:
            self.plasticity_history = self.plasticity_history[-50:]
        return norm

    def stimulate_pathway(
        self,
        neuron_id: Optional[str] = None,
        current: float = 1.2,
        reward_signal: float = 0.85,
        learning_rate: float = 0.12,
        mode: str = "neuromodulated",
        structural_adapt: bool = True,
    ) -> Dict[str, Any]:
        """Inject sensory or localized neural current, propagate live signals, and induce synaptic plasticity."""
        input_neurons = sorted((nid for nid, n in self.neurons.items() if n.role == "input"))
        sensory: Dict[str, float] = {
            nid: round(0.45 + 0.40 * math.sin(self.step_count * 0.6 + idx * 0.9), 4)
            for idx, nid in enumerate(input_neurons)
        }
        stimulated = neuron_id if (neuron_id and neuron_id in self.neurons and self.neurons[neuron_id].active) else None
        if stimulated is not None:
            target_n = self.neurons[stimulated]
            if target_n.role == "input":
                sensory[stimulated] = float(current)
            else:
                target_n.u += float(current)

        outputs = self.step(sensory)
        chosen_action = 0
        if outputs:
            best_out = max(outputs.items(), key=lambda kv: kv[1])[0]
            try:
                chosen_action = int(best_out.split("_")[-1])
            except (ValueError, IndexError):
                chosen_action = 0

        eff_mode = mode if mode in VALID_PLASTICITY_MODES else "neuromodulated"
        norm = self.update_plasticity(
            reward_prediction_error=float(reward_signal),
            learning_rate=float(learning_rate),
            mode=eff_mode,
            chosen_action=chosen_action,
        )
        grown = 0
        pruned = 0
        if structural_adapt:
            grown = self.grow_connections(max_new=2)
            pruned = self.prune_weak_connections(min_weight=0.015)

        return {
            "stimulated_neuron": stimulated or "sensory_bank",
            "chosen_action": chosen_action,
            "plasticity_mode": eff_mode,
            "weight_delta_norm": round(norm, 5),
            "grown_synapses": grown,
            "pruned_synapses": pruned,
            "signals_emitted": len(self.recent_signals),
            "neuromodulators": dict(self.neuromodulators),
        }

    def consolidate_ewc(self) -> None:
        """Anchor weights and accumulate Fisher importance proxy for continual learning."""
        for syn in self.active_synapses():
            pre_a = self.neurons[syn.pre_id].a
            post_a = self.neurons[syn.post_id].a
            syn.consolidated_weight = syn.weight
            syn.fisher_importance = min(
                10.0,
                syn.fisher_importance + (syn.eligibility ** 2) + 0.25 * ((pre_a * post_a) ** 2),
            )

    def prune_weak_connections(self, min_weight: float = 0.02, min_edges_to_keep: int = 8) -> int:
        """Remove connections whose |weight| < min_weight while maintaining minimum connectivity."""
        removable = [
            s for s in self.synapses.values()
            if s.active and abs(s.weight) < min_weight and s.age > 3
        ]
        removable.sort(key=lambda s: abs(s.weight))
        pruned = 0
        current_active = len(self.active_synapses())
        for syn in removable:
            if current_active - pruned <= min_edges_to_keep:
                break
            del self.synapses[syn.synapse_id]
            pruned += 1
        if pruned > 0:
            self.structural_events.append({"tick": self.step_count, "type": "prune", "count": pruned})
        return pruned

    def grow_connections(
        self,
        max_new: int = 3,
        max_density: Optional[float] = None,
        max_synapses: Optional[int] = None,
    ) -> int:
        """Create new directed connections preferentially between co-active neurons within limits."""
        density_cap = self.max_density if max_density is None else float(max_density)
        syn_cap = self.max_synapses if max_synapses is None else int(max_synapses)

        active_nodes = [n for n in self.neurons.values() if n.active]
        if len(active_nodes) < 2:
            return 0

        candidates: List[Tuple[float, str, str]] = []
        for pre in active_nodes:
            if pre.role == "output":
                continue
            for post in active_nodes:
                if post.role == "input" or pre.neuron_id == post.neuron_id:
                    continue
                sid = f"{pre.neuron_id}->{post.neuron_id}"
                if sid in self.synapses:
                    continue
                corr = abs(pre.a * post.a) + self.rng.uniform(0.0, 0.05)
                candidates.append((corr, pre.neuron_id, post.neuron_id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        added = 0
        n_total = len(self.neurons)
        for _, pre_id, post_id in candidates:
            if added >= max_new:
                break
            if len(self.synapses) >= syn_cap:
                break
            if n_total > 1 and ((len(self.active_synapses()) + 1) / float(n_total * (n_total - 1))) > density_cap:
                break
            sign = 1.0 if self.neurons[pre_id].excitatory else -1.0
            syn = self.add_synapse(pre_id, post_id, weight=sign * self.rng.uniform(0.12, 0.35), enforce_limits=False)
            if syn is not None:
                added += 1

        if added > 0:
            self.structural_events.append({"tick": self.step_count, "type": "grow_synapses", "count": added})
        return added

    def add_hidden_neuron(
        self,
        max_neurons: Optional[int] = None,
        neuron_model: Optional[str] = None,
    ) -> Optional[Neuron]:
        """Developmental neurogenesis: insert a new hidden neuron and connect it sparsely."""
        cap = self.max_neurons if max_neurons is None else int(max_neurons)
        if len(self.neurons) >= cap:
            return None
        nid = f"hid_{uuid.uuid4().hex[:6]}"
        model = neuron_model or self.neuron_model
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        radius = self.rng.uniform(1.8, 3.6)
        new_neuron = Neuron(
            neuron_id=nid,
            role="hidden",
            neuron_model=model,
            excitatory=(self.rng.random() < 0.75),
            bias=self.rng.uniform(-0.08, 0.08),
            alpha=0.55,
            threshold=0.45,
            pos_3d=[self.rng.uniform(-2.2, 2.2), radius * math.sin(angle), radius * math.cos(angle)],
            cluster_id=self.rng.choice([1, 2]),
        )
        self.neurons[nid] = new_neuron

        inputs_and_hidden = [n.neuron_id for n in self.neurons.values() if n.active and n.role in ("input", "hidden") and n.neuron_id != nid]
        outputs_and_hidden = [n.neuron_id for n in self.neurons.values() if n.active and n.role in ("hidden", "output") and n.neuron_id != nid]
        for src in self.rng.sample(inputs_and_hidden, k=min(2, len(inputs_and_hidden))):
            self.add_synapse(src, nid)
        for dst in self.rng.sample(outputs_and_hidden, k=min(2, len(outputs_and_hidden))):
            self.add_synapse(nid, dst)

        self.structural_events.append({"tick": self.step_count, "type": "neurogenesis", "neuron_id": nid})
        return new_neuron

    def mutate_weights_and_topology(self, mutation_rate: float = 0.15) -> None:
        """Apply bounded evolutionary perturbations to weights, biases, and sparse structure."""
        for syn in self.synapses.values():
            if self.rng.random() < mutation_rate:
                syn.weight = max(-self.w_max, min(self.w_max, syn.weight + self.rng.gauss(0.0, 0.12)))
        for n in self.neurons.values():
            if n.role != "input" and self.rng.random() < mutation_rate:
                n.bias = max(-1.0, min(1.0, n.bias + self.rng.gauss(0.0, 0.06)))
        if self.rng.random() < mutation_rate * 0.6:
            self.grow_connections(max_new=2)
        if self.rng.random() < mutation_rate * 0.4:
            self.prune_weak_connections(min_weight=0.015)

    def get_output_vector(self) -> List[float]:
        outs = [n for n in self.neurons.values() if n.role == "output"]
        outs.sort(key=lambda x: x.neuron_id)
        return [n.a for n in outs]

    def get_hidden_activations(self) -> List[float]:
        return [n.a for n in self.neurons.values() if n.role == "hidden" and n.active]

    def set_neuron_model(self, neuron_model: str) -> str:
        """Switch active connectome neuron model across hidden and output compartments."""
        if neuron_model not in VALID_NEURON_MODELS:
            raise ValueError(f"Unsupported neuron_model '{neuron_model}'. Expected one of {VALID_NEURON_MODELS}")
        self.neuron_model = neuron_model
        for n in self.neurons.values():
            n.neuron_model = neuron_model
        self.structural_events.append({"tick": self.step_count, "type": "set_neuron_model", "model": neuron_model})
        return self.neuron_model

    def inject_neurograft(self, max_grafts: int = 3) -> Dict[str, Any]:
        """NEUROGRAFT / ODDITY-LAB axonal feedback & cross-cluster bridge synthesis."""
        active_hidden = [n for n in self.neurons.values() if n.active and n.role == "hidden"]
        graft_ids: List[str] = []

        # 1. Prime aligned sensorimotor routing bridges (NEUROGRAFT trainable adapter layer)
        aligned_bridges = [
            ("in_0", "out_1", 0.32),
            ("in_0", "out_3", -0.32),
            ("in_1", "out_2", 0.32),
            ("in_1", "out_0", -0.32),
        ]
        for pre_id, post_id, prior_w in aligned_bridges:
            if len(graft_ids) >= max(1, int(max_grafts)):
                break
            sid = f"{pre_id}->{post_id}"
            if sid in self.synapses:
                syn = self.synapses[sid]
                syn.active = True
                if not syn.is_graft:
                    syn.is_graft = True
                    syn.delay_ticks = 2
                    syn.plasticity_trace = max(syn.plasticity_trace, 0.25)
                    if abs(syn.weight) < 0.15:
                        syn.weight = prior_w
                    graft_ids.append(sid)
            elif pre_id in self.neurons and post_id in self.neurons:
                syn = self.add_synapse(pre_id, post_id, weight=prior_w, enforce_limits=False)
                if syn is not None:
                    syn.is_graft = True
                    syn.delay_ticks = 2
                    syn.plasticity_trace = 0.25
                    graft_ids.append(syn.synapse_id)

        # 2. Cross-cluster hidden->hidden recurrence bridges
        candidate_pairs: List[Tuple[str, str, float]] = []
        for h1 in active_hidden:
            for h2 in active_hidden:
                if h1.neuron_id != h2.neuron_id and h1.cluster_id != h2.cluster_id:
                    sid = f"{h1.neuron_id}->{h2.neuron_id}"
                    if sid not in self.synapses:
                        score = abs(h1.a) + abs(h2.a) + 0.1
                        candidate_pairs.append((h1.neuron_id, h2.neuron_id, score))

        candidate_pairs.sort(key=lambda item: item[2], reverse=True)
        for pre_id, post_id, _ in candidate_pairs:
            if len(graft_ids) >= max(1, int(max_grafts)):
                break
            sign = 1.0 if self.neurons[pre_id].excitatory else -1.0
            syn = self.add_synapse(pre_id, post_id, weight=sign * 0.24, enforce_limits=True)
            if syn is not None:
                syn.is_graft = True
                syn.delay_ticks = 2
                syn.plasticity_trace = 0.25
                graft_ids.append(syn.synapse_id)

        if not graft_ids:
            for syn in sorted(self.active_synapses(), key=lambda s: abs(s.weight), reverse=True):
                if not syn.is_graft:
                    syn.is_graft = True
                    syn.delay_ticks = 2
                    graft_ids.append(syn.synapse_id)
                    if len(graft_ids) >= max(1, int(max_grafts)):
                        break

        if graft_ids:
            self.structural_events.append(
                {"tick": self.step_count, "type": "neurograft", "count": len(graft_ids)}
            )
        total_grafts = sum(1 for s in self.synapses.values() if s.active and s.is_graft)
        return {
            "grafts_added": len(graft_ids),
            "graft_ids": graft_ids,
            "total_grafts": total_grafts,
        }

    def reseed_fractal_topology(self, branch_factor: int = 2) -> Dict[str, Any]:
        """SONAPSE-FRACTAL hierarchical self-similar spatial & topological re-wiring."""
        hidden_nodes = [n for n in self.neurons.values() if n.role == "hidden"]
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for idx, n in enumerate(hidden_nodes):
            level = idx % 3
            radius = 1.55 + 0.85 * level
            theta = idx * golden_angle
            phi = (idx + 1) / float(max(1, len(hidden_nodes) + 1)) * math.pi
            n.pos_3d = [
                round(math.cos(phi) * 2.1, 4),
                round(radius * math.sin(theta), 4),
                round(radius * math.cos(theta), 4),
            ]
            n.cluster_id = level + 1

        bf = max(1, min(4, int(branch_factor)))
        added = 0
        for idx, src in enumerate(hidden_nodes):
            if not src.active:
                continue
            for step_scale in range(1, bf + 1):
                dst = hidden_nodes[(idx + (2 ** (step_scale - 1))) % len(hidden_nodes)]
                if dst.active and dst.neuron_id != src.neuron_id:
                    syn = self.add_synapse(src.neuron_id, dst.neuron_id, weight=0.32 / step_scale, enforce_limits=True)
                    if syn is not None:
                        syn.is_graft = True
                        added += 1

        self.structural_events.append(
            {"tick": self.step_count, "type": "fractal_reseed", "count": added}
        )
        return {
            "repositioned_hidden": len(hidden_nodes),
            "fractal_links_added": added,
            "branch_factor": bf,
        }

    def to_visualization_graph(self) -> Dict[str, Any]:
        """Return compact graph representation formatted for the 3D Connectome Explorer."""
        nodes = [n.to_dict() for n in self.neurons.values()]
        edges = [
            {
                "id": s.synapse_id,
                "source": s.pre_id,
                "target": s.post_id,
                "weight": round(s.weight, 4),
                "eligibility": round(s.eligibility, 4),
                "last_dw": round(s.last_dw, 6),
                "last_flux": round(s.last_flux, 5),
                "plasticity_trace": round(s.plasticity_trace, 5),
                "age": s.age,
                "active": s.active,
                "excitatory": s.weight >= 0,
                "is_graft": bool(s.is_graft),
                "delay_ticks": int(s.delay_ticks),
            }
            for s in self.synapses.values()
        ]
        return {
            "step_count": self.step_count,
            "total_synops": self.total_synops,
            "neuron_model": self.neuron_model,
            "density": round(self.current_density(), 5),
            "last_weight_delta_norm": round(self.last_weight_delta_norm, 6),
            "neuromodulators": {k: round(float(v), 4) for k, v in self.neuromodulators.items()},
            "nodes": nodes,
            "edges": edges,
            "recent_signals": self.recent_signals,
            "recent_plasticity_events": self.recent_plasticity_events[-30:],
            "plasticity_history": self.plasticity_history[-50:],
            "structural_events": self.structural_events[-25:],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "neuron_model": self.neuron_model,
            "w_max": self.w_max,
            "eligibility_decay": self.eligibility_decay,
            "max_neurons": self.max_neurons,
            "max_synapses": self.max_synapses,
            "max_density": self.max_density,
            "step_count": self.step_count,
            "total_synops": self.total_synops,
            "last_weight_delta_norm": self.last_weight_delta_norm,
            "neuromodulators": dict(self.neuromodulators),
            "recent_plasticity_events": list(self.recent_plasticity_events[-30:]),
            "plasticity_history": list(self.plasticity_history[-50:]),
            "neurons": {nid: n.to_dict() for nid, n in self.neurons.items()},
            "synapses": {sid: s.to_dict() for sid, s in self.synapses.items()},
            "structural_events": self.structural_events[-100:],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SonapseConnectome":
        conn = cls(
            input_dim=0,
            hidden_dim=0,
            output_dim=0,
            neuron_model=data.get("neuron_model", "rate_coded"),
            seed=data.get("seed", 42),
            w_max=data.get("w_max", 2.5),
            eligibility_decay=data.get("eligibility_decay", 0.7),
            max_neurons=data.get("max_neurons", 128),
            max_synapses=data.get("max_synapses", 512),
            max_density=data.get("max_density", 0.35),
        )
        conn.step_count = int(data.get("step_count", 0))
        conn.total_synops = int(data.get("total_synops", 0))
        conn.last_weight_delta_norm = float(data.get("last_weight_delta_norm", 0.0))
        if "neuromodulators" in data and isinstance(data["neuromodulators"], dict):
            conn.neuromodulators.update({k: float(v) for k, v in data["neuromodulators"].items()})
        conn.recent_plasticity_events = list(data.get("recent_plasticity_events", []))
        conn.plasticity_history = list(data.get("plasticity_history", []))
        conn.neurons = {nid: Neuron.from_dict(nd) for nid, nd in data["neurons"].items()}
        conn.synapses = {sid: Synapse.from_dict(sd) for sid, sd in data["synapses"].items()}
        conn.structural_events = list(data.get("structural_events", []))
        return conn
