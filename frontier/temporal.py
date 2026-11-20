"""Temporal frontier modules: Continuous Thought Machine (CTM) synchrony readouts & Kuramoto binding."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from frontier.contract import CapacityMatchedHarness


class ContinuousThoughtMachine:
    """Mini Continuous Thought Machine (Sakana AI 2025) with pairwise synchronization readout S_t = Z_t Z_t^T."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 12,
        output_dim: int = 2,
        thought_ticks: int = 5,
        seed: int = 42,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.thought_ticks = max(2, int(thought_ticks))
        rng = np.random.default_rng(seed)

        self.w_in = rng.normal(0.0, 0.45, size=(self.hidden_dim, self.input_dim))
        self.w_rec = rng.normal(0.0, 0.35, size=(self.hidden_dim, self.hidden_dim))
        # Per-neuron temporal history weights (Neuron-Level Models)
        self.nlm_decay = rng.uniform(0.35, 0.85, size=(self.hidden_dim,))
        # Synchronisation readout maps upper-triangular pairwise synchrony to logits
        self.sync_pairs = [(i, j) for i in range(self.hidden_dim) for j in range(i, self.hidden_dim)]
        self.w_sync = rng.normal(0.0, 0.25, size=(self.output_dim, len(self.sync_pairs)))
        self.b_out = np.zeros(self.output_dim, dtype=float)

    def synchrony_features(self, x: np.ndarray) -> np.ndarray:
        """Roll out internal thought ticks and compute temporal pairwise synchronization S = (Z Z^T) / T."""
        z_hist = np.zeros((self.hidden_dim, self.thought_ticks), dtype=float)
        h = np.zeros(self.hidden_dim, dtype=float)
        drive_in = self.w_in @ x
        for t in range(self.thought_ticks):
            # Neuron-level recursive temporal mixing
            pre = drive_in + (self.w_rec @ h) * 0.55
            h = self.nlm_decay * h + (1.0 - self.nlm_decay) * np.tanh(pre + 0.25 * math.sin(0.7 * (t + 1)))
            z_hist[:, t] = h
        sync_mat = (z_hist @ z_hist.T) / float(self.thought_ticks)
        return np.array([sync_mat[i, j] for i, j in self.sync_pairs], dtype=float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        s_vec = self.synchrony_features(x)
        return self.w_sync @ s_vec + self.b_out

    def fit_readout(self, x_train: np.ndarray, y_train: np.ndarray, reg: float = 1e-2) -> None:
        """Closed-form ridge fit of synchronization readout weights on training episodes."""
        feats = np.stack([self.synchrony_features(x) for x in x_train], axis=0)
        n_feat = feats.shape[1]
        a_mat = feats.T @ feats + reg * np.eye(n_feat)
        y_onehot = np.eye(self.output_dim)[y_train]
        w_solved = np.linalg.solve(a_mat, feats.T @ y_onehot)
        self.w_sync = w_solved.T
        self.b_out = np.mean(y_onehot - feats @ w_solved, axis=0)

    def accuracy(self, x_eval: np.ndarray, y_eval: np.ndarray) -> float:
        preds = [int(np.argmax(self.forward(x))) for x in x_eval]
        return float(sum(p == int(y) for p, y in zip(preds, y_eval)) / max(1, len(y_eval)))


class KuramotoOscillatorBank:
    """Coupled Kuramoto phase oscillators wired to a SONAPSE connectome for phase-coherence binding."""

    def __init__(
        self,
        num_oscillators: int = 16,
        coupling_k: float = 2.4,
        dt: float = 0.08,
        seed: int = 42,
    ) -> None:
        self.n = max(4, int(num_oscillators))
        self.coupling_k = float(coupling_k)
        self.dt = float(dt)
        rng = np.random.default_rng(seed)
        self.omega = rng.normal(1.0, 0.22, size=self.n)
        self.theta = rng.uniform(0.0, 2.0 * math.pi, size=self.n)
        self.adj = np.ones((self.n, self.n), dtype=float) - np.eye(self.n)

    def bind_connectome(self, connectome: Any) -> None:
        """Import live neuron count and normalized synaptic coupling matrix from a SonapseConnectome."""
        nodes = sorted(n.neuron_id for n in connectome.neurons.values() if n.active)
        if len(nodes) < 2:
            return
        idx = {nid: i for i, nid in enumerate(nodes)}
        n = len(nodes)
        self.n = n
        rng = np.random.default_rng(getattr(connectome, "seed", 42))
        self.omega = np.array(
            [1.0 + 0.15 * float(connectome.neurons[nid].cluster_id) + float(rng.normal(0.0, 0.08)) for nid in nodes],
            dtype=float,
        )
        self.theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
        mat = np.zeros((n, n), dtype=float)
        for syn in connectome.active_synapses():
            if syn.pre_id in idx and syn.post_id in idx:
                w = abs(float(syn.weight))
                mat[idx[syn.post_id], idx[syn.pre_id]] += w
                mat[idx[syn.pre_id], idx[syn.post_id]] += 0.5 * w
        # Weak mean-field background coupling preserves phase coherence across modular bridges
        mat += 0.15 * (np.ones((n, n)) - np.eye(n))
        row_sums = np.maximum(1e-6, mat.sum(axis=1, keepdims=True))
        self.adj = mat / row_sums

    def step(self, steps: int = 1) -> Dict[str, Any]:
        for _ in range(max(1, int(steps))):
            diff = self.theta[None, :] - self.theta[:, None]
            coupling_term = np.sum(self.adj * np.sin(diff), axis=1)
            dtheta = self.omega + self.coupling_k * coupling_term
            self.theta = np.mod(self.theta + self.dt * dtheta, 2.0 * math.pi)
        return self.measure()

    def measure(self) -> Dict[str, Any]:
        complex_mean = np.mean(np.exp(1j * self.theta))
        order_r = float(np.abs(complex_mean))
        mean_psi = float(np.angle(complex_mean))
        # Phase-locking value (PLV) on top coupled pairs
        diffs = self.theta[None, :] - self.theta[:, None]
        plv = float(np.mean(np.cos(diffs)))
        return {
            "num_oscillators": self.n,
            "coupling_k": round(self.coupling_k, 3),
            "order_parameter_R": round(order_r, 5),
            "mean_phase_rad": round(mean_psi, 5),
            "phase_locking_index": round(max(0.0, plv), 5),
            "phases": [round(float(p), 4) for p in self.theta.tolist()],
        }


def run_ctm_synchrony_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare CTM synchronization readout vs single-tick baseline and capacity-matched flat readout on nonlinear parity."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        n_train, n_eval = 140, 80
        x_tr = rng.choice([-1.0, 1.0], size=(n_train, 6))
        y_tr = (((x_tr[:, 0] * x_tr[:, 1] * x_tr[:, 2]) + 0.5 * (x_tr[:, 3] * x_tr[:, 4])) > 0).astype(int)
        x_ev = rng.choice([-1.0, 1.0], size=(n_eval, 6))
        y_ev = (((x_ev[:, 0] * x_ev[:, 1] * x_ev[:, 2]) + 0.5 * (x_ev[:, 3] * x_ev[:, 4])) > 0).astype(int)

        # Baseline: 1-tick linear readout (no synchronization)
        w_base = rng.normal(0.0, 0.5, size=(10, 6))
        h_tr = np.tanh(x_tr @ w_base.T)
        h_ev = np.tanh(x_ev @ w_base.T)
        y_oh = np.eye(2)[y_tr]
        w_b_out = np.linalg.solve(h_tr.T @ h_tr + 1e-2 * np.eye(10), h_tr.T @ y_oh)
        base_acc = float(np.mean(np.argmax(h_ev @ w_b_out, axis=1) == y_ev))

        # Candidate: Mini CTM with 5 internal thought ticks and pairwise synchronization readout S = Z Z^T
        ctm = ContinuousThoughtMachine(input_dim=6, hidden_dim=10, output_dim=2, thought_ticks=5, seed=seed)
        ctm.fit_readout(x_tr, y_tr, reg=1e-2)
        cand_acc = ctm.accuracy(x_ev, y_ev)

        # Capacity-matched control: single-tick wide random feature MLP matching len(sync_pairs) = 55 features
        n_ctrl = len(ctm.sync_pairs)
        w_ctrl = rng.normal(0.0, 0.5, size=(n_ctrl, 6))
        hc_tr = np.tanh(x_tr @ w_ctrl.T)
        hc_ev = np.tanh(x_ev @ w_ctrl.T)
        w_c_out = np.linalg.solve(hc_tr.T @ hc_tr + 1e-2 * np.eye(n_ctrl), hc_tr.T @ y_oh)
        ctrl_acc = float(np.mean(np.argmax(hc_ev @ w_c_out, axis=1) == y_ev))

        return {
            "baseline": base_acc,
            "candidate": cand_acc,
            "capacity_matched_control": ctrl_acc,
            "details": {"thought_ticks": 5, "sync_feature_dim": n_ctrl},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="ctm_synchrony",
        category="Temporal",
        metric_name="held_out_parity_accuracy",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=True,
    )


def run_kuramoto_binding_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare coupled Kuramoto phase coherence R(t) against uncoupled baseline (K=0)."""

    def _trial(seed: int) -> Dict[str, Any]:
        uncoupled = KuramotoOscillatorBank(num_oscillators=18, coupling_k=0.0, dt=0.08, seed=seed)
        coupled = KuramotoOscillatorBank(num_oscillators=18, coupling_k=2.6, dt=0.08, seed=seed)
        res_base = uncoupled.step(steps=45)
        res_cand = coupled.step(steps=45)
        return {
            "baseline": res_base["order_parameter_R"],
            "candidate": res_cand["order_parameter_R"],
            "capacity_matched_control": res_base["order_parameter_R"],
            "details": res_cand,
        }

    return CapacityMatchedHarness.evaluate(
        module_id="kuramoto_binding",
        category="Temporal",
        metric_name="kuramoto_order_parameter_R",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=False,
    )
