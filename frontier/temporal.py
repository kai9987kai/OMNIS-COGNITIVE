"""Temporal frontier modules: Continuous Thought Machine (CTM) synchrony readouts & Kuramoto binding."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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

    def sequence_synchrony_features(self, seq: np.ndarray, entropy_threshold: float = 0.0) -> Tuple[np.ndarray, int]:
        """Roll hidden states over temporal frames with optional certainty halting and compute S = (Z Z^T) / T."""
        t_len = max(1, int(seq.shape[0]))
        z_hist = np.zeros((self.hidden_dim, t_len), dtype=float)
        h = np.zeros(self.hidden_dim, dtype=float)
        ticks_used = t_len
        for t in range(t_len):
            drive_in = self.w_in @ seq[t]
            pre = drive_in + (self.w_rec @ h) * 0.65
            h = self.nlm_decay * h + (1.0 - self.nlm_decay) * np.tanh(pre)
            z_hist[:, t] = h
            if entropy_threshold > 0.0 and t >= 3:
                sync_p = (z_hist[:, : t + 1] @ z_hist[:, : t + 1].T) / float(t + 1)
                s_p = np.array([sync_p[i, j] for i, j in self.sync_pairs], dtype=float)
                logits = self.w_sync @ s_p + self.b_out
                logits = logits - float(np.max(logits))
                probs = np.exp(logits)
                probs /= float(np.sum(probs) + 1e-12)
                ent = -float(np.sum(probs * np.log(probs + 1e-12)))
                if ent < entropy_threshold:
                    ticks_used = t + 1
                    break
        sync_mat = (z_hist[:, :ticks_used] @ z_hist[:, :ticks_used].T) / float(ticks_used)
        return np.array([sync_mat[i, j] for i, j in self.sync_pairs], dtype=float), ticks_used

    def fit_sequence_readout(self, seqs_train: np.ndarray, y_train: np.ndarray, reg: float = 1e-2) -> None:
        """Closed-form ridge fit of synchronization readout weights over temporal frame sequences."""
        feats = np.stack([self.sequence_synchrony_features(s)[0] for s in seqs_train], axis=0)
        n_feat = feats.shape[1]
        a_mat = feats.T @ feats + reg * np.eye(n_feat)
        y_onehot = np.eye(self.output_dim)[y_train]
        w_solved = np.linalg.solve(a_mat, feats.T @ y_onehot)
        self.w_sync = w_solved.T
        self.b_out = np.mean(y_onehot - feats @ w_solved, axis=0)

    def sequence_accuracy(self, seqs_eval: np.ndarray, y_eval: np.ndarray, entropy_threshold: float = 0.0) -> Tuple[float, float]:
        """Evaluate accuracy and mean compute ticks on temporal sequences."""
        correct = 0
        total_ticks = 0
        for seq, y in zip(seqs_eval, y_eval):
            s_vec, ticks = self.sequence_synchrony_features(seq, entropy_threshold=entropy_threshold)
            pred = int(np.argmax(self.w_sync @ s_vec + self.b_out))
            correct += int(pred == int(y))
            total_ticks += ticks
        n = max(1, len(y_eval))
        return float(correct / n), float(total_ticks / n)


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
    """Compare CTM temporal synchronization readout vs final-frame baseline and capacity-matched frame-averaged control on delayed XOR."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        n_train, n_eval, seq_len, dim = 140, 80, 6, 6

        def _make_delayed_xor_sequences(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
            seqs = rng.normal(0.0, 0.35, size=(n_samples, seq_len, dim))
            cue_a = rng.choice([-1.0, 1.0], size=n_samples)
            cue_b = rng.choice([-1.0, 1.0], size=n_samples)
            seqs[:, 0, 0] = cue_a * 2.2
            seqs[:, 0, 1] = -cue_a * 2.2
            seqs[:, 3, 2] = cue_b * 2.2
            seqs[:, 3, 3] = -cue_b * 2.2
            labels = ((cue_a * cue_b) > 0).astype(int)
            return seqs, labels

        s_tr, y_tr = _make_delayed_xor_sequences(n_train)
        s_ev, y_ev = _make_delayed_xor_sequences(n_eval)
        y_oh = np.eye(2)[y_tr]

        # Baseline: single-tick readout on final frame only (no temporal binding across t=0 and t=3)
        w_base = rng.normal(0.0, 0.5, size=(10, dim))
        h_tr = np.tanh(s_tr[:, -1, :] @ w_base.T)
        h_ev = np.tanh(s_ev[:, -1, :] @ w_base.T)
        w_b_out = np.linalg.solve(h_tr.T @ h_tr + 1e-2 * np.eye(10), h_tr.T @ y_oh)
        base_acc = float(np.mean(np.argmax(h_ev @ w_b_out, axis=1) == y_ev))

        # Candidate: Mini CTM rolling multi-timescale NLM state across frames with pairwise synchronization S = Z Z^T / T
        ctm = ContinuousThoughtMachine(input_dim=dim, hidden_dim=10, output_dim=2, thought_ticks=seq_len, seed=seed)
        ctm.fit_sequence_readout(s_tr, y_tr, reg=1e-2)
        cand_acc, mean_ticks = ctm.sequence_accuracy(s_ev, y_ev, entropy_threshold=0.0)

        # Capacity-matched control: processes every frame through wide MLP matching len(sync_pairs)=55 features,
        # pooled additively across time without recurrent pairwise temporal synchrony
        n_ctrl = len(ctm.sync_pairs)
        w_ctrl = rng.normal(0.0, 0.5, size=(n_ctrl, dim))
        hc_tr = np.mean(np.tanh(s_tr @ w_ctrl.T), axis=1)
        hc_ev = np.mean(np.tanh(s_ev @ w_ctrl.T), axis=1)
        w_c_out = np.linalg.solve(hc_tr.T @ hc_tr + 1e-2 * np.eye(n_ctrl), hc_tr.T @ y_oh)
        ctrl_acc = float(np.mean(np.argmax(hc_ev @ w_c_out, axis=1) == y_ev))

        return {
            "baseline": base_acc,
            "candidate": cand_acc,
            "capacity_matched_control": ctrl_acc,
            "details": {"seq_len": seq_len, "mean_ticks": round(mean_ticks, 2), "sync_feature_dim": n_ctrl},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="ctm_synchrony",
        category="Temporal",
        metric_name="delayed_xor_sequence_accuracy",
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
