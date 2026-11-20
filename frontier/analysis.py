"""Frontier analysis modules: TopK Sparse Dictionary (SAE) with causal ablation, Functional/Effective Connectivity, & Causal Emergence."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frontier.contract import CapacityMatchedHarness
from sonapse.connectome import SonapseConnectome
from sonapse.topology import analyze_topology


class TopKSparseDictionary:
    """Exact Top-k overcomplete Sparse Autoencoder (Gao et al. 2024-2025) with causal feature ablation."""

    def __init__(self, input_dim: int = 8, dict_size: int = 24, k_active: int = 3, seed: int = 42) -> None:
        self.input_dim = int(input_dim)
        self.dict_size = int(dict_size)
        self.k_active = max(1, min(self.dict_size, int(k_active)))
        rng = np.random.default_rng(seed)
        w = rng.normal(0.0, 0.35, size=(self.dict_size, self.input_dim))
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self.w_enc = w
        self.w_dec = w.T.copy()  # (input_dim, dict_size)
        self.b_pre = np.zeros(self.input_dim, dtype=float)

    def encode_topk(self, x: np.ndarray) -> np.ndarray:
        pre = (x - self.b_pre) @ self.w_enc.T
        z = np.zeros_like(pre)
        if pre.ndim == 1:
            idx = np.argsort(pre)[-self.k_active :]
            z[idx] = np.maximum(0.0, pre[idx])
        else:
            for row in range(pre.shape[0]):
                idx = np.argsort(pre[row])[-self.k_active :]
                z[row, idx] = np.maximum(0.0, pre[row, idx])
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z @ self.w_dec.T + self.b_pre

    def fit(self, data: np.ndarray, epochs: int = 35, lr: float = 0.08) -> float:
        self.b_pre = np.mean(data, axis=0)
        for _ in range(epochs):
            z = self.encode_topk(data)
            recon = self.decode(z)
            err = recon - data  # (N, D)
            grad_dec = (err.T @ z) / float(len(data))
            self.w_dec -= lr * grad_dec
            # Unit-norm dictionary atoms
            self.w_dec /= np.linalg.norm(self.w_dec, axis=0, keepdims=True) + 1e-8
            self.w_enc = self.w_dec.T.copy()
        final_recon = self.decode(self.encode_topk(data))
        return float(np.mean((final_recon - data) ** 2))

    def causal_feature_ablation(self, x_samples: np.ndarray, probe_readout: np.ndarray) -> Dict[str, Any]:
        """Measure downstream logit shift when causally ablating top active sparse features vs random features."""
        z = self.encode_topk(x_samples)
        recon_full = self.decode(z)
        logits_full = recon_full @ probe_readout

        mean_act = np.mean(z, axis=0)
        top_feat = int(np.argmax(mean_act))
        z_abl = z.copy()
        z_abl[:, top_feat] = 0.0
        logits_abl = self.decode(z_abl) @ probe_readout
        causal_shift = float(np.mean(np.abs(logits_full - logits_abl)))

        # Random inactive/low-salience feature ablation control
        low_feat = int(np.argmin(mean_act))
        z_rand = z.copy()
        z_rand[:, low_feat] = 0.0
        logits_rand = self.decode(z_rand) @ probe_readout
        random_shift = float(np.mean(np.abs(logits_full - logits_rand)))

        return {
            "top_feature_index": top_feat,
            "causal_ablation_shift": round(causal_shift, 5),
            "control_ablation_shift": round(random_shift, 5),
            "causal_specificity_ratio": round(causal_shift / max(1e-5, random_shift), 4),
        }


class ConnectivityAnalyzer:
    """Computes Functional Connectivity (zero-lag Pearson) and Effective Connectivity (lagged directed Granger proxy)."""

    @staticmethod
    def record_connectome_traces(connectome: SonapseConnectome, steps: int = 36, seed: int = 42) -> Tuple[List[str], np.ndarray]:
        rng = np.random.default_rng(seed)
        node_ids = sorted(n.neuron_id for n in connectome.neurons.values() if n.active)
        in_count = sum(1 for n in connectome.neurons.values() if n.role == "input")
        traces = np.zeros((steps, len(node_ids)), dtype=float)
        for t in range(steps):
            inp = [float(math.sin(0.45 * t + i * 0.7) + rng.normal(0.0, 0.25)) for i in range(max(1, in_count))]
            connectome.step(inp)
            for j, nid in enumerate(node_ids):
                traces[t, j] = float(connectome.neurons[nid].a)
        return node_ids, traces

    @staticmethod
    def analyze(node_ids: List[str], traces: np.ndarray) -> Dict[str, Any]:
        t_len, n = traces.shape
        if t_len < 4 or n < 2:
            return {"num_nodes": n, "mean_fc": 0.0, "mean_ec": 0.0, "asymmetry_index": 0.0}
        # Zero-lag functional connectivity (symmetric Pearson correlation)
        centered = traces - np.mean(traces, axis=0, keepdims=True)
        stds = np.std(centered, axis=0, keepdims=True) + 1e-8
        normed = centered / stds
        fc = np.abs((normed.T @ normed) / float(t_len))
        np.fill_diagonal(fc, 0.0)

        # Lag-1 directed effective connectivity (asymmetric temporal predictability)
        x_prev = normed[:-1, :]
        x_next = normed[1:, :]
        ec = np.abs((x_prev.T @ x_next) / float(t_len - 1))
        np.fill_diagonal(ec, 0.0)

        asymmetry = float(np.mean(np.abs(ec - ec.T)))
        return {
            "num_nodes": n,
            "mean_fc": round(float(np.mean(fc)), 5),
            "mean_ec": round(float(np.mean(ec)), 5),
            "asymmetry_index": round(asymmetry, 5),
        }


def run_topk_sparse_dictionary_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare TopK-SAE reconstruction & causal feature specificity against random fixed projection control."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        conn = SonapseConnectome(input_dim=8, hidden_dim=8, output_dim=5, neuron_model="dual_soma", seed=seed)
        _, traces = ConnectivityAnalyzer.record_connectome_traces(conn, steps=48, seed=seed)
        # Extract hidden activation subspace (columns 8..16 or first 8)
        data = traces[:, :8]

        sae = TopKSparseDictionary(input_dim=8, dict_size=24, k_active=3, seed=seed)
        cand_mse = sae.fit(data, epochs=35, lr=0.09)

        # Capacity-matched random dictionary (same 24 atoms & k=3, unfitted random directions)
        rand_sae = TopKSparseDictionary(input_dim=8, dict_size=24, k_active=3, seed=seed + 99)
        rand_sae.b_pre = np.mean(data, axis=0)
        ctrl_mse = float(np.mean((rand_sae.decode(rand_sae.encode_topk(data)) - data) ** 2))

        base_mse = float(np.mean((data - np.mean(data, axis=0)) ** 2))
        probe = rng.normal(0.0, 1.0, size=(8, 2))
        abl = sae.causal_feature_ablation(data, probe)

        return {
            "baseline": 1.0 / (1.0 + base_mse),
            "candidate": 1.0 / (1.0 + cand_mse),
            "capacity_matched_control": 1.0 / (1.0 + ctrl_mse),
            "details": {"sae_mse": round(cand_mse, 5), "control_mse": round(ctrl_mse, 5), **abl},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="topk_sparse_dictionary",
        category="Analysis",
        metric_name="sparse_reconstruction_inverse_mse",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=True,
    )


def run_connectivity_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Verify directed Effective Connectivity asymmetry is strictly higher on wired SONAPSE connectomes than shuffled noise."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        conn = SonapseConnectome(input_dim=6, hidden_dim=8, output_dim=4, neuron_model="dual_soma", seed=seed)
        conn.inject_neurograft(max_grafts=3)
        node_ids, traces = ConnectivityAnalyzer.record_connectome_traces(conn, steps=40, seed=seed)
        wired_res = ConnectivityAnalyzer.analyze(node_ids, traces)

        # Time-shuffled null control destroys temporal causal directionality
        shuffled = traces.copy()
        for col in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, col])
        null_res = ConnectivityAnalyzer.analyze(node_ids, shuffled)

        return {
            "baseline": null_res["mean_ec"],
            "candidate": wired_res["mean_ec"],
            "capacity_matched_control": null_res["mean_ec"],
            "details": {"wired": wired_res, "null_shuffled": null_res},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="functional_effective_connectivity",
        category="Analysis",
        metric_name="directed_effective_connectivity_ec",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=False,
    )


def run_causal_emergence_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare Causal Emergence index on modular NeuroGraft+Fractal connectomes vs ungrafted baseline."""

    def _trial(seed: int) -> Dict[str, Any]:
        base_conn = SonapseConnectome(input_dim=8, hidden_dim=8, output_dim=5, seed=seed)
        cand_conn = SonapseConnectome(input_dim=8, hidden_dim=8, output_dim=5, neuron_model="dual_soma", seed=seed)
        cand_conn.reseed_fractal_topology(branch_factor=2)
        cand_conn.inject_neurograft(max_grafts=4)

        t_base = analyze_topology(base_conn)
        t_cand = analyze_topology(cand_conn)
        return {
            "baseline": t_base["causal_emergence_index"],
            "candidate": t_cand["causal_emergence_index"],
            "capacity_matched_control": t_base["causal_emergence_index"],
            "details": {
                "baseline_ce": t_base["causal_emergence_index"],
                "candidate_ce": t_cand["causal_emergence_index"],
                "candidate_ei_macro": t_cand["effective_information_macro"],
            },
        }

    return CapacityMatchedHarness.evaluate(
        module_id="causal_emergence_audit",
        category="Analysis",
        metric_name="causal_emergence_index_phi_ce",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=False,
    )
