"""Frontier optimizers: Muon (Newton-Schulz Orthogonalized Momentum) vs Adam & RigL Dynamic Sparse Training."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from frontier.contract import CapacityMatchedHarness


def newton_schulz_orthogonalize(g: np.ndarray, steps: int = 5) -> np.ndarray:
    """5-step quintic Newton-Schulz orthogonalization (Keller Jordan Muon optimizer)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = np.asarray(g, dtype=float)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    norm = float(np.linalg.norm(x))
    if norm < 1e-8:
        return g.copy()
    x = x / (norm + 1e-7)
    for _ in range(max(1, int(steps))):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    return x.T if transposed else x


class MuonOptimizer:
    """MomentUm Orthogonalized by Newton-Schulz (with Nesterov lookahead) for 2D neural-graft matrices."""

    def __init__(self, shape: tuple, lr: float = 0.08, momentum: float = 0.9, ns_steps: int = 5, nesterov: bool = True) -> None:
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        self.nesterov = bool(nesterov)
        self.buf = np.zeros(shape, dtype=float)

    def step(self, weights: np.ndarray, grad: np.ndarray, lr_scale: float = 1.0) -> np.ndarray:
        self.buf = self.momentum * self.buf + grad
        update_g = (grad + self.momentum * self.buf) if self.nesterov else self.buf
        ortho = newton_schulz_orthogonalize(update_g, steps=self.ns_steps)
        scale = math.sqrt(max(1.0, weights.shape[0] / float(max(1, weights.shape[1]))))
        return weights - (self.lr * float(lr_scale)) * scale * ortho


class RigLSparseTrainer:
    """Rigging the Lottery (RigL) dynamic sparse training maintaining exact sparsity budget."""

    def __init__(self, shape: tuple, sparsity: float = 0.70, lr: float = 0.12, seed: int = 42) -> None:
        self.shape = shape
        self.sparsity = float(sparsity)
        self.lr = float(lr)
        rng = np.random.default_rng(seed)
        total = int(shape[0] * shape[1])
        self.k_active = max(2, int(round(total * (1.0 - self.sparsity))))
        self.mask = np.zeros(total, dtype=bool)
        self.mask[rng.choice(total, size=self.k_active, replace=False)] = True
        self.mask = self.mask.reshape(shape)
        self.weights = rng.normal(0.0, 0.2, size=shape) * self.mask

    def train(self, x_data: np.ndarray, y_data: np.ndarray, epochs: int = 30, dynamic_rewire: bool = True) -> float:
        for ep in range(epochs):
            pred = x_data @ self.weights.T
            err = pred - y_data
            grad = (err.T @ x_data) / float(len(x_data))
            self.weights -= self.lr * (grad * self.mask)

            if dynamic_rewire and ep > 0 and ep % 5 == 0 and ep < epochs - 3:
                drop_frac = 0.30 * (0.5 * (1.0 + math.cos(math.pi * ep / float(epochs))))
                n_swap = max(1, int(round(self.k_active * drop_frac)))
                flat_w = np.abs(self.weights).reshape(-1)
                flat_g = np.abs(grad).reshape(-1)
                flat_m = self.mask.reshape(-1)

                active_idx = np.where(flat_m)[0]
                inactive_idx = np.where(~flat_m)[0]
                if len(active_idx) >= n_swap and len(inactive_idx) >= n_swap:
                    prune_idx = active_idx[np.argsort(flat_w[active_idx])[:n_swap]]
                    grow_idx = inactive_idx[np.argsort(flat_g[inactive_idx])[-n_swap:]]
                    flat_m[prune_idx] = False
                    flat_m[grow_idx] = True
                    self.mask = flat_m.reshape(self.shape)
                    self.weights *= self.mask

        final_pred = x_data @ self.weights.T
        return float(np.mean((final_pred - y_data) ** 2))


def run_muon_vs_adam_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare Nesterov Muon orthogonalized updates vs tuned Adam on non-axis-aligned ill-conditioned 2D graft weights."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        d_in, d_out = 16, 10
        # Non-axis-aligned ill-conditioned activations & spectral target typical inside 2D neural layers
        q_x, _ = np.linalg.qr(rng.normal(size=(d_in, d_in)))
        x = (rng.normal(0.0, 1.0, size=(120, d_in)) * np.geomspace(2.2, 0.12, num=d_in)) @ q_x.T
        u_m, _ = np.linalg.qr(rng.normal(size=(d_out, d_out)))
        v_m, _ = np.linalg.qr(rng.normal(size=(d_in, d_in)))
        sing = np.geomspace(3.0, 0.2, num=d_out)
        w_true = u_m @ np.diag(sing) @ v_m[:d_out, :]
        y = x @ w_true.T

        w_init = rng.normal(0.0, 0.1, size=(d_out, d_in))
        steps = 25

        # Baseline: tuned Adam optimizer (diagonal adaptive preconditioning)
        w_adam = w_init.copy()
        m_a = np.zeros_like(w_adam)
        v_a = np.zeros_like(w_adam)
        for t in range(1, steps + 1):
            g = ((x @ w_adam.T - y).T @ x) / float(len(x))
            m_a = 0.9 * m_a + 0.1 * g
            v_a = 0.999 * v_a + 0.001 * (g * g)
            m_hat = m_a / (1.0 - 0.9 ** t)
            v_hat = v_a / (1.0 - 0.999 ** t)
            w_adam -= 0.12 * m_hat / (np.sqrt(v_hat) + 1e-8)
        adam_loss = float(np.mean((x @ w_adam.T - y) ** 2))

        # Candidate: Muon optimizer with Nesterov lookahead and cosine step-size decay
        w_muon = w_init.copy()
        opt = MuonOptimizer(shape=w_muon.shape, lr=0.35, momentum=0.85, ns_steps=5, nesterov=True)
        for t in range(steps):
            g = ((x @ w_muon.T - y).T @ x) / float(len(x))
            lr_s = max(0.02, 0.5 * (1.0 + math.cos(math.pi * t / float(steps))))
            w_muon = opt.step(w_muon, g, lr_scale=lr_s)
        muon_loss = float(np.mean((x @ w_muon.T - y) ** 2))

        return {
            "baseline": 1.0 / (1.0 + adam_loss),
            "candidate": 1.0 / (1.0 + muon_loss),
            "capacity_matched_control": 1.0 / (1.0 + adam_loss),
            "details": {"adam_mse": round(adam_loss, 5), "muon_mse": round(muon_loss, 5), "steps": steps},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="muon_optimizer",
        category="Optimisers",
        metric_name="graft_convergence_inverse_mse",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=False,
    )


def run_rigl_sparse_training_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare RigL dynamic sparse training vs capacity-matched static random sparse mask at 70% sparsity."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        d_in, d_out = 16, 8
        # Sparse ground-truth connectome target (30% nonzero)
        true_mask = rng.random((d_out, d_in)) < 0.30
        w_true = rng.normal(0.0, 0.9, size=(d_out, d_in)) * true_mask
        x = rng.normal(0.0, 1.0, size=(90, d_in))
        y = x @ w_true.T

        rigl = RigLSparseTrainer(shape=(d_out, d_in), sparsity=0.70, lr=0.14, seed=seed)
        static_ctrl = RigLSparseTrainer(shape=(d_out, d_in), sparsity=0.70, lr=0.14, seed=seed)

        cand_mse = rigl.train(x, y, epochs=36, dynamic_rewire=True)
        ctrl_mse = static_ctrl.train(x, y, epochs=36, dynamic_rewire=False)
        base_mse = float(np.mean(y ** 2))

        return {
            "baseline": 1.0 / (1.0 + base_mse),
            "candidate": 1.0 / (1.0 + cand_mse),
            "capacity_matched_control": 1.0 / (1.0 + ctrl_mse),
            "details": {"sparsity": 0.70, "rigl_mse": round(cand_mse, 5), "static_sparse_mse": round(ctrl_mse, 5)},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="rigl_sparse_training",
        category="Optimisers",
        metric_name="sparse_inverse_mse_70pct",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=True,
    )
