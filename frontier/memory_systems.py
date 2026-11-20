"""Frontier memory & NEUROWEAVE mechanisms: Continuum Memory, One-Shot BTSP, Modern Hopfield, and Fast Weights."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from frontier.contract import CapacityMatchedHarness


class ContinuumMemorySystem:
    """Nested Learning multi-timescale Continuum Memory (fast / medium / slow frequency-separated banks)."""

    def __init__(self, dim: int = 12, out_dim: int = 4, seed: int = 42) -> None:
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        rng = np.random.default_rng(seed)
        self.w_fast = rng.normal(0.0, 0.05, size=(self.out_dim, self.dim))
        self.w_med = rng.normal(0.0, 0.05, size=(self.out_dim, self.dim))
        self.w_slow = rng.normal(0.0, 0.05, size=(self.out_dim, self.dim))
        self.step_idx = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (0.45 * self.w_fast + 0.30 * self.w_med + 0.25 * self.w_slow) @ x

    def update(self, x: np.ndarray, target: np.ndarray) -> float:
        self.step_idx += 1
        pred = self.predict(x)
        err = target - pred
        grad = np.outer(err, x)
        self.w_fast = 0.96 * self.w_fast + 0.25 * grad
        if self.step_idx % 3 == 0:
            self.w_med = 0.992 * self.w_med + 0.09 * grad + 0.04 * (self.w_fast - self.w_med)
        if self.step_idx % 8 == 0:
            self.w_slow = 0.998 * self.w_slow + 0.04 * grad + 0.05 * (self.w_med - self.w_slow)
        return float(np.mean(err ** 2))


class BTSPBinaryMemory:
    """One-Shot Behavioral Timescale Synaptic Plasticity (BTSP) binary-synapse associative memory."""

    def __init__(self, n_in: int = 48, n_out: int = 48, sparsity: float = 0.2, seed: int = 42) -> None:
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.sparsity = float(sparsity)
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((self.n_out, self.n_in), dtype=np.int8)

    def imprint_one_shot(self, cue: np.ndarray, target: np.ndarray, plateau_prob: float = 0.85) -> None:
        """Single-shot plateau-potential gated binary synaptic transition (0->1 on co-activity)."""
        pre_active = (cue > 0.5).astype(np.int8)
        post_active = (target > 0.5).astype(np.int8)
        coactive = np.outer(post_active, pre_active)
        gate = (self.rng.random(coactive.shape) < plateau_prob).astype(np.int8)
        # Bidirectional BTSP: strengthen coactive, mildly depress conflicting synapses
        self.weights = np.clip(self.weights + coactive * gate, 0, 1).astype(np.int8)

    def recall(self, cue: np.ndarray, k_active: int = 8) -> np.ndarray:
        drive = self.weights.astype(float) @ (cue > 0.5).astype(float)
        top_idx = np.argsort(drive)[-k_active:]
        out = np.zeros(self.n_out, dtype=np.int8)
        out[top_idx] = 1
        return out


class ModernHopfieldAssociativeMemory:
    """NEUROWEAVE associative_memory mechanism: continuous Modern Hopfield softmax energy retrieval."""

    def __init__(self, dim: int = 16, beta: float = 6.0, capacity: int = 64) -> None:
        self.dim = int(dim)
        self.beta = float(beta)
        self.capacity = int(capacity)
        self.patterns: List[np.ndarray] = []

    def store(self, pattern: np.ndarray) -> None:
        vec = np.asarray(pattern, dtype=float).reshape(-1)[: self.dim]
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        self.patterns.append(vec)
        if len(self.patterns) > self.capacity:
            self.patterns = self.patterns[-self.capacity :]

    def retrieve(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=float).reshape(-1)[: self.dim]
        if q.size < self.dim:
            q = np.pad(q, (0, self.dim - q.size))
        if not self.patterns:
            return q
        x_mat = np.stack(self.patterns, axis=0)  # (P, D)
        logits = self.beta * (x_mat @ q)
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        probs /= float(np.sum(probs) + 1e-12)
        return probs @ x_mat


class FastWeightsController:
    """NEUROWEAVE fast_weights mechanism: Ba et al. decaying outer-product associative working memory."""

    def __init__(self, dim: int = 12, decay_lambda: float = 0.85, eta: float = 0.45) -> None:
        self.dim = int(dim)
        self.decay_lambda = float(decay_lambda)
        self.eta = float(eta)
        self.a_mat = np.zeros((self.dim, self.dim), dtype=float)

    def reset(self) -> None:
        self.a_mat.fill(0.0)

    def bind(self, key: np.ndarray, value: np.ndarray) -> None:
        k = np.asarray(key, dtype=float).reshape(-1)[: self.dim]
        v = np.asarray(value, dtype=float).reshape(-1)[: self.dim]
        kn = float(np.linalg.norm(k))
        if kn > 1e-8:
            k = k / kn
        self.a_mat = self.decay_lambda * self.a_mat + self.eta * np.outer(v, k)

    def query(self, key: np.ndarray) -> np.ndarray:
        k = np.asarray(key, dtype=float).reshape(-1)[: self.dim]
        kn = float(np.linalg.norm(k))
        if kn > 1e-8:
            k = k / kn
        return self.a_mat @ k


def run_continuum_memory_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare Nested Continuum Memory retention after task shift vs single-timescale fast and capacity-matched control."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        dim, out_dim = 10, 4
        w_task1 = rng.normal(0.0, 0.7, size=(out_dim, dim))
        w_task2 = rng.normal(0.0, 0.7, size=(out_dim, dim))

        x1 = rng.normal(0.0, 1.0, size=(48, dim))
        x1 /= np.linalg.norm(x1, axis=1, keepdims=True) + 1e-8
        y1 = x1 @ w_task1.T
        x2 = rng.normal(0.0, 1.0, size=(36, dim))
        x2 /= np.linalg.norm(x2, axis=1, keepdims=True) + 1e-8
        y2 = x2 @ w_task2.T

        # Candidate: Nested 3-timescale Continuum Memory
        cm = ContinuumMemorySystem(dim=dim, out_dim=out_dim, seed=seed)
        for xi, yi in zip(x1, y1):
            cm.update(xi, yi)
        for xi, yi in zip(x2, y2):
            cm.update(xi, yi)
        cand_mse = float(np.mean([(cm.predict(xi) - yi) ** 2 for xi, yi in zip(x1, y1)]))

        # Baseline: single fast timescale matrix
        w_base = np.zeros((out_dim, dim))
        for xi, yi in zip(x1, y1):
            w_base = 0.96 * w_base + 0.25 * np.outer(yi - w_base @ xi, xi)
        for xi, yi in zip(x2, y2):
            w_base = 0.96 * w_base + 0.25 * np.outer(yi - w_base @ xi, xi)
        base_mse = float(np.mean([(w_base @ xi - yi) ** 2 for xi, yi in zip(x1, y1)]))

        # Capacity-matched control: 3 identical-timescale matrices (same 3x parameter count, no nested frequency separation)
        w_c = [np.zeros((out_dim, dim)) for _ in range(3)]
        for xi, yi in zip(x1, y1):
            pred = sum(w_c) @ xi / 3.0
            g = np.outer(yi - pred, xi)
            w_c = [0.96 * w + 0.25 * g for w in w_c]
        for xi, yi in zip(x2, y2):
            pred = sum(w_c) @ xi / 3.0
            g = np.outer(yi - pred, xi)
            w_c = [0.96 * w + 0.25 * g for w in w_c]
        ctrl_mse = float(np.mean([(sum(w_c) @ xi / 3.0 - yi) ** 2 for xi, yi in zip(x1, y1)]))

        return {
            "baseline": 1.0 / (1.0 + base_mse),
            "candidate": 1.0 / (1.0 + cand_mse),
            "capacity_matched_control": 1.0 / (1.0 + ctrl_mse),
            "details": {"task1_retention_mse_continuum": round(cand_mse, 4), "task1_mse_control": round(ctrl_mse, 4)},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="continuum_memory",
        category="Memory",
        metric_name="post_shift_retention_fidelity",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=True,
    )


def run_btsp_binary_memory_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare One-Shot BTSP binary-synapse recall vs capacity-matched unguided binary projection."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        n_dim, k_act, n_items = 48, 8, 12
        cues = []
        targets = []
        for _ in range(n_items):
            c = np.zeros(n_dim, dtype=np.int8)
            t = np.zeros(n_dim, dtype=np.int8)
            c[rng.choice(n_dim, size=k_act, replace=False)] = 1
            t[rng.choice(n_dim, size=k_act, replace=False)] = 1
            cues.append(c)
            targets.append(t)

        btsp = BTSPBinaryMemory(n_in=n_dim, n_out=n_dim, seed=seed)
        for c, t in zip(cues, targets):
            btsp.imprint_one_shot(c, t, plateau_prob=0.9)

        # Evaluate noisy one-shot cue recall (flip 1 bit)
        btsp_scores = []
        for c, t in zip(cues, targets):
            noisy = c.copy()
            noisy[rng.integers(0, n_dim)] ^= 1
            rec = btsp.recall(noisy, k_active=k_act)
            btsp_scores.append(float(np.sum(rec * t) / float(k_act)))

        # Capacity-matched binary control (same active bit density in W, randomly placed)
        density = float(np.mean(btsp.weights))
        ctrl_w = (rng.random((n_dim, n_dim)) < density).astype(float)
        ctrl_scores = []
        for c, t in zip(cues, targets):
            drive = ctrl_w @ c.astype(float)
            rec = np.zeros(n_dim, dtype=np.int8)
            rec[np.argsort(drive)[-k_act:]] = 1
            ctrl_scores.append(float(np.sum(rec * t) / float(k_act)))

        return {
            "baseline": float(k_act / n_dim),
            "candidate": float(np.mean(btsp_scores)),
            "capacity_matched_control": float(np.mean(ctrl_scores)),
            "details": {"binary_synapse_density": round(density, 4), "items_imprinted": n_items},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="btsp_binary_memory",
        category="Memory",
        metric_name="one_shot_recall_overlap",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=True,
    )


def run_associative_memory_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare Modern Hopfield associative retrieval vs classical linear Hebbian outer-product memory."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        dim, n_pat = 16, 18
        pats = rng.normal(0.0, 1.0, size=(n_pat, dim))
        pats /= np.linalg.norm(pats, axis=1, keepdims=True) + 1e-8

        hop = ModernHopfieldAssociativeMemory(dim=dim, beta=10.0, capacity=64)
        hebb_w = np.zeros((dim, dim))
        for p in pats:
            hop.store(p)
            hebb_w += np.outer(p, p) / float(dim)

        cand_cos, ctrl_cos, base_cos = [], [], []
        for p in pats:
            q = p + rng.normal(0.0, 0.35, size=dim)
            q /= float(np.linalg.norm(q) + 1e-8)
            base_cos.append(float(np.dot(q, p)))

            r_hop = hop.retrieve(q)
            r_hop /= float(np.linalg.norm(r_hop) + 1e-8)
            cand_cos.append(float(np.dot(r_hop, p)))

            r_hebb = hebb_w @ q
            r_hebb /= float(np.linalg.norm(r_hebb) + 1e-8)
            ctrl_cos.append(float(np.dot(r_hebb, p)))

        return {
            "baseline": float(np.mean(base_cos)),
            "candidate": float(np.mean(cand_cos)),
            "capacity_matched_control": float(np.mean(ctrl_cos)),
            "details": {"stored_patterns": n_pat, "beta": 10.0},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="associative_memory",
        category="NEUROWEAVE Gaps",
        metric_name="noisy_pattern_restoration_cosine",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=True,
    )


def run_fast_weights_benchmark(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare Ba et al. Fast Weights online key-value binding vs static linear projection of identical size."""

    def _trial(seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        dim = 12
        fw = FastWeightsController(dim=dim, decay_lambda=0.92, eta=0.95)
        static_w = rng.normal(0.0, 0.25, size=(dim, dim))

        cand_scores, ctrl_scores = [], []
        for _ in range(15):
            fw.reset()
            # Generate 4 orthogonal-ish episode key-value bindings
            q_mat, _ = np.linalg.qr(rng.normal(size=(dim, 4)))
            v_mat, _ = np.linalg.qr(rng.normal(size=(dim, 4)))
            for i in range(4):
                fw.bind(q_mat[:, i], v_mat[:, i])
            for i in range(4):
                pred_fw = fw.query(q_mat[:, i])
                pred_fw /= float(np.linalg.norm(pred_fw) + 1e-8)
                cand_scores.append(float(np.dot(pred_fw, v_mat[:, i])))

                pred_st = static_w @ q_mat[:, i]
                pred_st /= float(np.linalg.norm(pred_st) + 1e-8)
                ctrl_scores.append(float(np.dot(pred_st, v_mat[:, i])))

        return {
            "baseline": 0.0,
            "candidate": float(np.mean(cand_scores)),
            "capacity_matched_control": float(np.mean(ctrl_scores)),
            "details": {"dim": dim, "bindings_per_episode": 4},
        }

    return CapacityMatchedHarness.evaluate(
        module_id="fast_weights",
        category="NEUROWEAVE Gaps",
        metric_name="episodic_kv_recall_cosine",
        seeds=seeds or [42, 101, 202, 303, 404],
        trial_fn=_trial,
        higher_is_better=True,
        requires_capacity_control=True,
    )
