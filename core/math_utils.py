"""Mathematical utilities for information theory, multi-objective Pareto analysis, and statistics."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


def shannon_entropy(values: Sequence[float] | np.ndarray, bins: int = 16, is_probabilities: bool = False) -> float:
    """Compute discrete Shannon entropy H(X) = -sum_i p(x_i) log2 p(x_i).

    If `is_probabilities` is False and the input does not already form a probability distribution,
    continuous values are discretized into `bins` histogram bins.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0

    if is_probabilities or (
        np.all(arr >= 0.0) and abs(float(np.sum(arr)) - 1.0) < 1e-4 and arr.size <= 64
    ):
        probs = arr[arr > 0.0]
        if probs.size == 0:
            return 0.0
        probs = probs / float(np.sum(probs))
    else:
        if float(np.max(arr) - np.min(arr)) < 1e-12:
            return 0.0
        counts, _ = np.histogram(arr, bins=max(2, int(bins)))
        counts = counts[counts > 0]
        if counts.size == 0:
            return 0.0
        probs = counts.astype(float) / float(np.sum(counts))

    entropy = -float(np.sum(probs * np.log2(probs)))
    return max(0.0, entropy)


def multi_objective_score(
    l_task: float,
    c_compute: float,
    l_memory: float,
    l_instability: float,
    alpha: float = 1.0,
    beta: float = 0.2,
    gamma: float = 0.2,
    delta: float = 0.2,
) -> float:
    """Compute scalarized multi-objective loss J(theta) = alpha*L_task + beta*C_compute + gamma*L_memory + delta*L_instability."""
    score = (
        float(alpha) * float(l_task)
        + float(beta) * float(c_compute)
        + float(gamma) * float(l_memory)
        + float(delta) * float(l_instability)
    )
    return float(score)


def compute_pareto_front(points: Sequence[Sequence[float]]) -> List[int]:
    """Return indices of non-dominated points assuming minimization across all objectives."""
    if not points:
        return []
    arr = np.asarray(points, dtype=float)
    n = arr.shape[0]
    pareto_indices: List[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    return pareto_indices


def statistical_summary(samples: Iterable[float]) -> Dict[str, float]:
    """Compute sample count, mean, sample std, and 95% t-confidence interval."""
    arr = np.asarray(list(samples), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci_95_low": 0.0, "ci_95_high": 0.0}
    mean = float(np.mean(arr))
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci_95_low": mean, "ci_95_high": mean}

    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(n)
    if se == 0.0:
        margin = 0.0
    else:
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        margin = t_crit * se

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_95_low": mean - margin,
        "ci_95_high": mean + margin,
    }


def compare_distributions(
    sample_a: Iterable[float],
    sample_b: Iterable[float],
    alpha: float = 0.05,
    min_effect_size: float = 0.2,
) -> Dict[str, Any]:
    """Compare two sample sets (baseline A vs treatment B) via Welch's t-test and Cohen's d."""
    a = np.asarray(list(sample_a), dtype=float)
    b = np.asarray(list(sample_b), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    summary_a = statistical_summary(a)
    summary_b = statistical_summary(b)

    if a.size < 2 or b.size < 2:
        diff = summary_b["mean"] - summary_a["mean"]
        return {
            "baseline": summary_a,
            "treatment": summary_b,
            "mean_diff": diff,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "significant": False,
        }

    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    mean_diff = float(np.mean(b) - np.mean(a))

    if var_a < 1e-18 and var_b < 1e-18:
        if abs(mean_diff) < 1e-12:
            t_stat, p_val, cohens_d = 0.0, 1.0, 0.0
        else:
            t_stat = float(np.sign(mean_diff) * 1e6)
            p_val = 0.0
            cohens_d = float(np.sign(mean_diff) * 10.0)
    else:
        res = stats.ttest_ind(b, a, equal_var=False)
        t_stat = float(res.statistic) if np.isfinite(res.statistic) else 0.0
        p_val = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
        pooled_denom = ((a.size - 1) * var_a + (b.size - 1) * var_b) / max(1, a.size + b.size - 2)
        pooled_sd = math.sqrt(max(1e-18, pooled_denom))
        cohens_d = mean_diff / pooled_sd

    significant = bool((p_val < alpha) and (abs(cohens_d) >= min_effect_size))
    return {
        "baseline": summary_a,
        "treatment": summary_b,
        "mean_diff": mean_diff,
        "t_statistic": t_stat,
        "p_value": p_val,
        "cohens_d": cohens_d,
        "significant": significant,
    }
