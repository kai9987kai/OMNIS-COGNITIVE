"""Formal typed interface protocols for the GHOST ENGINE autonomous research laboratory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class LaboratoryProtocol(Protocol):
    """Contract for paired statistical hypothesis testing and ablation experimentation."""

    experiments: List[Dict[str, Any]]

    def run_experiment(
        self,
        name: str = "plasticity_vs_fixed",
        hypothesis: Optional[str] = None,
        seeds: Optional[List[int]] = None,
        steps: int = 45,
        min_mean_diff: float = 0.0,
        max_p_value: float = 0.05,
        min_effect_size: float = 0.2,
    ) -> Dict[str, Any]: ...
