"""Default reproducible scientific experiments driven by GHOST ENGINE."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ghost_engine.laboratory import GhostEngine


def run_standard_experiment_suite(
    ghost_engine: Optional[GhostEngine] = None,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run all four canonical OMNIS research experiments."""
    engine = ghost_engine or GhostEngine()
    seed_list = list(seeds or [42, 101, 202, 303, 404, 505])
    names = [
        "plasticity_vs_fixed",
        "memory_replay_retention",
        "developmental_vs_fixed",
        "morpheus_regeneration",
    ]
    reports = [engine.run_experiment(name=name, seeds=seed_list) for name in names]
    return {
        "count": len(reports),
        "experiments": reports,
    }
