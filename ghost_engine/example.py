"""Standalone runnable demonstration of GHOST ENGINE controlled scientific hypothesis testing."""

from __future__ import annotations

import json

from ghost_engine.interfaces import LaboratoryProtocol
from ghost_engine.laboratory import GhostEngine


def main() -> None:
    engine = GhostEngine()
    assert isinstance(engine, LaboratoryProtocol)

    report = engine.run_experiment(
        name="morpheus_regeneration",
        seeds=[42, 101, 202, 303, 404, 505],
    )
    stats = report["statistical_analysis"]
    print("=== GHOST ENGINE SCIENTIFIC EXPERIMENT EXAMPLE ===")
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "name": report["name"],
                "verdict": report["verdict"],
                "mean_diff": stats["mean_diff"],
                "p_value": stats["p_value"],
                "cohens_d": stats["cohens_d"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
