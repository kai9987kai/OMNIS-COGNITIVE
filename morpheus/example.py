"""Standalone runnable demonstration of MORPHEUS damage injection and Recovery(t)."""

from __future__ import annotations

import json

from diamond.environment import DiamondWorld
from morpheus.interfaces import RegeneratorProtocol
from morpheus.regenerator import MorpheusRegenerator
from neurohive.organism import ArtificialOrganism


def main() -> None:
    world = DiamondWorld(width=10, height=10, seed=55)
    org = ArtificialOrganism(organism_id="morpheus_demo", seed=55)
    world.register_agent(org.organism_id, energy=org.energy)
    for t in range(1, 15):
        org.step(world, tick=t)

    regen = MorpheusRegenerator(seed=55)
    assert isinstance(regen, RegeneratorProtocol)

    report = regen.run_regeneration_experiment(
        org,
        world,
        neuron_lesion_ratio=0.30,
        synapse_sever_ratio=0.30,
        recovery_rounds=3,
    )
    print("=== MORPHEUS REGENERATION EXAMPLE ===")
    print(
        json.dumps(
            {
                "performance_before": report["performance_before"],
                "initial_drop_ratio": report["initial_drop_ratio"],
                "final_recovery_ratio": report["final_recovery_ratio"],
                "rehab_gain": round(report["final_recovery_ratio"] - report["initial_drop_ratio"], 4),
                "rounds": len(report["recovery_curve"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
