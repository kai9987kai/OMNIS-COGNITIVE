"""Standalone runnable demonstration of GENESIS 8-stage development and evolution."""

from __future__ import annotations

import json

from diamond.environment import DiamondWorld
from genesis.interfaces import GenesisEngineProtocol
from genesis.lifecycle import DEVELOPMENTAL_STAGES, GenesisEngine
from neurohive.ecosystem import NeuroHiveEcosystem


def main() -> None:
    world = DiamondWorld(width=10, height=10, seed=31)
    ecosystem = NeuroHiveEcosystem(world=world, max_population=6)
    genesis = GenesisEngine(seed=31)
    assert isinstance(genesis, GenesisEngineProtocol)

    for idx in range(4):
        org = genesis.create_organism(seed=100 + idx, use_deep_rl=(idx == 3))
        ecosystem.add_organism(org)

    for _ in range(5):
        ecosystem.step()

    first_org = next(iter(ecosystem.organisms.values()))
    new_stage = genesis.advance_stage(first_org)
    evo_summary = genesis.evolve_generation(ecosystem)
    ckpt_diff = genesis.compare_checkpoints(first_org)

    print("=== GENESIS DEVELOPMENT & EVOLUTION EXAMPLE ===")
    print(
        json.dumps(
            {
                "stages_available": len(DEVELOPMENTAL_STAGES),
                "advanced_stage": new_stage,
                "generation": evo_summary["generation"],
                "population_size": evo_summary["population_size"],
                "best_fitness": evo_summary["best_fitness"],
                "checkpoint_delta_keys": list(ckpt_diff.keys()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
