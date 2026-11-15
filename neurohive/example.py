"""Standalone runnable demonstration of the NEUROHIVE multi-agent ecosystem."""

from __future__ import annotations

import json

from diamond.environment import DiamondWorld
from neurohive.ecosystem import NeuroHiveEcosystem
from neurohive.interfaces import EcosystemProtocol, OrganismProtocol
from neurohive.organism import ArtificialOrganism


def main() -> None:
    world = DiamondWorld(width=10, height=10, seed=19)
    ecosystem = NeuroHiveEcosystem(world=world, max_population=8)
    assert isinstance(ecosystem, EcosystemProtocol)

    org_a = ArtificialOrganism(organism_id="hive_tab", seed=19, species_id="alpha", use_deep_rl=False)
    org_b = ArtificialOrganism(organism_id="hive_dqn", seed=23, species_id="alpha", use_deep_rl=True)
    assert isinstance(org_a, OrganismProtocol)
    assert isinstance(org_b, OrganismProtocol)

    ecosystem.add_organism(org_a)
    ecosystem.add_organism(org_b)

    last_step = {}
    for _ in range(8):
        last_step = ecosystem.step()

    summary = ecosystem.get_population_summary()
    print("=== NEUROHIVE ECOSYSTEM EXAMPLE ===")
    print(
        json.dumps(
            {
                "tick": last_step.get("tick"),
                "alive_count": last_step.get("alive_count"),
                "mean_fitness": summary["mean_fitness"],
                "species_counts": summary["species_counts"],
                "learners": [o["learner_type"] for o in summary["organisms"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
