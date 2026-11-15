"""NEUROHIVE population ecosystem managing multi-agent competition, cooperation, speciation, and bounded lifecycle."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from core.events import EventBus
from core.governance import GovernanceManager
from diamond.environment import DiamondWorld
from mnemorph.memory import MemoryClass
from neurohive.organism import ArtificialOrganism


class NeuroHiveEcosystem:
    """Population manager for multiple interacting ArtificialOrganism instances."""

    def __init__(
        self,
        world: DiamondWorld,
        governance: Optional[GovernanceManager] = None,
        event_bus: Optional[EventBus] = None,
        max_population: int = 24,
    ) -> None:
        self.world = world
        self.governance = governance
        self.event_bus = event_bus
        self.max_population = int(max_population)
        self.organisms: Dict[str, ArtificialOrganism] = {}
        self.social_events: List[Dict[str, Any]] = []
        self.tick: int = 0

    def add_organism(self, organism: ArtificialOrganism) -> bool:
        """Register organism in ecosystem and DIAMOND world, respecting population cap."""
        cap = self.governance.quotas.max_population_size if self.governance else self.max_population
        alive_count = len([o for o in self.organisms.values() if o.alive])
        if len(self.organisms) >= cap or alive_count >= cap:
            if self.governance:
                allowed = self.governance.check_population_cap(len(self.organisms), 1)
                if allowed == 0:
                    return False
            else:
                return False

        self.organisms[organism.organism_id] = organism
        self.world.register_agent(organism.organism_id, energy=organism.energy)
        if self.event_bus:
            self.event_bus.emit(
                "neurohive.organism_added",
                source="neurohive",
                tick=self.tick,
                payload={"organism_id": organism.organism_id, "generation": organism.generation},
            )
        return True

    def remove_dead_organisms(self) -> List[str]:
        """Clean up dead organisms from active world simulation while returning their IDs."""
        dead = [oid for oid, org in self.organisms.items() if not org.alive]
        for oid in dead:
            self.world.unregister_agent(oid)
            self.organisms.pop(oid, None)
            if self.event_bus:
                self.event_bus.emit(
                    "neurohive.organism_died",
                    source="neurohive",
                    tick=self.tick,
                    payload={"organism_id": oid},
                )
        return dead

    def _assign_species(self) -> None:
        """Cluster organisms into species based on neuron model and connectome complexity."""
        for org in self.organisms.values():
            n_bucket = len(org.connectome.neurons) // 4
            model = org.genome.get("neuron_model", "rate_coded")
            org.species_id = f"sp_{model[:4]}_{n_bucket}"

    def _process_social_interactions(self) -> None:
        """Execute bounded cooperation (energy & memory sharing) and competition among nearby organisms."""
        alive_orgs = [o for o in self.organisms.values() if o.alive]
        for i in range(len(alive_orgs)):
            for j in range(i + 1, len(alive_orgs)):
                a = alive_orgs[i]
                b = alive_orgs[j]
                wa = self.world.agents.get(a.organism_id)
                wb = self.world.agents.get(b.organism_id)
                if not wa or not wb:
                    continue
                dist = math.hypot(float(wa["x"]) - float(wb["x"]), float(wa["y"]) - float(wb["y"]))
                if dist > 2.2:
                    continue

                # Kin or same-species cooperation vs cross-species competition
                if a.species_id == b.species_id or set(a.parent_ids) & set(b.parent_ids):
                    donor, recipient = (a, b) if a.energy >= b.energy else (b, a)
                    if donor.energy > 90.0 and recipient.energy < 80.0:
                        transfer = 6.0
                        donor.energy -= transfer
                        recipient.energy = min(150.0, recipient.energy + transfer)
                        self.world.agents[donor.organism_id]["energy"] = donor.energy
                        self.world.agents[recipient.organism_id]["energy"] = recipient.energy

                        # Share highest-importance semantic/procedural memory
                        if donor.memory.records:
                            best_mem = max(donor.memory.records.values(), key=lambda r: r.importance)
                            recipient.memory.encode(
                                MemoryClass.SEMANTIC,
                                content={"shared_from": donor.organism_id, **best_mem.content},
                                embedding=best_mem.embedding,
                                timestamp=float(self.tick),
                                reward_assoc=best_mem.reward_assoc,
                                significance=0.7,
                                provenance=f"social:{donor.organism_id}",
                            )
                        self.social_events.append(
                            {
                                "tick": self.tick,
                                "type": "cooperation",
                                "donor": donor.organism_id,
                                "recipient": recipient.organism_id,
                                "energy_shared": transfer,
                            }
                        )
                else:
                    # Mild resource competition when overlapping very closely
                    if dist < 1.1:
                        winner, loser = (a, b) if a.fitness >= b.fitness else (b, a)
                        drain = 2.0
                        loser.energy = max(0.0, loser.energy - drain)
                        winner.energy = min(150.0, winner.energy + 1.0)
                        self.world.agents[loser.organism_id]["energy"] = loser.energy
                        self.world.agents[winner.organism_id]["energy"] = winner.energy
                        self.social_events.append(
                            {
                                "tick": self.tick,
                                "type": "competition",
                                "winner": winner.organism_id,
                                "loser": loser.organism_id,
                            }
                        )
        if len(self.social_events) > 100:
            self.social_events = self.social_events[-100:]

    def step(self, fitness_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Advance all alive organisms and the DIAMOND world by one discrete tick."""
        self.tick += 1
        self.world.step_world_dynamics()

        step_results: List[Dict[str, Any]] = []
        for oid, org in list(self.organisms.items()):
            if org.alive:
                res = org.step(self.world, tick=self.tick, fitness_weights=fitness_weights)
                step_results.append(res)

        self._process_social_interactions()
        self._assign_species()
        dead_ids = self.remove_dead_organisms()

        return {
            "tick": self.tick,
            "alive_count": len(self.organisms),
            "died": dead_ids,
            "results": step_results,
            "mean_fitness": self.mean_fitness(),
        }

    def mean_fitness(self) -> float:
        alive = [o.fitness for o in self.organisms.values() if o.alive]
        return float(sum(alive) / len(alive)) if alive else 0.0

    def get_population_summary(self) -> Dict[str, Any]:
        species_counts: Dict[str, int] = {}
        fitness_values: List[float] = []
        for o in self.organisms.values():
            species_counts[o.species_id] = species_counts.get(o.species_id, 0) + 1
            fitness_values.append(round(o.fitness, 4))
        return {
            "tick": self.tick,
            "population_size": len(self.organisms),
            "max_population": self.governance.quotas.max_population_size if self.governance else self.max_population,
            "mean_fitness": round(self.mean_fitness(), 4),
            "species_counts": species_counts,
            "fitness_distribution": fitness_values,
            "organisms": [o.get_summary() for o in self.organisms.values()],
            "recent_social_events": self.social_events[-20:],
        }
