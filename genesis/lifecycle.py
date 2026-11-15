"""GENESIS ENGINE managing 8-stage organism development, evolutionary operators, and lineage DAG."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple
import uuid

from core.events import EventBus
from core.governance import GovernanceManager
from neurohive.ecosystem import NeuroHiveEcosystem
from neurohive.organism import ArtificialOrganism

DEVELOPMENTAL_STAGES = [
    "STAGE_1_GENESIS",
    "STAGE_2_DEVELOPMENT",
    "STAGE_3_LEARNING",
    "STAGE_4_MEMORY",
    "STAGE_5_SPECIALISATION",
    "STAGE_6_EVOLUTION",
    "STAGE_7_REGENERATION",
    "STAGE_8_EVALUATION",
]


class GenesisEngine:
    """Coordinates organism creation, developmental progression, selection, recombination, mutation, and lineage."""

    def __init__(
        self,
        seed: int = 42,
        governance: Optional[GovernanceManager] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.governance = governance
        self.event_bus = event_bus
        self.generation_counter = 0
        self.lineage: Dict[str, Dict[str, Any]] = {}
        self.evolution_history: List[Dict[str, Any]] = []

    def create_organism(
        self,
        organism_id: Optional[str] = None,
        generation: int = 0,
        parent_ids: Optional[List[str]] = None,
        neuron_model: str = "rate_coded",
        learning_rate: float = 0.08,
        reg_lambda: float = 0.002,
        seed: Optional[int] = None,
        use_deep_rl: bool = False,
    ) -> ArtificialOrganism:
        """STAGE 1: Create an organism with minimal initial architecture and record lineage."""
        org_seed = self.rng.randint(1, 999999) if seed is None else seed
        quotas = self.governance.quotas if self.governance else None
        org = ArtificialOrganism(
            organism_id=organism_id,
            generation=generation,
            parent_ids=parent_ids,
            seed=org_seed,
            neuron_model=neuron_model,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            max_neurons=quotas.max_neurons_per_organism if quotas else 128,
            max_synapses=quotas.max_synapses_per_organism if quotas else 512,
            max_memories=quotas.max_memories_per_organism if quotas else 500,
            use_deep_rl=use_deep_rl,
        )
        self.lineage[org.organism_id] = {
            "organism_id": org.organism_id,
            "generation": org.generation,
            "parent_ids": list(org.parent_ids),
            "species_id": org.species_id,
            "genome": dict(org.genome),
            "fitness": org.fitness,
            "stage": org.stage,
        }
        if self.event_bus:
            self.event_bus.emit(
                "genesis.organism_created",
                source="genesis",
                payload={"organism_id": org.organism_id, "generation": generation},
            )
        return org

    def advance_stage(self, organism: ArtificialOrganism, target_stage: Optional[str] = None) -> str:
        """Advance organism to the next (or specified) developmental stage with structural/cognitive actions."""
        if target_stage and target_stage in DEVELOPMENTAL_STAGES:
            next_stage = target_stage
        else:
            idx = DEVELOPMENTAL_STAGES.index(organism.stage) if organism.stage in DEVELOPMENTAL_STAGES else 0
            next_stage = DEVELOPMENTAL_STAGES[min(len(DEVELOPMENTAL_STAGES) - 1, idx + 1)]

        organism.stage = next_stage
        if next_stage == "STAGE_2_DEVELOPMENT":
            organism.connectome.add_hidden_neuron()
            organism.connectome.grow_connections(max_new=4)
        elif next_stage == "STAGE_4_MEMORY":
            organism.memory.consolidate()
        elif next_stage == "STAGE_5_SPECIALISATION":
            organism.connectome.prune_weak_connections(min_weight=0.02)

        organism.record_checkpoint(next_stage.lower())
        if organism.organism_id in self.lineage:
            self.lineage[organism.organism_id]["stage"] = next_stage
            self.lineage[organism.organism_id]["fitness"] = round(organism.fitness, 4)
        return next_stage

    def recombine(
        self,
        parent_a: ArtificialOrganism,
        parent_b: ArtificialOrganism,
        generation: int,
        crossover_rate: float = 0.65,
    ) -> ArtificialOrganism:
        """Create offspring via parameter & compatible synaptic weight crossover."""
        if self.rng.random() > crossover_rate:
            # Asexual clone + inherited baseline before mutation
            donor = parent_a if parent_a.fitness >= parent_b.fitness else parent_b
            child = self.create_organism(
                generation=generation,
                parent_ids=[donor.organism_id],
                neuron_model=donor.genome["neuron_model"],
                learning_rate=donor.genome["learning_rate"],
                reg_lambda=donor.genome["reg_lambda"],
                use_deep_rl=donor.use_deep_rl,
            )
            child.connectome = copy.deepcopy(donor.connectome)
            child.connectome.seed = child.seed
            return child

        lr = 0.5 * (parent_a.genome["learning_rate"] + parent_b.genome["learning_rate"])
        reg = 0.5 * (parent_a.genome["reg_lambda"] + parent_b.genome["reg_lambda"])
        model = parent_a.genome["neuron_model"] if self.rng.random() < 0.5 else parent_b.genome["neuron_model"]
        base = parent_a if parent_a.fitness >= parent_b.fitness else parent_b
        other = parent_b if base is parent_a else parent_a
        child = self.create_organism(
            generation=generation,
            parent_ids=[parent_a.organism_id, parent_b.organism_id],
            neuron_model=model,
            learning_rate=lr,
            reg_lambda=reg,
            use_deep_rl=base.use_deep_rl,
        )
        # Recombine shared synaptic connections from parent_a and parent_b
        child.connectome = copy.deepcopy(base.connectome)
        child.connectome.seed = child.seed
        for sid, syn in child.connectome.synapses.items():
            if sid in other.connectome.synapses and self.rng.random() < 0.5:
                syn.weight = other.connectome.synapses[sid].weight
        child.record_checkpoint("recombined")
        return child

    def mutate(self, organism: ArtificialOrganism, mutation_rate: float = 0.15) -> ArtificialOrganism:
        """Apply bounded hyperparameter and connectome mutations."""
        if self.rng.random() < mutation_rate:
            organism.genome["learning_rate"] = max(
                0.01, min(0.35, organism.genome["learning_rate"] + self.rng.gauss(0.0, 0.015))
            )
        if self.rng.random() < mutation_rate:
            organism.genome["reg_lambda"] = max(
                0.0, min(0.05, organism.genome["reg_lambda"] + self.rng.gauss(0.0, 0.001))
            )
        organism.connectome.mutate_weights_and_topology(mutation_rate=mutation_rate)
        organism.record_checkpoint("mutated")
        if organism.organism_id in self.lineage:
            self.lineage[organism.organism_id]["genome"] = dict(organism.genome)
        return organism

    def evolve_generation(
        self,
        ecosystem: NeuroHiveEcosystem,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.65,
        elite_fraction: float = 0.25,
        target_population: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute P_(t+1) = Select(Mutate(Recombine(P_t))) strictly bounded by population cap."""
        self.generation_counter += 1
        current = [o for o in ecosystem.organisms.values() if o.alive]
        if not current:
            return {"generation": self.generation_counter, "population": 0, "elites": [], "offspring": []}

        current.sort(key=lambda o: o.fitness, reverse=True)
        for o in current:
            if o.organism_id in self.lineage:
                self.lineage[o.organism_id]["fitness"] = round(o.fitness, 4)

        cap = self.governance.quotas.max_population_size if self.governance else ecosystem.max_population
        desired = min(cap, target_population or max(len(current), 4))

        num_elites = max(1, min(len(current), int(round(desired * elite_fraction))))
        elites = current[:num_elites]

        def tournament_pick() -> ArtificialOrganism:
            pool = self.rng.sample(current, k=min(3, len(current)))
            return max(pool, key=lambda o: o.fitness)

        new_population: List[ArtificialOrganism] = []
        for elite in elites:
            elite.generation = max(elite.generation, self.generation_counter - 1)
            elite.energy = max(elite.energy, 100.0)
            elite.stage = "STAGE_6_EVOLUTION"
            new_population.append(elite)

        offspring_ids: List[str] = []
        while len(new_population) < desired:
            pa = tournament_pick()
            pb = tournament_pick()
            child = self.recombine(pa, pb, generation=self.generation_counter, crossover_rate=crossover_rate)
            self.mutate(child, mutation_rate=mutation_rate)
            child.stage = "STAGE_2_DEVELOPMENT"
            new_population.append(child)
            offspring_ids.append(child.organism_id)

        # Replace ecosystem population cleanly within bounds
        for oid in list(ecosystem.organisms.keys()):
            if oid not in {o.organism_id for o in new_population}:
                ecosystem.world.unregister_agent(oid)
                ecosystem.organisms.pop(oid, None)

        for org in new_population:
            if org.organism_id not in ecosystem.organisms:
                ecosystem.add_organism(org)
            else:
                ecosystem.world.register_agent(org.organism_id, energy=org.energy)

        summary = {
            "generation": self.generation_counter,
            "population_size": len(ecosystem.organisms),
            "elites": [e.organism_id for e in elites],
            "offspring": offspring_ids,
            "best_fitness": round(elites[0].fitness, 4),
            "mean_fitness": round(ecosystem.mean_fitness(), 4),
        }
        self.evolution_history.append(summary)
        if self.event_bus:
            self.event_bus.emit("genesis.generation_evolved", source="genesis", payload=summary)
        return summary

    def compare_checkpoints(
        self,
        organism: ArtificialOrganism,
        idx_a: int = 0,
        idx_b: int = -1,
    ) -> Dict[str, Any]:
        """Compare two developmental checkpoints of an organism."""
        if not organism.developmental_history:
            return {}
        ckpt_a = organism.developmental_history[idx_a]
        ckpt_b = organism.developmental_history[idx_b]
        return {
            "organism_id": organism.organism_id,
            "checkpoint_a": ckpt_a["label"],
            "checkpoint_b": ckpt_b["label"],
            "delta_neurons": ckpt_b["num_neurons"] - ckpt_a["num_neurons"],
            "delta_synapses": ckpt_b["num_synapses"] - ckpt_a["num_synapses"],
            "delta_memories": ckpt_b["memory_count"] - ckpt_a["memory_count"],
            "delta_fitness": round(ckpt_b["fitness"] - ckpt_a["fitness"], 4),
            "delta_performance": round(ckpt_b["performance"] - ckpt_a["performance"], 4),
            "a": {k: v for k, v in ckpt_a.items() if k != "connectome_snapshot"},
            "b": {k: v for k, v in ckpt_b.items() if k != "connectome_snapshot"},
        }
