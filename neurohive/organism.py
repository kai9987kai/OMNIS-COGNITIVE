"""ArtificialOrganism integrating SONAPSE, MNEMORPH, CognitivePipeline, and DIAMOND RL."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
import uuid

from cognition.pipeline import CognitivePipeline
from core.schemas import CognitiveStateSnapshot, utc_now_iso
from diamond.environment import DiamondWorld
from diamond.deep_rl import DeepQAgentLearner
from diamond.rl import DynaWorldModel, QAgentLearner
from mnemorph.memory import MnemorphMemory
from sonapse.connectome import SonapseConnectome
from sonapse.topology import analyze_topology


class ArtificialOrganism:
    """Autonomous artificial organism with neural architecture, multi-level memory, budget, and developmental history."""

    def __init__(
        self,
        organism_id: Optional[str] = None,
        generation: int = 0,
        parent_ids: Optional[List[str]] = None,
        species_id: str = "sp_alpha",
        seed: int = 42,
        neuron_model: str = "rate_coded",
        learning_rate: float = 0.08,
        reg_lambda: float = 0.002,
        plasticity_mode: str = "three_factor",
        neural_weight: float = 0.35,
        max_neurons: int = 128,
        max_synapses: int = 512,
        max_memories: int = 500,
        use_deep_rl: bool = False,
    ) -> None:
        self.organism_id = organism_id or f"org_{uuid.uuid4().hex[:8]}"
        self.generation = int(generation)
        self.parent_ids = list(parent_ids or [])
        self.species_id = species_id
        self.seed = int(seed)
        self.stage = "STAGE_1_GENESIS"
        self.age = 0
        self.alive = True
        self.energy = 100.0
        self.specialisation = "generalist"
        self.use_deep_rl = bool(use_deep_rl)

        self.genome: Dict[str, Any] = {
            "neuron_model": neuron_model,
            "learning_rate": float(learning_rate),
            "reg_lambda": float(reg_lambda),
            "plasticity_mode": plasticity_mode,
            "neural_weight": float(neural_weight),
            "ewc_lambda": 0.25,
            "use_deep_rl": self.use_deep_rl,
        }

        self.connectome = SonapseConnectome(
            input_dim=8,
            hidden_dim=8,
            output_dim=5,
            neuron_model=neuron_model,
            seed=self.seed,
            max_neurons=max_neurons,
            max_synapses=max_synapses,
        )
        self.memory = MnemorphMemory(agent_id=self.organism_id, max_memories=max_memories, seed=self.seed)
        self.learner: Any = (
            DeepQAgentLearner(input_dim=8, num_actions=5, seed=self.seed)
            if self.use_deep_rl
            else QAgentLearner(num_actions=5, learning_rate=0.18, seed=self.seed)
        )
        self.world_model = DynaWorldModel(seed=self.seed)
        self.pipeline = CognitivePipeline()

        self.performance_score: float = 0.0
        self.adaptation_score: float = 0.0
        self.robustness_score: float = 1.0
        self.compute_cost: float = 0.0
        self.fitness: float = 0.0
        self.cumulative_reward: float = 0.0
        self.developmental_history: List[Dict[str, Any]] = []
        self.last_cognitive_state: Optional[CognitiveStateSnapshot] = None

        self.record_checkpoint("genesis_init")

    def compute_fitness(
        self,
        alpha: float = 1.0,
        beta: float = 0.4,
        gamma: float = 0.3,
        reg_lambda: float = 0.1,
    ) -> float:
        """Compute F_i = alpha * P_i + beta * A_i + gamma * R_i - lambda * C_i."""
        self.fitness = float(
            alpha * self.performance_score
            + beta * self.adaptation_score
            + gamma * self.robustness_score
            - reg_lambda * self.compute_cost
        )
        return self.fitness

    def step(
        self,
        world: DiamondWorld,
        tick: int,
        enable_plasticity: bool = True,
        enable_structural_adaptation: bool = True,
        enable_memory_ops: bool = True,
        ewc_lambda: float = 0.0,
        fitness_weights: Optional[Dict[str, float]] = None,
        frontier_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one cognitive-behavioral cycle in DIAMOND and update fitness metrics."""
        if not self.alive:
            return {"organism_id": self.organism_id, "alive": False, "reward": 0.0}

        self.age += 1
        out = self.pipeline.step(
            organism_id=self.organism_id,
            tick=tick,
            world=world,
            connectome=self.connectome,
            memory=self.memory,
            learner=self.learner,
            world_model=self.world_model,
            learning_rate=self.genome["learning_rate"],
            reg_lambda=self.genome["reg_lambda"],
            plasticity_mode=self.genome["plasticity_mode"],
            enable_plasticity=enable_plasticity,
            enable_structural_adaptation=enable_structural_adaptation,
            enable_memory_ops=enable_memory_ops,
            ewc_lambda=ewc_lambda,
            neural_weight=self.genome["neural_weight"],
            frontier_context=frontier_context,
        )
        self.last_cognitive_state = out["snapshot"]
        reward = float(out["reward"])
        self.cumulative_reward += reward

        # Sync energy & survival status from world
        world_agent = world.agents.get(self.organism_id, {})
        self.energy = float(world_agent.get("energy", self.energy))
        if out["done"] or self.energy <= 0.0:
            self.alive = False

        # Update multi-factor fitness components
        recent_rewards = self.learner.reward_history[-20:]
        mean_recent = sum(recent_rewards) / max(1, len(recent_rewards))
        self.performance_score = float(mean_recent + 0.05 * world_agent.get("harvested", 0))

        # Adaptation measures learning progress + plasticity responsiveness
        if len(self.learner.reward_history) >= 6:
            early = sum(self.learner.reward_history[:3]) / 3.0
            late = sum(self.learner.reward_history[-3:]) / 3.0
            self.adaptation_score = float(max(0.0, late - early) + 0.1 * len(self.memory.records) / 100.0)
        else:
            self.adaptation_score = float(0.05 + 0.5 * self.connectome.last_weight_delta_norm)

        active_ratio = self.connectome.active_neuron_count() / max(1, len(self.connectome.neurons))
        self.robustness_score = float(active_ratio * (0.5 + 0.5 * min(1.0, self.energy / 100.0)))
        self.compute_cost = float(
            (len(self.connectome.active_synapses()) / 128.0) + (len(self.memory.records) / 500.0)
        )

        # Infer specialisation tag from behavior & structure
        if world_agent.get("harvested", 0) >= 3:
            self.specialisation = "forager"
        elif self.memory.stats.get("consolidated", 0) >= 3:
            self.specialisation = "archivist"
        elif len(self.connectome.active_synapses()) > 55:
            self.specialisation = "integrator"

        fw = fitness_weights or {"alpha": 1.0, "beta": 0.4, "gamma": 0.3, "lambda": 0.1}
        self.compute_fitness(
            alpha=fw.get("alpha", 1.0),
            beta=fw.get("beta", 0.4),
            gamma=fw.get("gamma", 0.3),
            reg_lambda=fw.get("lambda", 0.1),
        )
        return {
            "organism_id": self.organism_id,
            "alive": self.alive,
            "action": out["action"],
            "reward": reward,
            "td_error": out["td_error"],
            "fitness": self.fitness,
            "energy": self.energy,
        }

    def record_checkpoint(self, label: str) -> Dict[str, Any]:
        """Record an immutable developmental snapshot for Development Timeline comparisons."""
        topo = analyze_topology(self.connectome)
        ckpt = {
            "label": label,
            "timestamp": utc_now_iso(),
            "age": self.age,
            "stage": self.stage,
            "fitness": round(self.fitness, 4),
            "performance": round(self.performance_score, 4),
            "adaptation": round(self.adaptation_score, 4),
            "robustness": round(self.robustness_score, 4),
            "compute_cost": round(self.compute_cost, 4),
            "num_neurons": topo["num_neurons"],
            "num_synapses": topo["num_synapses"],
            "density": round(topo["density"], 5),
            "memory_count": len(self.memory.records),
            "memory_classes": self.memory.get_class_counts(),
            "connectome_snapshot": copy.deepcopy(self.connectome.to_dict()),
        }
        self.developmental_history.append(ckpt)
        if len(self.developmental_history) > 30:
            self.developmental_history = self.developmental_history[-30:]
        return ckpt

    def get_summary(self) -> Dict[str, Any]:
        return {
            "organism_id": self.organism_id,
            "species_id": self.species_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "stage": self.stage,
            "specialisation": self.specialisation,
            "age": self.age,
            "alive": self.alive,
            "energy": round(self.energy, 2),
            "fitness": round(self.fitness, 4),
            "performance_score": round(self.performance_score, 4),
            "adaptation_score": round(self.adaptation_score, 4),
            "robustness_score": round(self.robustness_score, 4),
            "compute_cost": round(self.compute_cost, 4),
            "cumulative_reward": round(self.cumulative_reward, 3),
            "neurons": len(self.connectome.neurons),
            "active_neurons": self.connectome.active_neuron_count(),
            "synapses": len(self.connectome.active_synapses()),
            "memories": len(self.memory.records),
            "learner_type": "deep_q_network" if self.use_deep_rl else "tabular_q",
            "genome": dict(self.genome),
            "checkpoints_count": len(self.developmental_history),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organism_id": self.organism_id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "species_id": self.species_id,
            "seed": self.seed,
            "use_deep_rl": self.use_deep_rl,
            "stage": self.stage,
            "age": self.age,
            "alive": self.alive,
            "energy": self.energy,
            "specialisation": self.specialisation,
            "genome": dict(self.genome),
            "performance_score": self.performance_score,
            "adaptation_score": self.adaptation_score,
            "robustness_score": self.robustness_score,
            "compute_cost": self.compute_cost,
            "fitness": self.fitness,
            "cumulative_reward": self.cumulative_reward,
            "connectome": self.connectome.to_dict(),
            "learner": self.learner.to_dict(),
            "developmental_history": copy.deepcopy(self.developmental_history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtificialOrganism":
        use_deep_rl = data.get("use_deep_rl", data.get("learner", {}).get("learner_type") == "deep_q_network")
        org = cls(
            organism_id=data["organism_id"],
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            species_id=data.get("species_id", "sp_alpha"),
            seed=data.get("seed", 42),
            neuron_model=data.get("genome", {}).get("neuron_model", "rate_coded"),
            learning_rate=data.get("genome", {}).get("learning_rate", 0.08),
            reg_lambda=data.get("genome", {}).get("reg_lambda", 0.002),
            plasticity_mode=data.get("genome", {}).get("plasticity_mode", "three_factor"),
            neural_weight=data.get("genome", {}).get("neural_weight", 0.35),
            use_deep_rl=use_deep_rl,
        )
        org.stage = data.get("stage", "STAGE_1_GENESIS")
        org.age = int(data.get("age", 0))
        org.alive = bool(data.get("alive", True))
        org.energy = float(data.get("energy", 100.0))
        org.specialisation = data.get("specialisation", "generalist")
        org.performance_score = float(data.get("performance_score", 0.0))
        org.adaptation_score = float(data.get("adaptation_score", 0.0))
        org.robustness_score = float(data.get("robustness_score", 1.0))
        org.compute_cost = float(data.get("compute_cost", 0.0))
        org.fitness = float(data.get("fitness", 0.0))
        org.cumulative_reward = float(data.get("cumulative_reward", 0.0))
        org.connectome = SonapseConnectome.from_dict(data["connectome"])
        if data.get("learner", {}).get("learner_type") == "deep_q_network":
            org.learner = DeepQAgentLearner.from_dict(data["learner"], seed=org.seed)
        else:
            org.learner = QAgentLearner.from_dict(data["learner"], seed=org.seed)
        org.developmental_history = list(data.get("developmental_history", []))
        return org
