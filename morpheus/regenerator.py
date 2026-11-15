"""MORPHEUS fault detection, controlled lesioning, structural/plasticity regeneration, and Recovery(t) evaluation."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional

from core.events import EventBus
from diamond.environment import DiamondWorld
from neurohive.organism import ArtificialOrganism
from sonapse.topology import analyze_topology


class MorpheusRegenerator:
    """Investigates adaptive computational fault tolerance and functional recovery."""

    def __init__(self, event_bus: Optional[EventBus] = None, seed: int = 42) -> None:
        self.event_bus = event_bus
        self.rng = random.Random(seed)
        self.checkpoints: Dict[str, Dict[str, Any]] = {}

    def save_recovery_checkpoint(self, organism: ArtificialOrganism) -> str:
        """Store an explicit state checkpoint (used only when testing checkpoint_restore mode)."""
        self.checkpoints[organism.organism_id] = copy.deepcopy(organism.to_dict())
        return organism.organism_id

    def restore_from_checkpoint(self, organism: ArtificialOrganism) -> bool:
        """Restore organism from a previously saved checkpoint."""
        ckpt = self.checkpoints.get(organism.organism_id)
        if not ckpt:
            return False
        restored = ArtificialOrganism.from_dict(ckpt)
        organism.connectome = restored.connectome
        organism.learner = restored.learner
        organism.energy = restored.energy
        organism.alive = True
        organism.record_checkpoint("morpheus_checkpoint_restore")
        return True

    def detect_faults(self, organism: ArtificialOrganism) -> Dict[str, Any]:
        """Scan SONAPSE connectome for deactivated neurons, severed synapses, and isolated nodes."""
        deactivated_neurons = [nid for nid, n in organism.connectome.neurons.items() if not n.active]
        severed_synapses = [sid for sid, s in organism.connectome.synapses.items() if not s.active]
        topo = analyze_topology(organism.connectome)
        fault_detected = bool(deactivated_neurons or severed_synapses or topo["isolated_neurons"])
        return {
            "organism_id": organism.organism_id,
            "fault_detected": fault_detected,
            "deactivated_neurons": deactivated_neurons,
            "severed_synapses": severed_synapses,
            "isolated_neurons": topo["isolated_neurons"],
            "active_neurons": topo["num_neurons"],
            "active_synapses": topo["num_synapses"],
        }

    def introduce_damage(
        self,
        organism: ArtificialOrganism,
        neuron_lesion_ratio: float = 0.25,
        synapse_sever_ratio: float = 0.25,
    ) -> Dict[str, Any]:
        """Deactivate a controlled fraction of hidden neurons and sever active synapses."""
        hidden_active = [
            n.neuron_id
            for n in organism.connectome.neurons.values()
            if n.role == "hidden" and n.active
        ]
        n_lesion = min(len(hidden_active), max(1, int(round(len(hidden_active) * neuron_lesion_ratio)))) if hidden_active and neuron_lesion_ratio > 0 else 0
        lesioned_neurons = self.rng.sample(hidden_active, k=n_lesion) if n_lesion > 0 else []
        for nid in lesioned_neurons:
            organism.connectome.neurons[nid].active = False
            organism.connectome.neurons[nid].a = 0.0
            organism.connectome.neurons[nid].u = 0.0

        active_syns = [s.synapse_id for s in organism.connectome.active_synapses()]
        n_sever = min(len(active_syns), max(1, int(round(len(active_syns) * synapse_sever_ratio)))) if active_syns and synapse_sever_ratio > 0 else 0
        severed_syns = self.rng.sample(active_syns, k=n_sever) if n_sever > 0 else []
        for sid in severed_syns:
            organism.connectome.synapses[sid].active = False

        organism.stage = "STAGE_7_REGENERATION"
        organism.record_checkpoint("post_damage")
        report = {
            "organism_id": organism.organism_id,
            "lesioned_neurons": lesioned_neurons,
            "severed_synapses": severed_syns,
            "remaining_active_neurons": organism.connectome.active_neuron_count(),
            "remaining_active_synapses": len(organism.connectome.active_synapses()),
        }
        if self.event_bus:
            self.event_bus.emit("morpheus.damage_introduced", source="morpheus", payload=report)
        return report

    def evaluate_performance(
        self,
        organism: ArtificialOrganism,
        reference_world_state: Dict[str, Any],
        steps: int = 12,
    ) -> Dict[str, float]:
        """Measure non-zero performance under equivalent deterministic conditions without mutating policy."""
        eval_world = DiamondWorld(width=reference_world_state["width"], height=reference_world_state["height"])
        eval_world.load_state(reference_world_state)
        eval_world.register_agent(organism.organism_id, x=2, y=2, energy=100.0)

        # Clone connectome membrane state so evaluation doesn't leak side-effects
        probe_conn = copy.deepcopy(organism.connectome)
        total_reward = 0.0
        output_norms: List[float] = []

        for _ in range(steps):
            obs = eval_world.observe(organism.organism_id)
            probe_conn.step(obs["features"])
            out_vec = probe_conn.get_output_vector()
            out_norm = math.sqrt(sum(v * v for v in out_vec) / max(1, len(out_vec)))
            output_norms.append(out_norm)

            # Policy blends Q/DQN values and neural connectome signal
            q_vals = organism.learner.get_q_values(obs["state_key"], features=obs["features"])
            combined = [0.6 * q + 0.4 * n for q, n in zip(q_vals, out_vec)]
            action = int(max(range(len(combined)), key=lambda i: combined[i]))
            _, rew, _, _ = eval_world.step_agent(organism.organism_id, action)
            eval_world.step_world_dynamics()
            total_reward += rew

        mean_reward = total_reward / max(1, steps)
        # Positive normalized behavioral score + neural functional throughput
        behavioral_score = max(0.05, (mean_reward + 1.5) / 2.5)
        neural_throughput = sum(output_norms) / max(1, len(output_norms))
        active_hidden = sum(1 for n in probe_conn.neurons.values() if n.role == "hidden" and n.active)
        baseline_hidden = max(1, int(getattr(probe_conn, "initial_hidden_dim", 8)))
        structural_factor = min(1.0, active_hidden / float(baseline_hidden))

        composite = max(0.05, 0.55 * behavioral_score + 0.30 * neural_throughput + 0.15 * structural_factor)
        return {
            "performance": float(composite),
            "behavioral_score": float(behavioral_score),
            "neural_throughput": float(neural_throughput),
            "structural_factor": float(structural_factor),
            "mean_reward": float(mean_reward),
        }

    def regenerate_step(
        self,
        organism: ArtificialOrganism,
        world: Optional[DiamondWorld] = None,
        learning_steps: int = 8,
    ) -> Dict[str, Any]:
        """Perform non-reset regeneration: bypass synaptogenesis, neurogenesis, homeostatic scaling, and plasticity."""
        conn = organism.connectome
        baseline_hidden = max(8, int(getattr(conn, "initial_hidden_dim", 8)))

        # 1. Compensatory neurogenesis until active hidden capacity matches baseline
        born = 0
        while sum(1 for n in conn.neurons.values() if n.role == "hidden" and n.active) < baseline_hidden:
            if conn.add_hidden_neuron() is None:
                break
            born += 1

        # 2. Structural bypass routing around severed synapses and isolated neurons
        bypass_added = conn.grow_connections(max_new=8)

        # 3. Homeostatic synaptic scaling on healthy pathways to restore excitatory/inhibitory balance
        active_syns = conn.active_synapses()
        for syn in active_syns:
            syn.weight = max(-conn.w_max, min(conn.w_max, syn.weight * 1.18))

        # 4. Rehearsal + plasticity retuning using Mnemorph memories (prefer positive/high-importance experiences)
        replayed, _ = organism.memory.replay(batch_size=8)
        replayed_sorted = sorted(replayed, key=lambda m: (m.reward_assoc, m.importance), reverse=True)
        for mem in replayed_sorted:
            feats = mem.embedding[:8] if len(mem.embedding) >= 8 else mem.embedding
            conn.step(feats)
            act = mem.content.get("action", mem.content.get("best_action")) if isinstance(mem.content, dict) else None
            # Amplify reparative plasticity without mutating Q-table adversely
            rew_signal = mem.reward_assoc if mem.reward_assoc >= 0 else 0.15 * abs(mem.reward_assoc)
            conn.update_plasticity(
                reward_prediction_error=rew_signal,
                learning_rate=max(0.10, organism.genome["learning_rate"]),
                reg_lambda=organism.genome["reg_lambda"],
                mode="three_factor",
                chosen_action=int(act) if act is not None else None,
            )

        if world is not None and learning_steps > 0:
            # Keep Q-values safe while recalibrating connectome sensorimotor flow on live observations
            for _ in range(min(4, learning_steps)):
                obs = world.observe(organism.organism_id)
                conn.step(obs["features"])
                conn.update_plasticity(
                    reward_prediction_error=0.15,
                    learning_rate=0.05,
                    reg_lambda=organism.genome["reg_lambda"],
                    mode="hebbian",
                )

        organism.record_checkpoint("morpheus_regenerated")
        res = {
            "organism_id": organism.organism_id,
            "neurons_generated": born,
            "bypass_synapses_added": bypass_added,
            "replayed_memories": len(replayed),
            "active_neurons": conn.active_neuron_count(),
            "active_synapses": len(conn.active_synapses()),
        }
        if self.event_bus:
            self.event_bus.emit("morpheus.regenerated", source="morpheus", payload=res)
        return res

    def run_regeneration_experiment(
        self,
        organism: ArtificialOrganism,
        world: DiamondWorld,
        neuron_lesion_ratio: float = 0.30,
        synapse_sever_ratio: float = 0.30,
        recovery_rounds: int = 3,
    ) -> Dict[str, Any]:
        """Run complete damage & multi-round Recovery(t) measurement protocol."""
        ref_state = world.save_state()
        before_metrics = self.evaluate_performance(organism, ref_state)
        p_before = max(1e-6, before_metrics["performance"])

        damage_report = self.introduce_damage(
            organism,
            neuron_lesion_ratio=neuron_lesion_ratio,
            synapse_sever_ratio=synapse_sever_ratio,
        )
        fault_report = self.detect_faults(organism)
        damaged_metrics = self.evaluate_performance(organism, ref_state)
        p_damaged = damaged_metrics["performance"]

        recovery_curve: List[Dict[str, Any]] = [
            {
                "t": 0,
                "label": "damaged",
                "performance": round(p_damaged, 4),
                "recovery_ratio": round(p_damaged / p_before, 4),
            }
        ]

        # Run non-reset regeneration rounds on a sandbox copy of the world so ref_state stays clean
        rehab_world = DiamondWorld(width=ref_state["width"], height=ref_state["height"])
        rehab_world.load_state(ref_state)
        rehab_world.register_agent(organism.organism_id, energy=120.0)
        organism.alive = True
        organism.energy = 120.0

        for round_idx in range(1, recovery_rounds + 1):
            regen_info = self.regenerate_step(organism, world=rehab_world, learning_steps=6)
            after_metrics = self.evaluate_performance(organism, ref_state)
            p_after = after_metrics["performance"]
            recovery_curve.append(
                {
                    "t": round_idx,
                    "label": f"regen_round_{round_idx}",
                    "performance": round(p_after, 4),
                    "recovery_ratio": round(p_after / p_before, 4),
                    "bypass_added": regen_info["bypass_synapses_added"],
                    "neurons_generated": regen_info["neurons_generated"],
                }
            )

        final_recovery = recovery_curve[-1]["recovery_ratio"]
        organism.robustness_score = float(min(1.5, final_recovery))
        return {
            "organism_id": organism.organism_id,
            "performance_before": round(p_before, 4),
            "performance_damaged": round(p_damaged, 4),
            "performance_after": recovery_curve[-1]["performance"],
            "initial_drop_ratio": round(p_damaged / p_before, 4),
            "final_recovery_ratio": final_recovery,
            "damage_report": damage_report,
            "fault_report": fault_report,
            "recovery_curve": recovery_curve,
        }
