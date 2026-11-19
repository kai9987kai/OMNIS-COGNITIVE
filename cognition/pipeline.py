"""10-stage Cognitive Processing Pipeline producing C_t = {P_t, A_t, M_t, G_t, W_t, H_t}."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from core.math_utils import shannon_entropy
from core.schemas import CognitiveStateSnapshot
from diamond.environment import ACTION_NAMES, DiamondWorld
from diamond.rl import DynaWorldModel, QAgentLearner
from mnemorph.memory import MemoryClass, MnemorphMemory
from sonapse.connectome import SonapseConnectome

COGNITIVE_STAGES = [
    "PERCEPTION",
    "ATTENTION",
    "WORKING_MEMORY",
    "COGNITIVE_PROCESSING",
    "DECISION_MAKING",
    "ACTION",
    "ENVIRONMENTAL_FEEDBACK",
    "LEARNING",
    "MEMORY_CONSOLIDATION",
    "ARCHITECTURAL_ADAPTATION",
]


class CognitivePipeline:
    """Subsystem-agnostic cognitive loop bridging perception, memory, neural dynamics, RL, and plasticity."""

    def __init__(self, history_maxlen: int = 32) -> None:
        self.history: Deque[Dict[str, Any]] = deque(maxlen=history_maxlen)
        self.last_snapshot: Optional[CognitiveStateSnapshot] = None

    def step(
        self,
        organism_id: str,
        tick: int,
        world: DiamondWorld,
        connectome: SonapseConnectome,
        memory: MnemorphMemory,
        learner: QAgentLearner,
        world_model: DynaWorldModel,
        learning_rate: float = 0.08,
        reg_lambda: float = 0.002,
        plasticity_mode: str = "three_factor",
        enable_plasticity: bool = True,
        enable_structural_adaptation: bool = True,
        enable_memory_ops: bool = True,
        ewc_lambda: float = 0.0,
        neural_weight: float = 0.35,
    ) -> Dict[str, Any]:
        """Execute all 10 cognitive stages for one agent step and return telemetry + C_t snapshot."""

        # 1. PERCEPTION
        obs = world.observe(organism_id)
        features = list(obs["features"])
        p_t = {
            "position": obs["position"],
            "energy": obs["energy"],
            "features": features,
            "state_key": obs["state_key"],
            "task_id": obs["task_id"],
        }

        # 2. ATTENTION (salience gain on resource/hazard/energy channels)
        hazard_proximity = features[5] if len(features) > 5 else 0.0
        resource_proximity = features[2] if len(features) > 2 else 0.0
        energy_need = 1.0 - (features[7] if len(features) > 7 else 0.5)
        gains = [
            1.0 + 0.5 * resource_proximity,
            1.0 + 0.5 * resource_proximity,
            1.0 + 0.6 * energy_need,
            1.0 + 0.8 * hazard_proximity,
            1.0 + 0.8 * hazard_proximity,
            1.0 + 0.8 * hazard_proximity,
            1.0,
            1.0 + 0.4 * energy_need,
        ]
        attended_features = [round(f * g, 4) for f, g in zip(features, gains)]
        a_t = {
            "hazard_salience": round(hazard_proximity, 4),
            "resource_salience": round(resource_proximity, 4),
            "energy_urgency": round(energy_need, 4),
            "attended_vector": attended_features,
        }

        # 3. WORKING MEMORY
        retrieved_memories = []
        if enable_memory_ops:
            memory.encode(
                MemoryClass.SENSORY,
                content={"state_key": obs["state_key"], "pos": obs["position"]},
                embedding=features,
                timestamp=float(tick),
                significance=0.35 + 0.4 * hazard_proximity,
                provenance="perception",
            )
            retrieved_memories = memory.retrieve(features, top_k=3, current_time=float(tick))
        mem_bias = 0.0
        if retrieved_memories:
            mem_bias = sum(m.reward_assoc for m in retrieved_memories) / len(retrieved_memories)
        m_t = {
            "total_records": len(memory.records),
            "retrieved_count": len(retrieved_memories),
            "memory_reward_bias": round(mem_bias, 4),
        }

        # 4. COGNITIVE PROCESSING (SONAPSE neural connectome step)
        neural_inputs = list(attended_features)
        if neural_inputs:
            neural_inputs[-1] = max(-1.0, min(1.0, neural_inputs[-1] + 0.25 * mem_bias))
        output_map = connectome.step(neural_inputs)
        output_vec = connectome.get_output_vector()
        hidden_acts = connectome.get_hidden_activations()
        g_t = {
            "active_neurons": connectome.active_neuron_count(),
            "active_synapses": len(connectome.active_synapses()),
            "density": round(connectome.current_density(), 4),
            "activation_entropy": round(shannon_entropy(hidden_acts), 4),
            "output_activations": [round(v, 4) for v in output_vec],
        }

        # 5. DECISION MAKING (blend Q-values, SONAPSE outputs, and PROCEDURAL memory heuristics)
        blended_logits = list(output_vec) if len(output_vec) == learner.num_actions else [0.0] * learner.num_actions
        if enable_memory_ops and retrieved_memories:
            proc_hits = memory.retrieve(features, memory_class=MemoryClass.PROCEDURAL, top_k=3, current_time=float(tick))
            for pm in proc_hits:
                best_act = pm.content.get("best_action")
                if best_act is not None and 0 <= int(best_act) < learner.num_actions:
                    blended_logits[int(best_act)] += 0.35 * pm.reward_assoc

        action = learner.select_action(
            obs["state_key"],
            neural_logits=blended_logits,
            neural_weight=neural_weight,
            features=features,
        )

        # 6. ACTION & 7. ENVIRONMENTAL FEEDBACK
        next_obs, reward, done, info = world.step_agent(organism_id, action)

        # 8. LEARNING (RL TD update + Dyna World Model + SONAPSE Action-Gated Plasticity)
        if enable_plasticity:
            td_error = learner.update(
                obs["state_key"],
                action,
                reward,
                next_obs["state_key"],
                done=done,
                ewc_lambda=ewc_lambda,
                features=features,
                next_features=next_obs["features"],
            )
        else:
            q_now = learner.get_q_values(obs["state_key"], features=features)[action]
            q_next = 0.0 if done else max(learner.get_q_values(next_obs["state_key"], features=next_obs["features"]))
            td_error = float(reward) + learner.discount_gamma * q_next - q_now
            learner.cumulative_reward += float(reward)
            learner.reward_history.append(float(reward))
            learner.td_error_history.append(float(td_error))
            if len(learner.reward_history) > 500:
                learner.reward_history = learner.reward_history[-500:]
                learner.td_error_history = learner.td_error_history[-500:]

        model_err = (
            world_model.observe_transition(obs["state_key"], action, next_obs["state_key"], reward)
            if enable_plasticity
            else 0.0
        )
        planning_updates = (
            world_model.plan_updates(learner, planning_steps=3)
            if (enable_plasticity and enable_memory_ops)
            else 0
        )

        weight_delta_norm = 0.0
        if enable_plasticity:
            weight_delta_norm = connectome.update_plasticity(
                reward_prediction_error=td_error,
                learning_rate=learning_rate,
                reg_lambda=reg_lambda,
                mode=plasticity_mode,
                ewc_lambda=ewc_lambda,
                chosen_action=action,
            )
        # Active Inference Variational Free Energy F_t & Epistemic Foraging Drive G_pi
        surprise_term = 0.5 * (float(model_err) ** 2 + float(td_error) ** 2)
        homeostatic_kl = 0.25 * ((1.0 - min(1.2, float(obs["energy"]) / 100.0)) ** 2) + 0.15 * float(g_t["activation_entropy"])
        variational_free_energy = surprise_term + homeostatic_kl
        epistemic_drive = (1.0 / (1.0 + len(world_model.transitions) / 12.0)) + 0.35 * min(1.5, abs(float(model_err)))

        w_t = {
            "known_transitions": len(world_model.transitions),
            "model_prediction_error": round(model_err, 4),
            "planning_updates": planning_updates,
            "td_error": round(td_error, 4),
            "variational_free_energy": round(variational_free_energy, 4),
            "epistemic_foraging_drive": round(epistemic_drive, 4),
        }

        # 9. MEMORY CONSOLIDATION & REPLAY
        consolidated_count = 0
        replay_count = 0
        replay_ms = 0.0
        if enable_memory_ops:
            significance = min(1.0, 0.3 + 0.5 * abs(reward) + 0.2 * abs(td_error))
            memory.encode(
                MemoryClass.WORKING if abs(reward) < 0.5 else MemoryClass.EPISODIC,
                content={
                    "state_key": obs["state_key"],
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "reward": round(reward, 4),
                    "next_state_key": next_obs["state_key"],
                },
                embedding=features,
                timestamp=float(tick),
                reward_assoc=reward,
                significance=significance,
                usefulness=min(1.0, 0.3 + 0.4 * abs(td_error)),
                provenance="interaction",
            )
            if reward >= 0.8:
                memory.encode(
                    MemoryClass.PROCEDURAL,
                    content={"state_key": obs["state_key"], "best_action": action, "reward": round(reward, 4)},
                    embedding=features,
                    timestamp=float(tick),
                    reward_assoc=reward,
                    significance=0.85,
                    provenance="reinforcement",
                )
            if tick % 4 == 0:
                replayed, replay_ms = memory.replay(batch_size=6)
                replay_count = len(replayed)
                for r in replayed:
                    c = r.content
                    if "state_key" in c and "action" in c and "next_state_key" in c:
                        act_int = int(c["action"])
                        rew_val = float(c.get("reward", r.reward_assoc))
                        if enable_plasticity:
                            learner.update(
                                c["state_key"],
                                act_int,
                                rew_val,
                                c["next_state_key"],
                                done=False,
                                ewc_lambda=ewc_lambda,
                                record_stats=False,
                                features=r.embedding,
                                next_features=r.embedding,
                            )
            if tick % 8 == 0:
                consolidated_count = memory.consolidate()
                memory.decay_and_forget(current_time=float(tick), decay_scale=0.35)

        # 10. ARCHITECTURAL ADAPTATION
        grown_synapses = 0
        pruned_synapses = 0
        if enable_structural_adaptation and tick > 0 and tick % 10 == 0:
            pruned_synapses = connectome.prune_weak_connections(min_weight=0.025)
            if abs(td_error) > 0.25 or reward > 0.5:
                grown_synapses = connectome.grow_connections(max_new=2)

        hist_entry = {
            "tick": tick,
            "action": action,
            "reward": round(reward, 4),
            "td_error": round(td_error, 4),
        }
        self.history.append(hist_entry)
        h_t = {
            "recent_steps": list(self.history)[-6:],
            "mean_recent_reward": round(
                sum(x["reward"] for x in self.history) / max(1, len(self.history)), 4
            ),
            "action_entropy": round(
                shannon_entropy([x["action"] for x in self.history], bins=5), 4
            ),
        }

        snapshot = CognitiveStateSnapshot(
            organism_id=organism_id,
            tick=tick,
            P_t=p_t,
            A_t=a_t,
            M_t=m_t,
            G_t=g_t,
            W_t=w_t,
            H_t=h_t,
            stage_metrics={
                "stages_executed": COGNITIVE_STAGES,
                "weight_delta_norm": round(weight_delta_norm, 5),
                "consolidated_count": consolidated_count,
                "replay_count": replay_count,
                "replay_overhead_ms": round(replay_ms, 4),
                "grown_synapses": grown_synapses,
                "pruned_synapses": pruned_synapses,
                "output_map": output_map,
            },
        )
        self.last_snapshot = snapshot

        return {
            "snapshot": snapshot,
            "action": action,
            "reward": reward,
            "done": done,
            "td_error": td_error,
            "info": info,
        }
