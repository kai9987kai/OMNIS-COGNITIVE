"""OMNIS CORE central lifecycle, subsystem coordination, and Observatory API facade."""

from __future__ import annotations

from collections import deque
import json
import os
import threading
import time
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from core.events import EventBus
from core.governance import GovernanceManager, SelfImprovementEngine
from core.persistence import OmnisDatabase
from core.schemas import SCHEMA_VERSION, SystemConfig, utc_now_iso
from diamond.environment import DiamondWorld
from frontier import FrontierExtensionHook, build_frontier_registry
from frontier.memory_systems import (
    BTSPBinaryMemory,
    ContinuumMemorySystem,
    FastWeightsController,
    ModernHopfieldAssociativeMemory,
)
from frontier.models import ActiveDendriticLayer
from frontier.temporal import ContinuousThoughtMachine, KuramotoOscillatorBank
from genesis.lifecycle import GenesisEngine
from ghost_engine.laboratory import GhostEngine
from hardware.abstraction import HardwareManager
from morpheus.regenerator import MorpheusRegenerator
from neurohive.ecosystem import NeuroHiveEcosystem
from neurohive.organism import ArtificialOrganism
from sonapse.topology import analyze_topology
from supermix_aether.router import SupermixAether


class OmnisCore:
    """Central event-driven orchestration layer coordinating all OMNIS subsystems."""

    def __init__(self, config: Optional[SystemConfig] = None, config_path: Optional[str] = None) -> None:
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                self.config = SystemConfig(**json.load(fh))
        else:
            self.config = SystemConfig()

        self._lock = threading.RLock()
        self.running = True
        self.paused = False
        self.simulation_speed = 1.0
        self.tick = 0
        self.last_step_latency_ms = 0.0
        self.telemetry_history: Deque[Dict[str, Any]] = deque(maxlen=200)

        self.event_bus = EventBus(max_history=2000)
        self.db = OmnisDatabase(self.config.db_path)
        self.governance = GovernanceManager(
            quotas=self.config.quotas,
            db=self.db,
            event_bus=self.event_bus,
        )
        self.self_improvement = SelfImprovementEngine(self.governance, self.config)
        self.hardware = HardwareManager(active_backend=self.config.hardware_backend)
        self.supermix = SupermixAether(
            allow_external_apis=self.config.allow_external_apis,
            event_bus=self.event_bus,
        )

        self.world = DiamondWorld(
            width=self.config.world_width,
            height=self.config.world_height,
            seed=self.config.seed,
            continuous=self.config.continuous_world,
        )
        self.ecosystem = NeuroHiveEcosystem(
            world=self.world,
            governance=self.governance,
            event_bus=self.event_bus,
            max_population=self.config.quotas.max_population_size,
        )
        self.genesis = GenesisEngine(
            seed=self.config.seed,
            governance=self.governance,
            event_bus=self.event_bus,
        )
        self.morpheus = MorpheusRegenerator(event_bus=self.event_bus, seed=self.config.seed)
        self.ghost_engine = GhostEngine(db=self.db, event_bus=self.event_bus)
        self.frontier_hook = FrontierExtensionHook()
        self.frontier_registry = build_frontier_registry()
        self.kuramoto_bank = KuramotoOscillatorBank(seed=self.config.seed)
        self.fast_weights_ctrl = FastWeightsController(dim=8)
        self.hopfield_mem = ModernHopfieldAssociativeMemory(dim=8)
        self.ctm_live = ContinuousThoughtMachine(input_dim=8, hidden_dim=10, output_dim=5, thought_ticks=5, seed=self.config.seed)
        self.continuum_mem = ContinuumMemorySystem(dim=8, out_dim=5, seed=self.config.seed)
        self.btsp_mem = BTSPBinaryMemory(n_in=16, n_out=16, seed=self.config.seed)
        self.dendritic_ctrl = ActiveDendriticLayer(input_dim=8, context_dim=2, num_units=8, branches=3, seed=self.config.seed)
        self._register_frontier_observers()

        self._initialize_population()
        self.governance.record_audit(
            "omnis_core",
            "system_initialized",
            {"schema_version": SCHEMA_VERSION, "initial_population": len(self.ecosystem.organisms)},
        )

    def _register_frontier_observers(self) -> None:
        def _lead_feature_vector(lead: Any) -> Optional[np.ndarray]:
            if lead is None or lead.last_cognitive_state is None:
                return None
            feats = list(lead.last_cognitive_state.P_t.get("features", [0.0] * 8))[:8]
            while len(feats) < 8:
                feats.append(0.0)
            return np.asarray(feats, dtype=float)

        def _obs_kuramoto(ctx: Dict[str, Any]) -> Dict[str, Any]:
            lead = ctx.get("lead")
            if lead is not None and self.kuramoto_bank.n != lead.connectome.active_neuron_count():
                self.kuramoto_bank.bind_connectome(lead.connectome)
            res = self.kuramoto_bank.step(steps=4)
            if lead is not None:
                active_ids = sorted(n.neuron_id for n in lead.connectome.neurons.values() if n.active)
                for idx, nid in enumerate(active_ids):
                    if idx < len(self.kuramoto_bank.theta):
                        lead.connectome.neurons[nid].kuramoto_phase = round(float(self.kuramoto_bank.theta[idx]), 4)
            return res

        def _obs_fast_weights(ctx: Dict[str, Any]) -> Dict[str, Any]:
            feats = _lead_feature_vector(ctx.get("lead"))
            if feats is None:
                return {"bound": False}
            self.fast_weights_ctrl.bind(feats, feats)
            recalled = self.fast_weights_ctrl.query(feats)
            return {"bound": True, "recall_norm": round(float(np.linalg.norm(recalled)), 4)}

        def _obs_assoc(ctx: Dict[str, Any]) -> Dict[str, Any]:
            feats = _lead_feature_vector(ctx.get("lead"))
            if feats is None:
                return {"stored_patterns": len(self.hopfield_mem.patterns)}
            self.hopfield_mem.store(feats)
            recalled = self.hopfield_mem.retrieve(feats)
            return {"stored_patterns": len(self.hopfield_mem.patterns), "energy_norm": round(float(np.linalg.norm(recalled)), 4)}

        def _obs_ctm_synchrony(ctx: Dict[str, Any]) -> Dict[str, Any]:
            feats = _lead_feature_vector(ctx.get("lead"))
            if feats is None:
                return {"thought_ticks": self.ctm_live.thought_ticks, "sync_energy": 0.0, "top_action": 0}
            sync_vec = self.ctm_live.synchrony_features(feats)
            logits = self.ctm_live.w_sync @ sync_vec + self.ctm_live.b_out
            return {
                "thought_ticks": self.ctm_live.thought_ticks,
                "sync_energy": round(float(np.linalg.norm(sync_vec)), 4),
                "top_action": int(np.argmax(logits)),
            }

        def _obs_continuum_memory(ctx: Dict[str, Any]) -> Dict[str, Any]:
            lead = ctx.get("lead")
            feats = _lead_feature_vector(lead)
            if feats is None:
                return {"step_idx": self.continuum_mem.step_idx, "online_mse": 0.0, "slow_norm": 0.0}
            hist = list(getattr(lead.learner, "reward_history", []))
            r_last = float(hist[-1]) if hist else 0.0
            target = np.full(self.continuum_mem.out_dim, r_last * 0.25, dtype=float)
            mse = self.continuum_mem.update(feats, target)
            return {
                "step_idx": self.continuum_mem.step_idx,
                "online_mse": round(float(mse), 5),
                "slow_norm": round(float(np.linalg.norm(self.continuum_mem.w_slow)), 4),
            }

        def _obs_btsp(ctx: Dict[str, Any]) -> Dict[str, Any]:
            lead = ctx.get("lead")
            feats = _lead_feature_vector(lead)
            if feats is None:
                return {"active_binary_synapses": int(np.sum(self.btsp_mem.weights)), "recall_density": 0.0}
            # Quantize 8D sensor vector into 16D sparse binary pattern (pos/neg opponent coding)
            cue = np.zeros(16, dtype=float)
            for idx in range(min(8, len(feats))):
                cue[2 * idx] = 1.0 if feats[idx] >= 0.0 else 0.0
                cue[2 * idx + 1] = 1.0 if feats[idx] < 0.0 else 0.0
            hist = list(getattr(lead.learner, "reward_history", []))
            if hist and float(hist[-1]) > 0.2:
                self.btsp_mem.imprint_one_shot(cue, cue, plateau_prob=0.85)
            recalled = self.btsp_mem.recall(cue, k_active=4)
            return {
                "active_binary_synapses": int(np.sum(self.btsp_mem.weights)),
                "recall_density": round(float(np.mean(recalled)), 4),
            }

        def _obs_dendritic(ctx: Dict[str, Any]) -> Dict[str, Any]:
            lead = ctx.get("lead")
            feats = _lead_feature_vector(lead)
            if feats is None:
                return {"gated_output": 0.0, "active_branches": 0}
            task_id = int(lead.last_cognitive_state.P_t.get("task_id", 0))
            ctx_vec = np.array([1.0, 0.0] if task_id == 0 else [0.0, 1.0], dtype=float)
            out_val, _, gate_vec = self.dendritic_ctrl.forward(feats, ctx_vec)
            return {
                "gated_output": round(float(out_val), 4),
                "active_branches": int(np.sum(gate_vec > 0.0)),
            }

        self.frontier_hook.register_observer("kuramoto", _obs_kuramoto)
        self.frontier_hook.register_observer("fast_weights", _obs_fast_weights)
        self.frontier_hook.register_observer("associative_memory", _obs_assoc)
        self.frontier_hook.register_observer("ctm_synchrony", _obs_ctm_synchrony)
        self.frontier_hook.register_observer("continuum_memory", _obs_continuum_memory)
        self.frontier_hook.register_observer("btsp", _obs_btsp)
        self.frontier_hook.register_observer("dendritic", _obs_dendritic)

    def _initialize_population(self) -> None:
        models = ["rate_coded", "conventional", "lif", "threshold"]
        for i in range(self.config.initial_population):
            model = self.config.default_neuron_model if i == 0 else models[i % len(models)]
            org = self.genesis.create_organism(
                generation=0,
                neuron_model=model,
                learning_rate=self.config.learning_rate,
                reg_lambda=self.config.regularization_lambda,
            )
            self.genesis.advance_stage(org, "STAGE_2_DEVELOPMENT")
            self.ecosystem.add_organism(org)
            self.supermix.register_agent(org.organism_id, role=org.specialisation, capabilities=["forage", "learn"])

    def step(self, steps: int = 1) -> Dict[str, Any]:
        """Advance integrated simulation by `steps` ticks with resource & latency tracking."""
        with self._lock:
            last_eco: Dict[str, Any] = {}
            t0 = time.perf_counter()
            for _ in range(max(1, int(steps))):
                self.tick += 1
                last_eco = self.ecosystem.step(fitness_weights=self.config.fitness_weights)
                # Keep ecosystem populated if extinction occurs during long runs
                if not self.ecosystem.organisms:
                    org = self.genesis.create_organism(generation=self.genesis.generation_counter)
                    self.ecosystem.add_organism(org)
                # Periodic checkpoint & topology critic advice
                if self.tick % 10 == 0:
                    lead = self.get_organism()
                    if lead is not None:
                        lead.record_checkpoint(f"tick_{self.tick}")
                        topo = analyze_topology(lead.connectome)
                        self.supermix.route_task("topology_audit", topo)

                snap = {
                    "tick": self.tick,
                    "alive": len(self.ecosystem.organisms),
                    "mean_fitness": round(self.ecosystem.mean_fitness(), 4),
                    "total_memories": sum(len(o.memory.records) for o in self.ecosystem.organisms.values()),
                    "total_synapses": sum(len(o.connectome.active_synapses()) for o in self.ecosystem.organisms.values()),
                }
                self.telemetry_history.append(snap)
                if self.frontier_hook.spec.any_active():
                    self.frontier_hook.on_step({"tick": self.tick, "lead": self.get_organism()})

            self.last_step_latency_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, int(steps))
            self.event_bus.emit(
                "core.step_completed",
                source="omnis_core",
                tick=self.tick,
                payload={"steps": steps, "alive": len(self.ecosystem.organisms)},
            )
            return last_eco

    def get_organism(self, organism_id: Optional[str] = None) -> Optional[ArtificialOrganism]:
        with self._lock:
            if organism_id and organism_id in self.ecosystem.organisms:
                return self.ecosystem.organisms[organism_id]
            if not self.ecosystem.organisms:
                return None
            return max(self.ecosystem.organisms.values(), key=lambda o: o.fitness)

    def spawn_organism(self, neuron_model: Optional[str] = None, use_deep_rl: bool = False) -> Dict[str, Any]:
        with self._lock:
            org = self.genesis.create_organism(
                generation=self.genesis.generation_counter,
                neuron_model=neuron_model or self.config.default_neuron_model,
                learning_rate=self.config.learning_rate,
                reg_lambda=self.config.regularization_lambda,
                use_deep_rl=use_deep_rl,
            )
            added = self.ecosystem.add_organism(org)
            if not added:
                return {"added": False, "reason": "Population quota cap reached", "max_population": self.config.quotas.max_population_size}
            self.supermix.register_agent(org.organism_id, role=org.specialisation, capabilities=["forage", "learn"])
            return {"added": True, "organism": org.get_summary()}

    def run_evolution_cycle(self) -> Dict[str, Any]:
        with self._lock:
            return self.genesis.evolve_generation(
                self.ecosystem,
                mutation_rate=self.config.mutation_rate,
                crossover_rate=self.config.crossover_rate,
                elite_fraction=self.config.elite_fraction,
            )

    def run_damage_experiment(
        self,
        organism_id: Optional[str] = None,
        neuron_lesion_ratio: float = 0.30,
        synapse_sever_ratio: float = 0.30,
    ) -> Dict[str, Any]:
        with self._lock:
            org = self.get_organism(organism_id)
            if org is None:
                raise ValueError("No active organism available")
            return self.morpheus.run_regeneration_experiment(
                org,
                self.world,
                neuron_lesion_ratio=neuron_lesion_ratio,
                synapse_sever_ratio=synapse_sever_ratio,
            )

    def persist_state(self) -> Dict[str, Any]:
        with self._lock:
            saved_memories = 0
            saved_organisms = 0
            ts = utc_now_iso()
            for org in self.ecosystem.organisms.values():
                saved_memories += org.memory.persist_to_db(self.db)
                self.db.save_organism(org.to_dict(), ts)
                saved_organisms += 1
            snap_id = f"world_tick_{self.tick}"
            self.db.save_world_snapshot(snap_id, self.tick, ts, self.world.save_state())
            return {
                "saved_organisms": saved_organisms,
                "saved_memories": saved_memories,
                "world_snapshot_id": snap_id,
            }

    def get_command_centre_status(self) -> Dict[str, Any]:
        with self._lock:
            lead = self.get_organism()
            hw = self.hardware.get_telemetry(lead.connectome if lead else None)
            total_memories = sum(len(o.memory.records) for o in self.ecosystem.organisms.values())
            total_neurons = sum(o.connectome.active_neuron_count() for o in self.ecosystem.organisms.values())
            total_synapses = sum(len(o.connectome.active_synapses()) for o in self.ecosystem.organisms.values())
            mean_act = 0.0
            if lead is not None:
                acts = [abs(n.a) for n in lead.connectome.neurons.values() if n.active]
                mean_act = sum(acts) / max(1, len(acts))

            return {
                "schema_version": SCHEMA_VERSION,
                "status": "PAUSED" if self.paused else "OPERATIONAL",
                "tick": self.tick,
                "generation": self.genesis.generation_counter,
                "simulation_speed": self.simulation_speed,
                "active_organisms": len(self.ecosystem.organisms),
                "max_population": self.config.quotas.max_population_size,
                "active_experiments": len(self.ghost_engine.experiments),
                "mean_neural_activity": round(mean_act, 4),
                "total_active_neurons": total_neurons,
                "total_active_synapses": total_synapses,
                "total_memories": total_memories,
                "mean_fitness": round(self.ecosystem.mean_fitness(), 4),
                "step_latency_ms": round(self.last_step_latency_ms, 3),
                "throughput_steps_per_sec": round(1000.0 / max(0.05, self.last_step_latency_ms), 1) if self.last_step_latency_ms > 0 else 0.0,
                "hardware": hw,
                "event_counts": self.event_bus.get_counts(),
                "telemetry_history": list(self.telemetry_history)[-60:],
                "lead_organism_id": lead.organism_id if lead else None,
            }

    def reset(self) -> Dict[str, Any]:
        with self._lock:
            self.tick = 0
            self.telemetry_history.clear()
            self.world = DiamondWorld(
                width=self.config.world_width,
                height=self.config.world_height,
                seed=self.config.seed,
                continuous=self.config.continuous_world,
            )
            self.ecosystem = NeuroHiveEcosystem(
                world=self.world,
                governance=self.governance,
                event_bus=self.event_bus,
                max_population=self.config.quotas.max_population_size,
            )
            self.genesis = GenesisEngine(
                seed=self.config.seed,
                governance=self.governance,
                event_bus=self.event_bus,
            )
            self._initialize_population()
            return self.get_command_centre_status()

    def shutdown(self) -> None:
        with self._lock:
            self.running = False
            self.persist_state()
            self.db.close()
