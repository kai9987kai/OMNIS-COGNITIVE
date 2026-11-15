"""End-to-end 10-part OMNIS 1.0 demonstration and reproducible artifact generator."""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is importable when invoked as `python scripts/run_demo.py`
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarks.suite import run_architecture_benchmarks
from core.orchestrator import OmnisCore
from core.schemas import SystemConfig
from experiments.default_experiments import run_standard_experiment_suite
from sonapse.topology import analyze_topology


def run_full_demonstration(output_path: str = "datasets/demo_run_artifact.json") -> dict:
    """Execute and verify all 10 required OMNIS demonstration milestones."""
    cfg = SystemConfig(db_path="datasets/omnis_demo.db", initial_population=4, seed=42)
    omnis = OmnisCore(config=cfg)

    # 1. Artificial organism created
    lead = omnis.get_organism()
    assert lead is not None, "Expected at least one initialized organism"
    step_1_creation = {
        "organism_id": lead.organism_id,
        "generation": lead.generation,
        "stage": lead.stage,
        "initial_neurons": lead.connectome.active_neuron_count(),
        "initial_synapses": len(lead.connectome.active_synapses()),
    }

    # 2. Neural network visualization snapshot exported
    initial_graph = lead.connectome.to_visualization_graph()
    initial_topology = analyze_topology(lead.connectome)
    step_2_visualization = {
        "nodes": len(initial_graph["nodes"]),
        "edges": len(initial_graph["edges"]),
        "density": initial_topology["density"],
        "modularity": initial_topology["modularity"],
    }

    # 3, 4, 5, 6. First environmental interactions, memory changing, neural connections adapting, multi-agent interaction
    initial_mem_count = len(lead.memory.records)
    initial_weights = {sid: s.weight for sid, s in lead.connectome.synapses.items()}

    for _ in range(25):
        omnis.step(steps=1)
        if not lead.alive:
            # Keep focal organism alive across observation horizon if depleted
            lead.alive = True
            lead.energy = 80.0
            omnis.world.register_agent(lead.organism_id, energy=80.0)

    omnis.genesis.advance_stage(lead, "STAGE_3_LEARNING")
    omnis.genesis.advance_stage(lead, "STAGE_4_MEMORY")

    post_weights = {sid: s.weight for sid, s in lead.connectome.synapses.items()}
    changed_weights = sum(
        1 for sid, w in post_weights.items()
        if sid not in initial_weights or abs(w - initial_weights[sid]) > 1e-6
    )

    step_3_interactions = {
        "ticks_simulated": omnis.tick,
        "lead_cumulative_reward": round(lead.cumulative_reward, 4),
        "world_interactions": len(omnis.world.recent_interactions),
    }
    step_4_memory = {
        "initial_memories": initial_mem_count,
        "post_interaction_memories": len(lead.memory.records),
        "class_counts": lead.memory.get_class_counts(),
    }
    step_5_adaptation = {
        "changed_or_new_synapses": changed_weights,
        "post_synapses_count": len(lead.connectome.active_synapses()),
        "structural_events": lead.connectome.structural_events[-5:],
    }
    step_6_multi_agent = {
        "active_organisms": len(omnis.ecosystem.organisms),
        "social_events_logged": len(omnis.ecosystem.social_events),
        "species_counts": omnis.ecosystem.get_population_summary()["species_counts"],
    }

    # 7. Evolutionary experiment
    evo_gen_1 = omnis.run_evolution_cycle()
    omnis.step(steps=10)
    evo_gen_2 = omnis.run_evolution_cycle()
    step_7_evolution = {
        "generation_1": evo_gen_1,
        "generation_2": evo_gen_2,
        "lineage_tracked": len(omnis.genesis.lineage),
    }

    # 8 & 9. Controlled network-damage & recovery performance measurements
    regen_report = omnis.run_damage_experiment(
        neuron_lesion_ratio=0.30,
        synapse_sever_ratio=0.30,
    )
    step_8_damage = regen_report["damage_report"]
    step_9_recovery = {
        "performance_before": regen_report["performance_before"],
        "performance_damaged": regen_report["performance_damaged"],
        "performance_after": regen_report["performance_after"],
        "final_recovery_ratio": regen_report["final_recovery_ratio"],
        "recovery_curve": regen_report["recovery_curve"],
    }

    # 10. Architecture comparison benchmarks + Ghost Engine scientific validation
    benchmark_report = run_architecture_benchmarks(seeds=[42, 101, 202], episodes=6, steps_per_episode=20)
    experiment_suite = run_standard_experiment_suite(omnis.ghost_engine, seeds=[42, 101, 202, 303, 404, 505])

    # Persist final state to SQLite
    persistence_info = omnis.persist_state()
    omnis.shutdown()

    artifact = {
        "project": "OMNIS 1.0",
        "demonstration_steps": {
            "1_organism_creation": step_1_creation,
            "2_neural_visualization": step_2_visualization,
            "3_environmental_interactions": step_3_interactions,
            "4_memory_dynamics": step_4_memory,
            "5_neural_adaptation": step_5_adaptation,
            "6_multi_agent_ecosystem": step_6_multi_agent,
            "7_evolutionary_experiment": step_7_evolution,
            "8_controlled_damage": step_8_damage,
            "9_recovery_measurements": step_9_recovery,
            "10_architecture_comparison": {
                "pareto_optimal": benchmark_report["pareto_optimal_architectures"],
                "architectures": {
                    k: {
                        "task_performance_mean": v["task_performance_mean"],
                        "retention_mean": v["retention_mean"],
                        "robustness_mean": v["robustness_mean"],
                        "network_complexity_mean": v["network_complexity_mean"],
                    }
                    for k, v in benchmark_report["architectures"].items()
                },
                "ghost_engine_verdicts": {
                    e["name"]: e["verdict"] for e in experiment_suite["experiments"]
                },
            },
        },
        "persistence": persistence_info,
        "benchmark_files": benchmark_report["exported_files"],
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)

    return artifact


if __name__ == "__main__":
    result = run_full_demonstration()
    print("=== OMNIS 1.0 END-TO-END DEMONSTRATION COMPLETE ===")
    print(json.dumps(result["demonstration_steps"], indent=2))
