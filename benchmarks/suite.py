"""Benchmark suite comparing Fixed (A), Adaptive (B), Adaptive+Memory (C), Full OMNIS (D), and Deep-RL OMNIS (E)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from core.math_utils import (
    compare_distributions,
    compute_pareto_front,
    multi_objective_score,
    statistical_summary,
)
from core.schemas import SCHEMA_VERSION, utc_now_iso
from diamond.environment import DiamondWorld
from frontier.memory_systems import ContinuumMemorySystem, ModernHopfieldAssociativeMemory
from genesis.lifecycle import GenesisEngine
from morpheus.regenerator import MorpheusRegenerator
from neurohive.organism import ArtificialOrganism


ARCHITECTURES = [
    "A_FIXED",
    "B_ADAPTIVE",
    "C_ADAPTIVE_MEMORY",
    "D_FULL_OMNIS",
    "E_DEEP_RL_OMNIS",
    "F_FRONTIER_OMNIS",
]


def _evaluate_single_seed(arch: str, seed: int, episodes: int = 8, steps_per_episode: int = 24) -> Dict[str, float]:
    t0 = time.perf_counter()
    world = DiamondWorld(width=14, height=14, seed=seed)
    genesis = GenesisEngine(seed=seed)
    morph = MorpheusRegenerator(seed=seed)

    use_deep_rl = arch == "E_DEEP_RL_OMNIS"
    is_frontier = arch == "F_FRONTIER_OMNIS"
    neuron_model = "dual_soma" if is_frontier else "rate_coded"
    org = genesis.create_organism(seed=seed, neuron_model=neuron_model, learning_rate=0.09, use_deep_rl=use_deep_rl)
    enable_plasticity = arch in ("B_ADAPTIVE", "C_ADAPTIVE_MEMORY", "D_FULL_OMNIS", "E_DEEP_RL_OMNIS", "F_FRONTIER_OMNIS")
    enable_structural = arch in ("B_ADAPTIVE", "C_ADAPTIVE_MEMORY", "D_FULL_OMNIS", "E_DEEP_RL_OMNIS", "F_FRONTIER_OMNIS")
    enable_memory = arch in ("C_ADAPTIVE_MEMORY", "D_FULL_OMNIS", "E_DEEP_RL_OMNIS", "F_FRONTIER_OMNIS")
    enable_developmental = arch in ("D_FULL_OMNIS", "E_DEEP_RL_OMNIS", "F_FRONTIER_OMNIS")

    frontier_ctx: Optional[Dict[str, Any]] = None
    if is_frontier:
        org.genome["plasticity_mode"] = "bcm_metaplastic"
        org.connectome.inject_neurograft(max_grafts=4)
        frontier_ctx = {
            "hopfield_mem": ModernHopfieldAssociativeMemory(dim=8, beta=5.0),
            "continuum_mem": ContinuumMemorySystem(dim=8, out_dim=5, seed=seed),
        }

    if enable_developmental:
        genesis.advance_stage(org, "STAGE_2_DEVELOPMENT")
        org.connectome.grow_connections(max_new=4)

    world.register_agent(org.organism_id, energy=120.0)
    episode_rewards: List[float] = []
    task0_before: List[float] = []
    task0_after: List[float] = []

    mid_ep = max(2, episodes // 2)
    switch_back_ep = max(mid_ep + 1, episodes - 2)

    tick = 0
    for ep in range(episodes):
        if ep == mid_ep:
            if enable_memory:
                org.learner.consolidate_task()
                org.connectome.consolidate_ewc()
            world.set_task(1)
        elif ep == switch_back_ep:
            world.set_task(0)

        ep_rew = 0.0
        for _ in range(steps_per_episode):
            tick += 1
            res = org.step(
                world,
                tick=tick,
                enable_plasticity=enable_plasticity,
                enable_structural_adaptation=enable_structural,
                enable_memory_ops=enable_memory,
                ewc_lambda=0.25 if enable_memory else 0.0,
                frontier_context=frontier_ctx,
            )
            world.step_world_dynamics()
            ep_rew += float(res["reward"])
            if not org.alive:
                org.alive = True
                org.energy = 100.0
                world.agents[org.organism_id]["alive"] = True
                world.agents[org.organism_id]["energy"] = 100.0

        episode_rewards.append(ep_rew)
        if ep < mid_ep:
            task0_before.append(ep_rew)
        elif ep >= switch_back_ep:
            task0_after.append(ep_rew)

    # Generalisation probe on unseen seed world layout
    unseen_world = DiamondWorld(width=14, height=14, seed=seed + 777)
    unseen_world.register_agent(org.organism_id, energy=100.0)
    gen_rew = 0.0
    for s in range(12):
        obs = unseen_world.observe(org.organism_id)
        org.connectome.step(obs["features"])
        act = org.learner.select_action(
            obs["state_key"],
            neural_logits=org.connectome.get_output_vector(),
            features=obs["features"],
        )
        _, r, _, _ = unseen_world.step_agent(org.organism_id, act)
        unseen_world.step_world_dynamics()
        gen_rew += r

    # Robustness probe via Morpheus controlled lesion
    ref_state = world.save_state()
    p_before = morph.evaluate_performance(org, ref_state, steps=8)["performance"]
    morph.introduce_damage(org, neuron_lesion_ratio=0.25, synapse_sever_ratio=0.25)
    if enable_developmental:
        morph.regenerate_step(org, world=world, learning_steps=6)
    elif enable_plasticity:
        org.connectome.grow_connections(max_new=2)
    p_after = morph.evaluate_performance(org, ref_state, steps=8)["performance"]
    robustness = p_after / max(1e-6, p_before)

    early_mean = sum(episode_rewards[:2]) / max(1, len(episode_rewards[:2]))
    late_mean = sum(episode_rewards[-2:]) / max(1, len(episode_rewards[-2:]))
    learning_speed = late_mean - early_mean

    retention = (
        max(0.0, min(1.5, (sum(task0_after) / len(task0_after) + 5.0) / max(0.5, sum(task0_before) / len(task0_before) + 5.0)))
        if task0_before and task0_after
        else 1.0
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    complexity = float(len(org.connectome.neurons) + len(org.connectome.active_synapses()))
    task_perf = float(sum(episode_rewards) / max(1, len(episode_rewards)))

    return {
        "task_performance": task_perf,
        "learning_speed": float(learning_speed),
        "retention": float(retention),
        "generalisation": float(gen_rew / 12.0),
        "robustness": float(robustness),
        "compute_cost_ms": float(elapsed_ms),
        "network_complexity": complexity,
    }


def run_architecture_benchmarks(
    seeds: Optional[List[int]] = None,
    episodes: int = 8,
    steps_per_episode: int = 24,
    output_dir: str = "benchmarks/results",
) -> Dict[str, Any]:
    """Run controlled multi-seed benchmark across Architectures A, B, C, D and export JSON & Markdown."""
    seeds = list(seeds or [42, 101, 202])
    per_arch_raw: Dict[str, List[Dict[str, float]]] = {}
    arch_summaries: Dict[str, Dict[str, Any]] = {}
    pareto_points: List[List[float]] = []

    for arch in ARCHITECTURES:
        trials = [_evaluate_single_seed(arch, s, episodes=episodes, steps_per_episode=steps_per_episode) for s in seeds]
        per_arch_raw[arch] = trials

        perf_s = statistical_summary([t["task_performance"] for t in trials])
        speed_s = statistical_summary([t["learning_speed"] for t in trials])
        ret_s = statistical_summary([t["retention"] for t in trials])
        gen_s = statistical_summary([t["generalisation"] for t in trials])
        rob_s = statistical_summary([t["robustness"] for t in trials])
        cost_s = statistical_summary([t["compute_cost_ms"] for t in trials])
        comp_s = statistical_summary([t["network_complexity"] for t in trials])

        # Multi-objective loss J(theta) (lower is better)
        j_theta = multi_objective_score(
            l_task=-perf_s["mean"],
            c_compute=cost_s["mean"] / 1000.0,
            l_memory=1.0 - ret_s["mean"],
            l_instability=max(0.0, 1.0 - rob_s["mean"]),
        )
        pareto_points.append([-perf_s["mean"], cost_s["mean"], 1.0 - ret_s["mean"], 1.0 - rob_s["mean"]])

        arch_summaries[arch] = {
            "task_performance_mean": round(perf_s["mean"], 4),
            "task_performance_std": round(perf_s["std"], 4),
            "learning_speed_mean": round(speed_s["mean"], 4),
            "retention_mean": round(ret_s["mean"], 4),
            "generalisation_mean": round(gen_s["mean"], 4),
            "robustness_mean": round(rob_s["mean"], 4),
            "compute_cost_mean_ms": round(cost_s["mean"], 3),
            "compute_cost_mean": round(cost_s["mean"] / 1000.0, 4),
            "network_complexity_mean": round(comp_s["mean"], 2),
            "multi_objective_J": round(j_theta, 4),
            "trials": trials,
        }

    pareto_idx = compute_pareto_front(pareto_points)
    pareto_optimal = [ARCHITECTURES[i] for i in pareto_idx]

    # Statistical comparison of D_FULL_OMNIS vs A_FIXED
    comp_d_vs_a = compare_distributions(
        [t["task_performance"] for t in per_arch_raw["A_FIXED"]],
        [t["task_performance"] for t in per_arch_raw["D_FULL_OMNIS"]],
    )
    rob_d_vs_a = compare_distributions(
        [t["robustness"] for t in per_arch_raw["A_FIXED"]],
        [t["robustness"] for t in per_arch_raw["D_FULL_OMNIS"]],
    )
    comp_e_vs_a = compare_distributions(
        [t["task_performance"] for t in per_arch_raw["A_FIXED"]],
        [t["task_performance"] for t in per_arch_raw["E_DEEP_RL_OMNIS"]],
    )
    comp_e_vs_d = compare_distributions(
        [t["task_performance"] for t in per_arch_raw["D_FULL_OMNIS"]],
        [t["task_performance"] for t in per_arch_raw["E_DEEP_RL_OMNIS"]],
    )
    comp_f_vs_a = compare_distributions(
        [t["task_performance"] for t in per_arch_raw["A_FIXED"]],
        [t["task_performance"] for t in per_arch_raw["F_FRONTIER_OMNIS"]],
    )
    comp_f_vs_d = compare_distributions(
        [t["task_performance"] for t in per_arch_raw["D_FULL_OMNIS"]],
        [t["task_performance"] for t in per_arch_raw["F_FRONTIER_OMNIS"]],
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "seeds": seeds,
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "architectures": arch_summaries,
        "pareto_optimal_architectures": pareto_optimal,
        "comparisons": {
            "D_vs_A_task_performance": comp_d_vs_a,
            "D_vs_A_robustness": rob_d_vs_a,
            "E_vs_A_task_performance": comp_e_vs_a,
            "E_vs_D_task_performance": comp_e_vs_d,
            "F_vs_A_task_performance": comp_f_vs_a,
            "F_vs_D_task_performance": comp_f_vs_d,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "latest_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    md_path = os.path.join(output_dir, "latest_benchmark.md")
    lines = [
        "# OMNIS 1.0 Architecture Benchmark Report",
        "",
        f"- **Generated:** {report['generated_at']}",
        f"- **Seeds:** {seeds}",
        f"- **Episodes / Steps:** {episodes} x {steps_per_episode}",
        f"- **Pareto-Optimal Architectures:** {', '.join(pareto_optimal)}",
        "",
        "| Architecture | Task Perf (Mean ± Std) | Learning Speed | Retention | Generalisation | Robustness | Complexity | Compute (ms) | J(θ) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arch in ARCHITECTURES:
        s = arch_summaries[arch]
        lines.append(
            f"| `{arch}` | {s['task_performance_mean']:.3f} ± {s['task_performance_std']:.3f} | "
            f"{s['learning_speed_mean']:.3f} | {s['retention_mean']:.3f} | {s['generalisation_mean']:.3f} | "
            f"{s['robustness_mean']:.3f} | {s['network_complexity_mean']:.1f} | {s['compute_cost_mean_ms']:.2f} | {s['multi_objective_J']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Statistical Tests (D_FULL_OMNIS vs A_FIXED)",
            f"- **Task Performance:** diff = {comp_d_vs_a['mean_diff']:.4f}, p = {comp_d_vs_a['p_value']:.4f}, Cohen's d = {comp_d_vs_a['cohens_d']:.4f}, significant = {comp_d_vs_a['significant']}",
            f"- **Robustness (Recovery):** diff = {rob_d_vs_a['mean_diff']:.4f}, p = {rob_d_vs_a['p_value']:.4f}, Cohen's d = {rob_d_vs_a['cohens_d']:.4f}, significant = {rob_d_vs_a['significant']}",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    report["exported_files"] = {"json": json_path, "markdown": md_path}
    return report
