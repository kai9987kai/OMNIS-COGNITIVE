"""Parallel & distributed execution backend using ThreadPoolExecutor and ProcessPoolExecutor."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import time
from typing import Any, Callable, Dict, List, Optional


def _evaluate_organism_trial_worker(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level picklable worker for parallel OMNIS organism parameter sweeps."""
    from diamond.environment import DiamondWorld
    from neurohive.organism import ArtificialOrganism

    t0 = time.perf_counter()
    seed = int(spec.get("seed", 42))
    steps = int(spec.get("steps", 25))
    neuron_model = str(spec.get("neuron_model", "rate_coded"))
    learning_rate = float(spec.get("learning_rate", 0.08))
    reg_lambda = float(spec.get("reg_lambda", 0.002))
    use_deep_rl = bool(spec.get("use_deep_rl", False))
    world_size = int(spec.get("world_size", 10))

    world = DiamondWorld(width=world_size, height=world_size, seed=seed)
    org = ArtificialOrganism(
        organism_id=str(spec.get("label", f"sweep_{seed}_{neuron_model}")),
        seed=seed,
        neuron_model=neuron_model,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        use_deep_rl=use_deep_rl,
    )
    world.register_agent(org.organism_id, energy=org.energy)
    rewards: List[float] = []
    for tick in range(1, steps + 1):
        if not org.alive:
            break
        res = org.step(world, tick=tick)
        rewards.append(float(res["reward"]))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "label": org.organism_id,
        "seed": seed,
        "neuron_model": neuron_model,
        "use_deep_rl": use_deep_rl,
        "learning_rate": learning_rate,
        "reg_lambda": reg_lambda,
        "fitness": round(org.fitness, 4),
        "cumulative_reward": round(org.cumulative_reward, 4),
        "energy": round(org.energy, 2),
        "active_synapses": len(org.connectome.active_synapses()),
        "memories": len(org.memory.records),
        "steps_completed": len(rewards),
        "worker_time_ms": round(elapsed_ms, 3),
    }


class ParallelDistributedBackend:
    """Executes parallel OMNIS tasks across CPU threads or OS processes."""

    name = "parallel_workers"

    def __init__(self, max_workers: Optional[int] = None) -> None:
        cpu_n = os.cpu_count() or 2
        self.worker_count = max(1, min(max_workers or max(2, cpu_n // 2), 8))

    def map_parallel(
        self,
        fn: Callable[[Any], Any],
        items: List[Any],
        mode: str = "thread",
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        """Run a callable over items in parallel via ThreadPoolExecutor or ProcessPoolExecutor."""
        if not items:
            return {"results": [], "wall_time_ms": 0.0, "mode": mode, "workers": 0}

        t0 = time.perf_counter()
        executor_cls = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor
        workers = min(self.worker_count, len(items))
        results: List[Any] = [None] * len(items)
        with executor_cls(max_workers=workers) as pool:
            future_to_idx = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
            for fut in as_completed(future_to_idx, timeout=timeout_s):
                idx = future_to_idx[fut]
                results[idx] = fut.result()

        wall_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "results": results,
            "wall_time_ms": round(wall_ms, 3),
            "mode": mode,
            "workers": workers,
        }

    def run_parallel_organism_sweep(
        self,
        configs: Optional[List[Dict[str, Any]]] = None,
        steps: int = 25,
        mode: str = "thread",
    ) -> Dict[str, Any]:
        """Evaluate multiple organism/hyperparameter configurations concurrently."""
        if not configs:
            configs = [
                {"label": "rate_coded_s42", "seed": 42, "neuron_model": "rate_coded", "learning_rate": 0.08, "steps": steps},
                {"label": "lif_s101", "seed": 101, "neuron_model": "lif", "learning_rate": 0.09, "steps": steps},
                {"label": "threshold_s202", "seed": 202, "neuron_model": "threshold", "learning_rate": 0.07, "steps": steps},
                {"label": "deep_rl_s303", "seed": 303, "neuron_model": "rate_coded", "learning_rate": 0.06, "use_deep_rl": True, "steps": steps},
            ]
        else:
            configs = [{**c, "steps": int(c.get("steps", steps))} for c in configs]

        batch = self.map_parallel(_evaluate_organism_trial_worker, configs, mode=mode, timeout_s=60.0)
        results = batch["results"]
        worker_total_ms = sum(float(r.get("worker_time_ms", 0.0)) for r in results)
        wall_ms = max(0.001, float(batch["wall_time_ms"]))
        best = max(results, key=lambda r: float(r.get("fitness", 0.0))) if results else None
        return {
            "execution_mode": mode,
            "workers": batch["workers"],
            "trials_evaluated": len(results),
            "wall_time_ms": round(wall_ms, 3),
            "sum_worker_time_ms": round(worker_total_ms, 3),
            "speedup_factor": round(worker_total_ms / wall_ms, 3),
            "best_config": best,
            "results": results,
        }
