"""FastAPI router and application factory for the OMNIS Observatory."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from benchmarks.suite import run_architecture_benchmarks
from core.governance import verify_api_key
from core.orchestrator import OmnisCore
from core.schemas import SCHEMA_VERSION
from diamond.environment import DiamondWorld
from genesis.lifecycle import DEVELOPMENTAL_STAGES
from neurohive.organism import ArtificialOrganism
from sonapse.topology import analyze_topology


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=200)


class SpeedRequest(BaseModel):
    speed: float = Field(default=1.0, ge=0.1, le=20.0)


class TaskRequest(BaseModel):
    task_id: int = Field(default=0, ge=0, le=10)


class SpawnRequest(BaseModel):
    neuron_model: Optional[str] = Field(default=None)
    use_deep_rl: bool = Field(default=False)


class ParallelSweepRequest(BaseModel):
    steps: int = Field(default=20, ge=5, le=100)
    mode: str = Field(default="thread")


class QuantumOptimizeRequest(BaseModel):
    iterations: int = Field(default=14, ge=4, le=50)


class TensorPassRequest(BaseModel):
    precision: str = Field(default="float32")


class AdvanceStageRequest(BaseModel):
    organism_id: Optional[str] = None
    target_stage: Optional[str] = None


class DamageRequest(BaseModel):
    organism_id: Optional[str] = None
    neuron_lesion_ratio: float = Field(default=0.30, ge=0.05, le=0.80)
    synapse_sever_ratio: float = Field(default=0.30, ge=0.05, le=0.80)


class ExperimentRequest(BaseModel):
    name: str = Field(default="plasticity_vs_fixed")
    hypothesis: Optional[str] = None
    seeds: List[int] = Field(default_factory=lambda: [42, 101, 202])
    steps: int = Field(default=30, ge=10, le=200)


class BenchmarkRequest(BaseModel):
    seeds: List[int] = Field(default_factory=lambda: [42, 101, 202])
    episodes: int = Field(default=8, ge=4, le=30)
    steps_per_episode: int = Field(default=24, ge=10, le=80)


class ProposalApproveRequest(BaseModel):
    proposal_id: str
    approver: str = "researcher"


def create_observatory_app(core: Optional[OmnisCore] = None) -> FastAPI:
    """Create and configure the FastAPI application serving the OMNIS Observatory API and UI."""
    omnis = core or OmnisCore(config_path="configs/default_config.json")
    app = FastAPI(
        title="OMNIS 1.0 Scientific Observatory API",
        version=SCHEMA_VERSION,
        description="Unified Artificial Intelligence, Cognitive Architecture & Artificial Life Ecosystem API",
    )
    app.state.omnis = omnis

    def check_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
        if not verify_api_key(x_api_key, omnis.config.api_key, required=omnis.config.require_api_auth):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    api_router = APIRouter(prefix="/api", dependencies=[Depends(check_api_key)])

    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        index_path = os.path.join(static_dir, "index.html")
        if not os.path.exists(index_path):
            raise HTTPException(status_code=404, detail="Observatory frontend index.html not found")
        return FileResponse(index_path)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "tick": omnis.tick,
            "organisms": len(omnis.ecosystem.organisms),
        }

    # --- A. Command Centre & Simulation Lifecycle ---
    @api_router.get("/command-centre")
    def command_centre() -> Dict[str, Any]:
        return omnis.get_command_centre_status()

    @api_router.post("/simulation/step")
    def simulation_step(req: StepRequest) -> Dict[str, Any]:
        step_res = omnis.step(steps=req.steps)
        return {"step_result": step_res, "command_centre": omnis.get_command_centre_status()}

    @api_router.post("/simulation/pause")
    def simulation_pause() -> Dict[str, Any]:
        omnis.paused = True
        return {"paused": True, "status": "PAUSED"}

    @api_router.post("/simulation/resume")
    def simulation_resume() -> Dict[str, Any]:
        omnis.paused = False
        return {"paused": False, "status": "OPERATIONAL"}

    @api_router.post("/simulation/speed")
    def simulation_speed(req: SpeedRequest) -> Dict[str, Any]:
        omnis.simulation_speed = req.speed
        return {"simulation_speed": omnis.simulation_speed}

    @api_router.post("/simulation/reset")
    def simulation_reset() -> Dict[str, Any]:
        return omnis.reset()

    @api_router.post("/simulation/persist")
    def simulation_persist() -> Dict[str, Any]:
        return omnis.persist_state()

    @api_router.post("/world/task")
    def world_set_task(req: TaskRequest) -> Dict[str, Any]:
        omnis.world.set_task(req.task_id)
        return {"task_id": omnis.world.task_id, "beacon": list(omnis.world.beacon)}

    # --- B. Connectome Explorer (3D Neural Graph & Topology) ---
    @api_router.get("/connectome")
    def get_connectome(
        organism_id: Optional[str] = Query(default=None),
        checkpoint_index: Optional[int] = Query(default=None),
    ) -> Dict[str, Any]:
        org = omnis.get_organism(organism_id)
        if org is None:
            raise HTTPException(status_code=404, detail="Organism not found")
        target_conn = org.connectome
        if (
            checkpoint_index is not None
            and 0 <= checkpoint_index < len(org.developmental_history)
            and "connectome_snapshot" in org.developmental_history[checkpoint_index]
        ):
            from sonapse.connectome import SonapseConnectome
            target_conn = SonapseConnectome.from_dict(
                org.developmental_history[checkpoint_index]["connectome_snapshot"]
            )
        graph = target_conn.to_visualization_graph()
        topology = analyze_topology(target_conn)
        return {
            "organism_id": org.organism_id,
            "neuron_model": target_conn.neuron_model,
            "checkpoints_count": len(org.developmental_history),
            "checkpoint_index": checkpoint_index,
            "graph": graph,
            "topology": topology,
            "cognitive_state": org.last_cognitive_state.model_dump() if org.last_cognitive_state else None,
        }

    # --- C. Memory Observatory ---
    @api_router.get("/memory")
    def get_memory(organism_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
        org = omnis.get_organism(organism_id)
        if org is None:
            raise HTTPException(status_code=404, detail="Organism not found")
        return org.memory.get_observatory_snapshot()

    @api_router.post("/memory/consolidate")
    def trigger_memory_consolidation(organism_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
        org = omnis.get_organism(organism_id)
        if org is None:
            raise HTTPException(status_code=404, detail="Organism not found")
        promoted = org.memory.consolidate()
        replayed, overhead_ms = org.memory.replay(batch_size=8)
        return {
            "organism_id": org.organism_id,
            "promoted": promoted,
            "replayed": len(replayed),
            "replay_overhead_ms": round(overhead_ms, 4),
            "snapshot": org.memory.get_observatory_snapshot(),
        }

    # --- D. Artificial Life Laboratory ---
    @api_router.get("/ecosystem")
    def get_ecosystem() -> Dict[str, Any]:
        pop = omnis.ecosystem.get_population_summary()
        pop["evolution_history"] = omnis.genesis.evolution_history[-20:]
        pop["lineage_count"] = len(omnis.genesis.lineage)
        pop["lineage"] = list(omnis.genesis.lineage.values())[-40:]
        return pop

    @api_router.post("/ecosystem/spawn")
    def spawn_organism(req: SpawnRequest) -> Dict[str, Any]:
        return omnis.spawn_organism(neuron_model=req.neuron_model, use_deep_rl=req.use_deep_rl)

    @api_router.post("/ecosystem/evolve")
    def evolve_generation() -> Dict[str, Any]:
        summary = omnis.run_evolution_cycle()
        return {"evolution": summary, "ecosystem": omnis.ecosystem.get_population_summary()}

    # --- E. World Simulator ---
    @api_router.get("/world")
    def get_world() -> Dict[str, Any]:
        return omnis.world.get_observatory_view()

    # --- F. Research Laboratory & Self-Improvement ---
    @api_router.get("/research")
    def get_research() -> Dict[str, Any]:
        latest_bench = None
        bench_path = os.path.join("benchmarks", "results", "latest_benchmark.json")
        if os.path.exists(bench_path):
            import json
            with open(bench_path, "r", encoding="utf-8") as fh:
                latest_bench = json.load(fh)
        return {
            "experiments": omnis.ghost_engine.experiments[-20:],
            "latest_benchmark": latest_bench,
            "proposals": list(omnis.self_improvement.proposals.values()),
            "audit_logs": omnis.db.get_audit_logs(limit=30),
        }

    @api_router.post("/research/experiment")
    def run_experiment(req: ExperimentRequest) -> Dict[str, Any]:
        return omnis.ghost_engine.run_experiment(
            name=req.name,
            hypothesis=req.hypothesis,
            seeds=req.seeds,
            steps=req.steps,
        )

    @api_router.post("/research/benchmark")
    def run_benchmark(req: BenchmarkRequest) -> Dict[str, Any]:
        return run_architecture_benchmarks(
            seeds=req.seeds,
            episodes=req.episodes,
            steps_per_episode=req.steps_per_episode,
        )

    @api_router.post("/research/self-improve/propose")
    def propose_self_improvement() -> Dict[str, Any]:
        bench = run_architecture_benchmarks(seeds=[42, 101], episodes=6, steps_per_episode=16)
        prop = omnis.self_improvement.analyze_bottlenecks(bench)

        def sandbox_eval(cfg: Any) -> Dict[str, float]:
            from core.orchestrator import OmnisCore as SandboxCore
            cfg_copy = cfg.model_copy(update={"db_path": ":memory:", "initial_population": 3})
            sb = SandboxCore(config=cfg_copy)
            sb.step(steps=12)
            mean_fit = sb.ecosystem.mean_fitness()
            sb.db.close()
            return {"composite_score": mean_fit}

        evaluated = omnis.self_improvement.evaluate_proposal_sandbox(prop["proposal_id"], sandbox_eval)
        return evaluated

    @api_router.post("/research/self-improve/approve")
    def approve_self_improvement(req: ProposalApproveRequest) -> Dict[str, Any]:
        try:
            cfg = omnis.self_improvement.approve_and_deploy(req.proposal_id, approver=req.approver)
            omnis.config = cfg
            return {"status": "deployed", "config": cfg.model_dump()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # --- G. Hardware Observatory & Supermix-Aether ---
    @api_router.get("/hardware")
    def get_hardware() -> Dict[str, Any]:
        lead = omnis.get_organism()
        return {
            "hardware": omnis.hardware.get_telemetry(lead.connectome if lead else None),
            "supermix_aether": omnis.supermix.get_status(),
            "quotas": omnis.governance.quotas.model_dump(),
        }

    @api_router.post("/hardware/quantum-optimize")
    def run_quantum_optimize(req: QuantumOptimizeRequest) -> Dict[str, Any]:
        def objective(params: Dict[str, float]) -> float:
            w = DiamondWorld(width=8, height=8, seed=omnis.config.seed, num_obstacles=3, num_resources=10, num_hazards=2)
            org = ArtificialOrganism(
                seed=omnis.config.seed,
                learning_rate=params["learning_rate"],
                reg_lambda=params["reg_lambda"],
                neural_weight=params["neural_weight"],
            )
            w.register_agent(org.organism_id, energy=100.0)
            total_rew = 0.0
            for t in range(1, 9):
                res = org.step(w, tick=t)
                total_rew += float(res["reward"])
            return -total_rew + 0.5 * params["reg_lambda"]

        return omnis.hardware.quantum_inspired.optimize(
            objective_fn=objective,
            bounds={"learning_rate": [0.02, 0.22], "reg_lambda": [0.0005, 0.015], "neural_weight": [0.20, 0.75]},
            iterations=req.iterations,
        )

    @api_router.post("/hardware/parallel-sweep")
    def run_parallel_sweep(req: ParallelSweepRequest) -> Dict[str, Any]:
        mode = req.mode if req.mode in ("thread", "process") else "thread"
        return omnis.hardware.distributed.run_parallel_organism_sweep(steps=req.steps, mode=mode)

    @api_router.post("/hardware/tensor-pass")
    def run_tensor_pass(req: TensorPassRequest) -> Dict[str, Any]:
        lead = omnis.get_organism()
        conn = lead.connectome if lead is not None else ArtificialOrganism(seed=omnis.config.seed).connectome
        precision = req.precision if req.precision in ("float32", "float16", "int8") else "float32"
        return omnis.hardware.run_connectome_step(conn, [0.5] * 8, backend="pytorch_tensor", precision=precision)

    # --- H. Development Timeline & Morpheus Regeneration ---
    @api_router.get("/timeline")
    def get_timeline(organism_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
        org = omnis.get_organism(organism_id)
        if org is None:
            raise HTTPException(status_code=404, detail="Organism not found")
        checkpoints = [
            {k: v for k, v in ckpt.items() if k != "connectome_snapshot"}
            for ckpt in org.developmental_history
        ]
        comparison = (
            omnis.genesis.compare_checkpoints(org, 0, -1)
            if len(org.developmental_history) >= 2
            else None
        )
        return {
            "organism_id": org.organism_id,
            "current_stage": org.stage,
            "all_stages": DEVELOPMENTAL_STAGES,
            "checkpoints": checkpoints,
            "comparison_first_vs_latest": comparison,
        }

    @api_router.post("/timeline/advance-stage")
    def advance_developmental_stage(req: AdvanceStageRequest) -> Dict[str, Any]:
        org = omnis.get_organism(req.organism_id)
        if org is None:
            raise HTTPException(status_code=404, detail="Organism not found")
        new_stage = omnis.genesis.advance_stage(org, target_stage=req.target_stage)
        return {
            "organism_id": org.organism_id,
            "current_stage": new_stage,
            "all_stages": DEVELOPMENTAL_STAGES,
            "checkpoints_count": len(org.developmental_history),
        }

    @api_router.post("/morpheus/damage-recover")
    def run_morpheus_damage(req: DamageRequest) -> Dict[str, Any]:
        try:
            return omnis.run_damage_experiment(
                organism_id=req.organism_id,
                neuron_lesion_ratio=req.neuron_lesion_ratio,
                synapse_sever_ratio=req.synapse_sever_ratio,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @api_router.get("/events")
    def get_events(limit: int = Query(default=50, ge=1, le=500)) -> Dict[str, Any]:
        events = [e.model_dump() for e in omnis.event_bus.get_history(limit=limit)]
        return {"counts": omnis.event_bus.get_counts(), "events": events}

    app.include_router(api_router)
    return app
