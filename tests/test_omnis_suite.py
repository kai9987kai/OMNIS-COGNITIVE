"""Comprehensive 17-area automated verification suite for OMNIS 1.0."""

from __future__ import annotations

import math
import tempfile
import os

from fastapi.testclient import TestClient
import pytest

from core.events import EventBus
from core.governance import GovernanceManager, SelfImprovementEngine
from core.math_utils import compare_distributions, compute_pareto_front, multi_objective_score, shannon_entropy
from core.orchestrator import OmnisCore
from core.persistence import OmnisDatabase
from core.schemas import ResourceQuota, SystemConfig
from diamond.environment import DiamondWorld
from diamond.rl import DynaWorldModel, QAgentLearner
from genesis.lifecycle import GenesisEngine
from ghost_engine.laboratory import GhostEngine
from hardware.abstraction import HardwareManager
from mnemorph.memory import MemoryClass, MnemorphMemory, compute_importance
from morpheus.regenerator import MorpheusRegenerator
from neurohive.ecosystem import NeuroHiveEcosystem
from neurohive.organism import ArtificialOrganism
from observatory.api import create_observatory_app
from sonapse.connectome import Neuron, SonapseConnectome, activate_neuron
from sonapse.topology import analyze_topology
from supermix_aether.router import SpecialistModel, SupermixAether
import numpy as np

from frontier import (
    CapacityMatchedHarness,
    FrontierExtensionHook,
    FrontierSpec,
    build_frontier_registry,
)
from frontier.analysis import ConnectivityAnalyzer, TopKSparseDictionary, run_causal_emergence_benchmark
from frontier.biophysics import HodgkinHuxleyCable, export_connectome_to_nir, validate_nir_graph
from frontier.memory_systems import (
    BTSPBinaryMemory,
    ContinuumMemorySystem,
    FastWeightsController,
    ModernHopfieldAssociativeMemory,
)
from frontier.methodology import (
    AnytimeValidTester,
    DecisionSensitiveSelector,
    VerifierGatedQDSearch,
    run_faultline_ensemble_benchmark,
)
from frontier.models import ActiveDendriticLayer, TinyRecursiveModel, _generate_4x4_sudoku_pair
from frontier.optimizers import MuonOptimizer, RigLSparseTrainer, newton_schulz_orthogonalize
from frontier.temporal import ContinuousThoughtMachine, KuramotoOscillatorBank


# 1. Neuron activation correctness
def test_01_neuron_activation_correctness() -> None:
    a_conv, _ = activate_neuron(1.0, "conventional")
    assert math.isclose(a_conv, math.tanh(1.0), rel_tol=1e-6)

    a_rate, _ = activate_neuron(0.0, "rate_coded")
    assert math.isclose(a_rate, 0.5, rel_tol=1e-6)

    a_thr_hi, _ = activate_neuron(0.6, "threshold", threshold=0.5)
    a_thr_lo, _ = activate_neuron(0.4, "threshold", threshold=0.5)
    assert a_thr_hi == 1.0 and a_thr_lo == 0.0

    a_lif_spike, u_reset = activate_neuron(0.8, "lif", threshold=0.5)
    assert a_lif_spike == 1.0 and u_reset == 0.0

    conn = SonapseConnectome(input_dim=4, hidden_dim=4, output_dim=2, neuron_model="lif", seed=7)
    outs = conn.step([1.0, 0.8, -0.3, 0.5])
    assert len(outs) == 2


# 2. Synaptic update correctness
def test_02_synaptic_update_correctness() -> None:
    conn = SonapseConnectome(input_dim=3, hidden_dim=3, output_dim=2, neuron_model="rate_coded", seed=11)
    conn.step([0.9, 0.5, 0.2])
    conn.step([0.9, 0.5, 0.2])
    before = {sid: s.weight for sid, s in conn.synapses.items()}
    norm = conn.update_plasticity(reward_prediction_error=1.2, learning_rate=0.1, reg_lambda=0.001, mode="three_factor")
    assert norm > 0.0
    assert any(abs(conn.synapses[sid].weight - w) > 1e-6 for sid, w in before.items())
    for syn in conn.synapses.values():
        assert -conn.w_max <= syn.weight <= conn.w_max

    hebb_norm = conn.update_plasticity(reward_prediction_error=0.0, learning_rate=0.05, reg_lambda=0.001, mode="hebbian")
    assert hebb_norm >= 0.0


# 3. Connectome integrity
def test_03_connectome_integrity() -> None:
    conn = SonapseConnectome(
        input_dim=4, hidden_dim=6, output_dim=3,
        max_neurons=20, max_synapses=60, max_density=0.35, seed=19,
    )
    conn.grow_connections(max_new=20)
    topo = analyze_topology(conn)
    n = topo["num_neurons"]
    expected_density = topo["num_synapses"] / (n * (n - 1))
    assert math.isclose(topo["density"], expected_density, rel_tol=1e-6)
    assert topo["density"] <= conn.max_density + 1e-9
    assert len(conn.synapses) <= conn.max_synapses
    assert "clustering_coefficient" in topo and "modularity" in topo and "redundant_pathways" in topo

    # Isolated neuron detection
    conn.neurons["iso_0"] = Neuron(neuron_id="iso_0", role="hidden")
    topo_iso = analyze_topology(conn)
    assert "iso_0" in topo_iso["isolated_neurons"]


# 4. Memory encoding and retrieval
def test_04_memory_encoding_and_retrieval() -> None:
    imp = compute_importance(reward_assoc=1.0, novelty=0.8, significance=0.9, usefulness=0.7)
    assert 0.0 <= imp <= 1.0

    mem = MnemorphMemory(agent_id="agent_mem_test", max_memories=30, seed=5)
    for mc in MemoryClass:
        mem.encode(mc, content={"tag": mc.value}, embedding=[0.1, 0.2, 0.3, 0.4], reward_assoc=0.2)

    target = mem.encode(
        MemoryClass.EPISODIC,
        content={"goal": "harvest_north"},
        embedding=[0.9, 0.1, 0.0, 0.2],
        reward_assoc=1.5,
        significance=0.9,
    )
    # Contradictory memory to test conflict detection
    mem.encode(
        MemoryClass.EPISODIC,
        content={"goal": "hazard_north"},
        embedding=[0.91, 0.1, 0.0, 0.2],
        reward_assoc=-1.2,
        significance=0.8,
    )
    hits = mem.retrieve([0.9, 0.1, 0.0, 0.2], top_k=2)
    assert hits and hits[0].embedding[0] > 0.8
    assert target.retrieval_count >= 1 or hits[0].retrieval_count >= 1

    conflicts = mem.detect_conflicts(similarity_threshold=0.85)
    assert len(conflicts) >= 1

    promoted = mem.consolidate(importance_threshold=0.3)
    assert promoted >= 1
    replayed, overhead_ms = mem.replay(batch_size=4)
    assert len(replayed) > 0 and overhead_ms >= 0.0

    assert mem.delete_memory(target.memory_id) is True


# 5. Memory persistence
def test_05_memory_persistence(tmp_path) -> None:
    db_file = str(tmp_path / "mem_persist.db")
    db = OmnisDatabase(db_file)
    mem1 = MnemorphMemory(agent_id="agent_persist", max_memories=50)
    r1 = mem1.encode(MemoryClass.LONG_TERM, content={"fact": "beacon_nw"}, embedding=[1.0, 0.0, 0.5], reward_assoc=0.9)
    saved = mem1.persist_to_db(db)
    assert saved == 1

    mem2 = MnemorphMemory(agent_id="agent_persist", max_memories=50)
    restored = mem2.restore_from_db(db)
    assert restored == 1
    assert r1.memory_id in mem2.records
    assert mem2.records[r1.memory_id].content["fact"] == "beacon_nw"
    db.close()


# 6. Agent lifecycle
def test_06_agent_lifecycle() -> None:
    world = DiamondWorld(width=10, height=10, seed=12)
    org = ArtificialOrganism(seed=12)
    world.register_agent(org.organism_id, energy=10.0)
    res = org.step(world, tick=1)
    assert org.age == 1
    assert "fitness" in res
    assert len(org.developmental_history) >= 1

    # Simulate energy depletion -> death
    world.agents[org.organism_id]["energy"] = 0.2
    org.step(world, tick=2)
    assert org.alive is False


# 7. Evolutionary selection
def test_07_evolutionary_selection() -> None:
    world = DiamondWorld(width=12, height=12, seed=21)
    eco = NeuroHiveEcosystem(world=world, max_population=8)
    genesis = GenesisEngine(seed=21)

    for i in range(4):
        org = genesis.create_organism(seed=21 + i)
        org.fitness = float(i + 1)
        eco.add_organism(org)

    summary = genesis.evolve_generation(eco, target_population=6)
    assert summary["generation"] == 1
    assert summary["population_size"] == 6
    assert len(summary["offspring"]) >= 2
    assert len(genesis.lineage) >= 6


# 8. Population limits
def test_08_population_limits() -> None:
    quotas = ResourceQuota(max_population_size=4)
    gov = GovernanceManager(quotas=quotas)
    world = DiamondWorld(width=10, height=10, seed=3)
    eco = NeuroHiveEcosystem(world=world, governance=gov, max_population=4)
    genesis = GenesisEngine(seed=3, governance=gov)

    added_flags = []
    for i in range(7):
        added_flags.append(eco.add_organism(genesis.create_organism(seed=3 + i)))

    assert added_flags.count(True) == 4
    assert added_flags.count(False) == 3
    assert len(eco.organisms) <= 4


# 9. Reinforcement learning updates
def test_09_reinforcement_learning_updates() -> None:
    learner = QAgentLearner(num_actions=5, learning_rate=0.5, discount_gamma=0.9, seed=9)
    learner.q_table["s1"][2] = 1.0
    # delta_t = r + gamma * max_a Q(s1, a) - Q(s0, a0) = 2.0 + 0.9 * 1.0 - 0.0 = 2.9
    td = learner.update("s0", 1, reward=2.0, next_state_key="s1", done=False)
    assert math.isclose(td, 2.9, rel_tol=1e-6)
    assert math.isclose(learner.q_table["s0"][1], 0.5 * 2.9, rel_tol=1e-6)

    wm = DynaWorldModel(seed=9)
    wm.observe_transition("s0", 1, "s1", 2.0)
    updates = wm.plan_updates(learner, planning_steps=2)
    assert updates == 1


# 10. Environmental transitions
def test_10_environmental_transitions() -> None:
    world = DiamondWorld(width=10, height=10, seed=33)
    world.register_agent("ag1", x=2, y=2, energy=50.0)
    obs0 = world.observe("ag1")
    assert len(obs0["features"]) == 8

    obs1, reward, done, info = world.step_agent("ag1", 1)  # RIGHT
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "action_name" in info

    saved = world.save_state()
    world2 = DiamondWorld(width=10, height=10, seed=1)
    world2.load_state(saved)
    assert world2.agents["ag1"]["position" if "position" in world2.agents["ag1"] else "x"] == world.agents["ag1"]["x"]
    assert world2.obstacles == world.obstacles

    # Continuous world mode check
    c_world = DiamondWorld(width=10, height=10, seed=33, continuous=True)
    c_world.register_agent("c1", x=2.5, y=2.5)
    c_obs, _, _, _ = c_world.step_agent("c1", 1)
    assert isinstance(c_obs["position"][0], float)


# 11. Fault detection
def test_11_fault_detection() -> None:
    org = ArtificialOrganism(seed=44)
    morph = MorpheusRegenerator(seed=44)
    pre_faults = morph.detect_faults(org)
    assert pre_faults["fault_detected"] is False

    morph.introduce_damage(org, neuron_lesion_ratio=0.3, synapse_sever_ratio=0.3)
    post_faults = morph.detect_faults(org)
    assert post_faults["fault_detected"] is True
    assert len(post_faults["deactivated_neurons"]) >= 1
    assert len(post_faults["severed_synapses"]) >= 1


# 12. Network recovery
def test_12_network_recovery() -> None:
    world = DiamondWorld(width=12, height=12, seed=55)
    org = ArtificialOrganism(seed=55)
    world.register_agent(org.organism_id, energy=120.0)
    for t in range(10):
        org.step(world, tick=t + 1)
        world.step_world_dynamics()

    morph = MorpheusRegenerator(seed=55)
    res = morph.run_regeneration_experiment(org, world, neuron_lesion_ratio=0.35, synapse_sever_ratio=0.35, recovery_rounds=3)
    assert res["performance_before"] > 0.0
    assert res["performance_after"] >= res["performance_damaged"]
    assert res["final_recovery_ratio"] > 0.4
    assert len(res["recovery_curve"]) == 4


# 13. Event communication
def test_13_event_communication() -> None:
    bus = EventBus(max_history=50)
    received = []
    wildcard_received = []
    bus.subscribe("test.ping", lambda e: received.append(e.payload["val"]))
    bus.subscribe("*", lambda e: wildcard_received.append(e.event_type))

    bus.emit("test.ping", source="unit_test", payload={"val": 123}, tick=1)
    assert received == [123]
    assert wildcard_received == ["test.ping"]
    assert bus.get_counts()["test.ping"] == 1
    assert len(bus.get_history(limit=10)) == 1


# 14. Database persistence
def test_14_database_persistence(tmp_path) -> None:
    db = OmnisDatabase(str(tmp_path / "omnis_test.db"))
    org = ArtificialOrganism(seed=77)
    db.save_organism(org.to_dict(), "2026-11-15T00:00:00Z")
    loaded_org = db.load_organism(org.organism_id)
    assert loaded_org is not None and loaded_org["organism_id"] == org.organism_id

    db.save_experiment({"experiment_id": "exp_t1", "name": "test", "status": "completed", "created_at": "2026-11-15"})
    assert len(db.load_experiments()) == 1

    db.save_world_snapshot("snap_1", 10, "2026-11-15", {"width": 10, "height": 10})
    assert db.load_world_snapshot("snap_1")["width"] == 10

    db.append_audit_log("2026-11-15", "tester", "verify", {"ok": True})
    assert len(db.get_audit_logs()) == 1
    db.close()


# 15. API functionality
def test_15_api_functionality(tmp_path) -> None:
    cfg = SystemConfig(db_path=str(tmp_path / "api_test.db"), initial_population=4, seed=42)
    core = OmnisCore(config=cfg)
    app = create_observatory_app(core)
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/command-centre").status_code == 200
    assert client.post("/api/simulation/step", json={"steps": 3}).status_code == 200
    assert client.get("/api/connectome").status_code == 200
    assert client.get("/api/memory").status_code == 200
    assert client.post("/api/memory/consolidate").status_code == 200
    assert client.get("/api/ecosystem").status_code == 200
    assert client.post("/api/ecosystem/spawn", json={"neuron_model": "lif"}).status_code == 200
    assert client.post("/api/ecosystem/evolve").status_code == 200
    assert client.get("/api/world").status_code == 200
    assert client.get("/api/hardware").status_code == 200
    assert client.get("/api/timeline").status_code == 200
    assert client.post("/api/morpheus/damage-recover", json={"neuron_lesion_ratio": 0.25, "synapse_sever_ratio": 0.25}).status_code == 200
    assert client.post("/api/research/experiment", json={"name": "plasticity_vs_fixed", "seeds": [42, 101], "steps": 15}).status_code == 200
    core.shutdown()


# 16. Visualisation data integrity
def test_16_visualisation_data_integrity(tmp_path) -> None:
    cfg = SystemConfig(db_path=str(tmp_path / "viz_test.db"), initial_population=3, seed=8)
    core = OmnisCore(config=cfg)
    core.step(steps=5)
    app = create_observatory_app(core)
    client = TestClient(app)

    conn_data = client.get("/api/connectome").json()
    node_ids = {n["neuron_id"] for n in conn_data["graph"]["nodes"]}
    for n in conn_data["graph"]["nodes"]:
        assert len(n["pos_3d"]) == 3
        assert all(math.isfinite(coord) for coord in n["pos_3d"])
    for e in conn_data["graph"]["edges"]:
        assert e["source"] in node_ids and e["target"] in node_ids
        assert math.isfinite(e["weight"])

    mem_data = client.get("/api/memory").json()
    mem_node_ids = {n["id"] for n in mem_data["graph"]["nodes"]}
    for e in mem_data["graph"]["edges"]:
        assert e["source"] in mem_node_ids and e["target"] in mem_node_ids

    world_data = client.get("/api/world").json()
    assert world_data["width"] == cfg.world_width and world_data["height"] == cfg.world_height
    core.shutdown()


# 17. Reproducible experiment execution + Governance & Math verification
def test_17_reproducible_experiments_and_math(tmp_path) -> None:
    g1 = GhostEngine()
    r1 = g1.run_experiment(name="plasticity_vs_fixed", seeds=[42, 101], steps=18)
    g2 = GhostEngine()
    r2 = g2.run_experiment(name="plasticity_vs_fixed", seeds=[42, 101], steps=18)

    assert r1["results"]["control_values"] == r2["results"]["control_values"]
    assert r1["results"]["treatment_values"] == r2["results"]["treatment_values"]
    assert r1["statistical_analysis"]["p_value"] == r2["statistical_analysis"]["p_value"]

    # Math edge cases: zero variance & entropy & pareto
    cmp_zero = compare_distributions([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert math.isfinite(cmp_zero["p_value"]) and cmp_zero["significant"] is False
    assert shannon_entropy([0.5, 0.5], is_probabilities=True) == pytest.approx(1.0, rel=1e-6)
    assert compute_pareto_front([[1.0, 2.0], [2.0, 3.0], [2.0, 1.0]]) == [0, 2]
    assert math.isfinite(multi_objective_score(0.1, 0.2, 0.3, 0.4))

    # Hardware & Supermix-Aether checks
    hw = HardwareManager()
    tel = hw.get_telemetry()
    assert tel["neuromorphic_simulation"]["simulated"] is True
    assert tel["neuromorphic_simulation"]["physical_device_connected"] is False

    sm = SupermixAether(allow_external_apis=False)
    routed = sm.route_task("foraging", {"features": [0.4, 0.1, 0.5, 0.0, 0.0, 0.0, 0.1, 0.8]})
    assert routed["status"] == "ok" and len(routed["result"]["action_scores"]) == 5

    # Controlled Self-Improvement gate check
    db = OmnisDatabase(str(tmp_path / "gov_test.db"))
    gov = GovernanceManager(ResourceQuota(), db=db)
    sie = SelfImprovementEngine(gov, SystemConfig())
    prop = sie.analyze_bottlenecks({"architectures": {"D_FULL_OMNIS": {"retention_mean": 0.5, "task_performance_mean": 0.5}}})
    evaluated = sie.evaluate_proposal_sandbox(prop["proposal_id"], lambda c: {"composite_score": c.learning_rate})
    if evaluated["gate_passed"]:
        sie.approve_and_deploy(prop["proposal_id"])
        sie.rollback(0)
    db.close()


def test_18_deep_rl_learner_and_benchmark(tmp_path: Path):
    from benchmarks.suite import run_architecture_benchmarks
    from diamond.deep_rl import DeepQAgentLearner
    from diamond.environment import DiamondWorld
    from neurohive.organism import ArtificialOrganism

    learner = DeepQAgentLearner(seed=42)
    feats = [0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7]
    next_feats = [0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8]
    q0 = learner.get_q_values("s0", features=feats)
    assert len(q0) == 5
    td = learner.update("s0", 1, 1.5, "s1", done=False, features=feats, next_features=next_feats)
    assert isinstance(td, float)
    learner.consolidate_task()
    restored = DeepQAgentLearner.from_dict(learner.to_dict(), seed=42)
    assert len(restored.get_q_values("s0", features=feats)) == 5

    world = DiamondWorld(width=10, height=10, seed=42)
    org = ArtificialOrganism(organism_id="dqn_org", seed=42, use_deep_rl=True)
    world.register_agent(org.organism_id, energy=org.energy)
    for t in range(1, 6):
        org.step(world, tick=t)
    org_reloaded = ArtificialOrganism.from_dict(org.to_dict())
    assert org_reloaded.use_deep_rl is True
    assert isinstance(org_reloaded.learner, DeepQAgentLearner)

    bench = run_architecture_benchmarks(seeds=[42], episodes=4, steps_per_episode=12, output_dir=str(tmp_path))
    assert "E_DEEP_RL_OMNIS" in bench["architectures"]
    assert "E_vs_A_task_performance" in bench["comparisons"]
    assert "E_vs_D_task_performance" in bench["comparisons"]


def test_19_postgresql_translation_and_cursor_adapter():
    from core.persistence import PostgresCompatibleCursor, translate_sqlite_to_postgres

    sql = translate_sqlite_to_postgres(
        "CREATE TABLE IF NOT EXISTS demo (id INTEGER PRIMARY KEY AUTOINCREMENT, val REAL); "
        "SELECT * FROM demo WHERE id = ? AND val >= ?;"
    )
    assert "%s" in sql
    assert "?" not in sql
    assert "SERIAL PRIMARY KEY" in sql

    recorded: list = []

    class _MockRawCursor:
        description = [("id",), ("val",)]

        def execute(self, q, params=None):
            recorded.append((q, params))

        def executemany(self, q, seq):
            recorded.append((q, list(seq)))

        def fetchone(self):
            return (1, 3.5)

        def fetchall(self):
            return [(1, 3.5)]

        def close(self):
            pass

    class _MockRawConn:
        def cursor(self):
            return _MockRawCursor()

    adapter = PostgresCompatibleCursor(_MockRawConn())
    res_cur = adapter.execute("SELECT * FROM demo WHERE id = ?", (7,))
    assert recorded[-1][0] == "SELECT * FROM demo WHERE id = %s"
    assert recorded[-1][1] == (7,)
    assert res_cur.fetchone() == (1, 3.5)
    assert res_cur.fetchall() == [(1, 3.5)]


def test_20_governance_auth_timeout_and_manifest(tmp_path: Path):
    import pytest
    from fastapi.testclient import TestClient

    from core.governance import generate_reproducibility_manifest, run_with_timeout, verify_api_key
    from core.orchestrator import OmnisCore
    from core.schemas import SystemConfig
    from observatory.api import create_observatory_app

    open_cfg = SystemConfig(db_path=":memory:", require_api_auth=False)
    assert verify_api_key(None, open_cfg.api_key or "", required=open_cfg.require_api_auth) is True

    locked_cfg = SystemConfig(db_path=":memory:", initial_population=2, require_api_auth=True, api_key="omnis-secret-key")
    assert verify_api_key(None, locked_cfg.api_key, required=True) is False
    assert verify_api_key("wrong", locked_cfg.api_key, required=True) is False
    assert verify_api_key("omnis-secret-key", locked_cfg.api_key, required=True) is True

    omnis = OmnisCore(config=locked_cfg)
    client = TestClient(create_observatory_app(omnis))
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/command-centre").status_code == 401
    assert client.get("/api/command-centre", headers={"X-API-Key": "omnis-secret-key"}).status_code == 200
    omnis.shutdown()

    assert run_with_timeout(lambda: 99, timeout_sec=2.0) == 99
    with pytest.raises(TimeoutError):
        import time

        run_with_timeout(lambda: time.sleep(0.3), timeout_sec=0.05)

    manifest = generate_reproducibility_manifest(locked_cfg, extra={"suite": "test_20"})
    assert len(manifest["config_sha256"]) == 64
    assert manifest["extra"]["suite"] == "test_20"


def test_21_pytorch_tensor_backend_and_parallel_sweep():
    from hardware.abstraction import HardwareManager, PyTorchTensorBackend
    from hardware.distributed import ParallelDistributedBackend
    from sonapse.connectome import SonapseConnectome

    conn = SonapseConnectome(seed=42)
    pt = PyTorchTensorBackend()
    for precision in ("float32", "float16", "int8"):
        res = pt.run_connectome_tensor_pass(conn, [0.5] * 8, precision=precision)
        assert res["precision"] == precision
        assert res["neurons"] == len(conn.neurons)
        assert res["drive_norm"] >= 0.0

    mgr = HardwareManager()
    step_res = mgr.run_connectome_step(conn, [0.5] * 8, backend="pytorch_tensor", precision="float32")
    assert step_res["drive_norm"] >= 0.0

    dist = ParallelDistributedBackend(max_workers=2)
    sweep = dist.run_parallel_organism_sweep(steps=10, mode="thread")
    assert sweep["trials_evaluated"] == 4
    assert sweep["best_config"] is not None
    assert sweep["speedup_factor"] > 0


def test_22_supermix_torch_and_http_adapter():
    from supermix_aether.router import (
        ExternalHTTPModelAdapter,
        SpecialistModel,
        SupermixAether,
        TorchNeuralSpecialist,
    )

    torch_spec = TorchNeuralSpecialist()
    out = torch_spec({"features": [0.1, 0.4, 0.9, 0.2, 0.5, 0.3, 0.7, 0.8]})
    assert len(out["action_scores"]) == 5

    router = SupermixAether(allow_external_apis=False)
    http_adapter = ExternalHTTPModelAdapter(endpoint_url="http://127.0.0.1:9999/mock")
    ext_model = SpecialistModel(
        model_id="ext_test",
        task_types=["research_synthesis"],
        backend_type="external_optional",
        handler=http_adapter,
    )
    router.register_model(ext_model)
    assert router.models["ext_test"].healthy is False

    routed = router.route_task("decision_arbitration", {"features": [0.2] * 8})
    assert routed["status"] == "ok"
    assert routed["model_id"] is not None


def test_23_ghost_engine_all_canonical_hypotheses_supported():
    from experiments.default_experiments import run_standard_experiment_suite
    from ghost_engine.laboratory import GhostEngine

    engine = GhostEngine()
    suite = run_standard_experiment_suite(ghost_engine=engine)
    assert suite["count"] == 4
    for report in suite["experiments"]:
        assert report["verdict"] == "supported", f"Experiment {report['name']} failed: {report}"
        assert report["statistical_analysis"]["p_value"] < 0.05


def test_24_observatory_new_routes_and_all_12_package_protocols():
    from fastapi.testclient import TestClient

    from cognition.interfaces import CognitivePipelineProtocol
    from cognition.pipeline import CognitivePipeline
    from core.interfaces import EventBusProtocol, OrchestratorProtocol, PersistenceBackendProtocol
    from core.orchestrator import OmnisCore
    from core.schemas import SystemConfig
    from diamond.deep_rl import DeepQAgentLearner
    from diamond.environment import DiamondWorld
    from diamond.interfaces import RLLearnerProtocol, WorldEnvironmentProtocol, WorldModelProtocol
    from diamond.rl import DynaWorldModel, QAgentLearner
    from genesis.interfaces import GenesisEngineProtocol
    from ghost_engine.interfaces import LaboratoryProtocol
    from ghost_engine.laboratory import GhostEngine
    from hardware.distributed import ParallelDistributedBackend
    from hardware.interfaces import (
        DistributedSweepBackendProtocol,
        HardwareManagerProtocol,
        QuantumInspiredOptimizerProtocol,
        TensorComputeBackendProtocol,
        VectorComputeBackendProtocol,
    )
    from mnemorph.interfaces import MemoryHierarchyProtocol
    from morpheus.interfaces import RegeneratorProtocol
    from neurohive.interfaces import EcosystemProtocol, OrganismProtocol
    from observatory.api import create_observatory_app
    from observatory.interfaces import ObservatoryFactoryProtocol
    from sonapse.interfaces import ConnectomeProtocol, TopologyAnalyzerProtocol
    from sonapse.topology import analyze_topology
    from supermix_aether.interfaces import OrchestrationRouterProtocol, SpecialistModelCallableProtocol
    from supermix_aether.router import TorchNeuralSpecialist

    omnis = OmnisCore(config=SystemConfig(db_path=":memory:", initial_population=3, seed=42))
    org = next(iter(omnis.ecosystem.organisms.values()))

    assert isinstance(omnis.event_bus, EventBusProtocol)
    assert isinstance(omnis.db, PersistenceBackendProtocol)
    assert isinstance(omnis, OrchestratorProtocol)
    assert isinstance(CognitivePipeline(), CognitivePipelineProtocol)
    assert isinstance(org.connectome, ConnectomeProtocol)
    assert isinstance(analyze_topology, TopologyAnalyzerProtocol)
    assert isinstance(org.memory, MemoryHierarchyProtocol)
    assert isinstance(omnis.world, WorldEnvironmentProtocol)
    assert isinstance(org.learner, RLLearnerProtocol)
    assert isinstance(DeepQAgentLearner(seed=42), RLLearnerProtocol)
    assert isinstance(DynaWorldModel(seed=42), WorldModelProtocol)
    assert isinstance(org, OrganismProtocol)
    assert isinstance(omnis.ecosystem, EcosystemProtocol)
    assert isinstance(omnis.genesis, GenesisEngineProtocol)
    assert isinstance(omnis.morpheus, RegeneratorProtocol)
    assert isinstance(GhostEngine(), LaboratoryProtocol)
    assert isinstance(omnis.supermix, OrchestrationRouterProtocol)
    assert isinstance(TorchNeuralSpecialist(), SpecialistModelCallableProtocol)
    assert isinstance(omnis.hardware, HardwareManagerProtocol)
    assert isinstance(omnis.hardware.cpu_backend, VectorComputeBackendProtocol)
    assert isinstance(omnis.hardware.torch_backend, TensorComputeBackendProtocol)
    assert isinstance(omnis.hardware.quantum_inspired, QuantumInspiredOptimizerProtocol)
    assert isinstance(ParallelDistributedBackend(max_workers=2), DistributedSweepBackendProtocol)
    assert isinstance(create_observatory_app, ObservatoryFactoryProtocol)

    client = TestClient(create_observatory_app(omnis))
    assert client.post("/api/hardware/quantum-optimize", json={"iterations": 6}).status_code == 200
    assert client.post("/api/hardware/parallel-sweep", json={"steps": 10, "mode": "thread"}).status_code == 200
    assert client.post("/api/hardware/tensor-pass", json={"precision": "float32"}).status_code == 200
    assert client.post("/api/timeline/advance-stage", json={}).status_code == 200
    omnis.shutdown()


# 25. Live connectome signals, neuromodulated plasticity, and stimulation API
def test_25_live_signals_neuromodulated_plasticity_and_stimulation_api() -> None:
    conn = SonapseConnectome(input_dim=4, hidden_dim=6, output_dim=4, seed=42)
    stim = conn.stimulate_pathway(
        current=1.35,
        reward_signal=0.95,
        learning_rate=0.14,
        mode="neuromodulated",
        structural_adapt=True,
    )
    assert stim["plasticity_mode"] == "neuromodulated"
    assert stim["weight_delta_norm"] > 0.0
    assert stim["signals_emitted"] > 0
    for nm_key in ("dopamine", "acetylcholine", "norepinephrine", "serotonin"):
        assert nm_key in stim["neuromodulators"]

    assert len(conn.recent_signals) > 0
    sig0 = conn.recent_signals[0]
    assert {"pre_id", "post_id", "flux", "pre_role", "post_role"}.issubset(sig0.keys())
    assert len(conn.recent_plasticity_events) > 0
    ev0 = conn.recent_plasticity_events[0]
    assert {"synapse_id", "dw", "weight", "kind"}.issubset(ev0.keys())
    assert ev0["kind"] in ("LTP", "LTD")
    assert len(conn.plasticity_history) >= 1
    assert any(s.plasticity_trace > 0.0 for s in conn.synapses.values())

    vis = conn.to_visualization_graph()
    assert vis["last_weight_delta_norm"] > 0.0
    assert len(vis["recent_plasticity_events"]) > 0
    assert len(vis["plasticity_history"]) >= 1
    assert any("last_dw" in e and "last_flux" in e and "plasticity_trace" in e for e in vis["edges"])

    restored = SonapseConnectome.from_dict(conn.to_dict())
    assert restored.neuromodulators == conn.neuromodulators
    assert len(restored.plasticity_history) == len(conn.plasticity_history)
    assert len(restored.recent_plasticity_events) == len(conn.recent_plasticity_events)

    cfg = SystemConfig(
        seed=42,
        initial_population=4,
        grid_width=8,
        grid_height=8,
        db_path=":memory:",
    )
    omnis = OmnisCore(cfg)
    try:
        client = TestClient(create_observatory_app(omnis))

        bad_resp = client.post("/api/connectome/stimulate", json={"plasticity_mode": "invalid_mode"})
        assert bad_resp.status_code == 400

        stim_resp = client.post(
            "/api/connectome/stimulate",
            json={"plasticity_mode": "neuromodulated", "reward_signal": 0.9, "learning_rate": 0.12},
        )
        assert stim_resp.status_code == 200
        stim_data = stim_resp.json()
        assert stim_data["plasticity_mode"] == "neuromodulated"
        assert omnis.config.plasticity_mode == "neuromodulated"
        assert omnis.get_organism().genome["plasticity_mode"] == "neuromodulated"
        assert stim_data["stimulation"]["weight_delta_norm"] > 0.0
        assert len(stim_data["graph"]["recent_signals"]) > 0
        assert len(stim_data["graph"]["recent_plasticity_events"]) > 0

        tick_before = omnis.tick
        live_resp = client.post(
            "/api/connectome/plasticity-step",
            json={"steps": 2, "plasticity_mode": "hybrid", "reward_override": -0.75},
        )
        assert live_resp.status_code == 200
        live_data = live_resp.json()
        assert live_data["plasticity_mode"] == "hybrid"
        assert omnis.config.plasticity_mode == "hybrid"
        assert omnis.get_organism().genome["plasticity_mode"] == "hybrid"
        assert omnis.tick == tick_before + 2
        assert len(live_data["graph"]["plasticity_history"]) >= 1
        assert any(ev["kind"] == "LTD" for ev in live_data["graph"]["recent_plasticity_events"])
    finally:
        omnis.shutdown()


# 26. kai9987kai ecosystem innovations: Dual-Soma (DSAG), Liquid-LTC, NeuroGraft, Fractal, Causal Emergence, NexusSearch RRF, 2^4 Carrier Assay, Tether/FAULTLINE, and HYPERION-20
def test_26_kai_ecosystem_innovations_dual_soma_rrf_carrier_tether_hyperion() -> None:
    # 1. Dual-Soma & Liquid-LTC neuron activations
    a_ds, u_ds = activate_neuron(1.1, "dual_soma", threshold=0.45, soma2_u=0.6)
    a_ltc, u_ltc = activate_neuron(1.1, "liquid_ltc", threshold=0.45, tau=1.2)
    assert -1.0 <= a_ds <= 1.0 and math.isfinite(u_ds)
    assert -1.0 <= a_ltc <= 1.0 and math.isfinite(u_ltc)

    # 2. NeuroGraft, Fractal reseed, and Causal Emergence topology metrics
    conn = SonapseConnectome(input_dim=6, hidden_dim=8, output_dim=4, neuron_model="dual_soma", seed=42)
    graft_res = conn.inject_neurograft(max_grafts=3)
    assert graft_res["total_grafts"] >= 1
    frac_res = conn.reseed_fractal_topology(branch_factor=2)
    assert frac_res["repositioned_hidden"] == 8
    conn.step([0.6, -0.3, 0.8, 0.1, -0.4, 0.5])
    topo = analyze_topology(conn)
    assert topo["graft_synapses"] >= 1
    assert topo["causal_emergence_index"] >= 0.0
    assert topo["effective_information_macro"] > 0.0
    assert 1.0 <= topo["fractal_dimension_proxy"] <= 3.0

    # 3. NexusSearch BM25 + Cosine Vector Reciprocal Rank Fusion & 2^4 Carrier Archaeology Assay
    mem = MnemorphMemory(agent_id="org_kai_test", max_memories=100, seed=42)
    mem.encode(
        MemoryClass.EPISODIC,
        content={"action_name": "NORTH", "event": "resource_harvest_beacon", "reward": 1.25},
        embedding=[0.8, 0.2, 0.9, 0.1],
        reward_assoc=1.25,
        significance=0.9,
        provenance="interaction",
    )
    mem.encode(
        MemoryClass.WORKING,
        content={"action_name": "WEST", "event": "hazard_collision", "reward": -0.85},
        embedding=[-0.7, -0.4, 0.1, 0.8],
        reward_assoc=-0.85,
        significance=0.75,
        provenance="perception",
    )
    rrf_hits = mem.hybrid_retrieve(
        query_embedding=[0.75, 0.25, 0.85, 0.1],
        query_text="resource_harvest_beacon positive reward",
        top_k=2,
    )
    assert len(rrf_hits) == 2
    assert rrf_hits[0]["bm25_score"] > 0.0
    assert rrf_hits[0]["rrf_score"] >= rrf_hits[1]["rrf_score"]

    assay = mem.run_carrier_archaeology_assay(conn)
    assert len(assay["factorial_matrix"]) == 16
    assert assay["dominant_carrier"] in assay["carrier_capacities"]
    by_mask = {r["mask"]: r["retention_score"] for r in assay["factorial_matrix"]}
    assert by_mask["1111"] == 1.0 and by_mask["0000"] == 0.0

    # 4. Supermix-Tether & FAULTLINE audit + HYPERION-20 scheduler
    sm = SupermixAether(allow_external_apis=False)
    tether = sm.tether_arbitrate({"features": [0.55, -0.25, 0.8, -0.3, 0.2, 0.4, 0.1, 0.85]})
    assert tether["donors_evaluated"] >= 3
    assert 0 <= tether["consensus_action"] < 5
    assert 0.0 <= tether["faultline_correlated_risk"] <= 1.0

    hw = HardwareManager()
    sched = hw.schedule_workload(workload_type="dual_soma", complexity=1.6)
    assert sched["scheduler"] == "HYPERION-20"
    assert sched["total_service_classes"] == 20
    assert sched["assigned_service_class"] == "SC07_DUAL_SOMA_DSAG"

    # 5. GhostEngine new canonical hypothesis: dual_soma_graft_vs_baseline
    ge = GhostEngine()
    exp_rep = ge.run_experiment(name="dual_soma_graft_vs_baseline")
    assert exp_rep["verdict"] == "supported"
    assert exp_rep["statistical_analysis"]["mean_diff"] > 0.0

    # 6. Observatory API new routes + Active Inference Free Energy telemetry
    cfg = SystemConfig(seed=42, initial_population=3, grid_width=8, grid_height=8, db_path=":memory:")
    omnis = OmnisCore(cfg)
    try:
        omnis.step(steps=2)
        lead = omnis.get_organism()
        assert lead is not None and lead.last_cognitive_state is not None
        assert "variational_free_energy" in lead.last_cognitive_state.W_t
        assert "epistemic_foraging_drive" in lead.last_cognitive_state.W_t

        client = TestClient(create_observatory_app(omnis))
        r_graft = client.post(
            "/api/connectome/neurograft",
            json={"operation": "neurograft", "max_grafts": 3, "neuron_model": "dual_soma"},
        )
        assert r_graft.status_code == 200
        assert r_graft.json()["neuron_model"] == "dual_soma"
        assert r_graft.json()["topology"]["graft_synapses"] >= 1

        r_frac = client.post("/api/connectome/neurograft", json={"operation": "fractal", "branch_factor": 2})
        assert r_frac.status_code == 200

        r_hyb = client.post("/api/memory/hybrid-search", json={"query_text": "interaction", "top_k": 4})
        assert r_hyb.status_code == 200 and "results" in r_hyb.json()

        r_ca = client.post("/api/memory/carrier-assay")
        assert r_ca.status_code == 200 and len(r_ca.json()["factorial_matrix"]) == 16

        r_hyp = client.post("/api/hardware/hyperion-schedule", json={"workload_type": "neurograft", "complexity": 1.4})
        assert r_hyp.status_code == 200 and r_hyp.json()["assigned_service_class"] == "SC09_NEUROGRAFT_ROUTER"

        r_teth = client.post("/api/hardware/tether-audit", json={})
        assert r_teth.status_code == 200 and "faultline_correlated_risk" in r_teth.json()
    finally:
        omnis.shutdown()


# 27. CROSSWIRE Mealy safety contract, TESSERA rediscovery, COUNTERFORM system-ID, MOLT trace placement, and CableLab morphology
def test_27_crosswire_tessera_counterform_molt_cablelab_and_trophic_ecology() -> None:
    cfg = SystemConfig(seed=42, initial_population=4, grid_width=8, grid_height=8, db_path=":memory:")
    omnis = OmnisCore(cfg)
    try:
        # 1. Governance patch accepts neuromodulated, dual_soma, liquid_ltc
        patch = omnis.governance.validate_config_patch(
            {"plasticity_mode": "neuromodulated", "default_neuron_model": "dual_soma"}
        )
        assert patch["plasticity_mode"] == "neuromodulated"
        assert patch["default_neuron_model"] == "dual_soma"

        # 2. SYNAPSE-ATLAS & NeuroPulse-CableLab physical morphology metrics
        lead = omnis.get_organism()
        assert lead is not None
        topo = analyze_topology(lead.connectome)
        assert topo["total_wiring_length"] > 0.0
        assert topo["wiring_economy_efficiency"] > 0.0
        assert topo["mean_cable_delay_ms"] > 0.0

        # 3. TESSERA independent rediscovery & Evo-OMEGA trophic ecology
        omnis.step(steps=2)
        for org in omnis.ecosystem.organisms.values():
            org.connectome.inject_neurograft(max_grafts=2)
        pop_summary = omnis.ecosystem.get_population_summary()
        assert "trophic_guilds" in pop_summary
        assert "species_diversity_index" in pop_summary
        assert pop_summary["tessera_library"]["published_count"] >= 1

        # 4. COUNTERFORM competing physical-law system ID & tuned absorber
        cf = omnis.world.run_counterform_system_id(horizon=20)
        assert cf["winning_law"] == "LAW_2_DAMPED_DRIVEN_OSCILLATOR"
        assert cf["tuned_absorber"]["attenuation_db"] > 5.0
        assert len(cf["receipt_sha256"]) == 16

        # 5. CROSSWIRE Mealy transducer safety contract verification
        cw = omnis.governance.crosswire.verify_organism_contract(lead)
        assert cw["admitted_for_deployment"] is True
        assert cw["surviving_interpretations"] == 4

        # 6. MOLT trace-placement vs. tissue-loss, regrowth & world-shift assay
        molt = omnis.morpheus.run_molt_trace_placement_assay(lead, omnis.world, lesion_ratio=0.35)
        assert molt["best_trace_placement"] in molt["placements"]
        assert len(molt["placements"]) == 3

        # 7. Observatory API endpoints for all new capabilities
        client = TestClient(create_observatory_app(omnis))
        assert client.post("/api/ecosystem/tessera-discover").status_code == 200
        assert client.post("/api/world/counterform-id").status_code == 200
        assert client.post("/api/governance/crosswire-verify").status_code == 200
        assert client.post("/api/morpheus/molt-assay").status_code == 200
    finally:
        omnis.shutdown()


# 28. Frontier Research Pack, Capacity-Matched Control Harness, and NIR Export
def test_28_frontier_research_pack_capacity_matched_harness_and_nir_export() -> None:
    # 1. FrontierSpec specification contract & type safety
    spec = FrontierSpec()
    assert spec.enabled is False
    assert spec.any_active() is False
    spec_dict = spec.to_dict()
    assert "enabled" in spec_dict and "seed" in spec_dict

    with pytest.raises(TypeError, match="Unknown FrontierSpec fields"):
        FrontierSpec.from_dict({"unrecognized_frontier_field": 999})

    # 2. FrontierExtensionHook default-off neutrality & observer registration
    hook = FrontierExtensionHook(spec)
    assert hook.on_step({"tick": 0, "lead": None}) is None  # zero overhead when disabled
    cfg = hook.configure({"enabled": True, "enable_kuramoto": True})
    assert cfg["enabled"] is True
    assert hook.spec.any_active() is True
    hook_status = hook.get_status()
    assert hook_status["active"] is True

    # 3. Temporal: Kuramoto coupled oscillators & Continuous Thought Machine
    kb = KuramotoOscillatorBank(num_oscillators=8, seed=42)
    m = kb.step(steps=3)
    assert 0.0 <= m["order_parameter_R"] <= 1.0
    assert "phase_locking_index" in m

    ctm = ContinuousThoughtMachine(input_dim=4, hidden_dim=6, output_dim=2, thought_ticks=3, seed=42)
    s_feat = ctm.synchrony_features(np.ones(4))
    assert len(s_feat) == 21
    out_ctm = ctm.forward(np.ones(4))
    assert len(out_ctm) == 2

    # 4. Memory: Continuum Memory, One-Shot BTSP, Modern Hopfield & Fast Weights
    cms = ContinuumMemorySystem(dim=6, out_dim=2, seed=42)
    err = cms.update(np.ones(6), np.ones(2))
    assert isinstance(err, float)

    btsp = BTSPBinaryMemory(n_in=16, n_out=16, seed=42)
    btsp.imprint_one_shot(np.ones(16), np.ones(16))
    assert np.sum(btsp.recall(np.ones(16))) > 0

    mhop = ModernHopfieldAssociativeMemory(dim=8, beta=4.0)
    mhop.store(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    ret = mhop.retrieve(np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert ret[0] > ret[1]

    fw = FastWeightsController(dim=8)
    fw.bind(np.ones(8), np.ones(8))
    assert np.linalg.norm(fw.a_mat) > 0.0

    # 5. Models: Tiny Recursive Model on 4x4 Sudoku & Active Dendritic Layer
    puzz, sol = _generate_4x4_sudoku_pair(np.random.default_rng(42))
    trm = TinyRecursiveModel(latent_dim=8, recursion_steps=2, seed=42)
    solved = trm.solve(puzz)
    assert solved.shape == (4, 4)

    dend = ActiveDendriticLayer(input_dim=4, context_dim=3, num_units=4, branches=2, seed=42)
    out_val, h_val, gate_val = dend.forward(np.ones(4), np.ones(3))
    assert isinstance(out_val, float)
    loss = dend.train_step(np.ones(4), np.ones(3), target=1.0)
    assert isinstance(loss, float)

    # 6. Optimizers: Muon Newton-Schulz momentum & RigL dynamic sparse training
    g = np.random.randn(4, 6)
    orth = newton_schulz_orthogonalize(g, steps=3)
    assert orth.shape == (4, 6)
    muon = MuonOptimizer(shape=(4, 6))
    w_new = muon.step(np.zeros((4, 6)), g)
    assert w_new.shape == (4, 6)

    rigl = RigLSparseTrainer(shape=(4, 6), sparsity=0.67, seed=42)
    x_in = np.random.randn(10, 6)
    y_tgt = np.random.randn(10, 4)
    mse_rigl = rigl.train(x_in, y_tgt, epochs=5)
    assert isinstance(mse_rigl, float)
    assert np.sum(rigl.mask) <= 10

    # 7. Analysis & Methodology: TopK SAE, Connectivity, Anytime-Valid E-Values, EVSI, QD Search
    sae = TopKSparseDictionary(input_dim=6, dict_size=12, k_active=2, seed=42)
    z = sae.encode_topk(np.ones(6))
    assert np.count_nonzero(z) <= 2

    conn = SonapseConnectome(input_dim=4, hidden_dim=4, output_dim=2, seed=42)
    node_ids, traces = ConnectivityAnalyzer.record_connectome_traces(conn, steps=10, seed=42)
    conn_stats = ConnectivityAnalyzer.analyze(node_ids, traces)
    assert "mean_fc" in conn_stats and "mean_ec" in conn_stats

    tester = AnytimeValidTester(alpha=0.05, sigma=1.0)
    res_test = tester.test_stream([0.5, 0.8, 1.2])
    assert "final_e_value" in res_test
    assert res_test["threshold"] == 20.0

    selector = DecisionSensitiveSelector()
    scored = selector.score_candidates([{"prior_std": 0.8, "compute_cost": 1.0, "decision_stake": 2.0}])
    assert scored[0]["evsi_score"] > 0

    fault = run_faultline_ensemble_benchmark([42, 101])
    assert "verdict" in fault and "receipt_sha256" in fault

    qd = VerifierGatedQDSearch(grid_bins=2, seed=42)
    qd_res = qd.search(budget=4)
    assert "coverage" in qd_res

    # 8. Biophysics: Hodgkin-Huxley cable simulation & NIR neuromorphic export
    hh = HodgkinHuxleyCable(compartments=4, dt_ms=0.05)
    sim = hh.simulate(duration_ms=4.0, inject_current_uA=20.0)
    assert "propagation_fraction" in sim

    nir_graph = export_connectome_to_nir(conn)
    val = validate_nir_graph(nir_graph)
    assert val["valid"] is True
    assert val["nir_version"] == "0.4.0"

    # 9. Complete Frontier Module Registry (all 19 modules verified)
    reg = build_frontier_registry()
    assert len(reg) == 19
    expected_modules = {
        "ctm_synchrony",
        "kuramoto_binding",
        "continuum_memory",
        "btsp_binary_memory",
        "associative_memory",
        "fast_weights",
        "tiny_recursive_model",
        "dendritic_layer",
        "muon_optimizer",
        "rigl_sparse_training",
        "topk_sparse_dictionary",
        "functional_effective_connectivity",
        "causal_emergence_audit",
        "anytime_valid_statistics",
        "decision_sensitive_selection",
        "faultline_joint_failure",
        "verifier_gated_qd_search",
        "hodgkin_huxley_cable",
        "nir_interchange_export",
    }
    assert set(reg.keys()) == expected_modules

    # Capacity-matched evaluation receipt verification
    r_kur = reg["kuramoto_binding"].runner([42, 101, 202, 303, 404])
    assert r_kur["verdict"] == "improved"
    assert len(r_kur["receipt_sha256"]) == 64
    assert r_kur["vs_baseline"]["ci_excludes_zero"] is True

    # 10. Live OmnisCore Hook Integration & Observatory API Endpoints
    omnis = OmnisCore(SystemConfig(database_path=":memory:"))
    try:
        assert hasattr(omnis, "frontier_hook")
        assert hasattr(omnis, "frontier_registry")
        omnis.step(steps=2)
        assert len(omnis.frontier_hook.telemetry_history) == 0  # Default-off neutrality

        omnis.frontier_hook.configure({"enabled": True, "enable_kuramoto": True})
        omnis.step(steps=2)
        assert len(omnis.frontier_hook.telemetry_history) > 0  # Active observer telemetry

        client = TestClient(create_observatory_app(omnis))
        r_reg = client.get("/api/frontier/registry")
        assert r_reg.status_code == 200
        assert r_reg.json()["count"] == 19

        r_run = client.post("/api/frontier/run", json={"module_id": "kuramoto_binding", "seeds": [42, 101]})
        assert r_run.status_code == 200
        assert "receipt_sha256" in r_run.json()

        assert client.post("/api/frontier/run", json={"module_id": "nonexistent"}).status_code == 404

        r_suite = client.post(
            "/api/frontier/suite",
            json={"module_ids": ["kuramoto_binding", "nir_interchange_export"], "seeds": [42, 101]},
        )
        assert r_suite.status_code == 200
        assert r_suite.json()["modules_executed"] == 2

        r_nir = client.post("/api/frontier/nir-export")
        assert r_nir.status_code == 200
        assert r_nir.json()["validation"]["valid"] is True

        r_km = client.post("/api/frontier/kuramoto-step", json={"coupling_k": 2.5, "steps": 2})
        assert r_km.status_code == 200
        assert "order_parameter_R" in r_km.json()

        r_cfg = client.post("/api/frontier/hook-config", json={"enabled": True, "enable_kuramoto": True})
        assert r_cfg.status_code == 200
        assert r_cfg.json()["spec"]["enabled"] is True
    finally:
        omnis.shutdown()


# 29. Whole-Project Closed-Loop Frontier, BCM Metaplasticity, SSE Telemetry & All 19 Improved
def test_29_whole_project_closed_loop_frontier_bcm_sse_and_all_19_improved() -> None:
    from benchmarks.suite import ARCHITECTURES, _evaluate_single_seed

    # 1. All 19 frontier modules achieve strict 'improved' verdict under capacity-matched 5-seed CI discipline
    reg = build_frontier_registry()
    assert len(reg) == 19
    for mid, entry in reg.items():
        res = entry.runner([42, 101, 202, 303, 404])
        assert res["verdict"] == "improved", f"Module {mid} verdict was {res['verdict']}"
        assert len(res["receipt_sha256"]) == 64

    # 2. SONAPSE bcm_metaplastic sliding threshold & non-destructive FC/EC + Top-K SAE live diagnostics
    omnis = OmnisCore(SystemConfig(database_path=":memory:"))
    try:
        patch = omnis.governance.validate_config_patch({"plasticity_mode": "bcm_metaplastic"})
        assert patch["plasticity_mode"] == "bcm_metaplastic"

        lead = omnis.get_organism()
        assert lead is not None
        stim = lead.connectome.stimulate_pathway(mode="bcm_metaplastic", reward_signal=0.9, structural_adapt=True)
        assert stim["plasticity_mode"] == "bcm_metaplastic"
        diag = lead.connectome.compute_live_diagnostics(steps=12)
        assert diag["connectivity"]["mean_fc"] >= 0.0
        assert diag["connectivity"]["asymmetry_index"] >= 0.0
        assert diag["sparse_dictionary"]["causal_specificity_ratio"] >= 0.0

        # 3. Default-off hook neutrality vs all 7 frontier observers wired & phase sync onto neurons
        omnis.step(steps=1)
        assert len(omnis.frontier_hook.telemetry_history) == 0

        omnis.frontier_hook.configure(
            {
                "enabled": True,
                "enable_kuramoto": True,
                "enable_fast_weights": True,
                "enable_associative_memory": True,
                "enable_ctm_synchrony": True,
                "enable_continuum_memory": True,
                "enable_btsp": True,
                "enable_dendritic": True,
            }
        )
        omnis.step(steps=2)
        assert len(omnis.frontier_hook.telemetry_history) == 2
        latest_obs = omnis.frontier_hook.telemetry_history[-1]["observers"]
        assert set(latest_obs.keys()) == {
            "kuramoto",
            "fast_weights",
            "associative_memory",
            "ctm_synchrony",
            "continuum_memory",
            "btsp",
            "dendritic",
        }
        # Kuramoto phases written onto active lead connectome neurons
        current_lead = omnis.get_organism()
        assert current_lead is not None
        phases = [n.kuramoto_phase for n in current_lead.connectome.neurons.values() if n.active]
        assert any(abs(p) > 0.0 for p in phases)

        # 4. Architecture F closed-loop benchmark evaluation
        assert "F_FRONTIER_OMNIS" in ARCHITECTURES
        f_trial = _evaluate_single_seed("F_FRONTIER_OMNIS", seed=42, episodes=4, steps_per_episode=12)
        assert "task_performance" in f_trial and "robustness" in f_trial

        # 5. GhostEngine Anytime-Valid E-Value augmentation & 6th canonical hypothesis
        exp = omnis.ghost_engine.run_experiment(
            name="frontier_closed_loop_adaptation",
            seeds=[42, 101, 202, 303, 404, 505],
            steps=30,
        )
        assert "e_value" in exp["statistical_analysis"]
        assert "anytime_valid_reject" in exp["statistical_analysis"]
        assert exp["verdict"] in ("supported", "inconclusive")

        # 6. Observatory API: live diagnostics, SSE telemetry stream, 7-flag hook, and full 19-module suite
        client = TestClient(create_observatory_app(omnis))

        r_diag = client.post("/api/connectome/diagnostics")
        assert r_diag.status_code == 200
        d_json = r_diag.json()
        assert "connectivity" in d_json and "sparse_dictionary" in d_json

        r_sse = client.get("/api/stream/telemetry?max_events=2&interval_ms=150")
        assert r_sse.status_code == 200
        assert "text/event-stream" in r_sse.headers.get("content-type", "")
        assert "event: telemetry" in r_sse.text
        assert "data:" in r_sse.text

        r_hook = client.post(
            "/api/frontier/hook-config",
            json={
                "enabled": True,
                "enable_kuramoto": True,
                "enable_fast_weights": True,
                "enable_associative_memory": True,
                "enable_ctm_synchrony": True,
                "enable_continuum_memory": True,
                "enable_btsp": True,
                "enable_dendritic": True,
            },
        )
        assert r_hook.status_code == 200
        assert len(r_hook.json()["hook_status"]["registered_observers"]) == 7

        r_all = client.post("/api/frontier/suite", json={"seeds": [42, 101, 202, 303, 404]})
        assert r_all.status_code == 200
        s_json = r_all.json()
        assert s_json["modules_executed"] == 19
        assert s_json["improved_count"] == 19
    finally:
        omnis.shutdown()



