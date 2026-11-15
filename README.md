# OMNIS 1.0

**Unified Artificial Intelligence, Cognitive Architecture & Artificial Life Ecosystem**

OMNIS 1.0 is an experimental scientific and software engineering platform designed to investigate whether artificial agents can develop more efficient, adaptive, and fault-tolerant computational architectures through interactions among dynamic neural connectomes (**SONAPSE**), multi-level adaptive memory (**MNEMORPH**), reinforcement learning in persistent environments (**DIAMOND**), multi-agent ecosystems (**NEUROHIVE**), developmental evolution (**GENESIS**), regenerative computing (**MORPHEUS**), and autonomous hypothesis-driven experimentation (**GHOST ENGINE**).

---

## Implemented vs. Planned Capabilities

| Capability | Status in OMNIS 1.0 | Notes |
|---|---|---|
| Core Orchestrator, EventBus, Quotas, Audit Trail, Auth | **Implemented** | `core/` with versioned schemas (`1.0.0`), dual SQLite + PostgreSQL (`psycopg` DSN) persistence, optional `X-API-Key` auth, `run_with_timeout`, and SHA-256 reproducibility manifests |
| Dynamic Neural Connectome (`G(t) = (V, E(t), W(t))`) | **Implemented** | 4 neuron models (`conventional`, `rate_coded`, `threshold`, `lif`), action-gated 3-factor R-STDP & Hebbian plasticity, EWC consolidation, NetworkX topology |
| 7-Class Adaptive Memory & Replay | **Implemented** | Sensory, working, episodic, semantic, procedural, long-term, experimental + durable persistence, prioritized replay, and conflict detection |
| Persistent World & RL (`DIAMOND`) | **Implemented** | Grid & continuous modes, obstacles, moving entities, regenerating energy, hazards, dynamic tasks, tabular Q-learning, Dyna world model, and PyTorch `DeepQAgentLearner` |
| Multi-Agent Ecosystem & Evolution | **Implemented** | Bounded populations, kin/species cooperation, competition, speciation, 8-stage development, crossover, mutation, Deep RL inheritance, and lineage DAG tracking |
| Regenerative Computing (`MORPHEUS`) | **Implemented** | Lesion simulation, fault detection, compensatory neurogenesis, bypass synaptogenesis, homeostatic scaling, and equivalent-condition `Recovery(t)` curves |
| Autonomous Research Lab (`GHOST ENGINE`) | **Implemented** | Seeded paired control-vs-treatment experiments (`plasticity_vs_fixed`, `memory_replay_retention`, `developmental_vs_fixed`, `morpheus_regeneration`), Welch/paired *t*-test, Cohen's *d*, and predefined validation gates |
| Specialist Routing (`SUPERMIX-AETHER`) | **Implemented** | Local heuristic, critic, and `TorchNeuralSpecialist` models + opt-in `ExternalHTTPModelAdapter` (`allow_external_apis=False` default) |
| Interactive 3D Observatory (`FastAPI` + Web UI) | **Implemented** | Panels A–H, Three.js 3D Connectome Explorer, Lineage Tree Canvas, A–E Pareto Frontier Canvas, 8-stage development strip, hardware controls, and optional `X-API-Key` enforcement |
| Hardware, Tensor & Parallel Compute | **Implemented (Local + Simulated)** | `CPUVectorBackend`, `PyTorchTensorBackend` (`float32`/`float16`/`int8`), `ParallelDistributedBackend` (multi-thread/process sweeps), simulated AER neuromorphic estimator, and simulated `QuantumInspiredOptimizer` (no physical QPU/neuromorphic hardware claimed) |
| Multi-Node Cluster Orchestration (Ray/MPI) | **Planned / Out of Scope** | Single-node thread/process parallelism is implemented via `ParallelDistributedBackend`; multi-node cluster scheduling remains a future extension |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite (24 Verification Areas)

```bash
pytest -v
```

### 3. Run Standalone Per-Subsystem Examples

Every subsystem under `core/`, `cognition/`, `sonapse/`, `mnemorph/`, `diamond/`, `neurohive/`, `genesis/`, `morpheus/`, `ghost_engine/`, `supermix_aether/`, `hardware/`, and `observatory/` provides `interfaces.py`, `README.md`, and a standalone demo:

```bash
python -m <package>.example
```

### 4. Run the 10-Step End-to-End Scientific Demonstration

```bash
python scripts/run_demo.py
```

This executes all 10 demonstration milestones, validates all 4 `GhostEngine` hypotheses, exports `datasets/demo_run_artifact.json`, and writes Architecture `A–E` benchmark reports to `benchmarks/results/latest_benchmark.json` and `benchmarks/results/latest_benchmark.md`.

### 5. Launch the OMNIS Observatory Web Workstation

```bash
python main.py --mode serve --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

---

## Repository Layout

Each of the 12 core subsystems includes typed `@runtime_checkable` `interfaces.py`, runnable `example.py`, and subsystem `README.md`:

- `core/`: Schemas, EventBus, statistical/information-theoretic math, dual SQLite + PostgreSQL persistence, governance, API auth, manifests, & `OmnisCore` orchestrator
- `cognition/`: 10-stage cognitive processing pipeline producing `C_t = {P_t, A_t, M_t, G_t, W_t, H_t}` with procedural memory integration
- `sonapse/`: Dynamic neural connectome (`G(t) = (V, E(t), W(t))`), action-gated R-STDP, & NetworkX topology analyzer
- `mnemorph/`: 7-tier memory hierarchy, importance scoring, consolidation, prioritized replay, and forgetting
- `diamond/`: Persistent grid/continuous world simulation, tabular Q-learning, Dyna world model, and PyTorch `DeepQAgentLearner`
- `neurohive/`: Organism abstraction (`use_deep_rl` configurable) & multi-agent ecosystem dynamics
- `genesis/`: 8-stage developmental lifecycle, evolutionary operators, and lineage tracking
- `morpheus/`: Controlled lesioning, fault detection, structural regeneration, and `Recovery(t)` measurement
- `ghost_engine/`: Reproducible paired hypothesis testing with predefined statistical validation criteria
- `supermix_aether/`: Local heuristic, PyTorch `TorchNeuralSpecialist`, and opt-in `ExternalHTTPModelAdapter` orchestration router
- `hardware/`: CPU vector, PyTorch tensor (`float32`/`float16`/`int8`), simulated AER neuromorphic, simulated quantum-inspired optimization, and `ParallelDistributedBackend`
- `observatory/`: FastAPI backend (with optional `X-API-Key` auth) and 8-panel 3D scientific Observatory frontend
- `benchmarks/`: Architecture A–E (`A_FIXED` through `E_DEEP_RL_OMNIS`) multi-seed benchmark suite and exported reports
- `experiments/`: Preconfigured reproducible research experiments
- `tests/`: 24-area automated pytest verification suite
- `documentation/`: Detailed architecture, mathematics, scientific methodology, and engineering guides
