# OMNIS 1.0

**Unified Artificial Intelligence, Cognitive Architecture & Artificial Life Ecosystem**

OMNIS 1.0 is an experimental scientific and software engineering platform designed to investigate whether artificial agents can develop more efficient, adaptive, and fault-tolerant computational architectures through interactions among dynamic neural connectomes (**SONAPSE**), multi-level adaptive memory (**MNEMORPH**), reinforcement learning in persistent environments (**DIAMOND**), multi-agent ecosystems (**NEUROHIVE**), developmental evolution (**GENESIS**), regenerative computing (**MORPHEUS**), and autonomous hypothesis-driven experimentation (**GHOST ENGINE**).

---

## Implemented vs. Planned Capabilities

| Capability | Status in OMNIS 1.0 | Notes |
|---|---|---|
| Core Orchestrator, EventBus, Quotas, Audit Trail, Auth | **Implemented** | `core/` with versioned schemas (`1.0.0`), dual SQLite + PostgreSQL (`psycopg` DSN) persistence, optional `X-API-Key` auth, `run_with_timeout`, and SHA-256 reproducibility manifests |
| Dynamic Neural Connectome (`G(t) = (V, E(t), W(t))`) | **Implemented** | 6 neuron models (`conventional`, `rate_coded`, `threshold`, `lif`, `dual_soma`, `liquid_ltc`), 5 plasticity modes (`three_factor`, `hebbian`, `hybrid`, `neuromodulated`, `bcm_metaplastic`), NeuroGraft & fractal topology, EWC consolidation, and NetworkX/CableLab morphology |
| 7-Class Adaptive Memory, Carrier Assay & Hybrid RRF | **Implemented** | Sensory, working, episodic, semantic, procedural, long-term, experimental + NexusSearch BM25+Vector RRF, `2^4` carrier factorial assay, durable persistence, and prioritized replay |
| Persistent World & RL (`DIAMOND`) | **Implemented** | Grid & continuous modes, COUNTERFORM system-ID, obstacles, moving entities, regenerating energy, hazards, tabular Q-learning, Dyna world model, and PyTorch `DeepQAgentLearner` |
| Multi-Agent Ecosystem & Evolution | **Implemented** | Trophic guilds, TESSERA rediscovery, bounded populations, 8-stage development, crossover, mutation, Deep RL inheritance, and lineage DAG tracking |
| Regenerative Computing (`MORPHEUS` & `MOLT`) | **Implemented** | Lesion simulation, bypass synaptogenesis, MOLT trace-placement assay, homeostatic scaling, and equivalent-condition `Recovery(t)` curves |
| Autonomous Research Lab (`GHOST ENGINE` & `CROSSWIRE`) | **Implemented** | Seeded paired control-vs-treatment experiments across 6 canonical hypotheses (including `dual_soma_graft_vs_baseline` and `frontier_closed_loop_adaptation`), Student-*t*, Cohen's *d*, Anytime-Valid E-Values, and CROSSWIRE safety verifier |
| Frontier Research Pack (`frontier/`, 19 Modules) | **Implemented** | Capacity-matched harness with 95% CIs, martingale e-values, and SHA-256 receipts across Temporal (CTM synchrony, Kuramoto), Memory (Continuum, BTSP, Hopfield, Fast Weights), Models (TRM Sudoku, k-WTA Dendrites), Optimizers (Muon, RigL), Analysis (Top-K SAE, FC/EC, Causal Emergence), Methodology (E-Values, EVSI, FAULTLINE, QD Search), and Biophysics (Hodgkin-Huxley cable, NIR v0.4.0 export), plugged into a neutral default-off `OmnisCore` hook with all 7 live observers |
| Specialist Routing (`SUPERMIX-AETHER`) | **Implemented** | Local heuristic, critic, `TorchNeuralSpecialist`, Supermix-Tether causal safety audit + opt-in `ExternalHTTPModelAdapter` (`allow_external_apis=False` default) |
| Interactive 3D Observatory (`FastAPI` + Web UI) | **Implemented** | Panels A–I (Command Centre through Frontier Lab), Three.js 3D Connectome with Kuramoto phase overlay, FC/EC + Top-K SAE live diagnostics, SSE live telemetry stream (`/api/stream/telemetry`), light/dark theme toggle, and optional `X-API-Key` enforcement |
| Hardware, Tensor & Parallel Compute | **Implemented (Local + Simulated)** | `CPUVectorBackend`, `PyTorchTensorBackend` (`float32`/`float16`/`int8`), HYPERION-20 scheduler, `ParallelDistributedBackend`, simulated AER neuromorphic estimator, and simulated `QuantumInspiredOptimizer` |
| Multi-Node Cluster Orchestration (Ray/MPI) | **Planned / Out of Scope** | Single-node thread/process parallelism is implemented via `ParallelDistributedBackend`; multi-node cluster scheduling remains a future extension |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite (29 Verification Areas)

```bash
pytest -v
```

### 3. Run the Frontier Research Pack Benchmark CLI

```bash
python main.py --mode frontier
```

### 4. Run Standalone Per-Subsystem Examples

Every subsystem under `core/`, `cognition/`, `sonapse/`, `mnemorph/`, `diamond/`, `neurohive/`, `genesis/`, `morpheus/`, `ghost_engine/`, `supermix_aether/`, `hardware/`, and `observatory/` provides `interfaces.py`, `README.md`, and a standalone demo:

```bash
python -m <package>.example
```

### 5. Run the 10-Step End-to-End Scientific Demonstration

```bash
python scripts/run_demo.py
```

This executes all 10 demonstration milestones, validates `GhostEngine` hypotheses, exports `datasets/demo_run_artifact.json`, and writes Architecture `A–F` benchmark reports to `benchmarks/results/latest_benchmark.json` and `benchmarks/results/latest_benchmark.md`.

### 6. Launch the OMNIS Observatory Web Workstation

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
- `observatory/`: FastAPI backend (with optional `X-API-Key` auth and SSE telemetry stream), 9-panel 3D scientific Observatory frontend (A–I), and light/dark theme switcher
- `frontier/`: 19 frontier neuroscience & neuromorphic modules, neutral default-off `OmnisCore` hook (7 observers), capacity-matched control harness, and NIR v0.4.0 export
- `benchmarks/`: Architecture A–F (`A_FIXED` through `F_FRONTIER_OMNIS`) multi-seed benchmark suite and exported reports
- `experiments/`: Preconfigured reproducible research experiments
- `tests/`: 29-area automated pytest verification suite
- `documentation/`: Detailed architecture, mathematics, scientific methodology, and engineering guides
