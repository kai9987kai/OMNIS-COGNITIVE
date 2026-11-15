# OMNIS 1.0 Engineering, API & Deployment Guide

## 1. Installation

### Prerequisites
- Python 3.11+ (verified on Python 3.12.10)
- `pip`

### Setup

```bash
pip install -r requirements.txt
```

Active capabilities & optional extensions:
- `torch`: Enables CUDA GPU detection in `HardwareManager`, `DeepQAgentLearner` (`E_DEEP_RL_OMNIS`), `PyTorchTensorBackend`, and `TorchNeuralSpecialist` (with deterministic NumPy fallbacks).
- `psycopg[binary]`: Enables live PostgreSQL DSN connectivity alongside OMNIS's built-in SQLite-to-PostgreSQL placeholder/DDL translation adapter.

---

## 2. Configuration Guide

Configuration is governed by `configs/default_config.json` and validated by `SystemConfig` (`core/schemas.py`). Key settings:
- `seed`: Master deterministic seed
- `db_path`: SQLite path (`datasets/omnis_state.db`), `:memory:`, or PostgreSQL DSN (`postgresql://...`)
- `require_api_auth` & `api_key`: Enables optional `X-API-Key` header enforcement on `/api/*` Observatory routes (`/` and `/api/health` remain public)
- `world_width`, `world_height`, `continuous_world`: DIAMOND world geometry & coordinate mode
- `default_neuron_model`: `rate_coded`, `conventional`, `threshold`, or `lif`
- `plasticity_mode`: `three_factor`, `hebbian`, or `hybrid`
- `quotas`: Hard safety bounds (`max_neurons_per_organism`, `max_synapses_per_organism`, `max_memories_per_organism`, `max_population_size`, `max_connectome_density`)
- `allow_external_apis`: `false` by default (gates `ExternalHTTPModelAdapter`)

---

## 3. Running & Testing

```bash
# Run all 24 unit & integration tests
pytest -v

# Run standalone verification example for any of the 12 subsystems
python -m <package>.example

# Run 10-part demonstration & generate datasets/demo_run_artifact.json + A–E benchmarks
python scripts/run_demo.py

# Run benchmark CLI mode
python main.py --mode benchmark

# Run Observatory FastAPI server
python main.py --mode serve --host 127.0.0.1 --port 8000
```

---

## 4. FastAPI Observatory Endpoints

All `/api/*` endpoints (except `/api/health`) honor optional `X-API-Key` authentication when `require_api_auth=true`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Public health check, schema version, current tick, organism count |
| `GET` | `/api/command-centre` | Global KPIs, telemetry history, hardware summary, event counts |
| `POST` | `/api/simulation/step` | Advance simulation by `{ "steps": N }` |
| `POST` | `/api/simulation/pause` / `resume` / `reset` / `persist` | Lifecycle & SQLite/PostgreSQL snapshot control |
| `POST` | `/api/world/task` | Switch DIAMOND task objective `{ "task_id": 0 \| 1 }` |
| `GET` | `/api/connectome` | 3D graph nodes/edges, topology metrics, `C_t`, checkpoint playback (`?checkpoint_index=`) |
| `GET` / `POST` | `/api/memory` · `/api/memory/consolidate` | Memory tier counts, association graph, conflict detector, consolidation & replay |
| `GET` / `POST` | `/api/ecosystem` · `/api/ecosystem/spawn` · `/api/ecosystem/evolve` | Population roster, lineage, social interactions, spawning (`use_deep_rl` option), evolution |
| `GET` | `/api/world` | Live DIAMOND grid/continuous state |
| `GET` / `POST` | `/api/research` · `/api/research/experiment` · `/api/research/benchmark` | Ghost Engine experiments & A–E benchmarks (`A_FIXED`..`E_DEEP_RL_OMNIS`) |
| `POST` | `/api/research/self-improve/propose` · `/approve` | Sandboxed self-improvement proposal, benchmark gate, and approval |
| `GET` | `/api/hardware` | CPU/RAM, GPU probe, simulated AER neuromorphic metrics, Supermix-Aether registry |
| `POST` | `/api/hardware/quantum-optimize` | Run simulated quantum-inspired hyperparameter annealing |
| `POST` | `/api/hardware/parallel-sweep` | Execute multi-thread/process `ParallelDistributedBackend` organism sweep |
| `POST` | `/api/hardware/tensor-pass` | Execute `PyTorchTensorBackend` connectome pass across precisions (`float32`/`float16`/`int8`) |
| `GET` / `POST` | `/api/timeline` · `/api/timeline/advance-stage` | Developmental checkpoint diffs & GENESIS 8-stage organism advancement |
| `POST` | `/api/morpheus/damage-recover` | Controlled Morpheus lesion injection & `Recovery(t)` rehabilitation |

---

## 5. Docker Deployment

```bash
docker build -t omnis:1.0 .
docker run -p 8000:8000 omnis:1.0
```

---

## 6. Troubleshooting

- **Port 8000 already in use**: Launch with `--port 8080`.
- **Three.js CDN unavailable offline**: The Observatory automatically falls back to its built-in 3D perspective Canvas renderer with full mouse orbit, zoom, animated pulses, and node picking.
- **SQLite locked or stale state**: Stop running server processes or pass a fresh `db_path` in config; click **Reset Ecosystem** in the Command Centre to reinitialize cleanly.
