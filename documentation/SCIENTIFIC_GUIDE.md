# OMNIS 1.0 Scientific Methodology & Experimental Guide

## 1. Central Research Question

> *Can artificial agents develop increasingly efficient computational architectures through interaction between adaptive connectivity, memory, environmental experience, evolutionary selection, and autonomous experimentation?*

OMNIS does **not** assume that combining neural plasticity, memory, evolution, and reinforcement learning automatically guarantees superiority across all tasks. Instead, `GHOST ENGINE` and `benchmarks/suite.py` subject every hypothesis to controlled, multi-seed statistical testing.

---

## 2. Scientific Hypotheses Investigated

1. **H1 — Synaptic Plasticity (`plasticity_vs_fixed`)**: Reward-modulated three-factor plasticity + structural adaptation improves cumulative reward compared to fixed initial synaptic weights.
2. **H2 — Continual Learning & Replay (`memory_replay_retention`)**: Prioritized MNEMORPH replay and EWC anchoring improve retention on Task 0 after training on Task 1.
3. **H3 — Developmental Advantage (`developmental_vs_fixed`)**: Full developmental neurogenesis + memory + plasticity outperforms minimal fixed architectures.
4. **H4 — Regenerative Fault Tolerance (`morpheus_regeneration`)**: Non-reset structural bypass synaptogenesis + homeostatic scaling + replay improves `Recovery(t)` relative to unassisted lesioned networks.

---

## 3. Experimental Cycle (`GHOST ENGINE`)

Every experiment in `ghost_engine/laboratory.py` executes:

`OBSERVE → HYPOTHESISE → DESIGN → EXECUTE → MEASURE → VALIDATE → RECORD`

Each exported report records:
- Unique `experiment_id` and `schema_version`
- Explicit `hypothesis`, `independent_variables`, and `dependent_variables`
- `control_config` and `treatment_config`
- Exact `seeds` and `environment_version` (`DIAMOND-1.0`)
- Predefined `evaluation_criteria` (`min_mean_diff`, `max_p_value=0.05`, `min_effect_size=0.2`)
- Statistical summary (`Welch t`, `p_value`, `cohens_d`, `ci_95`)
- Honest `verdict`: `supported`, `rejected`, or `inconclusive`

If small sample sizes (e.g., 3 seeds) or short episode horizons yield $p \ge 0.05$, OMNIS accurately reports `inconclusive` rather than fabricating significance.

---

## 4. Benchmark Methodology (`benchmarks/suite.py`)

Four architectural ablations are evaluated under identical random seeds and world dynamics:
- `A_FIXED`: Tabular RL + fixed initial SONAPSE connectome (plasticity, structural growth, and memory disabled)
- `B_ADAPTIVE`: Adds SONAPSE 3-factor plasticity & structural adaptation (no MNEMORPH memory/replay)
- `C_ADAPTIVE_MEMORY`: Adds MNEMORPH 7-class memory, replay, and EWC task consolidation
- `D_FULL_OMNIS`: Full developmental pipeline (GENESIS stage progression, neurogenesis, SONAPSE, MNEMORPH, MORPHEUS recovery)

Measured metrics: `task_performance`, `learning_speed`, `retention`, `generalisation` (unseen world seed), `robustness` (`Recovery(t)` after 25% lesion), `compute_cost_ms`, `network_complexity`, and multi-objective loss $J(\theta)$ with Pareto-front identification.

---

## 5. Observed Limitations & Future Work

1. **Short-Horizon Tabular Discretization**: DIAMOND uses a compact discretized `state_key` for fast tabular Q-learning alongside continuous/normalized sensory features for SONAPSE. Replacing or augmenting tabular Q with PyTorch actor-critic / DQN policies is a natural extension for larger continuous worlds.
2. **Statistical Power at 3 Seeds**: Default interactive demos use 3–4 seeds and short episodes for sub-15s execution; rigorous publication experiments should configure 15–30 seeds and longer episode horizons.
3. **Simulated Neuromorphic & Quantum-Inspired Modules**: `SimulatedNeuromorphicBackend` and `QuantumInspiredOptimizer` run on CPU to estimate spike sparsity, SynOps, and tunneling search dynamics; physical neuromorphic chips (e.g., Loihi/SpiNNaker) and physical QPUs are not attached.
4. **Single-Process Persistence**: SQLite supports local single-node research persistence; large distributed populations would migrate `OmnisDatabase` to PostgreSQL.
