# OMNIS 1.0 System & Cognitive Architecture

## 1. Subsystem Decomposition

OMNIS 1.0 is structured as an event-driven, modular research architecture orchestrated by `OmnisCore` (`core/orchestrator.py`):

```text
                       +-----------------------+
                       |      OMNIS CORE       |
                       | Lifecycle, Quotas,    |
                       | EventBus, Persistence |
                       +-----------+-----------+
                                   |
        +--------------------------+--------------------------+
        |                          |                          |
+-------v--------+        +--------v-------+         +--------v--------+
|   NEUROHIVE    |        |    DIAMOND     |         |  GHOST ENGINE   |
|  Population,   | <----> | Persistent     | <-----> | Seeded Control  |
|  Coop/Compete  |        | Grid/Cont Env  |         | vs Treatment    |
+-------+--------+        +----------------+         +-----------------+
        |
+-------v--------------------------------------------------------------+
|                       ARTIFICIAL ORGANISM                            |
|  +-------------+   +-------------+   +-------------+   +----------+  |
|  |   SONAPSE   |   |  MNEMORPH   |   |   DIAMOND   |   | GENESIS  |  |
|  | Connectome  |   | 7-Class Mem |   | Q-Learner + |   | 8-Stage  |  |
|  | G(t)=(V,E,W)|   | & Replay    |   | Dyna Model  |   | Lineage  |  |
|  +------+------+   +------+------+   +------+------+   +----+-----+  |
|         |                 |                 |               |        |
|         +-----------------+--------+--------+---------------+        |
|                                    |                                 |
|                      +-------------v--------------+                  |
|                      | COGNITIVE PIPELINE (10x)   |                  |
|                      | C_t = {P_t,A_t,M_t,G_t,W_t,H_t}               |
|                      +----------------------------+                  |
+----------------------------------------------------------------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
    +--------v--------+                             +--------v--------+
    |    MORPHEUS     |                             | SUPERMIX-AETHER |
    | Lesion, Bypass  |                             | Specialist      |
    | & Recovery(t)   |                             | Model Router    |
    +-----------------+                             +-----------------+
```

## 2. Shared Event Schema & Message Passing

Every cross-subsystem signal uses `OmnisEvent` (`core/schemas.py`, `SCHEMA_VERSION = "1.0.0"`), routed via `EventBus` (`core/events.py`) with bounded ring-buffer history (`max_history=2000`) and wildcard observers (`"*"`).

## 3. Ten-Stage Cross-System Cognitive Pipeline

Each tick of `CognitivePipeline.step()` (`cognition/pipeline.py`) executes 10 explicit stages and constructs `CognitiveStateSnapshot` ($C_t = \{P_t, A_t, M_t, G_t, W_t, H_t\}$):

1. **PERCEPTION (`P_t`)**: Calls `DiamondWorld.observe(organism_id)` to extract 8-dimensional normalized sensory vectors (nearest resource dx/dy/proximity, nearest hazard dx/dy/proximity, task beacon offset, normalized energy) and discretized state keys.
2. **ATTENTION (`A_t`)**: Modulates sensory channels dynamically based on hazard urgency, resource proximity, and metabolic energy need.
3. **WORKING MEMORY (`M_t`)**: Encodes sensory frames into `MemoryClass.SENSORY` and retrieves top-$k$ associative memories from `MnemorphMemory` to compute a historical reward bias.
4. **COGNITIVE PROCESSING (`G_t`)**: Propagates attended inputs through `SonapseConnectome.step(...)`, updating membrane potentials $u_i(t)$ and activations $a_i(t)$, and recording hidden activation entropy.
5. **DECISION MAKING**: Combines `QAgentLearner` tabular Q-values with SONAPSE output logits (`neural_weight`) under epsilon-greedy or Boltzmann exploration.
6. **ACTION**: Executes one of 5 discrete actions (`UP`, `RIGHT`, `DOWN`, `LEFT`, `HARVEST`).
7. **ENVIRONMENTAL FEEDBACK**: Receives `(next_obs, reward, done, info)` from DIAMOND.
8. **LEARNING (`W_t`)**: Updates `QAgentLearner` via TD error $\delta_t$, updates `DynaWorldModel` transition predictions and performs model-based planning rollouts, and updates `SonapseConnectome` synaptic weights via 3-factor / Hebbian plasticity driven by $\delta_t$.
9. **MEMORY CONSOLIDATION**: Encodes working/episodic/procedural traces, runs prioritized experience replay every 4 ticks (measuring overhead in ms), consolidates salient traces into `LONG_TERM` and `SEMANTIC` prototypes every 8 ticks, and applies class-specific exponential forgetting.
10. **ARCHITECTURAL ADAPTATION**: Periodically prunes weak synapses (`|w| < 0.025`) and grows new correlated connections every 10 ticks within hard `ResourceQuota` limits (`max_connectome_density`, `max_synapses`).

## 4. Eight-Stage Developmental Lifecycle (`GENESIS`)

1. `STAGE_1_GENESIS`: Minimal initial architecture created with tracked lineage.
2. `STAGE_2_DEVELOPMENT`: Controlled neurogenesis (`add_hidden_neuron`) and sparse synaptogenesis (`grow_connections`).
3. `STAGE_3_LEARNING`: Environmental reinforcement learning and synaptic plasticity.
4. `STAGE_4_MEMORY`: Consolidation of working/episodic traces into semantic/long-term tiers.
5. `STAGE_5_SPECIALISATION`: Weak-edge pruning and behavioral role tagging (`forager`, `archivist`, `integrator`).
6. `STAGE_6_EVOLUTION`: Multi-generation selection, recombination, and bounded mutation.
7. `STAGE_7_REGENERATION`: Fault injection and structural/plasticity recovery via `MORPHEUS`.
8. `STAGE_8_EVALUATION`: Multi-seed quantitative assessment via `GHOST ENGINE` and `benchmarks/suite.py`.

## 5. Performance Considerations

- **Graph Size Bounds**: `ResourceQuota` enforces `max_neurons_per_organism=128`, `max_synapses_per_organism=512`, `max_connectome_density=0.35`, and `max_population_size=24` by default so topology analysis and 3D Observatory rendering remain interactive.
- **Replay Budget**: `MnemorphMemory.replay()` samples a bounded batch (`batch_size=6–8`) and records exact wall-clock overhead (`last_replay_overhead_ms`).
- **Persistence**: SQLite operations in `OmnisDatabase` use a reentrant lock (`RLock`) for thread-safe FastAPI operation.
