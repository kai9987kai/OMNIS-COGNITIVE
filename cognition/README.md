# cognition — 10-Stage Cognitive Processing Pipeline

Implements the unified perception-to-adaptation loop computing cognitive state vectors:

`C_t = {P_t, A_t, M_t, G_t, W_t, H_t}`

## Architecture
- `pipeline.py`: `CognitivePipeline.step(...)` orchestrates Perception -> Attention -> Working Memory -> Connectome Processing -> Decision Making -> Action -> Feedback -> RL/Plasticity Learning -> Memory Consolidation/Replay -> Structural Adaptation.
- Integrates `MemoryClass.PROCEDURAL` action heuristics, `QAgentLearner`/`DeepQAgentLearner`, `DynaWorldModel` planning, and action-gated R-STDP in `SonapseConnectome`.
- `interfaces.py`: `CognitivePipelineProtocol`.

## Standalone Execution
```bash
python -m cognition.example
```

## Performance & Design Notes
Supports toggleable `enable_plasticity`, `enable_structural_adaptation`, `enable_memory_ops`, and `ewc_lambda` for controlled ablation studies in `GhostEngine`.
