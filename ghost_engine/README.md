# ghost_engine — GHOST ENGINE Autonomous Research Laboratory

Executes controlled `OBSERVE -> HYPOTHESISE -> DESIGN -> EXECUTE -> MEASURE -> VALIDATE -> RECORD` experiments with non-circular statistical criteria (two-sided Welch's *t*-test, Cohen's *d* effect size, minimum mean difference thresholds) across the 4 canonical OMNIS research hypotheses.

## Architecture & Key Modules
- `laboratory.py`: `GhostEngine` implementing `plasticity_vs_fixed`, `memory_replay_retention`, `developmental_vs_fixed`, and `morpheus_regeneration`.
- `interfaces.py`: `LaboratoryProtocol`.

## Standalone Execution
```bash
python -m ghost_engine.example
```
