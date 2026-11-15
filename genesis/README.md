# genesis — GENESIS 8-Stage Organism Development & Evolutionary Engine

Manages the organism lifecycle across 8 developmental stages (`STAGE_1_GENESIS` through `STAGE_8_EVALUATION`), tournament selection, neural/parameter recombination (`recombine`), structural/weight mutation (`mutate`), bounded population replacement (`evolve_generation`), lineage DAG tracking, and checkpoint diff analysis.

## Architecture & Key Modules
- `lifecycle.py`: `GenesisEngine` and `DEVELOPMENTAL_STAGES`.
- `interfaces.py`: `GenesisEngineProtocol`.

## Standalone Execution
```bash
python -m genesis.example
```
