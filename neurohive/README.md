# neurohive — NEUROHIVE Multi-Agent Artificial Life Ecosystem

Coordinates bounded populations of embodied `ArtificialOrganism` agents inside a shared `DiamondWorld`, managing metabolic energy, multi-objective fitness `F_i = alpha*P_i + beta*A_i + gamma*R_i - lambda*C_i`, kin cooperation (energy & semantic memory sharing), resource competition, dynamic speciation, and serialization.

## Architecture & Key Modules
- `organism.py`: `ArtificialOrganism` integrating `SonapseConnectome`, `MnemorphMemory`, `QAgentLearner`/`DeepQAgentLearner`, `DynaWorldModel`, and `CognitivePipeline` (`use_deep_rl` configurable).
- `ecosystem.py`: `NeuroHiveEcosystem` managing population caps, social interactions, speciation, and death lifecycle.
- `interfaces.py`: `OrganismProtocol` and `EcosystemProtocol`.

## Standalone Execution
```bash
python -m neurohive.example
```
