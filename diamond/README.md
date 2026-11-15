# diamond — DIAMOND Persistent World Simulation & Reinforcement Learning

Configurable multi-agent 2D grid world with renewable energy resources, moving entities, environmental hazards, task curriculum switching (`task_id`), serializable state checkpoints, tabular Q-learning + Dyna world modeling, and PyTorch Deep Q-Learning (`DeepQAgentLearner`).

## Architecture & Key Modules
- `environment.py`: `DiamondWorld` implementing `O(s_t)`, `A_t`, transition dynamics, reward shaping, and Observatory views.
- `rl.py`: `QAgentLearner` (with EWC Fisher regularization and linear Q readout) and `DynaWorldModel` simulated experience replay.
- `deep_rl.py`: `DeepQNetwork`, `ReplayBuffer`, and `DeepQAgentLearner` (PyTorch MLP with target network and EWC continual regularization, graceful NumPy fallback when PyTorch is absent).
- `interfaces.py`: `WorldEnvironmentProtocol`, `RLLearnerProtocol`, `WorldModelProtocol`.

## Standalone Execution
```bash
python -m diamond.example
```
