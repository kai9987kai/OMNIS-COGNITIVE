# hardware — HARDWARE Abstraction, Neuromorphic AER, PyTorch Tensor & Distributed Workers

Provides honest telemetry and compute backends across CPU NumPy vectorization (`CPUVectorBackend`), PyTorch multi-precision tensor passes (`PyTorchTensorBackend`: `float32`, `float16`, `int8` with NumPy fallback), simulated Address-Event Representation (`SimulatedNeuromorphicBackend`), simulated quantum-annealing search (`QuantumInspiredOptimizer`), and parallel multi-thread/process sweeps (`ParallelDistributedBackend`).

Explicitly reports `physical_device_connected=False` and `physical_qpu_connected=False` unless real external hardware SDKs are connected.

## Architecture & Key Modules
- `abstraction.py`: `CPUVectorBackend`, `PyTorchTensorBackend`, `SimulatedNeuromorphicBackend`, `QuantumInspiredOptimizer`, `HardwareManager`.
- `distributed.py`: `ParallelDistributedBackend` and picklable `_evaluate_organism_trial_worker`.
- `interfaces.py`: Protocol contracts for vector, tensor, distributed, quantum-inspired, and manager backends.

## Standalone Execution
```bash
python -m hardware.example
```
