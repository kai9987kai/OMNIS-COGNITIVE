"""HARDWARE: Execution Abstraction Layer, Simulated Neuromorphic AER, Parallel Distributed Workers, and Quantum-Inspired Optimization."""

from hardware.abstraction import (
    CPUVectorBackend,
    HardwareManager,
    PyTorchTensorBackend,
    QuantumInspiredOptimizer,
    SimulatedNeuromorphicBackend,
)
from hardware.distributed import ParallelDistributedBackend

__all__ = [
    "CPUVectorBackend",
    "HardwareManager",
    "ParallelDistributedBackend",
    "PyTorchTensorBackend",
    "QuantumInspiredOptimizer",
    "SimulatedNeuromorphicBackend",
]
