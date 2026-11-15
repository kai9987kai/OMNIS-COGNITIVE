"""Formal typed interface protocols for HARDWARE abstraction, neuromorphic, tensor, quantum-inspired, and parallel compute."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class VectorComputeBackendProtocol(Protocol):
    """Contract for NumPy vectorized connectome propagation."""

    def run_connectome_matrix_pass(self, connectome: Any, inputs: List[float]) -> Dict[str, Any]: ...


@runtime_checkable
class TensorComputeBackendProtocol(Protocol):
    """Contract for PyTorch tensor connectome execution across precisions (float32, float16, int8)."""

    def run_connectome_tensor_pass(
        self,
        connectome: Any,
        inputs: List[float],
        precision: str = "float32",
    ) -> Dict[str, Any]: ...


@runtime_checkable
class DistributedSweepBackendProtocol(Protocol):
    """Contract for multi-thread / multi-process distributed experiment sweeps."""

    def map_parallel(
        self,
        fn: Callable[[Any], Any],
        items: List[Any],
        mode: str = "thread",
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]: ...
    def run_parallel_organism_sweep(
        self,
        configs: Optional[List[Dict[str, Any]]] = None,
        steps: int = 25,
        mode: str = "thread",
    ) -> Dict[str, Any]: ...


@runtime_checkable
class QuantumInspiredOptimizerProtocol(Protocol):
    """Contract for simulated quantum-annealing / tunneling hyperparameter optimizers."""

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, float]], float],
        bounds: Dict[str, Tuple[float, float]],
        iterations: int = 16,
    ) -> Dict[str, Any]: ...


@runtime_checkable
class HardwareManagerProtocol(Protocol):
    """Contract for unified hardware profiling, neuromorphic AER monitoring, and backend dispatch."""

    def run_connectome_step(
        self,
        connectome: Any,
        inputs: List[float],
        backend: str = "cpu_vector",
        precision: str = "float32",
    ) -> Dict[str, Any]: ...
    def get_telemetry(self) -> Dict[str, Any]: ...
