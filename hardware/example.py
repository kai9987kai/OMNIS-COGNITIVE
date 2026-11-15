"""Standalone runnable demonstration of HARDWARE vector, PyTorch tensor, parallel sweep, and quantum-inspired optimization."""

from __future__ import annotations

import json

from hardware.abstraction import HardwareManager
from hardware.distributed import ParallelDistributedBackend
from hardware.interfaces import (
    DistributedSweepBackendProtocol,
    HardwareManagerProtocol,
    QuantumInspiredOptimizerProtocol,
    TensorComputeBackendProtocol,
    VectorComputeBackendProtocol,
)
from sonapse.connectome import SonapseConnectome


def main() -> None:
    mgr = HardwareManager()
    dist = ParallelDistributedBackend(max_workers=2)

    assert isinstance(mgr, HardwareManagerProtocol)
    assert isinstance(mgr.cpu_backend, VectorComputeBackendProtocol)
    assert isinstance(mgr.torch_backend, TensorComputeBackendProtocol)
    assert isinstance(mgr.quantum_inspired, QuantumInspiredOptimizerProtocol)
    assert isinstance(dist, DistributedSweepBackendProtocol)

    conn = SonapseConnectome(seed=42)
    vec_res = mgr.run_connectome_step(conn, [0.5] * 8, backend="cpu_vector")
    torch_res = mgr.run_connectome_step(conn, [0.5] * 8, backend="pytorch_tensor", precision="float32")
    opt_res = mgr.quantum_inspired.optimize(
        lambda p: -((p["lr"] - 0.08) ** 2 + (p["decay"] - 0.85) ** 2),
        {"lr": (0.01, 0.2), "decay": (0.5, 0.99)},
        iterations=10,
    )
    sweep_res = dist.run_parallel_organism_sweep(steps=12, mode="thread")

    print("=== HARDWARE & DISTRIBUTED COMPUTE EXAMPLE ===")
    print(
        json.dumps(
            {
                "cpu_vector_backend": vec_res.get("backend"),
                "torch_tensor_backend": torch_res.get("backend"),
                "physical_qpu_connected": opt_res["physical_qpu_connected"],
                "q_opt_best_params": opt_res.get("best_params"),
                "parallel_trials": sweep_res["trials_evaluated"],
                "parallel_speedup": sweep_res["speedup_factor"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
