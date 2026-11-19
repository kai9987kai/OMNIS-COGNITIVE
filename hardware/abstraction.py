"""Hardware abstraction layer covering CPU, optional GPU probe, simulated neuromorphic AER, and quantum-inspired annealing."""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

from hardware.distributed import ParallelDistributedBackend
from sonapse.connectome import SonapseConnectome


class CPUVectorBackend:
    """Vectorized NumPy CPU execution and host resource telemetry."""

    name = "cpu_vector"

    def run_connectome_matrix_pass(self, connectome: SonapseConnectome, inputs: List[float]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        n_list = sorted(connectome.neurons.keys())
        idx = {nid: i for i, nid in enumerate(n_list)}
        n_count = len(n_list)
        w_mat = np.zeros((n_count, n_count), dtype=np.float32)
        for syn in connectome.active_synapses():
            w_mat[idx[syn.post_id], idx[syn.pre_id]] = np.float32(syn.weight)

        a_vec = np.array([connectome.neurons[nid].a for nid in n_list], dtype=np.float32)
        for i, nid in enumerate(n_list):
            if connectome.neurons[nid].role == "input":
                in_idx = int(nid.split("_")[1]) if "_" in nid else 0
                if in_idx < len(inputs):
                    a_vec[i] = np.float32(inputs[in_idx])
        drive = w_mat @ a_vec
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "backend": self.name,
            "precision": "float32",
            "neurons": n_count,
            "active_synapses": len(connectome.active_synapses()),
            "drive_norm": round(float(np.linalg.norm(drive)), 4),
            "latency_ms": round(elapsed_ms, 4),
        }


class PyTorchTensorBackend:
    """PyTorch CPU/CUDA tensor execution backend with float32, float16, and quantized int8 support."""

    name = "pytorch_tensor"

    def __init__(self, prefer_gpu: bool = True) -> None:
        import torch

        self.torch = torch
        self.gpu_available = bool(torch.cuda.is_available())
        self.device = "cuda" if (prefer_gpu and self.gpu_available) else "cpu"

    def run_connectome_tensor_pass(
        self,
        connectome: SonapseConnectome,
        inputs: List[float],
        precision: str = "float32",
    ) -> Dict[str, Any]:
        torch = self.torch
        t0 = time.perf_counter()
        n_list = sorted(connectome.neurons.keys())
        idx = {nid: i for i, nid in enumerate(n_list)}
        n_count = len(n_list)

        w_np = np.zeros((n_count, n_count), dtype=np.float32)
        for syn in connectome.active_synapses():
            w_np[idx[syn.post_id], idx[syn.pre_id]] = np.float32(syn.weight)

        a_np = np.array([connectome.neurons[nid].a for nid in n_list], dtype=np.float32)
        for i, nid in enumerate(n_list):
            if connectome.neurons[nid].role == "input":
                in_idx = int(nid.split("_")[1]) if "_" in nid else 0
                if in_idx < len(inputs):
                    a_np[i] = np.float32(inputs[in_idx])

        simulated_quant = False
        w_t = torch.from_numpy(w_np).to(self.device)
        a_t = torch.from_numpy(a_np).to(self.device)

        with torch.no_grad():
            if precision == "float16":
                try:
                    drive_t = torch.matmul(w_t.half(), a_t.half()).float()
                except RuntimeError:
                    drive_t = torch.matmul(w_t.half().float(), a_t.half().float())
            elif precision == "int8":
                simulated_quant = True
                w_scale = max(float(w_t.abs().max().item()) / 127.0, 1e-6)
                a_scale = max(float(a_t.abs().max().item()) / 127.0, 1e-6)
                w_q = torch.clamp(torch.round(w_t / w_scale), -127, 127).to(torch.int32)
                a_q = torch.clamp(torch.round(a_t / a_scale), -127, 127).to(torch.int32)
                drive_t = torch.matmul(w_q, a_q).float() * (w_scale * a_scale)
            else:
                precision = "float32"
                drive_t = torch.matmul(w_t, a_t)

            drive_norm = float(torch.linalg.vector_norm(drive_t).item())

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "backend": self.name,
            "device": self.device,
            "gpu_accelerated": self.device == "cuda",
            "precision": precision,
            "simulated_quantization": simulated_quant,
            "neurons": n_count,
            "active_synapses": len(connectome.active_synapses()),
            "drive_norm": round(drive_norm, 4),
            "latency_ms": round(elapsed_ms, 4),
        }


class SimulatedNeuromorphicBackend:
    """Simulated event-driven Address-Event Representation (AER) neuromorphic backend.

    Explicitly marks physical_device_connected=False and simulated=True so no physical
    neuromorphic execution is ever falsely claimed.
    """

    name = "simulated_neuromorphic_aer"

    def __init__(self, energy_per_synop_pj: float = 23.0) -> None:
        self.energy_per_synop_pj = float(energy_per_synop_pj)

    def simulate_event_step(self, connectome: SonapseConnectome, spike_threshold: float = 0.35) -> Dict[str, Any]:
        t0 = time.perf_counter()
        aer_events = [
            nid for nid, n in connectome.neurons.items()
            if n.active and abs(n.a) >= spike_threshold
        ]
        event_set = set(aer_events)
        synops = 0
        for syn in connectome.active_synapses():
            if syn.pre_id in event_set:
                synops += 1

        total_neurons = max(1, connectome.active_neuron_count())
        sparse_rate = len(aer_events) / float(total_neurons)
        est_energy_pj = synops * self.energy_per_synop_pj
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "backend": self.name,
            "simulated": True,
            "physical_device_connected": False,
            "aer_spike_events": len(aer_events),
            "sparse_activation_ratio": round(sparse_rate, 4),
            "synops_executed": synops,
            "estimated_energy_pj": round(est_energy_pj, 2),
            "latency_ms": round(elapsed_ms, 4),
        }


class QuantumInspiredOptimizer:
    """Simulated quantum-inspired tunneling annealing for hyperparameter optimization (CPU simulation)."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, float]], float],
        bounds: Dict[str, List[float]],
        iterations: int = 16,
    ) -> Dict[str, Any]:
        current = {k: 0.5 * (v[0] + v[1]) for k, v in bounds.items()}
        best = dict(current)
        best_score = float(objective_fn(best))
        curr_score = best_score

        for step in range(iterations):
            temp = max(0.02, 1.0 - step / float(max(1, iterations)))
            tunnel_prob = 0.25 * temp
            candidate = {}
            for k, (lo, hi) in bounds.items():
                span = hi - lo
                if self.rng.random() < tunnel_prob:
                    # Wide quantum-inspired tunneling jump
                    candidate[k] = lo + self.rng.random() * span
                else:
                    candidate[k] = max(lo, min(hi, current[k] + self.rng.gauss(0.0, 0.2 * span * temp)))
            score = float(objective_fn(candidate))
            if score < curr_score or self.rng.random() < math.exp(max(-20.0, (curr_score - score) / max(0.01, temp))):
                current = candidate
                curr_score = score
            if score < best_score:
                best = dict(candidate)
                best_score = score

        return {
            "quantum_inspired_simulated": True,
            "physical_qpu_connected": False,
            "best_params": {k: round(v, 5) for k, v in best.items()},
            "best_objective": round(best_score, 5),
            "iterations": iterations,
        }


HYPERION_20_SERVICE_CLASSES: List[Dict[str, Any]] = [
    {"id": "SC01_CPU_SIMD", "domain": "conventional", "backend": "cpu_vector", "cost_weight": 1.0},
    {"id": "SC02_TENSOR_FP32", "domain": "tensor", "backend": "pytorch_tensor", "cost_weight": 1.25},
    {"id": "SC03_TENSOR_FP16", "domain": "tensor", "backend": "pytorch_tensor", "cost_weight": 0.9},
    {"id": "SC04_QUANT_INT8", "domain": "tensor", "backend": "pytorch_tensor", "cost_weight": 0.75},
    {"id": "SC05_GRAPH_SPARSE", "domain": "connectome", "backend": "cpu_vector", "cost_weight": 0.85},
    {"id": "SC06_AER_SPIKE_EVENT", "domain": "neuromorphic", "backend": "simulated_neuromorphic_aer", "cost_weight": 0.55},
    {"id": "SC07_DUAL_SOMA_DSAG", "domain": "neuromorphic", "backend": "simulated_neuromorphic_aer", "cost_weight": 0.7},
    {"id": "SC08_LIQUID_LTC_ODE", "domain": "neuromorphic", "backend": "cpu_vector", "cost_weight": 1.1},
    {"id": "SC09_NEUROGRAFT_ROUTER", "domain": "connectome", "backend": "cpu_vector", "cost_weight": 0.95},
    {"id": "SC10_FRACTAL_LATTICE", "domain": "connectome", "backend": "cpu_vector", "cost_weight": 0.9},
    {"id": "SC11_ISING_ANNEALER", "domain": "probabilistic", "backend": "quantum_inspired", "cost_weight": 1.4},
    {"id": "SC12_ACTIVE_INFERENCE_FEP", "domain": "probabilistic", "backend": "cpu_vector", "cost_weight": 1.05},
    {"id": "SC13_CAUSAL_EMERGENCE_EI", "domain": "experimental", "backend": "cpu_vector", "cost_weight": 1.15},
    {"id": "SC14_HYBRID_RRF_SEARCH", "domain": "memory", "backend": "cpu_vector", "cost_weight": 0.65},
    {"id": "SC15_CARRIER_FACTORIAL_16", "domain": "memory", "backend": "cpu_vector", "cost_weight": 1.2},
    {"id": "SC16_TETHER_INTERVENTION", "domain": "orchestration", "backend": "pytorch_tensor", "cost_weight": 1.1},
    {"id": "SC17_FAULTLINE_ENSEMBLE", "domain": "orchestration", "backend": "cpu_vector", "cost_weight": 0.95},
    {"id": "SC18_MORPHEUS_REGEN", "domain": "resilience", "backend": "cpu_vector", "cost_weight": 1.1},
    {"id": "SC19_ALIFE_SWEEP_POOL", "domain": "distributed", "backend": "parallel_workers", "cost_weight": 1.35},
    {"id": "SC20_GHOST_HYPOTHESIS_LAB", "domain": "experimental", "backend": "parallel_workers", "cost_weight": 1.5},
]


class Hyperion20Scheduler:
    """HYPERION-20 heterogeneous 20-service-class workload router and telemetry tracker."""

    def __init__(self) -> None:
        self.service_classes = list(HYPERION_20_SERVICE_CLASSES)
        self.dispatches: int = 0
        self.class_counts: Dict[str, int] = {sc["id"]: 0 for sc in self.service_classes}

    def schedule(self, workload_type: str = "connectome_plasticity", complexity: float = 1.0) -> Dict[str, Any]:
        t0 = time.perf_counter()
        w_norm = str(workload_type).lower()
        mapping = {
            "connectome_plasticity": "SC07_DUAL_SOMA_DSAG",
            "dual_soma": "SC07_DUAL_SOMA_DSAG",
            "liquid_ltc": "SC08_LIQUID_LTC_ODE",
            "neurograft": "SC09_NEUROGRAFT_ROUTER",
            "fractal": "SC10_FRACTAL_LATTICE",
            "quantum_anneal": "SC11_ISING_ANNEALER",
            "active_inference": "SC12_ACTIVE_INFERENCE_FEP",
            "causal_emergence": "SC13_CAUSAL_EMERGENCE_EI",
            "hybrid_search": "SC14_HYBRID_RRF_SEARCH",
            "carrier_assay": "SC15_CARRIER_FACTORIAL_16",
            "tether_audit": "SC16_TETHER_INTERVENTION",
            "faultline": "SC17_FAULTLINE_ENSEMBLE",
            "morpheus": "SC18_MORPHEUS_REGEN",
            "parallel_sweep": "SC19_ALIFE_SWEEP_POOL",
            "ghost_experiment": "SC20_GHOST_HYPOTHESIS_LAB",
        }
        chosen_id = mapping.get(w_norm, "SC01_CPU_SIMD")
        sc = next((c for c in self.service_classes if c["id"] == chosen_id), self.service_classes[0])
        self.dispatches += 1
        self.class_counts[sc["id"]] = self.class_counts.get(sc["id"], 0) + 1
        comp = max(0.1, min(10.0, float(complexity)))
        est_cost = round(comp * float(sc["cost_weight"]), 4)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "scheduler": "HYPERION-20",
            "total_service_classes": len(self.service_classes),
            "workload_type": workload_type,
            "assigned_service_class": sc["id"],
            "domain": sc["domain"],
            "target_backend": sc["backend"],
            "estimated_cost_units": est_cost,
            "total_dispatches": self.dispatches,
            "scheduling_latency_ms": round(elapsed_ms, 4),
        }

    def get_summary(self) -> Dict[str, Any]:
        active = {k: v for k, v in self.class_counts.items() if v > 0}
        return {
            "total_service_classes": len(self.service_classes),
            "total_dispatches": self.dispatches,
            "active_service_classes": active,
        }


class HardwareManager:
    """Unified hardware abstraction & telemetry interface."""

    def __init__(self, active_backend: str = "cpu_vector") -> None:
        self.active_backend = active_backend
        self.cpu_backend = CPUVectorBackend()
        self.torch_backend = PyTorchTensorBackend()
        self.distributed = ParallelDistributedBackend()
        self.neuromorphic_sim = SimulatedNeuromorphicBackend()
        self.quantum_inspired = QuantumInspiredOptimizer()
        self.hyperion_scheduler = Hyperion20Scheduler()
        self.gpu_available = self._probe_gpu()
        self.process = psutil.Process(os.getpid())

    def schedule_workload(self, workload_type: str = "connectome_plasticity", complexity: float = 1.0) -> Dict[str, Any]:
        return self.hyperion_scheduler.schedule(workload_type=workload_type, complexity=complexity)

    @staticmethod
    def _probe_gpu() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def run_connectome_step(
        self,
        connectome: SonapseConnectome,
        inputs: List[float],
        backend: Optional[str] = None,
        precision: str = "float32",
    ) -> Dict[str, Any]:
        target = backend or self.active_backend
        if target in {"pytorch_tensor", "pytorch_cpu", "pytorch_cuda", "cuda_gpu"}:
            return self.torch_backend.run_connectome_tensor_pass(connectome, inputs, precision=precision)
        return self.cpu_backend.run_connectome_matrix_pass(connectome, inputs)

    def get_telemetry(self, connectome: Optional[SonapseConnectome] = None) -> Dict[str, Any]:
        mem_info = self.process.memory_info()
        cpu_percent = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        neuromorphic_metrics = (
            self.neuromorphic_sim.simulate_event_step(connectome)
            if connectome is not None
            else {"simulated": True, "physical_device_connected": False, "synops_executed": 0}
        )
        return {
            "active_backend": self.active_backend,
            "available_backends": [
                "cpu_vector",
                "pytorch_tensor",
                "simulated_neuromorphic_aer",
                "parallel_workers",
                *(["cuda_gpu"] if self.gpu_available else []),
            ],
            "pytorch_device": self.torch_backend.device,
            "distributed_workers": self.distributed.worker_count,
            "gpu_available": self.gpu_available,
            "gpu_utilization_percent": None,
            "cpu_percent": round(float(cpu_percent), 2),
            "cpu_count": os.cpu_count() or 1,
            "process_memory_mb": round(mem_info.rss / (1024.0 * 1024.0), 2),
            "system_memory_percent": round(float(vm.percent), 2),
            "neuromorphic_simulation": neuromorphic_metrics,
            "hyperion_20": self.hyperion_scheduler.get_summary(),
        }
