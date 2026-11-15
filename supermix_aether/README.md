# supermix_aether — SUPERMIX-AETHER Model & Agent Orchestration

Registers specialist heuristic, PyTorch neural (`TorchNeuralSpecialist`), and gated external HTTP (`ExternalHTTPModelAdapter`, disabled unless `allow_external_apis=True`) adapters; manages agent registrations; routes cognitive tasks by specialty and tracks latency/call counts without ever requiring external network APIs for core simulation.

## Architecture & Key Modules
- `router.py`: `SupermixAether`, `SpecialistModel`, `TorchNeuralSpecialist`, and `ExternalHTTPModelAdapter`.
- `interfaces.py`: `OrchestrationRouterProtocol` and `SpecialistModelCallableProtocol`.

## Standalone Execution
```bash
python -m supermix_aether.example
```
