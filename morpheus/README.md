# morpheus — MORPHEUS Regenerative Fault-Tolerance Subsystem

Simulates structural connectome lesions (neuron deactivation and synapse severing), audits degraded `Recovery(t) = Performance_after(t) / Performance_before` under frozen equivalent world snapshots, and drives plasticity-mediated recovery via compensatory neurogenesis, alternative pathway synaptogenesis, homeostatic synaptic scaling, and positive-prioritized Mnemorph replay.

## Architecture & Key Modules
- `regenerator.py`: `MorpheusRegenerator`.
- `interfaces.py`: `RegeneratorProtocol`.

## Standalone Execution
```bash
python -m morpheus.example
```
