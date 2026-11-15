# sonapse — Adaptive Neural Connectome (`G(t) = (V, E(t), W(t))`)

Implements dynamic directed neural computation graphs with bounded neurogenesis, synaptogenesis, synaptic pruning, and NetworkX topological telemetry.

## Capabilities
- **4 Neuron Models**: `conventional` (`tanh`), `rate_coded` (`sigmoid`), `threshold`, and spiking `lif` (leaky integrate-and-fire).
- **Synaptic Plasticity**: Action-gated three-factor R-STDP (`dw = eta * delta_t * e_ij - lambda * w_ij`), Hebbian (`dw = eta * a_i * a_j`), hybrid mode, and EWC importance anchoring.
- **Structural Homeostasis**: Enforced density caps (`max_density`), weak-weight pruning, correlation-driven synaptogenesis, and 3D spatial node positioning.
- **Topology Analyzer (`topology.py`)**: Computes graph density, clustering coefficient, modularity `Q`, average path length, redundant paths, and isolated node detection.

## Standalone Execution
```bash
python -m sonapse.example
```

## Performance Notes
Two-phase synchronous propagation evaluates recurrent hidden state and same-step feedforward hidden-to-output projection cleanly in linear `O(|V| + |E|)` time per tick.
