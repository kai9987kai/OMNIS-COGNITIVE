# OMNIS 1.0 Mathematical Specification

This document specifies the exact mathematical formulations implemented across OMNIS 1.0 subsystems.

---

## 1. SONAPSE Neural & Synaptic Dynamics (`sonapse/connectome.py`)

The neural network is represented as a directed weighted graph at time $t$:

$$G(t) = (V, E(t), W(t))$$

### 1.1 Neuron Membrane & Activation Models

For each non-input neuron $i \in V$:

$$u_i(t+1) = \alpha_i u_i(t) + \sum_{j \in \text{pre}(i)} w_{ij}(t) a_j(t) + b_i$$

$$a_i(t+1) = \phi(u_i(t+1))$$

Implemented $\phi(\cdot)$ variants:
- **Conventional**: $\phi(u) = \tanh(u)$
- **Rate-Coded**: $\phi(u) = \frac{1}{1 + e^{-u}}$
- **Threshold**: $\phi(u) = \mathbb{I}[u \ge \theta_i]$
- **Leaky Integrate-and-Fire (LIF)**: $\phi(u) = 1$ with post-spike reset $u_i \leftarrow 0$ if $u \ge \theta_i$, else $0$.

### 1.2 Synaptic Plasticity & Continual Regularization

Eligibility trace update:

$$e_{ij}(t+1) = \gamma_e e_{ij}(t) + a_i(t) a_j(t-1)$$

Three-factor reward-modulated plasticity (with optional EWC penalty):

$$\Delta w_{ij} = \eta \delta_t e_{ij} - \lambda w_{ij} - \eta \lambda_{\text{EWC}} F_{ij}(w_{ij} - w_{ij}^*)$$

Optional Hebbian mode:

$$\Delta w_{ij} = \eta a_i a_j - \lambda w_{ij}$$

Weights are clipped to $[-w_{\max}, w_{\max}]$.

### 1.3 Graph Topology Metrics (`sonapse/topology.py`)

- **Directed Graph Density** (without self-loops):

$$D = \frac{|E|}{N(N-1)}$$

- Also computed: in/out/total degree distributions, average directed shortest path length, undirected clustering coefficient, weakly/strongly connected components, degree & betweenness centrality, Newman modularity $Q$ over cluster partitions, isolated neurons, and redundant multi-path pairs.

---

## 2. MNEMORPH Memory Importance & Retrieval (`mnemorph/memory.py`)

Memory importance score $I_i \in [0, 1]$:

$$I_i = \text{clip}\left(0.35|R_i| + 0.25 N_i + 0.20 S_i + 0.20 U_i,\; 0,\; 1\right)$$

where $R_i$ is reward association, $N_i$ is embedding novelty ($1 - \max_k \cos(\mathbf{e}_i, \mathbf{e}_k)$), $S_i$ is significance, and $U_i$ is demonstrated retrieval usefulness.

Retrieval score for query embedding $\mathbf{q}$ at time $t$:

$$\text{Score}(i, \mathbf{q}) = 0.55 \max(0, \cos(\mathbf{q}, \mathbf{e}_i)) + 0.25 I_i + 0.10 \text{strength}_i + 0.10 e^{-0.01 (t - t_i)}$$

---

## 3. DIAMOND Reinforcement Learning (`diamond/rl.py`)

Temporal difference error:

$$\delta_t = r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)$$

Q-learning update (with optional EWC task anchor regularization):

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \eta \left(\delta_t - \lambda_{\text{EWC}} I(s_t, a_t)(Q(s_t, a_t) - Q^*(s_t, a_t))\right)$$

---

## 4. NEUROHIVE Fitness & GENESIS Evolution (`neurohive/organism.py`, `genesis/lifecycle.py`)

Organism multi-factor fitness:

$$F_i = \alpha P_i + \beta A_i + \gamma R_i - \lambda C_i$$

where $P_i$ is task performance, $A_i$ is adaptation progress, $R_i$ is structural/energy robustness, and $C_i$ is synaptic + memory computational cost.

Population evolution operator:

$$P_{t+1} = \text{Select}\left(\text{Mutate}\left(\text{Recombine}(P_t)\right)\right)$$

with configurable elite retention (`elite_fraction`), tournament selection, crossover rate, mutation rate, and hard population cap enforcement.

---

## 5. MORPHEUS Fault Recovery (`morpheus/regenerator.py`)

Functional recovery metric evaluated under equivalent deterministic world conditions ($P_{\text{before}} > 0$):

$$\text{Recovery}(t) = \frac{\text{Performance}_{\text{after}}(t)}{\text{Performance}_{\text{before}}}$$

---

## 6. Information Theory, Multi-Objective Optimisation & Statistics (`core/math_utils.py`)

- **Shannon Entropy**:

$$H(X) = -\sum_i p(x_i)\log_2 p(x_i)$$

- **Multi-Objective Loss & Pareto Front**:

$$J(\theta) = \alpha L_{\text{task}} + \beta C_{\text{compute}} + \gamma L_{\text{memory}} + \delta L_{\text{instability}}$$

Non-dominated Pareto optimal indices are computed directly via `compute_pareto_front()`.

- **Statistical Validation**: Sample mean, sample standard deviation ($s$, ddof=1), 95% Student-$t$ confidence intervals, two-sample Welch's $t$-test ($p$-value), and Cohen's $d$ effect size. Claims of improvement require both $p < \alpha$ and $|d| \ge d_{\min}$.
