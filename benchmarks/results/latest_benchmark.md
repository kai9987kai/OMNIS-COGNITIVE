# OMNIS 1.0 Architecture Benchmark Report

- **Generated:** 2026-11-15T18:19:32.278406+00:00
- **Seeds:** [42, 101, 202]
- **Episodes / Steps:** 6 x 20
- **Pareto-Optimal Architectures:** A_FIXED, B_ADAPTIVE, C_ADAPTIVE_MEMORY, D_FULL_OMNIS, E_DEEP_RL_OMNIS

| Architecture | Task Perf (Mean ± Std) | Learning Speed | Retention | Generalisation | Robustness | Complexity | Compute (ms) | J(θ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_FIXED` | -3.129 ± 5.034 | -1.752 | 0.653 | -0.153 | 0.817 | 63.3 | 211.67 | 3.278 |
| `B_ADAPTIVE` | 0.114 ± 0.911 | -0.242 | 0.952 | -0.164 | 0.789 | 62.3 | 229.27 | -0.017 |
| `C_ADAPTIVE_MEMORY` | 0.621 ± 0.309 | 0.053 | 1.152 | -0.080 | 0.940 | 60.7 | 890.66 | -0.461 |
| `D_FULL_OMNIS` | -0.319 ± 1.354 | -1.065 | 0.754 | -0.016 | 1.065 | 80.7 | 1054.04 | 0.579 |
| `E_DEEP_RL_OMNIS` | -0.187 ± 0.372 | 0.303 | 0.952 | -0.050 | 0.998 | 81.7 | 4715.27 | 1.140 |

## Statistical Tests (D_FULL_OMNIS vs A_FIXED)
- **Task Performance:** diff = 2.8100, p = 0.4382, Cohen's d = 0.7624, significant = False
- **Robustness (Recovery):** diff = 0.2476, p = 0.0933, Cohen's d = 2.3167, significant = False
