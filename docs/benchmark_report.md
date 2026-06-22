# Adaptive Semantic Chunking (ASC) Benchmark Report

This report compares the research-grade ASC method against traditional fixed-size recursive character splitters.

| Method | P@3 | R@3 | MRR | Coherence | Boundary Quality | Runtime (s) | Chunks |
|---|---|---|---|---|---|---|---|
| ASC | 0.3333 | 1.0000 | 1.0000 | 0.5711 | 1.0000 | 1.09 | 3 |
| Recursive_256 | 0.6111 | 1.8333 | 1.0000 | 0.7623 | 0.2881 | 0.00 | 9 |
| Recursive_512 | 0.3333 | 1.0000 | 1.0000 | 0.6700 | 0.2494 | 0.00 | 6 |
| Recursive_1024 | 0.3333 | 1.0000 | 1.0000 | 0.5711 | 1.0000 | 0.00 | 3 |

### Statistical Significance (t-test vs Baselines)
- **MRR p-value**: nan (Not Significant)
- **P@3 p-value**: 2.0412e-02 (Statistically Significant, p < 0.05)
- **R@3 p-value**: 2.0412e-02 (Statistically Significant, p < 0.05)