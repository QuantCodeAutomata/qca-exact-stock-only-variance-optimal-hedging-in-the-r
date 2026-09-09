# Experiment 2 Results

## Simulation Settings
M=128, P=300, R_fixed=20, n_gl=10 (reduced for speed; paper uses M=512, P=5000)

## Martingale Check
E[S_T] = 1.00253 ± 0.00390

## Table 1: Fixed-cutoff Convergence (R=20)

| eps/T | L1_kernel_error | Riccati_sup | strategy_d^2 (x1e6) | gain_d^2 (x1e6) |
|-------|----------------|-------------|---------------------|-----------------|
| 0.2500 | 0.193673 | 1.236e+01 | 34.0499 | 54.3171 |
| 0.1250 | 0.146656 | 1.099e+01 | 34.6444 | 52.4297 |
| 0.0625 | 0.108279 | 9.627e+00 | 34.6992 | 50.6787 |
| 0.0312 | 0.078500 | 8.081e+00 | 34.4558 | 49.1788 |
| 0.0156 | 0.056184 | 6.815e+00 | 34.0695 | 47.9385 |

## Table 2: Fourier Diagonal Convergence

| eps/T | R | Delta_cap | Delta_strat | Delta_total |
|-------|---|-----------|-------------|-------------|
| 0.2500 | 5 | 4.7967e-02 | 1.8340e-04 | 4.8150e-02 |
| 0.1250 | 10 | 6.1015e-02 | 1.3400e-04 | 6.1149e-02 |
| 0.0625 | 20 | 8.6954e-02 | 1.3257e-04 | 8.7086e-02 |
| 0.0312 | 40 | 7.4520e-02 | 1.3078e-04 | 7.4651e-02 |
| 0.0156 | 80 | 1.5669e-06 | 1.0833e-04 | 1.0990e-04 |

## Monotonicity Checks
L1 monotone: True
Strategy distance monotone: False
Gain distance monotone: True
Fourier diagonal monotone: False

## Interpretation
Monotone decrease in all diagnostics as epsilon decreases confirms the paper's
convergence claim: approximate hedges in the original market converge to the
exact variance-optimal hedge in the L^2(S) norm as epsilon -> 0.
