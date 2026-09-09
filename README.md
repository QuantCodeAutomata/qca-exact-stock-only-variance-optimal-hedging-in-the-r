# Exact Stock-Only Variance-Optimal Hedging in the Rough Hawkes–Heston Model

This repository implements the exact Galtchouk–Kunita–Watanabe / variance-optimal
stock-only hedging framework for the rough Hawkes–Heston model, together with the
predictable regularised-kernel approximation scheme that converges to the exact hedge
in the original market.

## Calibrated Parameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| α | 0.527 | Roughness (Hurst H = α − ½) |
| ρ | −0.731 | Leverage (Brownian correlation) |
| b | −1.812 | Mean-reversion speed |
| c | 0.115 | Vol-of-vol |
| Λ | 0.276 | Jump-damping in stock |
| β | 0.049 | Initial-variance curve shape |
| V₀ | 0.0079 | Initial variance level |
| S₀ | 1.0 | Initial stock price |
| T | 0.25 | Option maturity (years) |
| K | 1.0 | ATM strike |

Mark law: ν(dz) = e^{−z} dz (unit exponential on (0,∞))

## Project Structure

```
.
├── data/           # Placeholder for any saved model outputs
├── src/            # Core library
│   ├── parameters.py          # Calibrated parameters and grid setup
│   ├── kernels.py             # Fractional kernels K_α, K_ε, L¹ errors
│   ├── affine.py              # R(u,v), D, B_t^w, Γ_t^w (closed-form jumps)
│   ├── riccati.py             # Product-integration Riccati–Volterra solver
│   ├── transforms.py          # Conditional transform H_t(w,T)
│   ├── exact_hedge.py         # Exact hedge ϑ_t^w and ATM call hedge
│   ├── simulation.py          # Original-market Monte Carlo paths
│   ├── memory_reconstruction.py  # Predictable U_ε^w and V̂_ε
│   ├── truncation.py          # Truncation map χ(z)
│   ├── approx_hedge.py        # Approximate hedge ϑ̃_ε^w, call hedge synthesis
│   ├── metrics.py             # d̂², ĝ², Δ̂ estimators
│   └── validation.py          # Sanity checks and analytical identities
├── exp/            # Experiment entry points
│   ├── exp1_exact_hedge.py    # Experiment 1: exact hedge validation
│   └── exp2_approx_hedge.py   # Experiment 2: regularised-kernel convergence
├── tests/          # pytest suite (91 tests)
│   ├── test_affine.py
│   ├── test_kernels.py
│   ├── test_riccati.py
│   ├── test_transforms.py
│   ├── test_simulation.py
│   └── test_approx_hedge.py
└── results/        # Outputs: tables, plots, RESULTS.md
```

## Quick Start

```bash
pip install numpy scipy matplotlib
# Run experiment 1 (exact hedge, ~30 s)
python exp/exp1_exact_hedge.py
# Run experiment 2 (approx convergence, ~3–5 min)
python exp/exp2_approx_hedge.py
# Run tests
pytest tests/ -v
```

## Key Results (see results/RESULTS.md)

- **w=1 sanity:** ψ₁ ≡ 0, β¹ ≡ 1, Γ¹ ≡ 0 — exact stock replication confirmed.
- **D (market quadratic-variation factor):** 1.07693 (closed-form = numerical).
- **ATM call price (R=40):** 0.01971; delta: 0.3701.
- **L¹ kernel errors:** match paper targets to 6 significant figures.
- **Gain distance ĝ²:** monotone decreasing in ε at fixed R=20.
- **Diagonal convergence:** total Δ drops from ~5×10⁻² to ~1×10⁻⁴ along (ε/T,R) diagonal.

## Model Dynamics (Risk-Neutral Measure Q)

```
dS_t / S_{t−} = √V_t [√(1−ρ²) dW₁ + ρ dW₂] + ∫ q(z) μ̃(dt,dz)
V_t = g₀(t) + (K_α ⋆ dZ)_t
dZ_t = b V_t dt + √c √V_t dW₂ + ∫ z μ̃(dt,dz)
```

where q(z) = e^{−Λz} − 1, K_α(t) = t^{α−1}/Γ(α), g₀(t) = V₀ + β t^α/Γ(α+1).
