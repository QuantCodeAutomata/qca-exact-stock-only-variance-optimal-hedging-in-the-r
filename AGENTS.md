# Rough Hawkes-Heston Variance-Optimal Hedging — Project Notes

## Project Structure

```
src/
  __init__.py        – empty package marker
  parameters.py      – ModelParams dataclass + grid/Fourier/GL utilities
  kernels.py         – fractional kernels, product-integration weights, g0/G0
  affine.py          – closed-form affine coefficients (D, R, B, A, Gamma)
  riccati.py         – Riccati-Volterra solver via product integration
  transforms.py      – conditional MGF H_0(w, T)
  exact_hedge.py     – Fourier-based call pricing and variance-optimal hedge
```

## Key Mathematical Objects

| Symbol | Location | Description |
|--------|----------|-------------|
| `ModelParams` | `parameters.py` | Calibrated parameters (alpha=0.527, rho=-0.731, …) |
| `K_alpha(t)` | `kernels.py` | Fractional kernel t^(α-1)/Γ(α) |
| `g_0(t)` | `kernels.py` | Deterministic variance curve V0 + β·t^α/Γ(α+1) |
| `W[m]` | `kernels.py` | Product-integration weights h^α(m^α-(m-1)^α)/Γ(α+1) |
| `D` | `affine.py` | Jump quadratic variation = 2+1/(1+2Λ)-2/(1+Λ) |
| `R(u,v)` | `affine.py` | Riccati drift (affine HJM structure) |
| `psi_w(t)` | `riccati.py` | Riccati-Volterra solution on [0,T] |
| `H_0(w,T)` | `transforms.py` | Conditional MGF exp(∫R(w,psi_w(T-s))g_0(s)ds) |
| `β_t^w` | `exact_hedge.py` | Exponential hedge = B(w,psi_w)/D |

## Design Decisions

- **Product integration** scheme is O(M²) per Fourier mode; batching over λ
  is done by sequential loop (each mode is independent).
- **Conjugate symmetry** used throughout: integration over [0,R] and doubling
  the real part, exploiting a_K(-λ) = a_K(λ)* and H_0(w̄) = H̄_0(w).
- `kernel_alpha` returns 0 at t=0 (integrable singularity for α<1); the
  `np.where` triggers a harmless divide-by-zero warning at t=0 — expected.
- `jump_integral_R` has a scipy.integrate.quad fallback when Re(1-a)<1e-10
  (essentially never triggered with the default calibration).
- `solve_riccati` stores R_vals incrementally to avoid redundant evaluations.

## Numerical Verification (default params, M=100, n_pts=20, R=30)

- ATM call price ≈ 0.0226  (small, consistent with V0=0.0079 over T=0.25)
- Δ (delta) ≈ 0.366  (in (0,1) as expected for ATM call)
- Γ density ≥ 0 at all tested Fourier modes ✓
- L¹ kernel error matches analytic formula [T^α-(T+ε)^α+ε^α]/Γ(α+1) ✓
- D matches numerical integral of (e^{-Λz}-1)² e^{-z} dz ✓

## Dependencies

- Python ≥ 3.10
- numpy ≥ 1.24
- scipy ≥ 1.10
