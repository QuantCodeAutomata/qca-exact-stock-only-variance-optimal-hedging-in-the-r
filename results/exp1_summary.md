# Experiment 1 Results

## Model Parameters
alpha=0.527, rho=-0.731, b=-1.812, c=0.115, Lambda=0.276, beta=0.049, V0=0.0079

## Key Numerical Results

### D (Diffusion coefficient)
D = 1.0769317778
D_error = 0.00e+00

### w=1 Sanity Check
max|psi_1| = 0.00e+00
|beta^1 - 1| = 0.00e+00
|Gamma^1| = 0.00e+00
PASSED = True

### ATM Call Price (R=40, n_gl=32)
Call price = 0.019713
Call delta = 0.370073

### Residual Density
min Gamma = 1.757977e+00
max Gamma = 2.296148e+00
Non-negative: True

### Martingale Check (Monte Carlo)
E[S_T] = 1.002020 +/- 0.002749
Std(S_T) = 0.061476
Martingale PASSED = True
Min V = -3.437130e-01
Positivity PASSED = False

### H_0 Checks
H_0(w=0, T) = (1+0j)
H_0(w=1, T) = (1+0j)
