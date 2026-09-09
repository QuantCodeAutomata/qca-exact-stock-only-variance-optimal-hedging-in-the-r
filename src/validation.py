"""Sanity checks and validation utilities for the rough Hawkes-Heston model."""
import numpy as np

from .affine import compute_D, B_coefficient, Gamma_density, R_affine
from .riccati import solve_riccati


def check_w1_sanity(params, M: int, tol: float = 1e-10) -> dict:
    """Verify the w=1 exact replication sanity check.

    For w = 1:
      - psi_1(t) should be identically 0 (since R(1,0) = 0)
      - beta^1 = B(1,0)/D should equal 1
      - Gamma^1 = A(1,0) - |B(1,0)|^2/D should be 0

    Parameters
    ----------
    params : ModelParams
    M : int
    tol : float

    Returns
    -------
    dict with keys 'psi1_max_error', 'beta1_error', 'gamma1_error', 'passed'.
    """
    psi1 = solve_riccati(1.0 + 0j, params, M)
    psi1_max = float(np.max(np.abs(psi1)))

    D = compute_D(params.Lambda)
    B1 = B_coefficient(1.0 + 0j, 0.0 + 0j, params.rho, params.c, params.Lambda)
    beta1 = float(abs(B1 / D - 1.0))

    gamma1 = Gamma_density(1.0 + 0j, 0.0 + 0j, params.b, params.rho, params.c, params.Lambda)

    # R(1,0) should be 0
    R10 = R_affine(1.0 + 0j, 0.0 + 0j, params.b, params.rho, params.c, params.Lambda)

    passed = (psi1_max < tol) and (beta1 < tol) and (abs(gamma1) < tol)
    return {
        "psi1_max_error": psi1_max,
        "beta1_error": beta1,
        "gamma1_error": float(gamma1),
        "R10": float(abs(R10)),
        "D": D,
        "B1": float(abs(B1)),
        "passed": passed,
    }


def check_martingale_sanity(S_paths: np.ndarray, target: float = 1.0, tol: float = 0.05) -> dict:
    """Check that E[S_T] ≈ S_0 (martingale property under Q).

    Parameters
    ----------
    S_paths : (P, M+1) float array
    target : float, expected S_0 = 1.
    tol : float, tolerance for pass/fail.

    Returns
    -------
    dict
    """
    S_T = S_paths[:, -1]
    mean_ST = float(np.mean(S_T))
    std_ST = float(np.std(S_T, ddof=1))
    se_ST = std_ST / np.sqrt(len(S_T))
    passed = abs(mean_ST - target) < tol
    return {
        "mean_ST": mean_ST,
        "std_ST": std_ST,
        "se_ST": float(se_ST),
        "passed": passed,
    }


def check_D_value(params) -> dict:
    """Validate the closed-form formula for D against direct integration.

    D = 1 + integral_0^inf (exp(-Lambda*z) - 1)^2 * exp(-z) dz
      = 2 + 1/(1+2*Lambda) - 2/(1+Lambda)  [closed form]

    Parameters
    ----------
    params : ModelParams

    Returns
    -------
    dict
    """
    from scipy.integrate import quad

    Lambda = params.Lambda
    D_closed = compute_D(Lambda)

    def integrand(z):
        return (np.exp(-Lambda * z) - 1.0) ** 2 * np.exp(-z)

    D_numerical, _ = quad(integrand, 0.0, 50.0, limit=200)
    D_numerical = 1.0 + D_numerical

    return {
        "D_computed": D_closed,
        "D_numerical": D_numerical,
        "D_expected": 2.0 + 1.0 / (1.0 + 2.0 * Lambda) - 2.0 / (1.0 + Lambda),
        "error": abs(D_closed - D_numerical),
    }


def check_positivity(V_paths: np.ndarray, tol: float = -1e-4) -> dict:
    """Check that variance paths remain non-negative.

    Parameters
    ----------
    V_paths : (P, M+1) float array
    tol : float, threshold below which we flag negative variance.

    Returns
    -------
    dict
    """
    min_V = float(V_paths.min())
    negative_frac = float(np.mean(V_paths < tol))
    passed = min_V > tol
    return {
        "min_V": min_V,
        "negative_fraction": negative_frac,
        "passed": passed,
    }


def check_gamma_nonneg(params, M: int, R_cutoff: float = 20.0, n_check: int = 20) -> dict:
    """Check that Gamma_t^w >= 0 for a grid of Fourier modes.

    Parameters
    ----------
    params : ModelParams
    M : int
    R_cutoff : float
    n_check : int

    Returns
    -------
    dict
    """
    lam_vals = np.linspace(0.0, R_cutoff, n_check + 1)[1:]
    min_gamma = np.inf
    for lam in lam_vals:
        w = 0.5 + 1j * lam
        psi = solve_riccati(w, params, M)
        # Check across time steps
        for k in range(0, M + 1, M // 10 + 1):
            g = Gamma_density(w, psi[M - k] if k < M else 0j,
                              params.b, params.rho, params.c, params.Lambda)
            if g < min_gamma:
                min_gamma = g
    passed = min_gamma >= -1e-10
    return {"min_Gamma": float(min_gamma), "passed": passed}
