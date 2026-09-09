import numpy as np

from .affine import R_affine
from .kernels import g0
from .parameters import time_grid


def compute_H0(w: complex, psi_w: np.ndarray, params, M: int) -> complex:
    """Compute the conditional moment generating function H_0(w, T) at t = 0.

    H_0(w, T) = exp(w * X_0 + integral_0^T R(w, psi_w(T - s)) * g_0(s) ds)

    where X_0 = log(S_0 / S_0) = 0.

    The integral is approximated by the left-endpoint Riemann sum

        h * sum_{k=0}^{M-1} R(w, psi_w[M - k]) * g_0(t_k),  t_k = k * h.

    Parameters
    ----------
    w : complex
        Fourier variable.
    psi_w : np.ndarray, shape (M+1,)
        Solution of the Riccati-Volterra equation; psi_w[k] = psi_w(k * h).
    params : ModelParams
        Calibrated model parameters.
    M : int
        Number of time steps.

    Returns
    -------
    H0 : complex
    """
    h = params.T / M
    b, rho, c, Lambda = params.b, params.rho, params.c, params.Lambda
    V0, beta, alpha = params.V0, params.beta, params.alpha

    # t_k = k*h for k = 0, ..., M-1.
    t_vals = time_grid(params.T, M)[:-1]  # shape (M,)

    # psi_w(T - t_k) = psi_w[M - k] for k = 0, ..., M-1.
    psi_reversed = psi_w[M:0:-1]  # [psi_w[M], psi_w[M-1], ..., psi_w[1]], shape (M,)

    R_vals = np.array(
        [R_affine(w, psi_reversed[k], b, rho, c, Lambda) for k in range(M)],
        dtype=complex,
    )
    g0_vals = g0(t_vals, V0, beta, alpha)  # shape (M,)

    integral = h * np.dot(R_vals, g0_vals)
    # X_0 = 0 so the w*X_0 term vanishes.
    return complex(np.exp(integral))


def compute_log_H0_integrand(
    w: complex, psi_w: np.ndarray, params, M: int
) -> float:
    """Return the value of the exponent integral in H_0 (diagnostic).

    This is h * sum_{k=0}^{M-1} R(w, psi_w[M-k]) * g_0(t_k) as a real number
    (takes the real part, since the imaginary part cancels by symmetry for
    the ATM call price computation).
    """
    h = params.T / M
    b, rho, c, Lambda = params.b, params.rho, params.c, params.Lambda
    V0, beta, alpha = params.V0, params.beta, params.alpha

    t_vals = time_grid(params.T, M)[:-1]
    psi_reversed = psi_w[M:0:-1]

    R_vals = np.array(
        [R_affine(w, psi_reversed[k], b, rho, c, Lambda) for k in range(M)],
        dtype=complex,
    )
    g0_vals = g0(t_vals, V0, beta, alpha)
    return float((h * np.dot(R_vals, g0_vals)).real)
