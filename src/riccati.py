import numpy as np
from scipy.special import gamma

from .kernels import product_integration_weights
from .affine import R_affine


def _epsilon_weights(M: int, alpha: float, h: float, epsilon: float) -> np.ndarray:
    """Product-integration weights for the regularised kernel K_epsilon.

    W_eps[0] = 0.
    W_eps[m] = [(m*h + epsilon)^alpha - ((m-1)*h + epsilon)^alpha] / Gamma(alpha+1)
               for m = 1, ..., M.
    """
    W = np.zeros(M + 1, dtype=float)
    m = np.arange(1, M + 1, dtype=float)
    W[1:] = ((m * h + epsilon) ** alpha - ((m - 1) * h + epsilon) ** alpha) / gamma(
        alpha + 1
    )
    return W


def solve_riccati(
    w: complex,
    params,
    M: int,
    use_epsilon: bool = False,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Solve the Riccati-Volterra equation for a single w.

    Discretises
        psi_w(t_k) = integral_0^{t_k} K(t_k - s) * R(w, psi_w(s)) ds

    on the uniform grid t_k = k*h, h = T/M, using the product-integration
    recurrence

        psi[0]   = 0
        psi[k+1] = sum_{j=0}^k  W[k+1-j] * R(w, psi[j]),  k = 0,...,M-1

    where W are the convolution weights for either K_alpha (use_epsilon=False)
    or K_epsilon (use_epsilon=True).

    Parameters
    ----------
    w : complex
        Fourier variable on the dampened contour.
    params : ModelParams
        Calibrated model parameters.
    M : int
        Number of time steps.
    use_epsilon : bool
        If True use the regularised kernel K_epsilon.
    epsilon : float
        Regularisation shift (only used when use_epsilon=True).

    Returns
    -------
    psi : np.ndarray, complex128, shape (M+1,)
        psi[k] = psi_w(k * h) for k = 0, ..., M.
    """
    alpha = params.alpha
    T = params.T
    h = T / M
    b, rho, c, Lambda = params.b, params.rho, params.c, params.Lambda

    if use_epsilon:
        W = _epsilon_weights(M, alpha, h, epsilon)
    else:
        W = product_integration_weights(M, alpha, h)

    psi = np.zeros(M + 1, dtype=complex)
    # R_vals[j] = R(w, psi[j]) accumulated as the iteration proceeds.
    R_vals = np.zeros(M, dtype=complex)

    for k in range(M):
        R_vals[k] = R_affine(w, psi[k], b, rho, c, Lambda)
        # W[k+1:0:-1] = [W[k+1], W[k], ..., W[1]]  (length k+1)
        psi[k + 1] = np.dot(W[k + 1 : 0 : -1], R_vals[: k + 1])

    return psi


def solve_riccati_batch(
    w_array: np.ndarray,
    params,
    M: int,
    use_epsilon: bool = False,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Solve the Riccati-Volterra equation for multiple w values.

    Parameters
    ----------
    w_array : np.ndarray
        Array of complex Fourier variables.
    params : ModelParams
        Calibrated model parameters.
    M : int
        Number of time steps.
    use_epsilon : bool
        Use regularised kernel if True.
    epsilon : float
        Regularisation shift.

    Returns
    -------
    psi_batch : np.ndarray, complex128, shape (len(w_array), M+1)
        psi_batch[i, k] = psi_{w_array[i]}(k * h).
    """
    n = len(w_array)
    psi_batch = np.zeros((n, M + 1), dtype=complex)
    for i, w in enumerate(w_array):
        psi_batch[i] = solve_riccati(w, params, M, use_epsilon, epsilon)
    return psi_batch
