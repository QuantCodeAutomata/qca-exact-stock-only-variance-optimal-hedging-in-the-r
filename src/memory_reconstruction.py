"""Approximate memory reconstruction using regularised kernels K_epsilon.

For each Fourier node w and regularisation shift epsilon, build the predictable
memory process U_epsilon^w(t_k) for all paths using only information through [0, t_k).
"""
import numpy as np
from scipy.special import gamma

from .affine import R_affine
from .kernels import g0


def compute_eps_conv_weights(M: int, alpha: float, h: float, epsilon: float) -> np.ndarray:
    """Convolution weights for the regularised kernel K_epsilon = (t+epsilon)^(alpha-1)/Gamma(alpha).

    conv_w_eps[m] = integral_{m*h}^{(m+1)*h} K_eps(s) ds / h
                 = [(m*h + epsilon + h)^alpha - (m*h + epsilon)^alpha]
                   / (h * Gamma(alpha+1))

    Returns array of shape (M,), indexed by lag m = 0, ..., M-1.
    """
    m_arr = np.arange(M, dtype=float)
    left = m_arr * h + epsilon
    right = left + h
    conv_w = (right ** alpha - left ** alpha) / (h * gamma(alpha + 1))
    return conv_w


def compute_Vhat_epsilon(
    dZ: np.ndarray,
    g0_vals: np.ndarray,
    conv_w_eps: np.ndarray,
    M: int,
) -> np.ndarray:
    """Regularised variance proxy for a single path.

    Vhat_eps[k] = g_0(t_k) + sum_{j=0}^{k-1} conv_w_eps[k-1-j] * dZ[j]

    Uses only information through [0, t_k) (predictable).

    Parameters
    ----------
    dZ : (M,) float array, driver increments.
    g0_vals : (M+1,) float array, g_0 on time grid.
    conv_w_eps : (M,) float array, regularised conv weights.
    M : int

    Returns
    -------
    Vhat : (M+1,) float array.
    """
    Vhat = np.zeros(M + 1, dtype=float)
    Vhat[0] = g0_vals[0]
    for k in range(1, M + 1):
        # Sum over j=0,...,k-1: conv_w_eps[k-1-j] * dZ[j]
        # lags: (k-1), (k-2), ..., 0 → conv_w_eps indices: k-1, k-2, ..., 0
        Vhat[k] = g0_vals[k] + np.dot(conv_w_eps[k - 1 :: -1], dZ[:k])
    return Vhat


def compute_Vhat_epsilon_batch(
    dZ: np.ndarray,
    g0_vals: np.ndarray,
    conv_w_eps: np.ndarray,
    M: int,
) -> np.ndarray:
    """Vectorised regularised variance proxy for P paths.

    Parameters
    ----------
    dZ : (P, M) float array.
    g0_vals : (M+1,) float array.
    conv_w_eps : (M,) float array.
    M : int

    Returns
    -------
    Vhat : (P, M+1) float array.
    """
    P = dZ.shape[0]
    Vhat = np.zeros((P, M + 1), dtype=float)
    Vhat[:, 0] = g0_vals[0]
    for k in range(1, M + 1):
        Vhat[:, k] = g0_vals[k] + (dZ[:, :k] * conv_w_eps[k - 1 :: -1]).sum(axis=1)
    return Vhat


def compute_U_epsilon_batch(
    w: complex,
    psi_eps: np.ndarray,
    F_eps_arr: np.ndarray,
    dZ: np.ndarray,
    g0_vals: np.ndarray,
    Vhat_eps: np.ndarray,
    M: int,
    h: float,
) -> np.ndarray:
    """Approximate predictable memory U_epsilon^w(t_k) for P paths.

    Uses the recursive semimartingale update (predictable convention):
        U[0] = h * sum_{k=0}^{M-1} F_eps[k] * g0[k]    (initial deterministic value)
        U[k+1] = U[k] + psi_eps[M-k] * dZ[k] - F_eps[k] * Vhat_eps[k+1] * h

    where F_eps[k] = R(w, psi_eps[M-k]) and psi_eps is the regularised Riccati
    solution on [0,T] with psi_eps[j] = psi_eps(j*h).

    Note: U[k] uses information strictly before t_{k+1}, so it is predictable
    at t_k (left-limit convention: hedge at t_k uses U[k] not U[k+1]).

    Parameters
    ----------
    w : complex
        Fourier variable w = 0.5 + i*lambda.
    psi_eps : (M+1,) complex array, regularised Riccati solution.
    F_eps_arr : (M,) complex array, F_eps[k] = R(w, psi_eps[M-k]).
    dZ : (P, M) float array, driver increments.
    g0_vals : (M+1,) float array.
    Vhat_eps : (P, M+1) float array, regularised variance proxy.
    M : int
    h : float

    Returns
    -------
    U_eps : (P, M+1) complex array.  U_eps[:, k] = U_epsilon^w(t_k) for all paths.
    """
    P = dZ.shape[0]
    U_eps = np.zeros((P, M + 1), dtype=complex)

    # Deterministic initial value: integral_0^T F_eps(T-s) g0(s) ds
    # ≈ h * sum_{k=0}^{M-1} F_eps[k] * g0[k]
    tgrid_vals = g0_vals[:M]   # g0(t_0), ..., g0(t_{M-1})
    U0_det = h * np.dot(F_eps_arr, tgrid_vals)
    U_eps[:, 0] = U0_det  # same for all paths (deterministic initial value)

    for k in range(M):
        # psi_eps(T - t_k) = psi_eps[M - k]
        psi_k = psi_eps[M - k]   # complex scalar
        F_k = F_eps_arr[k]        # complex scalar = R(w, psi_eps[M-k])
        dZk = dZ[:, k]            # (P,) float

        # Predictable update: position at t_k, informed by Vhat at t_{k+1}
        U_eps[:, k + 1] = (
            U_eps[:, k]
            + psi_k * dZk
            - F_k * Vhat_eps[:, k + 1] * h
        )

    return U_eps


def build_F_eps_array(w: complex, psi_eps: np.ndarray, params, M: int) -> np.ndarray:
    """Compute F_eps[k] = R(w, psi_eps(T - t_k)) for k = 0, ..., M-1.

    Parameters
    ----------
    w : complex
    psi_eps : (M+1,) complex array.
    params : ModelParams
    M : int

    Returns
    -------
    F_eps : (M,) complex array.
    """
    b, rho, c, Lambda = params.b, params.rho, params.c, params.Lambda
    # psi_eps(T - t_k) = psi_eps[M-k] for k=0,...,M-1
    F_eps = np.array(
        [R_affine(w, psi_eps[M - k], b, rho, c, Lambda) for k in range(M)],
        dtype=complex,
    )
    return F_eps
