"""Convergence diagnostics for the approximate hedging experiment."""
import numpy as np

from .affine import compute_D


def strategy_l2_distance(
    vartheta_eps: np.ndarray,
    vartheta_ref: np.ndarray,
    V_paths: np.ndarray,
    D: float,
    h: float,
) -> float:
    """Squared L^2(S) distance between two hedge strategies.

    d_hat^2 = (1/P) * sum_p sum_{k=0}^{M-1}
              |vartheta_eps[p,k] - vartheta_ref[p,k]|^2 * V[p,k] * D * h

    This approximates the integral E[ integral_0^T |Δvartheta|^2 d<S>_t/S_{t-}^2 ]
    where d<S>_t/S_{t-}^2 = V_t * D dt.

    Parameters
    ----------
    vartheta_eps : (P, M+1) float array
    vartheta_ref : (P, M+1) float array
    V_paths : (P, M+1) float array, variance paths.
    D : float, diffusion coefficient D = 1 + int q^2 dnu.
    h : float, time step.

    Returns
    -------
    float, non-negative.
    """
    P, M1 = vartheta_eps.shape
    M = M1 - 1
    diff = vartheta_eps[:, :M] - vartheta_ref[:, :M]   # (P, M)
    V_k = V_paths[:, :M]                                # (P, M)
    return float(np.mean(np.sum(diff ** 2 * V_k * D * h, axis=1)))


def gain_sup_distance(
    vartheta_eps: np.ndarray,
    vartheta_ref: np.ndarray,
    dS: np.ndarray,
) -> float:
    """Squared sup-norm gain distance.

    g_hat^2 = (1/P) * sum_p max_{0<=k<=M}
              |sum_{j<k} (vartheta_eps[p,j] - vartheta_ref[p,j]) * dS[p,j]|^2

    Parameters
    ----------
    vartheta_eps : (P, M+1) float array
    vartheta_ref : (P, M+1) float array
    dS : (P, M) float array, stock increments S[k+1] - S[k].

    Returns
    -------
    float, non-negative.
    """
    P, M1 = vartheta_eps.shape
    M = M1 - 1
    diff = vartheta_eps[:, :M] - vartheta_ref[:, :M]   # (P, M)
    gain_increments = diff * dS                          # (P, M)
    cumulative_gains = np.cumsum(gain_increments, axis=1)  # (P, M)
    # Add k=0 case (empty sum = 0)
    all_gains = np.hstack([np.zeros((P, 1)), cumulative_gains])  # (P, M+1)
    max_sq = np.max(all_gains ** 2, axis=1)              # (P,)
    return float(np.mean(max_sq))


def capital_distance(x_eps: float, x_ref: float) -> float:
    """Squared capital distance (x_eps - x_ref)^2."""
    return (x_eps - x_ref) ** 2


def total_benchmark_distance(
    x_eps: float,
    x_ref: float,
    vartheta_eps: np.ndarray,
    vartheta_ref: np.ndarray,
    V_paths: np.ndarray,
    D: float,
    h: float,
) -> float:
    """Total benchmark distance: capital + strategy L^2(S) distance.

    Delta_hat = (x_eps - x_ref)^2 + d_hat^2_{eps,R}

    Parameters
    ----------
    x_eps, x_ref : float
    vartheta_eps, vartheta_ref : (P, M+1) float arrays
    V_paths : (P, M+1) float array
    D : float
    h : float

    Returns
    -------
    float, non-negative.
    """
    return capital_distance(x_eps, x_ref) + strategy_l2_distance(
        vartheta_eps, vartheta_ref, V_paths, D, h
    )


def mc_se(values: np.ndarray) -> float:
    """Standard error of the Monte Carlo mean.

    Parameters
    ----------
    values : (P,) float array

    Returns
    -------
    float
    """
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))
