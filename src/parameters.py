import numpy as np
from dataclasses import dataclass
import scipy.special  # noqa: F401 (kept for consistency; gamma used via np.polynomial)
from numpy.polynomial.legendre import leggauss


@dataclass
class ModelParams:
    alpha: float = 0.527
    rho: float = -0.731
    b: float = -1.812
    c: float = 0.115
    Lambda: float = 0.276
    beta: float = 0.049
    V0: float = 0.0079
    S0: float = 1.0
    T: float = 0.25
    K: float = 1.0


def default_params() -> ModelParams:
    """Return the calibrated model parameters."""
    return ModelParams()


def time_grid(T: float, M: int) -> np.ndarray:
    """Return uniform grid [0, h, 2h, ..., T] of length M+1 where h = T/M."""
    return np.linspace(0.0, T, M + 1)


def fourier_contour(lambda_val: float) -> complex:
    """Return the dampened Fourier contour point w = 0.5 + i*lambda."""
    return 0.5 + 1j * lambda_val


def fourier_weights_atm(
    lambda_vals: np.ndarray, S0: float, K: float
) -> np.ndarray:
    """Compute ATM Fourier weights a_K(lambda).

    a_K(lambda) = sqrt(S0*K) / (2*pi) * exp(1j*lambda*log(S0/K)) / (lambda^2 + 0.25)

    For K = S0 = 1 this simplifies to 1 / (2*pi*(lambda^2 + 0.25)).
    """
    log_moneyness = np.log(S0 / K)
    return (
        np.sqrt(S0 * K)
        / (2.0 * np.pi)
        * np.exp(1j * lambda_vals * log_moneyness)
        / (lambda_vals ** 2 + 0.25)
    )


def gauss_legendre_nodes_weights(
    n_pts: int, a: float, b: float
) -> tuple:
    """Gauss-Legendre nodes and weights mapped from [-1,1] to [a, b].

    Returns
    -------
    nodes : np.ndarray, shape (n_pts,)
    weights : np.ndarray, shape (n_pts,)
    """
    xi, wi = leggauss(n_pts)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    nodes = mid + half * xi
    weights = half * wi
    return nodes, weights
