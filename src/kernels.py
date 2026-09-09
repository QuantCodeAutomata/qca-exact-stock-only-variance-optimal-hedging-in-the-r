import numpy as np
from scipy.special import gamma
from scipy.integrate import quad


def kernel_alpha(t: np.ndarray, alpha: float) -> np.ndarray:
    """Fractional kernel K_alpha(t) = t^(alpha-1) / Gamma(alpha).

    Returns 0 at t = 0 (the singularity is integrable for alpha in (0,1)).
    """
    t = np.asarray(t, dtype=float)
    return np.where(t > 0, t ** (alpha - 1) / gamma(alpha), 0.0)


def kernel_epsilon(t: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
    """Regularised kernel K_epsilon(t) = (t + epsilon)^(alpha-1) / Gamma(alpha)."""
    t = np.asarray(t, dtype=float)
    return (t + epsilon) ** (alpha - 1) / gamma(alpha)


def g0(t: np.ndarray, V0: float, beta: float, alpha: float) -> np.ndarray:
    """Deterministic variance forward curve.

    g_0(t) = V0 + beta * t^alpha / Gamma(alpha + 1).
    At t = 0: g_0(0) = V0.
    """
    t = np.asarray(t, dtype=float)
    return V0 + beta * np.where(t > 0, t ** alpha, 0.0) / gamma(alpha + 1)


def G0(t: np.ndarray, V0: float, beta: float, alpha: float) -> np.ndarray:
    """Integral of g_0: G_0(t) = V0*t + beta * t^(alpha+1) / Gamma(alpha + 2)."""
    t = np.asarray(t, dtype=float)
    return V0 * t + beta * t ** (alpha + 1) / gamma(alpha + 2)


def product_integration_weights(M: int, alpha: float, h: float) -> np.ndarray:
    """Product-integration weights for the fractional kernel K_alpha.

    W[0] = 0.
    W[m] = h^alpha * (m^alpha - (m-1)^alpha) / Gamma(alpha + 1)  for m = 1, ..., M.

    These are the integrals
        integral_{(m-1)h}^{mh} u^(alpha-1) / Gamma(alpha) du
    used in the Riccati-Volterra product-integration scheme.
    """
    W = np.zeros(M + 1, dtype=float)
    m = np.arange(1, M + 1, dtype=float)
    W[1:] = h ** alpha * (m ** alpha - (m - 1) ** alpha) / gamma(alpha + 1)
    return W


def l1_kernel_error(alpha: float, T: float, epsilon: float) -> float:
    """L^1(0,T) error between K_epsilon and K_alpha.

    ||K_epsilon - K_alpha||_{L^1(0,T)}
        = integral_0^T |K_epsilon(t) - K_alpha(t)| dt.

    Computed numerically via scipy.integrate.quad.
    """
    def integrand(t: float) -> float:
        return abs(kernel_epsilon(t, alpha, epsilon) - kernel_alpha(t, alpha))

    # The integrand has an integrable singularity at 0 for alpha < 1.
    result, _ = quad(integrand, 0.0, T, limit=200, points=[0.0])
    return float(result)
