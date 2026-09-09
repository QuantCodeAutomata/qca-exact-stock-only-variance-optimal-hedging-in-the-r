import numpy as np

from .affine import B_coefficient, compute_D, Gamma_density
from .transforms import compute_H0
from .parameters import fourier_contour, fourier_weights_atm


def exact_hedge_ratio_exp(
    w: complex, psi_w_val: complex, params
) -> complex:
    """Hedge ratio for the exponential building block e^{w*X}.

    beta_t^w = B_coefficient(w, psi_w(T-t), ...) / D

    Parameters
    ----------
    w : complex
        Fourier variable.
    psi_w_val : complex
        Value of psi_w at the relevant time-to-maturity, i.e. psi_w(T - t).
    params : ModelParams
        Calibrated model parameters.

    Returns
    -------
    beta : complex
    """
    B = B_coefficient(w, psi_w_val, params.rho, params.c, params.Lambda)
    D = compute_D(params.Lambda)
    return B / D


def call_price_fourier(
    psi_w_batch: np.ndarray,
    lambda_vals: np.ndarray,
    weights_gl: np.ndarray,
    params,
    M: int,
) -> float:
    """ATM call price at t = 0 via Gauss-Legendre Fourier inversion.

    C_0 = S_0 - integral_{-R}^{R} a_K(lambda) * H_0(w_lambda, T) d lambda.

    Conjugate symmetry (a_K real-symmetric, H_0(w_{-lambda}) = conj(H_0(w_lambda)))
    reduces this to

        C_0 = S_0 - 2 * Re( sum_i weights_gl[i] * a_K(lambda_i) * H_0(w_i, T) )

    where lambda_vals covers [0, R].

    Parameters
    ----------
    psi_w_batch : np.ndarray, shape (n_pts, M+1)
        Riccati solutions; psi_w_batch[i, k] = psi_{w_i}(k * h).
    lambda_vals : np.ndarray, shape (n_pts,)
        Gauss-Legendre nodes on [0, R].
    weights_gl : np.ndarray, shape (n_pts,)
        Gauss-Legendre weights.
    params : ModelParams
        Calibrated model parameters.
    M : int
        Number of time steps.

    Returns
    -------
    price : float
    """
    a_K = fourier_weights_atm(lambda_vals, params.S0, params.K)  # shape (n_pts,)

    n_pts = len(lambda_vals)
    H0_vals = np.zeros(n_pts, dtype=complex)
    for i in range(n_pts):
        w_i = fourier_contour(lambda_vals[i])
        H0_vals[i] = compute_H0(w_i, psi_w_batch[i], params, M)

    integral = np.dot(weights_gl, a_K * H0_vals)
    return float(params.S0 - 2.0 * integral.real)


def call_hedge_ratio_fourier(
    psi_w_batch: np.ndarray,
    lambda_vals: np.ndarray,
    weights_gl: np.ndarray,
    params,
    M: int,
) -> float:
    """Variance-optimal hedge ratio for the ATM call at t = 0.

    vartheta_0^{call} = 1
        - integral_{-R}^{R} a_K(lambda) * (H_0(w,T) / S_0) * beta_0^w d lambda

    where beta_0^w = B_coefficient(w, psi_w(T), ...) / D.

    Conjugate symmetry reduces to

        vartheta_0^{call} = 1
            - 2 * Re( sum_i w_i * a_K(lambda_i) * H_0(w_i,T) / S_0 * beta_0^{w_i} )

    Parameters
    ----------
    psi_w_batch : np.ndarray, shape (n_pts, M+1)
        Riccati solutions.
    lambda_vals : np.ndarray, shape (n_pts,)
        Gauss-Legendre nodes on [0, R].
    weights_gl : np.ndarray, shape (n_pts,)
        Gauss-Legendre weights.
    params : ModelParams
        Calibrated model parameters.
    M : int
        Number of time steps.

    Returns
    -------
    delta : float
    """
    a_K = fourier_weights_atm(lambda_vals, params.S0, params.K)  # shape (n_pts,)

    n_pts = len(lambda_vals)
    integrand_vals = np.zeros(n_pts, dtype=complex)
    for i in range(n_pts):
        w_i = fourier_contour(lambda_vals[i])
        psi_T = psi_w_batch[i, -1]  # psi_w(T) = psi_w[M]
        H0_i = compute_H0(w_i, psi_w_batch[i], params, M)
        beta_i = exact_hedge_ratio_exp(w_i, psi_T, params)
        integrand_vals[i] = a_K[i] * H0_i / params.S0 * beta_i

    integral = np.dot(weights_gl, integrand_vals)
    return float(1.0 - 2.0 * integral.real)


def minimum_error_density(
    w: complex, psi_w_val: complex, params
) -> float:
    """Instantaneous minimum-variance residual density Gamma_t^w.

    Gamma_t^w = A_t^w - |B_t^w|^2 / D >= 0.

    Parameters
    ----------
    w : complex
        Fourier variable.
    psi_w_val : complex
        Value of psi_w at time-to-maturity T - t.
    params : ModelParams
        Calibrated model parameters.

    Returns
    -------
    gamma : float  (non-negative)
    """
    return Gamma_density(
        w, psi_w_val, params.b, params.rho, params.c, params.Lambda
    )
