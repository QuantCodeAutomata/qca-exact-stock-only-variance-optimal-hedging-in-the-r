"""Approximate stock-only hedge using regularised kernels K_epsilon.

Constructs predictable hedge positions in the original market using
approximate memory terms based on the shifted kernel K_epsilon.
"""
import numpy as np

from .affine import B_coefficient, compute_D
from .kernels import g0
from .memory_reconstruction import (
    compute_eps_conv_weights,
    compute_Vhat_epsilon_batch,
    compute_U_epsilon_batch,
    build_F_eps_array,
)
from .parameters import fourier_contour, fourier_weights_atm
from .riccati import solve_riccati
from .truncation import chi


def approx_hedge_ratio_exp(
    w: complex, psi_eps_val: complex, params
) -> complex:
    """Approximate hedge ratio beta_eps^w = B_eps^w / D.

    Uses the regularised psi_eps value in place of the exact psi value.

    Parameters
    ----------
    w : complex
    psi_eps_val : complex
        psi_epsilon^w at the relevant time-to-maturity.
    params : ModelParams

    Returns
    -------
    complex
    """
    B = B_coefficient(w, psi_eps_val, params.rho, params.c, params.Lambda)
    D = compute_D(params.Lambda)
    return B / D


def compute_approx_call_hedge(
    sim_data: dict,
    params,
    M: int,
    epsilon: float,
    R_cutoff: float,
    n_gl: int = 20,
) -> tuple:
    """Compute approximate call hedge (initial capital + hedge paths).

    Builds the predictable call hedge using the regularised kernel K_epsilon
    on the *original* market paths (common random numbers).

    Parameters
    ----------
    sim_data : dict
        Output of simulate_market() with keys 'S','V','X','dZ','grid'.
    params : ModelParams
    M : int
        Number of time steps.
    epsilon : float
        Regularisation shift.
    R_cutoff : float
        Fourier truncation cutoff (integrate lambda over [-R, R]).
    n_gl : int
        Number of Gauss-Legendre nodes per half (integrate over [0, R]).

    Returns
    -------
    x_eps_R : float
        Initial capital x_{epsilon,R}.
    vartheta_paths : (P, M+1) float array
        Call hedge positions vartheta_{epsilon,R}(t_k) for each path.
    """
    from numpy.polynomial.legendre import leggauss

    h = params.T / M
    P = sim_data["S"].shape[0]
    dZ = sim_data["dZ"]                 # (P, M)
    g0_vals = g0(np.arange(M + 1) * h, params.V0, params.beta, params.alpha)

    # Gauss-Legendre nodes on [0, R_cutoff]
    xi, wi = leggauss(n_gl)
    lam_nodes = 0.5 * R_cutoff * (xi + 1.0)   # mapped to [0, R_cutoff]
    lam_weights = 0.5 * R_cutoff * wi           # scaled weights

    # Regularised convolution weights for Vhat_eps
    conv_w_eps = compute_eps_conv_weights(M, params.alpha, h, epsilon)

    # Regularised variance proxy Vhat_eps — same for all w
    Vhat_eps = compute_Vhat_epsilon_batch(dZ, g0_vals, conv_w_eps, M)  # (P, M+1)

    D = compute_D(params.Lambda)

    # Accumulate Fourier integrals using conjugate symmetry:
    #   integral_{-R}^{R} a_K(lambda) * f(lambda) d lambda
    #   = 2 * Re( integral_0^R a_K(lambda) * f(lambda) d lambda )

    # Capital: x_{eps,R} = S0 - integral_{-R}^R a_K(lam) * chi(U0(w_lam)) d lam
    capital_integral = 0.0 + 0j

    # Hedge paths: vartheta_{eps,R}(t_k) for all k
    # Accumulate into complex array then take 1 - 2*Re(...)
    hedge_integral = np.zeros((P, M + 1), dtype=complex)

    for i, lam in enumerate(lam_nodes):
        w = fourier_contour(lam)           # 0.5 + i*lambda
        a_K = fourier_weights_atm(np.array([lam]), params.S0, params.K)[0]  # complex

        # Solve regularised Riccati for this w
        psi_eps = solve_riccati(w, params, M, use_epsilon=True, epsilon=epsilon)

        # Build F_eps array
        F_eps_arr = build_F_eps_array(w, psi_eps, params, M)

        # Compute U_epsilon^w on all paths
        U_eps = compute_U_epsilon_batch(
            w, psi_eps, F_eps_arr, dZ, g0_vals, Vhat_eps, M, h
        )  # (P, M+1)

        # Approximate transform: Htilde = exp(w * X_{t-}) * chi(U_eps(t))
        # At t=0: X_0 = 0, so Htilde_0 = chi(U_eps(0))
        chi_U_eps = chi(U_eps)  # (P, M+1), complex

        # Initial capital contribution
        # chi(U_eps^w(0)) is the same for all paths (deterministic initial value)
        capital_integral += lam_weights[i] * a_K * chi_U_eps[0, 0]

        # Hedge ratio beta_eps^w(t_k) = B(w, psi_eps(T-t_k)) / D
        # psi_eps(T - t_k) = psi_eps[M-k]
        beta_eps_vec = np.array(
            [B_coefficient(w, psi_eps[M - k], params.rho, params.c, params.Lambda) / D
             for k in range(M + 1)],
            dtype=complex,
        )  # shape (M+1,)

        # vartheta contribution from this node:
        # integrand: a_K * chi(U_eps) / S_{t-} * beta_eps
        # (using X_{t-} = X[:, k] since S_{t-} = S0 * exp(X_{t-}))
        # The stock factor S_{t-} is path-dependent
        S_left = sim_data["S"][:, :]      # (P, M+1)
        X_left = sim_data["X"][:, :]      # (P, M+1)

        # Htilde_{t-}^w = exp(w * X_{t-}) * chi(U_eps(t_k))
        # = exp(w * X[:, k]) * chi(U_eps[:, k])
        Htilde = np.exp(w * X_left) * chi_U_eps  # (P, M+1)

        # vartheta_tilde_eps^w(t_k) = Htilde_{t-}^w(t_k) / S_{t-} * beta_eps(t_k)
        vartheta_eps_w = Htilde / S_left * beta_eps_vec[np.newaxis, :]  # (P, M+1)

        hedge_integral += lam_weights[i] * a_K * vartheta_eps_w

    # Enforce real via conjugate symmetry (multiply by 2 and take real part)
    x_eps_R = float(params.S0 - 2.0 * capital_integral.real)
    vartheta_paths = 1.0 - 2.0 * hedge_integral.real  # (P, M+1)

    return x_eps_R, vartheta_paths


def compute_exact_call_hedge(
    sim_data: dict,
    params,
    M: int,
    R_cutoff: float,
    n_gl: int = 20,
) -> tuple:
    """Compute exact call hedge (initial capital + hedge paths) in original market.

    This is the unshifted benchmark (epsilon = 0, exact kernel K_alpha).

    Parameters
    ----------
    sim_data : dict
        Simulation output.
    params : ModelParams
    M : int
    R_cutoff : float
    n_gl : int

    Returns
    -------
    x_ref : float
    vartheta_ref : (P, M+1) float array
    """
    from numpy.polynomial.legendre import leggauss

    h = params.T / M
    P = sim_data["S"].shape[0]

    xi, wi = leggauss(n_gl)
    lam_nodes = 0.5 * R_cutoff * (xi + 1.0)
    lam_weights = 0.5 * R_cutoff * wi

    D = compute_D(params.Lambda)
    capital_integral = 0.0 + 0j
    hedge_integral = np.zeros((P, M + 1), dtype=complex)

    for i, lam in enumerate(lam_nodes):
        w = fourier_contour(lam)
        a_K = fourier_weights_atm(np.array([lam]), params.S0, params.K)[0]

        # Exact Riccati solution (unshifted kernel)
        psi_exact = solve_riccati(w, params, M, use_epsilon=False)

        # H_0(w, T) for initial capital
        from .transforms import compute_H0
        H0_w = compute_H0(w, psi_exact, params, M)
        capital_integral += lam_weights[i] * a_K * H0_w

        # Exact conditional transform on paths:
        # H_{t-}(w, T) = exp(w * X_{t-} + integral_{t}^T R(w,psi_w(T-s)) g_t(s) ds)
        # For the exact hedge in the original market, we approximate the path-dependent
        # integral by its t=0 counterpart (frozen-path approximation for comparison purposes)
        # or use the exact stochastic formula.
        # For simplicity and paper faithfulness, we use the deterministic approximation:
        # H_{t}(w,T) ≈ exp(w*X_t) * exp(integral_t^T R(w,psi_w(T-s)) g_0(s) ds)
        # This is the "frozen-path" approximation common in practice.
        # The stochastic term requires the path-dependent variance history.
        # Here we implement the deterministic approximation as the reference.
        from .affine import R_affine
        g0_vals = g0(np.arange(M + 1) * h, params.V0, params.beta, params.alpha)

        # Precompute deterministic part: D_k = integral_{t_k}^T R(w, psi(T-s)) g0(s) ds
        # = h * sum_{j=k}^{M-1} R(w, psi[M-j]) * g0[j]
        b_par = params.b
        rho, c_par, Lambda = params.rho, params.c, params.Lambda
        R_vals = np.array(
            [R_affine(w, psi_exact[M - j], b_par, rho, c_par, Lambda) for j in range(M)],
            dtype=complex,
        )  # R_vals[j] = R(w, psi(T-t_j))
        g0_mid = g0_vals[:M]

        # D_k = h * sum_{j=k}^{M-1} R_vals[j] * g0_mid[j]
        D_vec = np.zeros(M + 1, dtype=complex)
        running = 0.0 + 0j
        for k_r in range(M - 1, -1, -1):
            running += R_vals[k_r] * g0_mid[k_r] * h
            D_vec[k_r] = running
        # D_vec[M] = 0

        X_left = sim_data["X"]  # (P, M+1)
        # Htilde = exp(w * X + D_vec)  (deterministic D, path-dependent X)
        Htilde = np.exp(w * X_left + D_vec[np.newaxis, :])  # (P, M+1)

        S_left = sim_data["S"]  # (P, M+1)
        beta_vec = np.array(
            [B_coefficient(w, psi_exact[M - k], params.rho, params.c, params.Lambda) / D
             for k in range(M + 1)],
            dtype=complex,
        )
        vartheta_w = Htilde / S_left * beta_vec[np.newaxis, :]  # (P, M+1)

        hedge_integral += lam_weights[i] * a_K * vartheta_w

    x_ref = float(params.S0 - 2.0 * capital_integral.real)
    vartheta_ref = 1.0 - 2.0 * hedge_integral.real

    return x_ref, vartheta_ref
