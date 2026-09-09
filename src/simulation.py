"""Monte Carlo simulation of the original rough Hawkes-Heston market under Q."""
import numpy as np
from scipy.special import gamma

from .kernels import g0, product_integration_weights


def simulate_market(params, M: int, P: int, seed: int = 42) -> dict:
    """Simulate P paths of the rough Hawkes-Heston market on [0, T] with M steps.

    Uses a product-integration Euler-Maruyama scheme.

    Parameters
    ----------
    params : ModelParams
        Calibrated model parameters.
    M : int
        Number of time steps; h = T/M.
    P : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'X'    : (P, M+1) float array, log-price paths X_t = log(S_t/S_0).
        'S'    : (P, M+1) float array, stock price paths.
        'V'    : (P, M+1) float array, variance paths.
        'dZ'   : (P, M)   float array, driver increments Delta Z_k.
        'grid' : (M+1,)   float array, time grid.
    """
    rng = np.random.default_rng(seed)

    h = params.T / M
    alpha = params.alpha
    b, rho, c, Lambda = params.b, params.rho, params.c, params.Lambda
    V0, beta_p, S0 = params.V0, params.beta, params.S0

    tgrid = np.arange(M + 1, dtype=float) * h

    # g_0 values on the full grid
    g0_vals = g0(tgrid, V0, beta_p, alpha)  # shape (M+1,)

    # Product-integration convolution weights:
    # conv_w[m] = h^(alpha-1) * [m^alpha - (m-1)^alpha] / Gamma(alpha+1), m=1..M
    # These satisfy: V(t_k) = g0(t_k) + sum_{j<k} conv_w[k-j] * dZ[j]
    W = product_integration_weights(M, alpha, h)  # shape (M+1,), W[0]=0
    # conv_w[m] = W[m] / h so that  sum_j conv_w[k-j]*dZ[j] = (K * dZ)(t_k)
    conv_w = W[1:] / h  # shape (M,): conv_w[m-1] corresponds to lag m

    # Allocate
    X = np.zeros((P, M + 1), dtype=float)
    V = np.zeros((P, M + 1), dtype=float)
    dZ_arr = np.zeros((P, M), dtype=float)

    V[:, 0] = V0

    # Pre-generate all Brownian increments at once (memory-efficient for large M,P)
    dW1 = rng.standard_normal((P, M)) * np.sqrt(h)  # (P, M)
    dW2 = rng.standard_normal((P, M)) * np.sqrt(h)  # (P, M)

    # Expected jumps per step is very small (≈ V0*h ≈ 4e-6), so
    # we generate only a total Poisson count and assign marks.
    # Drift correction constants:
    #   log(1+q(z)) = -Lambda*z  =>  compensation term = +Lambda*V*h
    #   net X drift per unit V*h = (-0.5 + Lambda)
    x_drift_coef = -0.5 + Lambda  # coefficient of V_k * h in ΔX

    for k in range(M):
        Vk = np.maximum(V[:, k], 0.0)  # (P,)
        sqrt_Vk = np.sqrt(Vk)          # (P,)

        # --- Jump component (Poisson with intensity V_k * h) ---
        # For rare jumps, vectorise Poisson sampling over P.
        lam_k = Vk * h                                          # (P,)
        N_jumps = rng.poisson(lam_k)                            # (P,)
        total_jumps = int(N_jumps.sum())
        if total_jumps > 0:
            all_marks = rng.exponential(1.0, total_jumps)
            jump_sums = np.zeros(P, dtype=float)
            idx = 0
            ends = np.cumsum(N_jumps)
            starts = ends - N_jumps
            for p in range(P):
                s, e = starts[p], ends[p]
                if e > s:
                    jump_sums[p] = all_marks[s:e].sum()
        else:
            jump_sums = np.zeros(P, dtype=float)

        dW1_k = dW1[:, k]  # (P,)
        dW2_k = dW2[:, k]  # (P,)

        # --- Driver increment: dZ = (b-1)*V*h + sqrt(c)*sqrt(V)*dW2 + jump_sum ---
        # (the -V*h compensates the Exp(1) mean of raw jumps)
        dZk = (
            (b - 1.0) * Vk * h
            + np.sqrt(c) * sqrt_Vk * dW2_k
            + jump_sums
        )

        # --- Log-price increment ---
        # ΔX = x_drift_coef * V*h + sqrt(1-rho^2)*sqrt(V)*dW1 + rho*sqrt(V)*dW2
        #       - Lambda * jump_sum
        dXk = (
            x_drift_coef * Vk * h
            + np.sqrt(1.0 - rho ** 2) * sqrt_Vk * dW1_k
            + rho * sqrt_Vk * dW2_k
            - Lambda * jump_sums
        )

        X[:, k + 1] = X[:, k] + dXk
        dZ_arr[:, k] = dZk

        # --- Variance update via product-integration convolution ---
        # V(t_{k+1}) = g0(t_{k+1}) + sum_{j=0}^{k} conv_w[k-j] * dZ[j]
        # conv_w has shape (M,): conv_w[m] for m=0,...,M-1 where m corresponds to lag m+1
        # So conv_w[k-j] for j=0..k is conv_w[k], conv_w[k-1], ..., conv_w[0]
        V[:, k + 1] = g0_vals[k + 1] + (
            dZ_arr[:, : k + 1] * conv_w[k::-1]
        ).sum(axis=1)

    S = S0 * np.exp(X)
    return {
        "X": X,
        "S": S,
        "V": V,
        "dZ": dZ_arr,
        "grid": tgrid,
    }
