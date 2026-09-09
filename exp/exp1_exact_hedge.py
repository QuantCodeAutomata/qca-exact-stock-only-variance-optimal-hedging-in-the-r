"""Experiment 1 — Exact variance-optimal stock-only hedging in the rough Hawkes-Heston model.

Validates:
- Riccati-Volterra solution for the exact kernel
- w=1 sanity check (psi_1 = 0, beta^1 = 1, residual = 0)
- ATM call price and hedge ratio via Lewis-Fourier inversion
- Non-negativity of residual density Gamma_t^w
- Martingale check via Monte Carlo
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parameters import default_params, time_grid, gauss_legendre_nodes_weights, fourier_contour
from src.kernels import kernel_alpha, g0, product_integration_weights
from src.affine import compute_D, R_affine, Gamma_density, B_coefficient
from src.riccati import solve_riccati, solve_riccati_batch
from src.transforms import compute_H0
from src.exact_hedge import (
    exact_hedge_ratio_exp,
    call_price_fourier,
    call_hedge_ratio_fourier,
    minimum_error_density,
)
from src.validation import (
    check_w1_sanity,
    check_martingale_sanity,
    check_D_value,
    check_positivity,
    check_gamma_nonneg,
)
from src.simulation import simulate_market

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp1(M: int = 512, R: float = 40.0, n_gl: int = 32, P_mc: int = 1000):
    """Run all experiment 1 diagnostics and save results.

    Parameters
    ----------
    M : int
        Number of time steps (production = 512).
    R : float
        Fourier cutoff radius.
    n_gl : int
        Number of Gauss-Legendre nodes on [0, R].
    P_mc : int
        Number of Monte Carlo paths for martingale check.
    """
    params = default_params()
    h = params.T / M
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Exact Variance-Optimal Hedging")
    print(f"  alpha={params.alpha}, rho={params.rho}, b={params.b}, c={params.c}")
    print(f"  Lambda={params.Lambda}, beta={params.beta}, V0={params.V0}")
    print(f"  T={params.T}, M={M}, h={h:.6f}, R={R}, n_gl={n_gl}")
    print("="*60)

    results = {}

    # ------------------------------------------------------------------
    # 1. Validate D
    # ------------------------------------------------------------------
    D_check = check_D_value(params)
    D = D_check["D_computed"]
    print(f"\n[1] D = {D:.10f}")
    print(f"    D_numerical = {D_check['D_numerical']:.10f}")
    print(f"    Error = {D_check['error']:.2e}")
    results["D"] = D
    results["D_error"] = D_check["error"]

    # ------------------------------------------------------------------
    # 2. w=1 sanity check
    # ------------------------------------------------------------------
    w1_check = check_w1_sanity(params, M, tol=1e-10)
    print(f"\n[2] w=1 sanity check:")
    print(f"    max|psi_1| = {w1_check['psi1_max_error']:.2e}  (target: 0)")
    print(f"    |beta^1 - 1| = {w1_check['beta1_error']:.2e}  (target: 0)")
    print(f"    |Gamma^1| = {abs(w1_check['gamma1_error']):.2e}  (target: 0)")
    print(f"    B(1,0) = {w1_check['B1']:.10f}  D = {w1_check['D']:.10f}")
    print(f"    PASSED: {w1_check['passed']}")
    results.update({f"w1_{k}": v for k, v in w1_check.items()})

    # ------------------------------------------------------------------
    # 3. Solve Riccati for Fourier nodes and compute H0
    # ------------------------------------------------------------------
    lam_nodes, gl_weights = gauss_legendre_nodes_weights(n_gl, 0.0, R)
    w_array = np.array([fourier_contour(lam) for lam in lam_nodes])

    print(f"\n[3] Solving Riccati-Volterra for {n_gl} Fourier nodes on [0, {R}]...")
    psi_batch = solve_riccati_batch(w_array, params, M)  # (n_gl, M+1)
    print(f"    Done. psi shape: {psi_batch.shape}")

    # ------------------------------------------------------------------
    # 4. ATM call price
    # ------------------------------------------------------------------
    call_price = call_price_fourier(psi_batch, lam_nodes, gl_weights, params, M)
    print(f"\n[4] ATM call price (Fourier, R={R}): {call_price:.6f}")
    results["call_price_R40"] = call_price

    # Convergence over cutoff R
    for R_test in [10.0, 20.0, 40.0]:
        if R_test > R:
            continue
        n_test = n_gl
        lam_t, wt = gauss_legendre_nodes_weights(n_test, 0.0, R_test)
        w_t = np.array([fourier_contour(l) for l in lam_t])
        psi_t = solve_riccati_batch(w_t, params, M)
        cp_t = call_price_fourier(psi_t, lam_t, wt, params, M)
        print(f"    R={R_test:.0f}: call price = {cp_t:.6f}")
        results[f"call_price_R{int(R_test)}"] = cp_t

    # ------------------------------------------------------------------
    # 5. ATM call hedge (delta at t=0)
    # ------------------------------------------------------------------
    call_delta = call_hedge_ratio_fourier(psi_batch, lam_nodes, gl_weights, params, M)
    print(f"\n[5] ATM call hedge ratio (delta): {call_delta:.6f}")
    print(f"    Expected range: (0, 1)")
    results["call_delta_R40"] = call_delta

    # ------------------------------------------------------------------
    # 6. Residual density Gamma_t^w along time for a fixed mode
    # ------------------------------------------------------------------
    lam_test = 5.0
    w_test = fourier_contour(lam_test)
    psi_test = solve_riccati(w_test, params, M)
    tgrid = time_grid(params.T, M)

    gamma_vals = np.zeros(M + 1, dtype=float)
    for k in range(M + 1):
        # psi_test at T - t_k = psi_test[M - k]
        psi_val = psi_test[M - k] if k < M else 0.0 + 0j
        gamma_vals[k] = Gamma_density(
            w_test, psi_val, params.b, params.rho, params.c, params.Lambda
        )

    min_gamma = float(gamma_vals.min())
    print(f"\n[6] Residual density Gamma_t^w for lambda={lam_test}:")
    print(f"    min Gamma = {min_gamma:.6e}  (should be >= 0)")
    print(f"    max Gamma = {float(gamma_vals.max()):.6e}")
    results["min_gamma"] = min_gamma
    results["max_gamma"] = float(gamma_vals.max())

    # Plot Gamma over time
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tgrid, gamma_vals, label=f"$\\Gamma_t^w$, $\\lambda={lam_test}$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("$\\Gamma_t^w$")
    ax.set_title("Residual Density $\\Gamma_t^w$ vs. Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_gamma_density.png"), dpi=150)
    plt.close(fig)
    print(f"    Plot saved: results/exp1_gamma_density.png")

    # ------------------------------------------------------------------
    # 7. H_0(w, T) diagnostics: check H_0(0, T) = 1
    # ------------------------------------------------------------------
    psi_w0 = solve_riccati(0.0 + 0j, params, M)
    H0_w0 = compute_H0(0.0 + 0j, psi_w0, params, M)
    print(f"\n[7] H_0(w=0, T) = {H0_w0}  (should be 1)")
    results["H0_w0"] = complex(H0_w0)

    # H_0(1, T) = E[S_T / S_0] = 1 (martingale, approximate)
    psi_w1 = solve_riccati(1.0 + 0j, params, M)
    H0_w1 = compute_H0(1.0 + 0j, psi_w1, params, M)
    print(f"    H_0(w=1, T) = {H0_w1}  (≈ 1, martingale check)")
    results["H0_w1"] = complex(H0_w1)

    # ------------------------------------------------------------------
    # 8. Psi_w profile along time for several modes
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lam_p in [1.0, 5.0, 10.0, 20.0]:
        wp = fourier_contour(lam_p)
        psi_p = solve_riccati(wp, params, M)
        axes[0].plot(tgrid, psi_p.real, label=f"$\\lambda={lam_p}$")
        axes[1].plot(tgrid, psi_p.imag, label=f"$\\lambda={lam_p}$")
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("Re($\\psi_w(t)$)")
    axes[0].set_title("Re($\\psi_w$) vs. time")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("Im($\\psi_w(t)$)")
    axes[1].set_title("Im($\\psi_w$) vs. time")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_psi_profiles.png"), dpi=150)
    plt.close(fig)
    print(f"\n[8] Psi profiles saved: results/exp1_psi_profiles.png")

    # ------------------------------------------------------------------
    # 9. Gamma nonneg check over grid of modes
    # ------------------------------------------------------------------
    gamma_check = check_gamma_nonneg(params, M // 4, R_cutoff=20.0, n_check=20)
    print(f"\n[9] Gamma non-negativity check (grid of modes):")
    print(f"    min Gamma = {gamma_check['min_Gamma']:.2e}  (should be >= 0)")
    print(f"    PASSED: {gamma_check['passed']}")
    results["gamma_nonneg_passed"] = gamma_check["passed"]

    # ------------------------------------------------------------------
    # 10. Monte Carlo martingale check
    # ------------------------------------------------------------------
    print(f"\n[10] Monte Carlo martingale check (P={P_mc} paths, M={min(M, 128)})...")
    M_mc = min(M, 128)
    sim = simulate_market(params, M_mc, P_mc, seed=42)
    mcheck = check_martingale_sanity(sim["S"], target=params.S0)
    pos_check = check_positivity(sim["V"])
    print(f"     E[S_T] = {mcheck['mean_ST']:.6f} ± {mcheck['se_ST']:.6f}  (target: {params.S0})")
    print(f"     Std(S_T) = {mcheck['std_ST']:.6f}")
    print(f"     Martingale PASSED: {mcheck['passed']}")
    print(f"     Min V = {pos_check['min_V']:.6e}")
    print(f"     Positivity PASSED: {pos_check['passed']}")
    results["mean_ST"] = mcheck["mean_ST"]
    results["std_ST"] = mcheck["std_ST"]
    results["se_ST"] = mcheck["se_ST"]
    results["martingale_passed"] = mcheck["passed"]
    results["min_V"] = pos_check["min_V"]
    results["positivity_passed"] = pos_check["passed"]

    # ------------------------------------------------------------------
    # 11. Beta profile for ATM call
    # ------------------------------------------------------------------
    beta_profile = np.zeros(M + 1, dtype=float)
    for k in range(M + 1):
        beta_acc = 0.0 + 0j
        for i in range(n_gl):
            w_i = w_array[i]
            psi_val = psi_batch[i, M - k] if k < M else 0.0 + 0j
            beta_i = exact_hedge_ratio_exp(w_i, psi_val, params)
            H0_i = compute_H0(w_i, psi_batch[i], params, M)
            from src.parameters import fourier_weights_atm
            a_K_i = fourier_weights_atm(np.array([lam_nodes[i]]), params.S0, params.K)[0]
            beta_acc += gl_weights[i] * a_K_i * H0_i / params.S0 * beta_i
        beta_profile[k] = float(1.0 - 2.0 * beta_acc.real)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tgrid, beta_profile, label="Call hedge ratio $\\vartheta_t^{call}$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(1, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Hedge ratio")
    ax.set_title("Exact ATM Call Hedge Ratio Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_call_hedge_ratio.png"), dpi=150)
    plt.close(fig)
    print(f"\n[11] Call hedge ratio profile saved: results/exp1_call_hedge_ratio.png")
    results["call_hedge_t0"] = float(beta_profile[0])
    results["call_hedge_tT"] = float(beta_profile[-1])

    return results


def main():
    results = run_exp1(M=256, R=40.0, n_gl=32, P_mc=500)

    # Save summary
    summary_lines = [
        "# Experiment 1 Results\n",
        "\n## Model Parameters\n",
        "alpha=0.527, rho=-0.731, b=-1.812, c=0.115, Lambda=0.276, beta=0.049, V0=0.0079\n",
        "\n## Key Numerical Results\n",
        "\n### D (Diffusion coefficient)\n",
        f"D = {results.get('D', 'N/A'):.10f}\n",
        f"D_error = {results.get('D_error', 'N/A'):.2e}\n",
        "\n### w=1 Sanity Check\n",
        f"max|psi_1| = {results.get('w1_psi1_max_error', 'N/A'):.2e}\n",
        f"|beta^1 - 1| = {results.get('w1_beta1_error', 'N/A'):.2e}\n",
        f"|Gamma^1| = {abs(results.get('w1_gamma1_error', 0)):.2e}\n",
        f"PASSED = {results.get('w1_passed', False)}\n",
        "\n### ATM Call Price (R=40, n_gl=32)\n",
        f"Call price = {results.get('call_price_R40', 'N/A'):.6f}\n",
        f"Call delta = {results.get('call_delta_R40', 'N/A'):.6f}\n",
        "\n### Residual Density\n",
        f"min Gamma = {results.get('min_gamma', 'N/A'):.6e}\n",
        f"max Gamma = {results.get('max_gamma', 'N/A'):.6e}\n",
        f"Non-negative: {results.get('gamma_nonneg_passed', False)}\n",
        "\n### Martingale Check (Monte Carlo)\n",
        f"E[S_T] = {results.get('mean_ST', 'N/A'):.6f} +/- {results.get('se_ST', 'N/A'):.6f}\n",
        f"Std(S_T) = {results.get('std_ST', 'N/A'):.6f}\n",
        f"Martingale PASSED = {results.get('martingale_passed', False)}\n",
        f"Min V = {results.get('min_V', 'N/A'):.6e}\n",
        f"Positivity PASSED = {results.get('positivity_passed', False)}\n",
        "\n### H_0 Checks\n",
        f"H_0(w=0, T) = {results.get('H0_w0', 'N/A')}\n",
        f"H_0(w=1, T) = {results.get('H0_w1', 'N/A')}\n",
    ]

    results_path = os.path.join(RESULTS_DIR, "exp1_summary.md")
    with open(results_path, "w") as f:
        f.writelines(summary_lines)
    print(f"\nResults saved to: {results_path}")
    return results


if __name__ == "__main__":
    main()
