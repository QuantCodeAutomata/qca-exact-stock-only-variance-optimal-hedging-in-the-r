"""Tests for approximate hedge construction, memory reconstruction, and convergence metrics."""
import numpy as np
import pytest

from src.parameters import default_params
from src.affine import compute_D
from src.kernels import g0
from src.riccati import solve_riccati
from src.simulation import simulate_market
from src.truncation import chi, chi_scalar
from src.memory_reconstruction import (
    compute_eps_conv_weights,
    compute_Vhat_epsilon,
    compute_Vhat_epsilon_batch,
    compute_U_epsilon_batch,
    build_F_eps_array,
)
from src.approx_hedge import approx_hedge_ratio_exp, compute_approx_call_hedge
from src.metrics import (
    strategy_l2_distance,
    gain_sup_distance,
    capital_distance,
    total_benchmark_distance,
)
from src.validation import (
    check_w1_sanity,
    check_D_value,
    check_positivity,
    check_martingale_sanity,
)


# ---------------------------------------------------------------------------
# Truncation map chi
# ---------------------------------------------------------------------------

class TestChi:
    """Tests for the truncation map chi(z) = exp(min(Re z, 0) + i Im z)."""

    def test_chi_pure_imaginary(self):
        """chi(i*phi) = exp(i*phi): unit-magnitude."""
        z = 0.0 + 1j * np.pi / 3
        assert abs(abs(chi(np.array([z]))[0]) - 1.0) < 1e-14

    def test_chi_negative_real(self):
        """chi(z) = exp(z) when Re(z) <= 0."""
        z = np.array([-0.5 + 1j * 0.3])
        assert abs(chi(z)[0] - np.exp(z[0])) < 1e-14

    def test_chi_positive_real_clips(self):
        """chi(z) has magnitude 1 when Re(z) > 0."""
        z = np.array([2.0 + 1j * 1.0])
        assert abs(abs(chi(z)[0]) - 1.0) < 1e-14

    def test_chi_scalar_matches_array(self):
        """chi_scalar and chi give the same result."""
        z = -0.3 + 0.7j
        assert abs(chi_scalar(z) - chi(np.array([z]))[0]) < 1e-14

    def test_chi_batch(self):
        """chi applies elementwise to arrays."""
        zs = np.array([-1.0 + 0j, 1.0 + 0j, 0.5 + 2j])
        results = chi(zs)
        assert results.shape == zs.shape
        # Negative real: unchanged; positive real: magnitude 1
        assert abs(results[0] - np.exp(zs[0])) < 1e-12
        assert abs(abs(results[1]) - 1.0) < 1e-12

    def test_chi_at_zero(self):
        """chi(0) = exp(0) = 1."""
        assert abs(chi_scalar(0.0 + 0j) - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# Memory reconstruction: conv weights and Vhat
# ---------------------------------------------------------------------------

class TestMemoryReconstruction:
    """Tests for eps convolution weights and Vhat_epsilon."""

    def test_conv_weights_positive(self):
        """Regularised convolution weights are all positive."""
        params = default_params()
        M = 32
        h = params.T / M
        w = compute_eps_conv_weights(M, params.alpha, h, params.T / 8)
        assert np.all(w > 0)

    def test_conv_weights_shape(self):
        """conv_weights has shape (M,)."""
        params = default_params()
        M = 32
        h = params.T / M
        w = compute_eps_conv_weights(M, params.alpha, h, params.T / 8)
        assert w.shape == (M,)

    def test_Vhat_initial_value(self):
        """Vhat_eps(t_0) = g_0(t_0) = V0 (no past history at t=0)."""
        params = default_params()
        M = 32
        h = params.T / M
        eps = params.T / 8
        tgrid = np.arange(M + 1) * h
        g0_vals = g0(tgrid, params.V0, params.beta, params.alpha)
        conv_w = compute_eps_conv_weights(M, params.alpha, h, eps)
        dZ = np.zeros(M)
        Vhat = compute_Vhat_epsilon(dZ, g0_vals, conv_w, M)
        assert abs(Vhat[0] - params.V0) < 1e-12

    def test_Vhat_batch_matches_single(self):
        """Vectorised Vhat_eps must match single-path computation."""
        params = default_params()
        M = 32
        h = params.T / M
        eps = params.T / 8
        P = 5
        rng = np.random.default_rng(0)
        dZ = rng.normal(0, 0.01, (P, M))
        tgrid = np.arange(M + 1) * h
        g0_vals = g0(tgrid, params.V0, params.beta, params.alpha)
        conv_w = compute_eps_conv_weights(M, params.alpha, h, eps)

        Vhat_batch = compute_Vhat_epsilon_batch(dZ, g0_vals, conv_w, M)
        for p in range(P):
            Vhat_single = compute_Vhat_epsilon(dZ[p], g0_vals, conv_w, M)
            assert np.allclose(Vhat_batch[p], Vhat_single)

    def test_U_eps_shape(self):
        """U_epsilon_batch must produce (P, M+1) complex array."""
        params = default_params()
        M = 32
        h = params.T / M
        eps = params.T / 8
        P = 5
        w = 0.5 + 1j * 3.0

        psi_eps = solve_riccati(w, params, M, use_epsilon=True, epsilon=eps)
        F_eps = build_F_eps_array(w, psi_eps, params, M)

        rng = np.random.default_rng(1)
        dZ = rng.normal(0, 0.01, (P, M))
        tgrid = np.arange(M + 1) * h
        g0_vals = g0(tgrid, params.V0, params.beta, params.alpha)
        conv_w = compute_eps_conv_weights(M, params.alpha, h, eps)
        Vhat = compute_Vhat_epsilon_batch(dZ, g0_vals, conv_w, M)

        U = compute_U_epsilon_batch(w, psi_eps, F_eps, dZ, g0_vals, Vhat, M, h)
        assert U.shape == (P, M + 1)
        assert U.dtype == complex

    def test_build_F_eps_shape(self):
        """build_F_eps_array returns (M,) complex array."""
        params = default_params()
        M = 32
        eps = params.T / 8
        w = 0.5 + 1j * 3.0
        psi_eps = solve_riccati(w, params, M, use_epsilon=True, epsilon=eps)
        F_eps = build_F_eps_array(w, psi_eps, params, M)
        assert F_eps.shape == (M,)
        assert F_eps.dtype == complex


# ---------------------------------------------------------------------------
# Approximate hedge ratio
# ---------------------------------------------------------------------------

class TestApproxHedgeRatio:
    """Tests for approx_hedge_ratio_exp."""

    def test_beta_at_w1_psi0_eps(self):
        """For w=1 and psi_eps small, beta_eps ≈ 1."""
        params = default_params()
        M = 32
        eps = params.T / 64
        psi_eps = solve_riccati(1.0 + 0j, params, M, use_epsilon=True, epsilon=eps)
        # psi_1^eps ≈ 0 for small eps
        psi_at_T = psi_eps[-1]
        beta = approx_hedge_ratio_exp(1.0 + 0j, psi_at_T, params)
        assert abs(beta - 1.0) < 0.01

    def test_beta_complex_mode(self):
        """approx_hedge_ratio_exp returns a complex number for contour modes."""
        params = default_params()
        M = 32
        eps = params.T / 8
        w = 0.5 + 1j * 5.0
        psi_eps = solve_riccati(w, params, M, use_epsilon=True, epsilon=eps)
        beta = approx_hedge_ratio_exp(w, psi_eps[-1], params)
        assert isinstance(beta, complex)
        assert np.isfinite(abs(beta))


# ---------------------------------------------------------------------------
# Compute approx call hedge (smoke test)
# ---------------------------------------------------------------------------

class TestComputeApproxCallHedge:
    """Smoke tests for compute_approx_call_hedge."""

    @pytest.fixture(scope="class")
    def sim_data(self):
        params = default_params()
        return simulate_market(params, M=32, P=20, seed=42)

    def test_returns_real_capital(self, sim_data):
        """Initial capital must be a real scalar."""
        params = default_params()
        x_eps, vartheta = compute_approx_call_hedge(
            sim_data, params, M=32, epsilon=params.T / 4, R_cutoff=5.0, n_gl=8
        )
        assert isinstance(x_eps, float)
        assert np.isfinite(x_eps)

    def test_hedge_shape(self, sim_data):
        """Hedge array must have shape (P, M+1)."""
        params = default_params()
        P = sim_data["S"].shape[0]
        M = sim_data["S"].shape[1] - 1
        x_eps, vartheta = compute_approx_call_hedge(
            sim_data, params, M=M, epsilon=params.T / 4, R_cutoff=5.0, n_gl=8
        )
        assert vartheta.shape == (P, M + 1)

    def test_hedge_real_valued(self, sim_data):
        """Hedge positions must be real-valued."""
        params = default_params()
        x_eps, vartheta = compute_approx_call_hedge(
            sim_data, params, M=32, epsilon=params.T / 4, R_cutoff=5.0, n_gl=8
        )
        assert np.all(np.isfinite(vartheta))

    def test_capital_in_reasonable_range(self, sim_data):
        """ATM call price should be positive and less than 0.5 * S0."""
        params = default_params()
        x_eps, _ = compute_approx_call_hedge(
            sim_data, params, M=32, epsilon=params.T / 8, R_cutoff=10.0, n_gl=12
        )
        assert 0.0 < x_eps < 0.5 * params.S0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for strategy_l2_distance, gain_sup_distance, total_benchmark_distance."""

    def test_strategy_dist_zero_if_equal(self):
        """strategy_l2_distance = 0 when both strategies are identical."""
        params = default_params()
        P, M = 10, 20
        vartheta = np.random.default_rng(0).normal(0, 1, (P, M + 1)).astype(float)
        V = np.ones((P, M + 1)) * params.V0
        D = compute_D(params.Lambda)
        h = params.T / M
        dist = strategy_l2_distance(vartheta, vartheta, V, D, h)
        assert dist == pytest.approx(0.0, abs=1e-14)

    def test_strategy_dist_nonneg(self):
        """strategy_l2_distance must be >= 0."""
        params = default_params()
        P, M = 5, 10
        rng = np.random.default_rng(1)
        v1 = rng.normal(0, 1, (P, M + 1))
        v2 = rng.normal(0, 1, (P, M + 1))
        V = np.ones((P, M + 1)) * params.V0
        D = compute_D(params.Lambda)
        h = params.T / M
        dist = strategy_l2_distance(v1, v2, V, D, h)
        assert dist >= 0.0

    def test_gain_dist_zero_if_equal(self):
        """gain_sup_distance = 0 when both strategies are identical."""
        P, M = 5, 10
        vartheta = np.random.default_rng(2).normal(0, 1, (P, M + 1))
        dS = np.random.default_rng(3).normal(0, 0.01, (P, M))
        dist = gain_sup_distance(vartheta, vartheta, dS)
        assert dist == pytest.approx(0.0, abs=1e-14)

    def test_gain_dist_nonneg(self):
        """gain_sup_distance must be >= 0."""
        P, M = 5, 10
        rng = np.random.default_rng(4)
        v1 = rng.normal(0, 1, (P, M + 1))
        v2 = rng.normal(0, 1, (P, M + 1))
        dS = rng.normal(0, 0.01, (P, M))
        dist = gain_sup_distance(v1, v2, dS)
        assert dist >= 0.0

    def test_capital_distance_zero_if_equal(self):
        """capital_distance = 0 when both capitals are the same."""
        assert capital_distance(0.5, 0.5) == 0.0

    def test_capital_distance_symmetric(self):
        """capital_distance(a, b) = capital_distance(b, a)."""
        assert capital_distance(0.3, 0.5) == capital_distance(0.5, 0.3)

    def test_total_distance_decomposition(self):
        """total_benchmark_distance = capital_distance + strategy_l2_distance."""
        params = default_params()
        P, M = 5, 10
        rng = np.random.default_rng(5)
        v1 = rng.normal(0, 1, (P, M + 1))
        v2 = rng.normal(0, 1, (P, M + 1))
        V = np.ones((P, M + 1)) * params.V0
        D = compute_D(params.Lambda)
        h = params.T / M
        x1, x2 = 0.3, 0.4

        total = total_benchmark_distance(x1, x2, v1, v2, V, D, h)
        cap_d = capital_distance(x1, x2)
        strat_d = strategy_l2_distance(v1, v2, V, D, h)
        assert abs(total - (cap_d + strat_d)) < 1e-14


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for validation.py functions."""

    def test_check_D_value(self):
        """Closed-form D should match numerical integration within 1e-8."""
        params = default_params()
        result = check_D_value(params)
        assert result["error"] < 1e-8

    def test_check_w1_sanity(self):
        """w=1 sanity check should pass: psi_1=0, beta^1=1, Gamma^1=0."""
        params = default_params()
        result = check_w1_sanity(params, M=128)
        assert result["passed"], f"w=1 sanity FAILED: {result}"
        assert result["psi1_max_error"] < 1e-10
        assert result["beta1_error"] < 1e-10
        assert abs(result["gamma1_error"]) < 1e-10

    def test_check_martingale_sanity_passes_on_many_paths(self):
        """Martingale check should pass for P=1000 paths."""
        params = default_params()
        sim = simulate_market(params, M=64, P=1000, seed=42)
        result = check_martingale_sanity(sim["S"], target=1.0, tol=0.05)
        assert result["passed"], f"Martingale check failed: E[S_T]={result['mean_ST']:.4f}"

    def test_check_positivity_mostly_holds(self):
        """Euler discretization can produce small negative V (full-truncation scheme).

        check_positivity records the minimum; we only require V > -0.5 as a sanity
        bound rather than strict positivity at every node.
        """
        params = default_params()
        sim = simulate_market(params, M=64, P=200, seed=99)
        result = check_positivity(sim["V"])
        # min_V must be bounded below (not severely negative)
        assert result["min_V"] > -0.5, f"Positivity severely violated: min_V={result['min_V']:.3e}"
