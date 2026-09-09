"""Tests for kernel functions in src/kernels.py."""
import numpy as np
import pytest
from scipy.integrate import quad

from src.parameters import default_params
from src.kernels import (
    kernel_alpha,
    kernel_epsilon,
    g0,
    G0,
    product_integration_weights,
    l1_kernel_error,
)


class TestKernelAlpha:
    """Tests for the fractional kernel K_alpha(t) = t^(alpha-1)/Gamma(alpha)."""

    def test_zero_at_t_zero(self):
        """K_alpha(0) = 0 (singularity handled by returning 0)."""
        assert kernel_alpha(np.array([0.0]), 0.527) == pytest.approx(0.0)

    def test_positive_for_t_positive(self):
        """K_alpha(t) > 0 for all t > 0 and alpha in (0, 1)."""
        t = np.array([0.01, 0.1, 0.5, 1.0])
        vals = kernel_alpha(t, 0.527)
        assert np.all(vals > 0)

    def test_monotone_decreasing_for_alpha_less_one(self):
        """For alpha < 1, K_alpha is decreasing in t."""
        t = np.array([0.01, 0.05, 0.1, 0.25])
        vals = kernel_alpha(t, 0.527)
        assert np.all(np.diff(vals) < 0)

    def test_shape_preserved(self):
        """Output shape matches input shape."""
        t = np.linspace(0, 0.25, 100)
        vals = kernel_alpha(t, 0.527)
        assert vals.shape == t.shape

    def test_integrability(self):
        """integral_0^T K_alpha(t) dt = T^alpha / Gamma(alpha+1)."""
        alpha = 0.527
        T = 0.25
        integral, _ = quad(lambda t: kernel_alpha(np.array([t]), alpha)[0], 1e-10, T)
        expected = T ** alpha / float(__import__('scipy.special', fromlist=['gamma']).gamma(alpha + 1))
        assert abs(integral - expected) < 1e-6


class TestKernelEpsilon:
    """Tests for the regularised kernel K_epsilon(t) = (t+epsilon)^(alpha-1)/Gamma(alpha)."""

    def test_finite_at_zero(self):
        """K_epsilon(0) = epsilon^(alpha-1)/Gamma(alpha) is finite."""
        from scipy.special import gamma
        alpha = 0.527
        eps = 0.01
        expected = eps ** (alpha - 1) / gamma(alpha)
        val = kernel_epsilon(np.array([0.0]), alpha, eps)[0]
        assert abs(val - expected) < 1e-12

    def test_approaches_K_alpha_as_eps_shrinks(self):
        """For large t, K_eps and K_alpha are close."""
        alpha = 0.527
        t = np.array([0.1, 0.15, 0.2, 0.25])
        for eps in [1e-3, 1e-4]:
            diff = np.abs(kernel_epsilon(t, alpha, eps) - kernel_alpha(t, alpha))
            assert np.all(diff < 0.01)


class TestG0AndG0Integral:
    """Tests for the forward variance curve g_0 and G_0."""

    def test_g0_at_zero(self):
        """g_0(0) = V0."""
        params = default_params()
        val = g0(np.array([0.0]), params.V0, params.beta, params.alpha)
        assert abs(val[0] - params.V0) < 1e-12

    def test_g0_positive(self):
        """g_0 is positive for all t >= 0 with positive parameters."""
        params = default_params()
        t = np.linspace(0, params.T, 50)
        vals = g0(t, params.V0, params.beta, params.alpha)
        assert np.all(vals > 0)

    def test_G0_is_antiderivative_of_g0(self):
        """G_0(T) should equal integral_0^T g_0(t) dt."""
        params = default_params()
        T = params.T
        from scipy.integrate import quad
        integral, _ = quad(
            lambda t: g0(np.array([t]), params.V0, params.beta, params.alpha)[0],
            0.0, T
        )
        G0_val = G0(np.array([T]), params.V0, params.beta, params.alpha)[0]
        assert abs(G0_val - integral) < 1e-8

    def test_g0_increasing(self):
        """g_0 is increasing with beta > 0."""
        params = default_params()
        t = np.linspace(0.0, params.T, 20)
        vals = g0(t, params.V0, params.beta, params.alpha)
        assert np.all(np.diff(vals) >= 0)


class TestProductIntegrationWeights:
    """Tests for the product-integration weights W[m]."""

    def test_W_zero_at_m_zero(self):
        """W[0] = 0."""
        params = default_params()
        M = 64
        h = params.T / M
        W = product_integration_weights(M, params.alpha, h)
        assert W[0] == 0.0

    def test_W_shape(self):
        """W has shape (M+1,)."""
        M = 64
        alpha = 0.527
        h = 0.25 / 64
        W = product_integration_weights(M, alpha, h)
        assert W.shape == (M + 1,)

    def test_W_positive(self):
        """All W[m] > 0 for m >= 1."""
        M = 64
        alpha = 0.527
        h = 0.25 / 64
        W = product_integration_weights(M, alpha, h)
        assert np.all(W[1:] > 0)

    def test_W_sum_equals_kernel_integral(self):
        """Sum of W[1:M+1] should equal G(alpha+1)^{-1} * (M*h)^alpha = T^alpha/Gamma(alpha+1)."""
        from scipy.special import gamma
        M = 128
        alpha = 0.527
        T = 0.25
        h = T / M
        W = product_integration_weights(M, alpha, h)
        expected = T ** alpha / gamma(alpha + 1)
        assert abs(W[1:].sum() - expected) < 1e-10

    def test_W_decreasing_for_alpha_less_one(self):
        """For alpha < 1, the weights W[m] are decreasing in m (the kernel is singular)."""
        M = 64
        alpha = 0.527
        h = 0.25 / 64
        W = product_integration_weights(M, alpha, h)
        assert np.all(np.diff(W[1:]) < 0)


class TestL1KernelError:
    """Tests for the L1 error between K_epsilon and K_alpha."""

    def test_error_decreases_with_epsilon(self):
        """||K_eps - K||_1 should be monotone decreasing as eps -> 0."""
        params = default_params()
        T = params.T
        alpha = params.alpha
        epsilons = [T / 4, T / 8, T / 16, T / 32]
        errors = [l1_kernel_error(alpha, T, eps) for eps in epsilons]
        assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))

    def test_error_positive(self):
        """L1 error is positive for any epsilon > 0."""
        params = default_params()
        err = l1_kernel_error(params.alpha, params.T, params.T / 4)
        assert err > 0.0

    def test_target_values_approx(self):
        """Paper target L1 errors for eps/T = {1/4,...,1/64} are approximately
        [0.193673, 0.146656, 0.108279, 0.078500, 0.056184]."""
        params = default_params()
        T = params.T
        alpha = params.alpha
        eps_ratios = [1/4, 1/8, 1/16, 1/32, 1/64]
        targets = [0.1937, 0.1467, 0.1083, 0.0785, 0.0562]
        for ratio, target in zip(eps_ratios, targets):
            err = l1_kernel_error(alpha, T, ratio * T)
            # Allow 10% tolerance vs paper target
            assert abs(err - target) < 0.1 * target + 0.005, \
                f"L1 error {err:.4f} far from target {target:.4f} at eps/T={ratio}"
