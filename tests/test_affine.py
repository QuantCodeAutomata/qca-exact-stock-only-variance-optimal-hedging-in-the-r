"""Tests for affine coefficient functions in src/affine.py."""
import pytest
import numpy as np
from scipy.integrate import quad

from src.parameters import default_params
from src.affine import (
    compute_D,
    jump_integral_R,
    R_affine,
    J_jump,
    B_coefficient,
    compute_A_density,
    Gamma_density,
)


# ---------------------------------------------------------------------------
# compute_D
# ---------------------------------------------------------------------------

class TestComputeD:
    """Tests for the D = 1 + integral q(z)^2 nu(dz) coefficient."""

    def test_formula_versus_numerical(self):
        """Closed-form D must match numerical integration."""
        Lambda = 0.276
        D_formula = compute_D(Lambda)

        def integrand(z):
            return (np.exp(-Lambda * z) - 1.0) ** 2 * np.exp(-z)

        D_num, _ = quad(integrand, 0.0, 60.0, limit=400)
        D_num += 1.0

        assert abs(D_formula - D_num) < 1e-8

    def test_D_greater_than_one(self):
        """D must be > 1 since it adds a non-negative quantity to 1."""
        assert compute_D(0.276) > 1.0

    def test_D_calibrated_value(self):
        """With Lambda=0.276, D should be close to documented value."""
        D = compute_D(0.276)
        # Closed form: 2 + 1/(1+2*0.276) - 2/(1+0.276)
        expected = 2.0 + 1.0 / 1.552 - 2.0 / 1.276
        assert abs(D - expected) < 1e-12

    def test_D_monotone_in_Lambda(self):
        """D should be an increasing function of Lambda for Lambda > 0.

        D(Lambda) = 2 + 1/(1+2L) - 2/(1+L); D'(L) > 0 for L > 0 because
        2/(1+L)^2 > 2/(1+2L)^2 iff 1+2L > 1+L (always true).
        """
        Lambdas = [0.1, 0.2, 0.3, 0.5, 1.0]
        Ds = [compute_D(L) for L in Lambdas]
        assert all(Ds[i] < Ds[i + 1] for i in range(len(Ds) - 1))

    def test_D_at_Lambda_zero_limit(self):
        """As Lambda -> 0, q(z) -> 0, so D -> 1."""
        D_small = compute_D(0.001)
        assert D_small > 1.0
        assert D_small < 1.01


# ---------------------------------------------------------------------------
# jump_integral_R
# ---------------------------------------------------------------------------

class TestJumpIntegralR:
    """Tests for the exponential-mark jump integral in R(u,v)."""

    def test_real_arguments_versus_numerical(self):
        """Closed form must match numerical quad for real (u,v) with Re(a)<1."""
        u, v, Lambda = 0.5, -0.3, 0.276
        analytical = jump_integral_R(u, v, Lambda)

        def integrand(z):
            val = np.exp((v - Lambda * u) * z) - u * (np.exp(-Lambda * z) - 1.0) - 1.0 - v * z
            return val * np.exp(-z)

        numerical, _ = quad(integrand, 0.0, 80.0, limit=400)
        assert abs(analytical - numerical) < 1e-8

    def test_complex_arguments_versus_numerical(self):
        """Should match numerical integration for complex (u,v)."""
        u = 0.5 + 1j * 5.0
        v = -0.2 + 1j * 0.3
        Lambda = 0.276
        analytical = jump_integral_R(u, v, Lambda)

        def re_integrand(z):
            val = np.exp((v - Lambda * u) * z) - u * (np.exp(-Lambda * z) - 1.0) - 1.0 - v * z
            return (val * np.exp(-z)).real

        def im_integrand(z):
            val = np.exp((v - Lambda * u) * z) - u * (np.exp(-Lambda * z) - 1.0) - 1.0 - v * z
            return (val * np.exp(-z)).imag

        re_num, _ = quad(re_integrand, 0.0, 60.0, limit=400)
        im_num, _ = quad(im_integrand, 0.0, 60.0, limit=400)
        assert abs(analytical.real - re_num) < 1e-6
        assert abs(analytical.imag - im_num) < 1e-6

    def test_vanishes_at_u_zero_v_zero(self):
        """Jump integral at u=0, v=0: integrand = exp(0) - 0 - 1 - 0 = 0."""
        val = jump_integral_R(0.0, 0.0, 0.276)
        assert abs(val) < 1e-12


# ---------------------------------------------------------------------------
# J_jump
# ---------------------------------------------------------------------------

class TestJJump:
    """Tests for the cross-jump term J(a, Lambda)."""

    def test_versus_numerical(self):
        """J(a) should match numerical integration over (0, inf)."""
        a = -0.5 + 0.3j
        Lambda = 0.276
        J_analytical = J_jump(a, Lambda)

        def re_integrand(z):
            return ((np.exp(a * z) - 1.0) * (np.exp(-Lambda * z) - 1.0) * np.exp(-z)).real

        def im_integrand(z):
            return ((np.exp(a * z) - 1.0) * (np.exp(-Lambda * z) - 1.0) * np.exp(-z)).imag

        re_num, _ = quad(re_integrand, 0.0, 60.0, limit=400)
        im_num, _ = quad(im_integrand, 0.0, 60.0, limit=400)

        assert abs(J_analytical.real - re_num) < 1e-8
        assert abs(J_analytical.imag - im_num) < 1e-8

    def test_J_at_a_zero(self):
        """When a=0: exp(0)-1 = 0, so J(0,Lambda) = 0."""
        val = J_jump(0.0 + 0j, 0.276)
        assert abs(val) < 1e-12


# ---------------------------------------------------------------------------
# R_affine
# ---------------------------------------------------------------------------

class TestRAffine:
    """Tests for the Riccati drift R(u,v)."""

    def test_R_at_zero_u_v(self):
        """R(0, 0) = 0 (no free term in the affine structure)."""
        params = default_params()
        R = R_affine(0.0 + 0j, 0.0 + 0j, params.b, params.rho, params.c, params.Lambda)
        assert abs(R) < 1e-12

    def test_R_at_w1_psi0(self):
        """R(1, 0) must equal 0 (which ensures psi_1 = 0 identically).

        R(1,0) = 1/2*(1-1) + b*0 + c/2*0 + J_integral(u=1,v=0)
        J_integral at u=1,v=0:
          a = 0 - Lambda*1 = -Lambda
          = 1/(1-(-Lambda)) + 1*Lambda/(1+Lambda) - 1 - 0
          = 1/(1+Lambda) + Lambda/(1+Lambda) - 1
          = (1+Lambda)/(1+Lambda) - 1 = 0
        """
        params = default_params()
        R = R_affine(1.0 + 0j, 0.0 + 0j, params.b, params.rho, params.c, params.Lambda)
        assert abs(R) < 1e-12

    def test_R_is_analytic_in_v(self):
        """R should be a smooth function of v for fixed u on the contour."""
        params = default_params()
        u = 0.5 + 1j * 3.0
        vs = [-0.5, -0.3, -0.1, 0.0, 0.1]
        R_vals = [R_affine(u, v, params.b, params.rho, params.c, params.Lambda) for v in vs]
        # Just check they are finite complex numbers
        for R_v in R_vals:
            assert np.isfinite(R_v.real)
            assert np.isfinite(R_v.imag)


# ---------------------------------------------------------------------------
# B_coefficient and Gamma_density
# ---------------------------------------------------------------------------

class TestBAndGamma:
    """Tests for the hedge numerator B and residual Gamma."""

    def test_B_at_w1_psi0_equals_D(self):
        """B(1, 0) must equal D (so beta^1 = B/D = 1)."""
        params = default_params()
        B = B_coefficient(1.0 + 0j, 0.0 + 0j, params.rho, params.c, params.Lambda)
        D = compute_D(params.Lambda)
        assert abs(B - D) < 1e-10

    def test_Gamma_nonneg_on_contour(self):
        """Residual density Gamma must be non-negative on the Lewis contour."""
        params = default_params()
        for lam in [0.5, 2.0, 5.0, 10.0, 20.0]:
            w = 0.5 + 1j * lam
            # psi = 0 at t=T (boundary condition)
            Gamma = Gamma_density(w, 0.0 + 0j, params.b, params.rho, params.c, params.Lambda)
            assert Gamma >= -1e-10, f"Gamma < 0 at lambda={lam}: {Gamma}"

    def test_Gamma_zero_at_w1(self):
        """For w=1, psi_1=0: Gamma(1,0) must be 0 (exact replication)."""
        params = default_params()
        Gamma = Gamma_density(1.0 + 0j, 0.0 + 0j, params.b, params.rho, params.c, params.Lambda)
        assert abs(Gamma) < 1e-10
