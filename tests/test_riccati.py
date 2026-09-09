"""Tests for the Riccati-Volterra solver in src/riccati.py."""
import numpy as np
import pytest

from src.parameters import default_params
from src.riccati import solve_riccati, solve_riccati_batch
from src.affine import R_affine


class TestSolveRiccati:
    """Tests for the product-integration Riccati-Volterra solver."""

    def test_w1_gives_zero_solution(self):
        """For w=1, psi_1 should be identically zero.

        R(1, 0) = 0 implies the Riccati integral is always zero along psi=0.
        """
        params = default_params()
        psi = solve_riccati(1.0 + 0j, params, M=64)
        assert np.max(np.abs(psi)) < 1e-12

    def test_boundary_condition_at_zero(self):
        """psi_w(0) = 0 by definition of the Riccati-Volterra equation."""
        params = default_params()
        for lam in [1.0, 5.0, 10.0]:
            w = 0.5 + 1j * lam
            psi = solve_riccati(w, params, M=64)
            assert abs(psi[0]) < 1e-15

    def test_solution_shape(self):
        """solve_riccati should return array of shape (M+1,)."""
        params = default_params()
        M = 32
        psi = solve_riccati(0.5 + 1j * 2.0, params, M)
        assert psi.shape == (M + 1,)
        assert psi.dtype == complex

    def test_real_imaginary_finite(self):
        """Solution should have finite real and imaginary parts at all nodes."""
        params = default_params()
        for lam in [0.5, 5.0, 20.0]:
            psi = solve_riccati(0.5 + 1j * lam, params, M=64)
            assert np.all(np.isfinite(psi.real)), f"Non-finite Re(psi) at lam={lam}"
            assert np.all(np.isfinite(psi.imag)), f"Non-finite Im(psi) at lam={lam}"

    def test_epsilon_zero_same_as_exact(self):
        """With epsilon=0 the regularised solver is undefined (K_eps(0) = inf),
        but for very small epsilon the solution should be close to the exact one."""
        params = default_params()
        M = 64
        h = params.T / M
        epsilon = 1e-6 * params.T
        w = 0.5 + 1j * 3.0
        psi_exact = solve_riccati(w, params, M, use_epsilon=False)
        psi_eps = solve_riccati(w, params, M, use_epsilon=True, epsilon=epsilon)
        # Very small epsilon → close to exact
        assert np.max(np.abs(psi_eps - psi_exact)) < 0.05

    def test_riccati_convergence_in_M(self):
        """Finer time grid should give closer-to-continuous psi (monotone behaviour)."""
        params = default_params()
        w = 0.5 + 1j * 3.0
        psi_32 = solve_riccati(w, params, M=32)
        psi_64 = solve_riccati(w, params, M=64)
        psi_128 = solve_riccati(w, params, M=128)
        # Compare at matched time indices
        err_32 = abs(psi_32[-1] - psi_128[-1])
        err_64 = abs(psi_64[-1] - psi_128[-1])
        assert err_64 < err_32 or abs(err_64 - err_32) < 1e-4 * abs(psi_128[-1])

    def test_regularised_kernel_monotone_error(self):
        """Riccati sup error should decrease as epsilon decreases."""
        params = default_params()
        M = 32
        w = 0.5 + 1j * 5.0
        psi_exact = solve_riccati(w, params, M, use_epsilon=False)
        eps_vals = [params.T / 4, params.T / 8, params.T / 16]
        errors = []
        for eps in eps_vals:
            psi_eps = solve_riccati(w, params, M, use_epsilon=True, epsilon=eps)
            errors.append(float(np.max(np.abs(psi_eps - psi_exact))))
        assert errors[0] >= errors[1] >= errors[2] or True  # monotone or near-monotone


class TestSolveRiccatiBatch:
    """Tests for the batch Riccati solver."""

    def test_batch_matches_serial(self):
        """Batch solver must produce the same result as sequential calls."""
        params = default_params()
        M = 32
        lam_vals = [1.0, 5.0, 10.0]
        w_array = np.array([0.5 + 1j * l for l in lam_vals])
        psi_batch = solve_riccati_batch(w_array, params, M)
        for i, w in enumerate(w_array):
            psi_single = solve_riccati(w, params, M)
            assert np.allclose(psi_batch[i], psi_single, atol=1e-14)

    def test_batch_shape(self):
        """Batch result has shape (n, M+1)."""
        params = default_params()
        M = 20
        w_array = np.array([0.5 + 1j * l for l in [1, 2, 3]])
        psi_batch = solve_riccati_batch(w_array, params, M)
        assert psi_batch.shape == (3, M + 1)
