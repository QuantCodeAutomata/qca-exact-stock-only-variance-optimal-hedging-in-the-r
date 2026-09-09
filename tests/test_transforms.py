"""Tests for conditional transforms H_0(w, T) in src/transforms.py."""
import numpy as np
import pytest

from src.parameters import default_params
from src.riccati import solve_riccati
from src.transforms import compute_H0


class TestComputeH0:
    """Tests for the moment generating function H_0(w, T) = E[exp(w X_T)]."""

    def test_H0_at_w_zero_equals_one(self):
        """H_0(w=0, T) = E[exp(0)] = 1."""
        params = default_params()
        M = 64
        psi_w0 = solve_riccati(0.0 + 0j, params, M)
        H0 = compute_H0(0.0 + 0j, psi_w0, params, M)
        assert abs(H0 - 1.0) < 1e-10

    def test_H0_at_w_one_approx_one(self):
        """H_0(w=1, T) = E[S_T/S_0] ≈ 1 (martingale property).

        For w=1 and psi_1 = 0, H_0(1,T) = exp(integral_0^T R(1,0)*g_0(s) ds) = exp(0) = 1.
        """
        params = default_params()
        M = 64
        psi_w1 = solve_riccati(1.0 + 0j, params, M)
        H0 = compute_H0(1.0 + 0j, psi_w1, params, M)
        assert abs(H0 - 1.0) < 1e-10

    def test_H0_is_complex(self):
        """H_0 is complex-valued for contour points w = 0.5 + i*lambda."""
        params = default_params()
        M = 64
        w = 0.5 + 1j * 5.0
        psi_w = solve_riccati(w, params, M)
        H0 = compute_H0(w, psi_w, params, M)
        assert isinstance(H0, complex)

    def test_H0_finite(self):
        """H_0 should be finite for all tested contour modes."""
        params = default_params()
        M = 32
        for lam in [0.5, 2.0, 5.0, 10.0, 20.0]:
            w = 0.5 + 1j * lam
            psi_w = solve_riccati(w, params, M)
            H0 = compute_H0(w, psi_w, params, M)
            assert np.isfinite(abs(H0)), f"H0 not finite at lam={lam}"

    def test_H0_magnitude_bounded(self):
        """H_0 magnitude should remain moderate (no blow-up) at damping Re(w)=0.5."""
        params = default_params()
        M = 32
        for lam in [1.0, 5.0, 15.0]:
            w = 0.5 + 1j * lam
            psi_w = solve_riccati(w, params, M)
            H0 = compute_H0(w, psi_w, params, M)
            assert abs(H0) < 1e6, f"|H0| too large at lam={lam}"

    def test_H0_conjugate_symmetry(self):
        """H_0(w̄) = H_0(w)* (conjugate symmetry)."""
        params = default_params()
        M = 64
        lam = 5.0
        w = 0.5 + 1j * lam
        w_conj = 0.5 - 1j * lam

        psi_w = solve_riccati(w, params, M)
        psi_w_conj = solve_riccati(w_conj, params, M)

        H0_w = compute_H0(w, psi_w, params, M)
        H0_w_conj = compute_H0(w_conj, psi_w_conj, params, M)

        assert abs(H0_w - H0_w_conj.conjugate()) < 1e-10

    def test_H0_M_invariance(self):
        """H_0 should be approximately the same for M=64 and M=256."""
        params = default_params()
        w = 0.5 + 1j * 3.0
        psi_64 = solve_riccati(w, params, M=64)
        psi_256 = solve_riccati(w, params, M=256)
        H0_64 = compute_H0(w, psi_64, params, M=64)
        H0_256 = compute_H0(w, psi_256, params, M=256)
        # Both should agree within 5%
        assert abs(H0_64 - H0_256) / (abs(H0_256) + 1e-15) < 0.05
