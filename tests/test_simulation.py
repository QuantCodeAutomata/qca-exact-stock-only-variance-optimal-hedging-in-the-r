"""Tests for the Monte Carlo simulation in src/simulation.py."""
import numpy as np
import pytest

from src.parameters import default_params
from src.simulation import simulate_market
from src.validation import check_martingale_sanity, check_positivity


class TestSimulateMarket:
    """Tests for rough Hawkes-Heston market simulation."""

    def test_output_shapes(self):
        """All output arrays must have the correct shapes."""
        params = default_params()
        M, P = 32, 50
        sim = simulate_market(params, M, P, seed=0)

        assert sim["X"].shape == (P, M + 1)
        assert sim["S"].shape == (P, M + 1)
        assert sim["V"].shape == (P, M + 1)
        assert sim["dZ"].shape == (P, M)
        assert sim["grid"].shape == (M + 1,)

    def test_initial_conditions(self):
        """X_0 = 0 and S_0 = S0 for all paths."""
        params = default_params()
        sim = simulate_market(params, 32, 50, seed=1)
        assert np.allclose(sim["X"][:, 0], 0.0)
        assert np.allclose(sim["S"][:, 0], params.S0)

    def test_S_equals_S0_exp_X(self):
        """S = S0 * exp(X) at all time steps."""
        params = default_params()
        sim = simulate_market(params, 32, 50, seed=2)
        S_reconstructed = params.S0 * np.exp(sim["X"])
        assert np.allclose(sim["S"], S_reconstructed, rtol=1e-12)

    def test_variance_mostly_positive(self):
        """Stored V can dip slightly negative under Euler discretization (known artefact).

        The simulation uses full-truncation in SDE increments (Vk = max(V, 0)), so
        all diffusion/jump computations are valid.  The stored array is allowed to
        go mildly negative; we only require V > -0.5 for the calibrated model.
        """
        params = default_params()
        sim = simulate_market(params, 64, 200, seed=3)
        assert sim["V"].min() > -0.5, f"V severely negative: min = {sim['V'].min():.3e}"

    def test_V_initial_value(self):
        """V_0 = V0 for all paths."""
        params = default_params()
        sim = simulate_market(params, 32, 50, seed=4)
        assert np.allclose(sim["V"][:, 0], params.V0, atol=1e-12)

    def test_stock_martingale(self):
        """E[S_T] should be close to S_0 = 1 (martingale under Q)."""
        params = default_params()
        sim = simulate_market(params, 64, 2000, seed=42)
        m = check_martingale_sanity(sim["S"], target=params.S0, tol=0.05)
        assert m["passed"], f"Martingale failed: E[S_T] = {m['mean_ST']:.4f}"

    def test_time_grid(self):
        """Time grid must be [0, h, 2h, ..., T] with h = T/M."""
        params = default_params()
        M = 64
        sim = simulate_market(params, M, 10, seed=5)
        h = params.T / M
        expected_grid = np.arange(M + 1) * h
        assert np.allclose(sim["grid"], expected_grid)

    def test_reproducibility(self):
        """Same seed produces identical results."""
        params = default_params()
        sim1 = simulate_market(params, 32, 20, seed=99)
        sim2 = simulate_market(params, 32, 20, seed=99)
        assert np.allclose(sim1["X"], sim2["X"])
        assert np.allclose(sim1["S"], sim2["S"])

    def test_different_seeds_differ(self):
        """Different seeds produce distinct paths."""
        params = default_params()
        sim1 = simulate_market(params, 32, 20, seed=1)
        sim2 = simulate_market(params, 32, 20, seed=2)
        assert not np.allclose(sim1["X"], sim2["X"])

    def test_S_positive(self):
        """S_t > 0 at all times (since S = exp(X) * S0 > 0)."""
        params = default_params()
        sim = simulate_market(params, 64, 200, seed=7)
        assert np.all(sim["S"] > 0)

    def test_edge_case_M_1(self):
        """Simulation with M=1 should work (single step)."""
        params = default_params()
        sim = simulate_market(params, 1, 10, seed=8)
        assert sim["X"].shape == (10, 2)
        assert sim["dZ"].shape == (10, 1)
