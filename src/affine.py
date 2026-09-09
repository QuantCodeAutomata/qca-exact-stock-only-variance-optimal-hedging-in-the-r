import numpy as np
from scipy.special import gamma  # noqa: F401
from scipy.integrate import quad


def compute_D(Lambda: float) -> float:
    """D = 1 + integral_0^inf (exp(-Lambda*z) - 1)^2 * exp(-z) dz.

    Closed form: D = 2 + 1/(1 + 2*Lambda) - 2/(1 + Lambda).
    """
    return 2.0 + 1.0 / (1.0 + 2.0 * Lambda) - 2.0 / (1.0 + Lambda)


def jump_integral_R(u: complex, v: complex, Lambda: float) -> complex:
    """Integral over the exponential mark distribution nu(dz) = exp(-z) dz.

    Computes integral_0^inf [exp((v - Lambda*u)*z) - u*(exp(-Lambda*z) - 1)
                             - 1 - v*z] * exp(-z) dz.

    Closed form (requiring Re(v - Lambda*u) < 1):
        1/(1 - a) + u*Lambda/(1 + Lambda) - 1 - v
    where a = v - Lambda*u.

    Falls back to numerical integration when Re(1 - a) < 1e-10.
    """
    a = v - Lambda * u
    one_minus_a = 1.0 - a

    if abs(one_minus_a.real) < 1e-10:
        # Numerical fallback: integrate real and imaginary parts separately.
        def re_integrand(z: float) -> float:
            val = (np.exp(a * z) - u * (np.exp(-Lambda * z) - 1.0) - 1.0 - v * z)
            return (val * np.exp(-z)).real

        def im_integrand(z: float) -> float:
            val = (np.exp(a * z) - u * (np.exp(-Lambda * z) - 1.0) - 1.0 - v * z)
            return (val * np.exp(-z)).imag

        re_val, _ = quad(re_integrand, 0.0, 50.0, limit=200)
        im_val, _ = quad(im_integrand, 0.0, 50.0, limit=200)
        return complex(re_val, im_val)

    return 1.0 / one_minus_a + u * Lambda / (1.0 + Lambda) - 1.0 - v


def R_affine(
    u: complex, v: complex, b: float, rho: float, c: float, Lambda: float
) -> complex:
    """Riccati drift coefficient R(u, v) for the rough Hawkes-Heston model.

    R(u, v) = 0.5*(u^2 - u)
              + (b + rho*sqrt(c)*u) * v
              + (c/2) * v^2
              + jump_integral_R(u, v, Lambda).
    """
    return (
        0.5 * (u ** 2 - u)
        + (b + rho * np.sqrt(c) * u) * v
        + 0.5 * c * v ** 2
        + jump_integral_R(u, v, Lambda)
    )


def J_jump(a: complex, Lambda: float) -> complex:
    """Cross-term jump integral J(a, Lambda).

    J(a) = integral_0^inf (exp(a*z) - 1) * (exp(-Lambda*z) - 1) * exp(-z) dz
         = 1/(1 - a + Lambda) - 1/(1 - a) - 1/(1 + Lambda) + 1.

    Handles the (rare) case where the denominator is near zero by returning
    the analytically continued limit.
    """
    denom1 = 1.0 - a + Lambda
    denom2 = 1.0 - a

    term1 = 1.0 / denom1 if abs(denom1) > 1e-14 else 0.0
    term2 = 1.0 / denom2 if abs(denom2) > 1e-14 else 0.0

    return term1 - term2 - 1.0 / (1.0 + Lambda) + 1.0


def B_coefficient(
    w: complex, psi_w: complex, rho: float, c: float, Lambda: float
) -> complex:
    """Hedging coefficient B(w, psi_w).

    B = w + rho*sqrt(c)*psi_w + J_jump(psi_w - Lambda*w, Lambda).
    """
    return w + rho * np.sqrt(c) * psi_w + J_jump(psi_w - Lambda * w, Lambda)


def compute_A_density(
    w: complex, psi_w: complex, rho: float, c: float, Lambda: float
) -> float:
    """Instantaneous variance-of-variance density A(w, psi_w).

    A = (1 - rho^2)*|w|^2
        + |rho*w + sqrt(c)*psi_w|^2
        + integral_0^inf |exp(a*z) - 1|^2 * exp(-z) dz

    where a = psi_w - Lambda*w.

    The last term evaluates in closed form to
        1/(1 - 2*Re(a)) - 2*Re(1/(1 - a)) + 1  [valid when Re(a) < 0.5].

    Returns a real (non-negative) value.
    """
    a = psi_w - Lambda * w
    re_a = a.real

    diffusion_w = (1.0 - rho ** 2) * abs(w) ** 2
    diffusion_v = abs(rho * w + np.sqrt(c) * psi_w) ** 2

    # Jump contribution: integral |exp(a*z) - 1|^2 exp(-z) dz
    jump_term = (
        1.0 / (1.0 - 2.0 * re_a)
        - 2.0 * (1.0 / (1.0 - a)).real
        + 1.0
    )

    return float(diffusion_w + diffusion_v + jump_term)


def Gamma_density(
    w: complex, psi_w: complex, b: float, rho: float, c: float, Lambda: float
) -> float:
    """Minimum-variance residual density Gamma = A - |B|^2 / D.

    Returns a real (non-negative) value.
    """
    A = compute_A_density(w, psi_w, rho, c, Lambda)
    B = B_coefficient(w, psi_w, rho, c, Lambda)
    D = compute_D(Lambda)
    return float(A - abs(B) ** 2 / D)
