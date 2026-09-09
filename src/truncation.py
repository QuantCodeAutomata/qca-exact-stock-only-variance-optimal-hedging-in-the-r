"""Truncation map chi for approximate transform memory stabilisation."""
import numpy as np


def chi(z: np.ndarray) -> np.ndarray:
    """Elementwise truncation map.

    chi(z) = exp(min(Re(z), 0) + i * Im(z))

    When Re(z) <= 0: chi(z) = exp(z)  (unchanged).
    When Re(z) >  0: chi(z) = exp(0 + i*Im(z))  (magnitude clipped to 1).

    Parameters
    ----------
    z : np.ndarray (complex)
        Input array of complex numbers.

    Returns
    -------
    np.ndarray, dtype complex128
    """
    z = np.asarray(z, dtype=complex)
    re_z = z.real
    im_z = z.imag
    # Clip the real part at 0
    clipped_re = np.minimum(re_z, 0.0)
    return np.exp(clipped_re + 1j * im_z)


def chi_scalar(z: complex) -> complex:
    """Scalar version of the truncation map chi.

    Parameters
    ----------
    z : complex

    Returns
    -------
    complex
    """
    re_z = z.real
    im_z = z.imag
    return complex(np.exp(min(re_z, 0.0) + 1j * im_z))
