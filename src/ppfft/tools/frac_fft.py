"""
This module provides the implementation of the 
Fractional Fast Fourier Transform and its adjoint.
"""

import numpy as np
from scipy.linalg import matmul_toeplitz

from ppfft.tools.grids import domain


def fast_frac_fft(x: np.ndarray, beta: float, m=None) -> np.ndarray:
    """Fast Fractional Fourier Transform.

    Parameters
    ----------
    x : np.ndarray
        Input array, one dimensional.
    beta : float
        Factor in the complex exponential.
    m : int | None, optional
        Outpout size, by default the same as the input.

    Returns
    -------
    out : np.ndarray
        Fractional Fourier Transform of `x`.
    """
    n = len(x)
    w = np.exp(-2j * np.pi * beta)
    # Output size is the same as input.
    if m is None:
        dom = domain(n)
        w_powers = w ** (0.5 * dom**2)  # this may not be optimal
        c = w ** (-0.5 * (dom - dom[0]) ** 2)
        return w_powers * matmul_toeplitz((c, c), w_powers * x)
    # Output size given by m.
    else:
        dom_n = domain(n)
        dom_m = domain(m)
        w_powers_n = w ** (0.5 * dom_n**2)  # this may not be optimal
        w_powers_m = w ** (0.5 * dom_m**2)  # this may not be optimal
        c = w ** (-0.5 * (dom_m - dom_n[0]) ** 2)
        r = w ** (-0.5 * (dom_n - dom_m[0]) ** 2)
        return w_powers_m * matmul_toeplitz((c, r), w_powers_n * x)


def frac_fft_for_ppfft(x: np.ndarray, alpha: float) -> np.ndarray:
    """Variant of FracFFT used in `ppfft`.

    Parameters
    ----------
    x : np.ndarray
        Input array, one dimensional.
    alpha : float
        Factor in the exponential (will be divided by input size).

    Returns
    -------
    out : np.ndarray
        FracFFT of input, at (input size) + 1 points.
    """
    n = len(x)
    return fast_frac_fft(x, alpha / n, m=n + 1)


def adj_frac_fft_for_ppfft(y: np.ndarray, alpha: float) -> np.ndarray:
    """Adjoint of `frac_fft_for_ppfft`.

    Parameters
    ----------
    y : np.ndarray
        Input array, one dimensional.
    alpha : float
        Factor in the exponential (will be divided by (input size) - 1).

    Returns
    -------
    out : np.ndarray
        Adjoint FracFFT of input, of size (input size) - 1.
    """
    n = len(y) - 1
    return fast_frac_fft(y, -alpha / n, m=n)
