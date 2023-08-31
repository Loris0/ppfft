"""
This module provides the implementation of the 
Fractional Fast Fourier Transform and its adjoint.
"""

import numpy as np
from scipy.linalg import matmul_toeplitz

from ppfft.tools.grids import domain


def frac_fft(x: np.ndarray, beta: float, m=None) -> np.ndarray:
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


def adj_frac_fft(y, beta, n=None):
    return frac_fft(y, -beta, m=n)


def new_fast_frac_fft(x: np.ndarray, beta: float, m=None) -> np.ndarray:
    n = len(x)
    w = np.exp(-2j * np.pi * beta)
    dom_in = np.arange(n)
    # Output size is the same as input.
    if m is None:
        dom_out = domain(n)
    # Output size given by m.
    else:
        dom_out = domain(m)
    w_powers_in = w ** (0.5 * dom_in**2)
    w_powers_out = w ** (0.5 * dom_out**2)
    c = w ** (-0.5 * (dom_out - dom_in[0]) ** 2)
    r = w ** (-0.5 * (dom_in - dom_out[0]) ** 2)
    return w_powers_out * matmul_toeplitz((c, r), w_powers_in * x)
