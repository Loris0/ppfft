"""
This module provides the implementation of:
- the forward horizontal ppfft
- the forward vertical ppfft
- the forward ppfft
- the adjoint ppfft
"""

import numpy as np

from ..tools.grids import domain
from ..tools.pad import pad, adj_pad
from ..tools.new_fft import new_fft, adj_new_fft
from ..tools.frac_fft import frac_fft, adj_frac_fft


def ppfft_horizontal(a: np.ndarray) -> np.ndarray:
    n, _ = a.shape

    res = np.empty((n + 1, n + 1), dtype=complex)

    # 1D FFT of each zero-padded line. Shape = (n, m)
    fft_col = new_fft(pad(a, new_shape=(n, n + 1)), axis=-1)

    # Frac FFT on each col
    for k, col in zip(domain(n + 1), fft_col.T):
        res[k + n // 2, :] = frac_fft(col, beta=-2 * k / (n * (n + 1)), m=n + 1)
    return res


def ppfft_vertical(a: np.ndarray) -> np.ndarray:
    """Pseudo-Polar Fast Fourier Transform on the basically vertical lines.

    Parameters
    ----------
    a : np.ndarray
        Input array of shape (n, n).

    Returns
    -------
    y : np.ndarray
        Ouput array of shape (n+1, 2n+1).

    See Also
    --------
    ppfft_horizontal : PPFFT on the basically horizontal lines.
    """
    return ppfft_horizontal(a.T)


def ppfft(a: np.ndarray):
    """Pseudo-Polar Fast Fourier Transform.

    Parameters
    ----------
    a : np.ndarray
        Input array of shape (n, n).

    Returns
    -------
    hori : np.ndarray
        Horizontal ppfft of shape (n+1, 2n+1)
    vert : np.ndarray
        Vertical ppfft of shape (n+1, 2n+1)
    """
    return ppfft_horizontal(a), ppfft_vertical(a)


def adj_ppfft(hori_ppfft: np.ndarray, vert_ppfft: np.ndarray) -> np.ndarray:
    """Adjoint of `ppfft`.

    Parameters
    ----------
    hori_ppfft : np.ndarray
        Horizontal PPFFT.
    vert_ppfft : np.ndarray
        Vertical PPFFT.

    Returns
    -------
    out : np.ndarray
        Adjoint PPFFT applied to input.
    """
    n, m = hori_ppfft.shape[0] - 1, hori_ppfft.shape[1]
    hori_aux = np.empty(shape=(n, m), dtype=complex)
    vert_aux = np.empty(shape=(n, m), dtype=complex)

    for k, (col_h, col_v) in enumerate(zip(hori_ppfft.T, vert_ppfft.T)):
        alpha = -2 * (k - n) / (n * m)
        hori_aux[:, k] = adj_frac_fft(col_h, alpha, n=n)
        vert_aux[:, k] = adj_frac_fft(col_v, alpha, n=n)

    hori_aux = adj_pad(adj_new_fft(hori_aux), (n, n))
    vert_aux = adj_pad(adj_new_fft(vert_aux), (n, n))

    return hori_aux + vert_aux.T
