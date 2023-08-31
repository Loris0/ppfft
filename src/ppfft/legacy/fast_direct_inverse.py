"""
Module to compute an image from I_D.
"""

import numpy as np

from ..tools.pad import pad, adj_pad
from ..tools.new_fft import adj_new_fft
from ..toeplitz.inverse_toeplitz import InverseToeplitz
from .fast_onion_peeling import fast_onion_peeling, precompute_onion_peeling


def adj_F_D(y: np.ndarray) -> np.ndarray:
    """Adjoint of F_D transformation.

    Parameters
    ----------
    y : np.ndarray
        Input array.

    Returns
    -------
    out : np.ndarray
        Adjoint of F_D applied to y.
    """
    n = len(y) - 1
    m = 2 * n + 1
    aux = np.zeros(shape=(m,), dtype=complex)
    aux[::2] = y

    if n % 2 == 0:
        return adj_pad(adj_new_fft(aux), original_shape=(n,))

    else:
        return adj_pad(adj_new_fft(np.roll(aux, -1)), original_shape=(n,))


def compute_col(n: int) -> np.ndarray:
    """Computes first columns of Toeplitz matrix used to invert F_D.

    Parameters
    ----------
    n : int
        Size of image to reconstruct.

    Returns
    -------
    out : np.ndarray
        First column of T.
    """
    m = 2 * n + 1
    one = np.ones(n + 1)
    pad_one = pad(one, (m,))

    q_m, r_m = divmod(m, 2)

    indices = (2 * np.arange(0, n)) % m
    indices[indices >= q_m + r_m] -= m
    indices += q_m

    return np.take(adj_new_fft(pad_one), indices)


def precompute_inverse_Id(n: int) -> InverseToeplitz:
    """Builds the Toeplitz inverse needed to invert F_D.

    Parameters
    ----------
    n : int
        Size of image to reconstruct.

    Returns
    -------
    out : InverseToeplitz
        Inverse of T
    """
    c = compute_col(n)

    return InverseToeplitz(col=c)


def fast_inverse_Id(Id: np.ndarray, toeplitz: InverseToeplitz) -> np.ndarray:
    """Fast inversion of Id.

    Parameters
    ----------
    Id : np.ndarray
        Id array.
    toeplitz : InverseToeplitz
        Inverse of Toeplitz matrix corresponding to F_D.

    Returns
    -------
    out : np.ndarray
        Reconstruction of original image.
    """
    n = Id.shape[0] - 1

    A = np.zeros(shape=(n, n + 1), dtype=complex)
    res = np.zeros(shape=(n, n), dtype=complex)

    for l, col in enumerate(Id.T):
        A[:, l] = toeplitz.apply_inverse(adj_F_D(col))

    for u, row in enumerate(A):
        res[u, :] = toeplitz.apply_inverse(adj_F_D(row))

    return res


def precompute_all(n: int) -> tuple:
    """Precomputes all InverseToeplitz and NUFFT.

    Parameters
    ----------
    n : int
        Size of image to reconstruct.

    Returns
    -------
    toeplitz_list : list[InverseToeplitz]
        List of InverseToeplitz for onion_peeling.
    nufft_list : list[NUFFT]
        List of NUFFT for onion-peeling.
    toeplitz : InverseToeplitz
        Inverse of Toeplitz matrix corresponding to F_D.
    """
    toeplitz_list, nufft_list = precompute_onion_peeling(n)
    toeplitz = precompute_inverse_Id(n)
    return toeplitz_list, nufft_list, toeplitz


def fast_direct_inversion(
    hori_ppfft: np.ndarray, vert_ppfft: np.ndarray, precomputations: tuple
) -> np.ndarray:
    """Fast Direct Inversion using onion-peeling.

    Parameters
    ----------
    hori_ppfft : np.ndarray
        Horizontal ppfft.
    vert_ppfft : np.ndarray
        Vertical ppfft.
    precomputations : tuple
        Precomputations, output of `precompute_all(n)`.

    Returns
    -------
    sol : np.ndarray
        Reconstructed image.
    """
    toeplitz_list, nufft_list, toeplitz = precomputations
    Id = fast_onion_peeling(hori_ppfft, vert_ppfft, toeplitz_list, nufft_list)
    sol = fast_inverse_Id(Id, toeplitz)
    return sol
