"""
This module provides implementations of the alternative definition of the FFT.
It includes forward transforms, inverses and adjoint, in 1D and 2D.
"""

import numpy as np


def new_fft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Alternative definition of the 1D FFT.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis over which to compute the FFT, by default -1.

    Returns
    -------
    out : np.ndarray
        1D FFT.
    """
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
    )


def adj_new_fft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Adjoint of `new_fft`.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis over which to compute the FFT, by default -1.

    Returns
    -------
    out : np.ndarray
        Adjoint of the 1D FFT.
    """
    return new_ifft(x, axis) * np.shape(x)[axis]


def new_ifft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Inverse of `new_fft`.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis over which to compute the iFFT, by default -1.

    Returns
    -------
    out : np.ndarray
        Inverse of the 1D FFT.
    """
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
    )


def adj_new_ifft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Adjoint of `new_ifft`.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis over which to compute the FFT, by default -1.

    Returns
    -------
    out : np.ndarray
        Adjoint of the 1D iFFT.
    """
    return new_fft(x, axis) / np.shape(x)[axis]


def new_fft2(a: np.ndarray) -> np.ndarray:
    """Alternative definition of the 2D FFT.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    out : np.ndarray
        2D FFT.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))


def adj_new_fft2(a: np.ndarray) -> np.ndarray:
    """Adjoint of `new_fft2`.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    out : np.ndarray
        Adjoint of the 2D FFT.
    """
    return new_ifft2(a) * np.size(a)


def new_ifft2(a: np.ndarray) -> np.ndarray:
    """Inverse of `new_fft2`.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    out : np.ndarray
        Inverse of the 2D FFT.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))


def adj_new_ifft2(a: np.ndarray) -> np.ndarray:
    """Adjoint of `new_ifft2`.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    out : np.ndarray
        Adjoint of the 2D iFFT.
    """
    return new_fft2(a) / np.size(a)
