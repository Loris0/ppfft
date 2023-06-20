import numpy as np


def new_fft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Alternative definition of the 1D fft.
    """
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
    )


def adj_new_fft(x: np.ndarray, axis: int = -1):
    """
    Adjoint operator of ``new_fft``.
    """
    return new_ifft(x, axis) * np.shape(x)[axis]


def new_ifft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Alternative definition of the 1D ifft.
    """
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
    )


def adj_new_ifft(x: np.ndarray, axis: int = -1):
    """
    Adjoint operator of ``new_ifft``.
    """
    return new_fft(x, axis) / np.shape(x)[axis]


def new_fft2(a: np.ndarray) -> np.ndarray:
    """
    Alternative definition of the 2D fft.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))


def adj_new_fft2(a):
    """
    Adjoint operator of ``new_fft2``.
    """
    return new_ifft2(a) * np.size(a)


def new_ifft2(a: np.ndarray) -> np.ndarray:
    """
    Alternative definition of the 2D ifft.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))


def adj_new_ifft2(a):
    """
    Adjoint operator of ``new_ifft2``.
    """
    return new_ifft2(a) / np.size(a)
