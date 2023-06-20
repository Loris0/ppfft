import numpy as np

from ..tools.pad import pad
from ..tools.new_fft import new_fft


def frac_fft(c: np.ndarray, a: int, b: int, axis=-1):
    """
    frac FFT with alpha = a / b
    """
    shape = np.shape(c)
    n = shape[axis]
    q_n, r_n = divmod(n, 2)
    m = b * n
    q_m, r_m = divmod(m, 2)

    aux_shape = list(shape)
    aux_shape[axis] = m
    aux_shape = tuple(aux_shape)
    aux = pad(c, aux_shape)

    indices = (a * np.arange(-q_n, q_n + r_n)) % m  # between 0 and m - 1
    indices[indices >= q_m + r_m] -= m  # between -m//2 and -1
    indices += q_m

    return np.take(new_fft(aux, axis), indices, axis=axis)


def adj_frac_fft(c: np.ndarray, a: int, b: int, axis=-1) -> np.ndarray:
    """
    Adjoint operator of ``frac_fft``.
    """
    return frac_fft(c, -a, b, axis)
