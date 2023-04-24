import numpy as np

from pad import pad
from new_fft import new_fft


def frac_fft(c: np.ndarray, a: int, b: int) -> np.ndarray:
    """
    Fractional FFT with alpha = a / b.
    """
    n = len(c)
    q_n, r_n = divmod(n, 2)
    m = b * len(c)
    q_m, r_m = divmod(m, 2)
    aux = pad(c, (m,))

    indices = (a * np.arange(-q_n, q_n + r_n)) % m  # between 0 and m - 1
    indices[indices >= q_m + r_m] -= m  # between -m//2 and -1
    indices += q_m

    return new_fft(aux)[indices]


def adj_frac_fft(c: np.ndarray, a: int, b: int) -> np.ndarray:
    """
    Adjoint operator of ``frac_fft``.
    """
    return frac_fft(c, -a, b)
