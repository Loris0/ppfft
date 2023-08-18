import numpy as np
import scipy.fft as fft

from ppfft.tools.frac_fft import new_fast_frac_fft


def new_ppfft_horizontal(a: np.ndarray) -> np.ndarray:
    n, _ = a.shape

    res = np.empty((1 + n // 2, n + 1), dtype=complex)

    # 1D FFT of each zero-padded line. Shape = (n, n + 1)
    fft_col = fft.fft(a, n=n + 1, axis=-1)[:, : 1 + n // 2]
    # Frac FFT on each col
    for k, col in enumerate(fft_col.T):
        res[k, :] = new_fast_frac_fft(col, beta=-2 * k / (n * (n + 1)), m=n + 1)
    return res


def new_ppfft_vertical(a):
    return new_ppfft_horizontal(a.T)


def new_ppfft(a):
    return new_ppfft_horizontal(a), new_ppfft_vertical(a)
