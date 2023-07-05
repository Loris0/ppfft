import numpy as np

from ppfft.tools.frac_fft import fast_frac_fft
from ppfft.tools.grids import domain
from ppfft.tools.new_fft import new_fft
from ppfft.tools.pad import pad


def new_ppfft_horizontal(a: np.ndarray) -> np.ndarray:
    n, _ = a.shape

    res = np.empty((n + 1, n + 1), dtype=complex)

    # 1D FFT of each zero-padded line. Shape = (n, m)
    fft_col = new_fft(pad(a, new_shape=(n, n + 1)), axis=-1)

    # Frac FFT on each col
    for k, col in zip(domain(n + 1), fft_col.T):
        res[k + n // 2, :] = fast_frac_fft(col, beta=-2 * k / (n * (n + 1)), m=n + 1)
    return res


def new_ppfft_vertical(a):
    return new_ppfft_horizontal(a.T)


def new_ppfft(a):
    return new_ppfft_horizontal(a), new_ppfft_vertical(a)
