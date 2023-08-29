import numpy as np

from ..tools.new_fft import new_fft
from ..tools.grids import domain
from ..tools.pad import pad
from ..tools.frac_fft import fast_frac_fft


def ppfft_horizontal(im):
    n = im.shape[0]
    m = 2 * n + 1
    res = np.zeros(shape=(2 * n + 1, n + 1), dtype=complex)  # k, l

    fft_col = new_fft(pad(im, new_shape=(n, m)), axis=-1)

    for k, col in zip(domain(m), fft_col.T):
        res[k + n, :] = fast_frac_fft(col, beta=-2 * k / (n * (2 * n + 1)), m=n + 1)
    return res.T


def ppfft_vertical(im):
    return ppfft_horizontal(im.T)


def ppfft(im):
    return ppfft_horizontal(im), ppfft_vertical(im)
