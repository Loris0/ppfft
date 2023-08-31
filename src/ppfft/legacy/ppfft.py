import numpy as np

from ..tools.new_fft import new_fft, adj_new_fft
from ..tools.grids import domain
from ..tools.pad import pad, adj_pad
from ..tools.frac_fft import frac_fft, adj_frac_fft


def ppfft_horizontal(im):
    n = im.shape[0]
    m = 2 * n + 1
    res = np.zeros(shape=(n + 1, 2 * n + 1), dtype=complex)  # l, k
    fft_col = new_fft(pad(im, new_shape=(n, m)), axis=-1)

    for k, col in zip(domain(m), fft_col.T):
        res[:, k + n] = frac_fft(col, beta=-2 * k / (n * (2 * n + 1)), m=n + 1)
    return res


def ppfft_vertical(im):
    return ppfft_horizontal(im.T)


def ppfft(im):
    return ppfft_horizontal(im), ppfft_vertical(im)


def adj_ppfft_horizontal(hori_ppfft):
    n = hori_ppfft.shape[0] - 1
    m = 2 * n + 1

    hori_aux = np.empty(shape=(n, m), dtype=complex)

    for k, col in zip(domain(m), hori_ppfft.T):
        hori_aux[:, k + n] = adj_frac_fft(col, beta=-2 * k / (n * (2 * n + 1)), n=n)

    hori_aux = adj_pad(adj_new_fft(hori_aux), (n, n))

    return hori_aux


def adj_ppfft_vertical(vert_ppfft):
    return adj_ppfft_horizontal(vert_ppfft).T


def adj_ppfft(hori_ppfft, vert_ppfft):
    return adj_ppfft_horizontal(hori_ppfft) + adj_ppfft_vertical(vert_ppfft)
