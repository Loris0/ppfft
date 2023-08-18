import numpy as np
import scipy.fft as fft

from .new_onion_peeling import new_onion_peeling


def new_direct_inversion(hori_ppfft, vert_ppfft, precomputations, workers: int = None):
    toeplitz_list, nufft_list = precomputations
    n = np.shape(hori_ppfft)[1] - 1
    fft_row = new_onion_peeling(
        hori_ppfft, vert_ppfft, toeplitz_list, nufft_list, workers
    )
    return fft.irfft(fft_row, n=n + 1, workers=workers)[:, :-1]
