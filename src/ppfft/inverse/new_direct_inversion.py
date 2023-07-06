from ppfft.inverse.new_onion_peeling import new_onion_peeling, precompute_onion_peeling
from ppfft.tools.new_fft import new_ifft


def new_direct_inversion(hori_ppfft, vert_ppfft, precomputations, workers: int = None):
    toeplitz_list, nufft_list = precomputations
    fft_col = new_onion_peeling(
        hori_ppfft, vert_ppfft, toeplitz_list, nufft_list, workers=workers
    )
    return new_ifft(fft_col, axis=0, workers=workers)[:-1].real
