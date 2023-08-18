from ppfft.inverse.onion_peeling import onion_peeling
from ppfft.tools.new_fft import new_ifft


def direct_inversion(hori_ppfft, vert_ppfft, precomputations, workers: int = None):
    toeplitz_list, nufft_list = precomputations
    fft_col = onion_peeling(
        hori_ppfft, vert_ppfft, toeplitz_list, nufft_list, workers=workers
    )
    return new_ifft(fft_col, axis=0, workers=workers)[:-1].real
