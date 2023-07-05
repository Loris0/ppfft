from ppfft.inverse.new_onion_peeling import new_onion_peeling, precompute_onion_peeling
from ppfft.tools.new_fft import new_ifft


def new_direct_inversion(hori_ppfft, vert_ppfft, precomputations):
    toeplitz_list, nufft_list = precomputations
    fft_col = new_onion_peeling(hori_ppfft, vert_ppfft, toeplitz_list, nufft_list)
    return new_ifft(fft_col, axis=0)[:-1].real
