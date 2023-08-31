from ppfft.inverse.onion_peeling import onion_peeling_col, onion_peeling_row
from ppfft.tools.new_fft import new_ifft


def direct_inversion_col(
    hori_ppfft, vert_ppfft, toeplitz_list, nufft_list, workers: int = None
):
    fft_col = onion_peeling_col(
        hori_ppfft, vert_ppfft, toeplitz_list, nufft_list, workers=workers
    )
    return new_ifft(fft_col, axis=0, workers=workers)[:-1].real


def direct_inversion_row(
    hori_ppfft, vert_ppfft, toeplitz_list, nufft_list, workers: int = None
):
    fft_row = onion_peeling_row(
        hori_ppfft, vert_ppfft, toeplitz_list, nufft_list, workers=workers
    )
    return new_ifft(fft_row, axis=1, workers=workers)[:, :-1].real
