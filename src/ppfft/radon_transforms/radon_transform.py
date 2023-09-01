import numpy as np

from ..tools.new_fft import new_fft, new_ifft
from ..legacy.ppfft import ppfft as legacy_ppfft
from ..ppfft.ppfft import ppfft


def legacy_radon_transform(im):
    hori, vert = legacy_ppfft(im)
    fft_hori = new_ifft(hori.T, axis=0)
    fft_vert = new_ifft(vert.T, axis=0)

    return np.concatenate((fft_hori, fft_vert[::-1, ::-1]), axis=1).real


def legacy_ppfft_from_legacy_radon(sino):
    n = sino.shape[0] // 2
    fft_hori, fft_vert = sino[:, : n + 1], sino[:, n + 1 :][::-1, ::-1]
    hori, vert = new_fft(fft_hori, axis=0), new_fft(fft_vert, axis=0)

    return hori.T, vert.T


def radon_transform(im):
    hori, vert = ppfft(im)
    fft_hori = new_ifft(hori, axis=0)
    fft_vert = new_ifft(vert, axis=0)

    return np.concatenate((fft_hori, fft_vert[::-1, ::-1]), axis=1).real


def ppfft_from_radon(sino):
    n = sino.shape[0] - 1
    fft_hori, fft_vert = sino[:, : n + 1], sino[:, n + 1 :][::-1, ::-1]
    hori, vert = new_fft(fft_hori, axis=0), new_fft(fft_vert, axis=0)

    return hori, vert


def radon_from_legacy_radon(legacy_radon):
    n = legacy_radon.shape[0] // 2  # old_radon.shape = (2 * n + 1, 2 * n + 2)
    res = np.copy(legacy_radon[n - n // 2 : n + n // 2 + 1])
    res[: n // 2] += legacy_radon[-(n // 2) :]
    res[-(n // 2) :] += legacy_radon[: n // 2]

    return res
