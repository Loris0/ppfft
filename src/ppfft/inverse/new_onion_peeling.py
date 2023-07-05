import numpy as np
from pynufft import NUFFT

from ppfft.resampling.inverse_toeplitz import InverseToeplitz
from ppfft.tools.pad import pad
from ppfft.tools.new_fft import new_fft, new_ifft


def select_points(k: int, n: int) -> np.ndarray:
    """Point selection for onion-peeling.

    Parameters
    ----------
    k : int
        Step of the onion-peeling method: -n//2 < k < n//2
    n : int
        Size of the image to reconstruct.

    Returns
    -------
    out : np.ndarray
        Indices of the points to take.
    """
    l = np.arange(k, -np.sign(k) - k, step=-np.sign(k))
    return n // 2 + np.rint(-n * l / (2 * k)).astype(int)


def precompute_onion_peeling(n: int, oversampling_factor: int = 5):
    half_n = n // 2
    toeplitz_list = []
    nufft_list = []

    for k in range(-half_n + 1, 0):
        y_ppfft = 4 * np.pi * np.arange(-(n // 2), n // 2 + 1) * k / (n * (n + 1))
        indices = select_points(k, n)
        y_ppfft = np.take(y_ppfft, indices)

        y_pos = -2 * np.pi * np.arange(-(n // 2), k) / (n + 1)
        y_neg = 2 * np.pi * np.arange(-(n // 2), k)[::-1] / (n + 1)

        y = np.concatenate((y_pos, y_ppfft, y_neg))

        c = np.einsum("lj->l", np.exp(-1j * np.arange(0, n)[:, None] * y[None, :]))

        toeplitz_list.append(InverseToeplitz(col=c))

        NufftObj = NUFFT()
        NufftObj.plan(om=-y[:, None], Nd=(n,), Kd=(oversampling_factor * n,), Jd=(6,))
        nufft_list.append(NufftObj)

    return toeplitz_list, nufft_list


def initialize_I_hat(hori_ppfft, vert_ppfft):
    n = np.shape(hori_ppfft)[0] - 1
    I_hat = np.zeros_like(hori_ppfft)  # shape (n + 1, n + 1)

    I_hat[0, :] = vert_ppfft[0, :]
    I_hat[-1, :] = vert_ppfft[-1, ::-1]
    I_hat[:, 0] = hori_ppfft[0, :]
    I_hat[:, -1] = hori_ppfft[-1, ::-1]
    I_hat[n // 2, :] = hori_ppfft[:, n // 2]
    I_hat[:, n // 2] = vert_ppfft[:, n // 2]
    np.fill_diagonal(I_hat, hori_ppfft[:, 0])
    np.fill_diagonal(np.fliplr(I_hat), vert_ppfft[:, -1])

    return I_hat


def compute_rows(
    k, I_hat, vert_ppfft_samples, toeplitz_inv: InverseToeplitz, NufftObj: NUFFT
):
    """
    -n//2 < k < 0
    """
    n = np.shape(I_hat)[0] - 1
    true_k = k + n // 2

    I_hat_pos = I_hat[true_k, :true_k]
    I_hat_neg = I_hat[true_k, -true_k:]

    known_samples = np.concatenate((I_hat_pos, vert_ppfft_samples, I_hat_neg))

    alpha = toeplitz_inv.apply_inverse(NufftObj.Kd * NufftObj.adjoint(known_samples))

    res = new_fft(pad(alpha, (n + 1,)))

    # Negative row
    I_hat[true_k, true_k + 1 : n // 2] = res[true_k + 1 : n // 2]
    I_hat[true_k, n // 2 + 1 : -true_k - 1] = res[n // 2 + 1 : -true_k - 1]
    # Positive row
    I_hat[-1 - true_k] = np.conjugate(I_hat[true_k, ::-1])

    return alpha


def compute_cols(
    k, I_hat, hori_ppfft_samples, toeplitz_inv: InverseToeplitz, NufftObj: NUFFT
):
    """
    -(n//2) < k < 0
    """
    n = np.shape(I_hat)[0] - 1
    true_k = k + n // 2

    I_hat_pos = I_hat[:true_k, true_k]
    I_hat_neg = I_hat[-true_k:, true_k]

    known_samples = np.concatenate((I_hat_pos, hori_ppfft_samples, I_hat_neg))

    alpha = toeplitz_inv.apply_inverse(NufftObj.Kd * NufftObj.adjoint(known_samples))

    res = new_fft(pad(alpha, (n + 1,)))

    # Negative column
    I_hat[true_k + 1 : n // 2, true_k] = res[true_k + 1 : n // 2]
    I_hat[n // 2 + 1 : -true_k - 1, true_k] = res[n // 2 + 1 : -true_k - 1]
    # Positive column
    I_hat[:, -1 - true_k] = np.conjugate(I_hat[::-1, true_k])

    return alpha


def new_onion_peeling(hori_ppfft, vert_ppfft, toeplitz_list, nufft_list):
    I_hat = initialize_I_hat(hori_ppfft, vert_ppfft)
    n = hori_ppfft.shape[0] - 1

    fft_col = np.zeros(shape=(n + 1, n), dtype=complex)
    fft_col[0, :] = new_ifft(I_hat[0])[:-1]
    fft_col[-1, :] = np.conjugate(fft_col[0, :])
    fft_col[n // 2, :] = new_ifft(I_hat[n // 2])[:-1]

    for toeplitz_inv, NufftObj, k in zip(
        toeplitz_list, nufft_list, range(-(n // 2) + 1, 0)
    ):
        indices = select_points(k, n)
        vert_ppfft_samples = np.take(vert_ppfft[k + n // 2], indices)
        hori_ppfft_samples = np.take(hori_ppfft[k + n // 2], indices)

        fft_col[k + n // 2, :] = compute_rows(
            k, I_hat, vert_ppfft_samples, toeplitz_inv, NufftObj
        )
        fft_col[-(k + n // 2) - 1, :] = np.conjugate(fft_col[k + n // 2, :])

        compute_cols(k, I_hat, hori_ppfft_samples, toeplitz_inv, NufftObj)

    return fft_col
