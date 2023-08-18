import numpy as np
import scipy.fft as fft
from pynufft import NUFFT

from ppfft.resampling.inverse_toeplitz import InverseToeplitz


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


def initialize_I_hat(hori_ppfft, vert_ppfft):
    n = np.shape(hori_ppfft)[1] - 1
    I_hat = np.zeros(shape=(n + 1, n + 1), dtype=complex)  # shape (n + 1, n + 1)

    # k = 0
    I_hat[0, n // 2 + 1 :] = np.conjugate(hori_ppfft[1:, n // 2])[::-1]
    I_hat[0, : n // 2 + 1] = hori_ppfft[:, n // 2]

    # l = 0
    I_hat[n // 2 + 1 :, 0] = np.conjugate(vert_ppfft[1:, n // 2])[::-1]
    I_hat[: n // 2 + 1, 0] = vert_ppfft[:, n // 2]

    # k = - n//2
    I_hat[n // 2 + 1, :] = fft.ifftshift(np.conjugate(vert_ppfft[-1, :]))

    # k = n//2
    I_hat[n // 2, :] = fft.ifftshift(vert_ppfft[-1, ::-1])

    # l = - n//2
    I_hat[:, n // 2 + 1] = fft.ifftshift(np.conjugate(hori_ppfft[-1, :]))

    # l = n//2
    I_hat[:, n // 2] = fft.ifftshift(hori_ppfft[-1, ::-1])

    np.fill_diagonal(
        I_hat, np.concatenate((hori_ppfft[:, 0], np.conjugate(hori_ppfft[1:, 0])[::-1]))
    )

    np.fill_diagonal(
        np.fliplr(I_hat)[1:],
        np.concatenate((vert_ppfft[1:, -1], np.conjugate(vert_ppfft[1:, -1])[::-1])),
    )

    return I_hat


def precompute_new_onion_peeling(n: int, oversampling_factor: int = 5):
    half_n = n // 2
    toeplitz_list = []
    nufft_list = []

    for k in range(half_n - 1, 0, -1):
        y_ppfft = 4 * np.pi * np.arange(-(n // 2), n // 2 + 1) * k / (n * (n + 1))
        indices = select_points(k, n)
        y_ppfft = np.take(y_ppfft, indices)

        y_pos = -2 * np.pi * np.arange(-(n // 2), -k) / (n + 1)
        y_neg = 2 * np.pi * np.arange(-(n // 2), -k)[::-1] / (n + 1)

        y = np.concatenate((y_pos, y_ppfft, y_neg))

        c = np.einsum("lj->l", np.exp(-1j * np.arange(0, n)[:, None] * y[None, :]))

        toeplitz_list.append(InverseToeplitz(col=c))

        NufftObj = NUFFT()
        NufftObj.plan(om=-y[:, None], Nd=(n,), Kd=(oversampling_factor * n,), Jd=(6,))
        NufftObj.arr = np.exp(-1j * y * (n // 2))

        nufft_list.append(NufftObj)

    return toeplitz_list, nufft_list


def compute_rows(
    k,
    I_hat,
    vert_ppfft_samples,
    toeplitz_inv: InverseToeplitz,
    NufftObj: NUFFT,
    workers: int = None,
):
    """
    0 < k < n//2
    """
    n = np.shape(I_hat)[0] - 1
    # Step: 1, 2, ..., n//2 - 1

    I_hat_pos = I_hat[k, n // 2 + 1 : n - k + 1]
    I_hat_neg = I_hat[k, k + 1 : n // 2 + 1]

    known_samples = np.concatenate((I_hat_pos, vert_ppfft_samples, I_hat_neg))

    alpha = toeplitz_inv.apply_inverse(
        NufftObj.Kd * NufftObj.adjoint(NufftObj.arr * known_samples), workers=workers
    )

    res = fft.fft(alpha, n=n + 1, workers=workers)

    # Positive row
    I_hat[k, n - k + 2 : n + 1] = res[n - k + 2 : n + 1]
    I_hat[k, 1:k] = res[1:k]

    # Negative row
    I_hat[-k, n - k + 2 : n + 1] = np.conjugate(res[1:k])[::-1]
    I_hat[-k, 1:k] = np.conjugate(res[n - k + 2 : n + 1])[::-1]

    return alpha


def compute_cols(
    k,
    I_hat,
    hori_ppfft_samples,
    toeplitz_inv: InverseToeplitz,
    NufftObj: NUFFT,
    workers: int = None,
):
    """
    0 < k < n//2
    """
    n = np.shape(I_hat)[0] - 1
    # Step: 1, 2, ..., n - 1

    I_hat_pos = I_hat[n // 2 + 1 : n - k + 1, k]
    I_hat_neg = I_hat[k + 1 : n // 2 + 1, k]

    known_samples = np.concatenate((I_hat_pos, hori_ppfft_samples, I_hat_neg))

    alpha = toeplitz_inv.apply_inverse(
        NufftObj.Kd * NufftObj.adjoint(NufftObj.arr * known_samples), workers=workers
    )

    res = fft.fft(alpha, n=n + 1, workers=workers)

    # Positive row
    I_hat[n - k + 2 : n + 1, k] = res[n - k + 2 : n + 1]
    I_hat[1:k, k] = res[1:k]

    # Negative row
    I_hat[n - k + 2 : n + 1, -k] = np.conjugate(res[1:k])[::-1]
    I_hat[1:k, -k] = np.conjugate(res[n - k + 2 : n + 1])[::-1]

    return alpha


def new_onion_peeling(
    hori_ppfft, vert_ppfft, toeplitz_list, nufft_list, workers: int = None
):
    I_hat = initialize_I_hat(hori_ppfft, vert_ppfft)
    n = hori_ppfft.shape[1] - 1

    fft_row = np.zeros(shape=(n, n // 2 + 1), dtype=complex)
    fft_row[:, 0] = fft.ifft(I_hat[:, 0])[:-1]
    fft_row[:, -1] = fft.ifft(I_hat[:, n // 2])[:-1]

    for toeplitz_inv, NufftObj, k in zip(
        toeplitz_list, nufft_list, range(n // 2 - 1, 0, -1)
    ):
        indices = select_points(k, n)
        vert_ppfft_samples = np.take(vert_ppfft[k], indices)
        hori_ppfft_samples = np.take(hori_ppfft[k], indices)

        compute_rows(k, I_hat, vert_ppfft_samples, toeplitz_inv, NufftObj, workers)
        alpha = compute_cols(
            k, I_hat, hori_ppfft_samples, toeplitz_inv, NufftObj, workers
        )

        fft_row[:, k] = alpha

    return fft_row
