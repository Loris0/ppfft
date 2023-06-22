import numpy as np

from ..tools.pad import pad
from ..tools.new_fft import new_fft
from ..resampling.inverse_toeplitz import InverseToeplitz
from pynufft import NUFFT


def new_find_closest(k, n):
    l = np.arange(k, -np.sign(k) - k, step=-np.sign(k))
    return n // 2 + np.rint(-n * l / (2 * k)).astype(int)


def resample_row(alpha):
    n = len(alpha)
    pad_alpha = pad(alpha, new_shape=(2 * n + 1,))
    fft_alpha = new_fft(pad_alpha)
    return fft_alpha[::2]


def fast_initialize(hori_ppfft, vert_ppfft):
    n = hori_ppfft.shape[0] - 1

    I_d = np.zeros(shape=(n + 1, n + 1), dtype=complex)

    I_d[0] = vert_ppfft[:, 0]  # x = -n/2
    I_d[-1] = vert_ppfft[::-1, -1]  # x = n/2
    I_d[:, 0] = hori_ppfft[:, 0]  # y = -n/2
    I_d[:, -1] = hori_ppfft[::-1, -1]  # y = n/2

    np.fill_diagonal(I_d, vert_ppfft[0, ::2])
    np.fill_diagonal(np.fliplr(I_d), vert_ppfft[-1, ::2])

    I_d[n // 2] = hori_ppfft[n // 2, ::2]
    I_d[:, n // 2] = vert_ppfft[n // 2, ::2]

    return I_d


def precompute_onion_peeling(n, oversampling_factor=5):
    """
    Computes and stores:
    - all the Toeplitz inverses
    - all the Nufft objects
    needed for the onion-peeling algorithm.

    ## Parameters
    n : int
        Size of the image to reconstruct.
    oversampling factor : int
        Oversampling factor used by NUFFT.

    ## Returns
    toeplitz_list : list[InverseToeplitz]
        List of inverses of the Toeplitz matrices.
    nufft_list : list[NUFFT]
        List of NUFFT objects.
    """

    half_n = n // 2
    m = 2 * n + 1

    toeplitz_list = []
    nufft_list = []

    for k in range(-half_n + 1, 0):
        y_ppfft = 8 * np.pi * k * np.arange(-half_n, half_n + 1) / (n * m)
        indices = new_find_closest(k, n)
        y_ppfft = np.take(y_ppfft, indices)

        y_pos = -4 * np.pi * np.arange(-half_n, k) / m
        y_neg = 4 * np.pi * np.arange(-half_n, k)[::-1] / m

        y = np.concatenate((y_pos, y_ppfft, y_neg))

        c = np.einsum("lj->l", np.exp(-1j * np.arange(0, n)[:, None] * y[None, :]))

        toeplitz_list.append(InverseToeplitz(col=c))

        NufftObj = NUFFT()
        NufftObj.plan(om=-y[:, None], Nd=(n,), Kd=(oversampling_factor * n,), Jd=(6,))
        nufft_list.append(NufftObj)

    return toeplitz_list, nufft_list


def fast_recover_row_negative(
    k, indices, vert_ppfft, Id, toeplitz_inv: InverseToeplitz, NufftObj: NUFFT
):
    """
    Recovers row  -(n//2) < k < 0 of Id.
    Id is modified in place.
    """
    n = vert_ppfft.shape[0] - 1
    half_n = n // 2
    true_k = k + half_n

    known_ppfft = vert_ppfft[:, 2 * k + n]
    known_ppfft = np.take(known_ppfft, indices)

    known_I_D_pos = Id[true_k, :true_k]

    known_I_D_neg = Id[true_k, -true_k:]

    known_samples = np.concatenate((known_I_D_pos, known_ppfft, known_I_D_neg))

    alpha = toeplitz_inv.apply_inverse(NufftObj.Kd * NufftObj.adjoint(known_samples))

    res = resample_row(alpha)

    Id[true_k, true_k + 1 : n // 2] = res[true_k + 1 : n // 2]
    Id[true_k, n // 2 + 1 : -true_k - 1] = res[n // 2 + 1 : -true_k - 1]


def fast_recover_row_positive(
    k, indices, vert_ppfft, Id, toeplitz_inv: InverseToeplitz, NufftObj: NUFFT
):
    """
    Recovers row 0 < k < n//2 of Id.
    Id is modified in place.
    """
    n = vert_ppfft.shape[0] - 1
    half_n = n // 2
    true_k = k + half_n

    known_ppfft = vert_ppfft[:, 2 * k + n]
    known_ppfft = np.take(known_ppfft, indices)

    known_I_D_pos = Id[true_k, : (n - true_k)]

    known_I_D_neg = Id[true_k, (true_k - n) :]

    known_samples = np.concatenate((known_I_D_pos, known_ppfft[::-1], known_I_D_neg))

    alpha = toeplitz_inv.apply_inverse(NufftObj.Kd * NufftObj.adjoint(known_samples))

    res = resample_row(alpha)

    Id[true_k, (n - true_k) + 1 : n // 2] = res[(n - true_k) + 1 : n // 2]
    Id[true_k, n // 2 + 1 : true_k - n - 1] = res[n // 2 + 1 : true_k - n - 1]


def fast_recover_col_negative(
    k, indices, hori_ppfft, Id, toeplitz_inv: InverseToeplitz, NufftObj: NUFFT
):
    """
    Recovers column -(n//2) < k < 0 of Id.
    Id is modified in place.
    """
    n = hori_ppfft.shape[0] - 1
    half_n = n // 2
    true_k = k + half_n

    known_ppfft = hori_ppfft[:, 2 * k + n]  # n + 1 elements
    known_ppfft = np.take(known_ppfft, indices)

    known_I_D_pos = Id[:true_k, true_k]

    known_I_D_neg = Id[-true_k:, true_k]

    known_samples = np.concatenate((known_I_D_pos, known_ppfft, known_I_D_neg))

    alpha = toeplitz_inv.apply_inverse(NufftObj.Kd * NufftObj.adjoint(known_samples))

    res = resample_row(alpha)

    Id[true_k + 1 : n // 2, true_k] = res[true_k + 1 : n // 2]
    Id[n // 2 + 1 : -true_k - 1, true_k] = res[n // 2 + 1 : -true_k - 1]


def fast_recover_col_positive(
    k, indices, hori_ppfft, Id, toeplitz_inv: InverseToeplitz, NufftObj: NUFFT
):
    """
    Recovers column 0 < k < n//2 Id.
    Id is modified in place.
    """
    n = hori_ppfft.shape[0] - 1
    half_n = n // 2
    true_k = k + half_n

    known_ppfft = hori_ppfft[:, 2 * k + n]  # n + 1 elements
    known_ppfft = np.take(known_ppfft, indices)

    known_I_D_pos = Id[: (n - true_k), true_k]

    known_I_D_neg = Id[(true_k - n) :, true_k]

    known_samples = np.concatenate((known_I_D_pos, known_ppfft[::-1], known_I_D_neg))

    alpha = toeplitz_inv.apply_inverse(NufftObj.Kd * NufftObj.adjoint(known_samples))

    res = resample_row(alpha)

    Id[(n - true_k) + 1 : n // 2, true_k] = res[(n - true_k) + 1 : n // 2]
    Id[n // 2 + 1 : true_k - n - 1, true_k] = res[n // 2 + 1 : true_k - n - 1]


def fast_onion_peeling(hori_ppfft, vert_ppfft, toeplitz_list, nufft_list):
    Id = fast_initialize(hori_ppfft, vert_ppfft)
    n = hori_ppfft.shape[0] - 1
    half_n = n // 2

    for toeplitz_inv, NufftObj, k in zip(
        toeplitz_list, nufft_list, range(-half_n + 1, 0)
    ):
        indices = new_find_closest(k, n)  # we could precompute this

        fast_recover_row_negative(k, indices, vert_ppfft, Id, toeplitz_inv, NufftObj)
        fast_recover_row_positive(-k, indices, vert_ppfft, Id, toeplitz_inv, NufftObj)

        fast_recover_col_negative(k, indices, hori_ppfft, Id, toeplitz_inv, NufftObj)
        fast_recover_col_positive(-k, indices, hori_ppfft, Id, toeplitz_inv, NufftObj)

    return Id
