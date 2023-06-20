import numpy as np

from ..tools.pad import pad
from ..tools.new_fft import new_fft
from ..resampling.fast_resampling import compute_alpha


def find_closest(y, n):
    m = 2 * n + 1
    target = -4 * np.pi * np.arange(-(n // 2), n // 2 + 1) / m
    return np.argmin(np.abs(y[:, None] - target[None, :]), axis=0)


def new_find_closest(k, n):
    l = np.arange(k, -np.sign(k) - k, step=-np.sign(k))
    return n // 2 + np.rint(-n * l / (2 * k)).astype(int)


def initialize(hori_ppfft, vert_ppfft):
    n = hori_ppfft.shape[0] - 1

    I_d = np.zeros(shape=(n + 1, n + 1), dtype=complex)

    I_d[0] = vert_ppfft[:, 0]  # x = -n/2
    I_d[-1] = vert_ppfft[::-1, -1]  # x = n/2
    I_d[:, 0] = hori_ppfft[:, 0]  # y = -n/2
    I_d[:, -1] = hori_ppfft[::-1, -1]  # y = n/2

    return I_d


def resample_row(alpha):
    n = len(alpha)
    pad_alpha = pad(alpha, new_shape=(2 * n + 1,))
    fft_alpha = new_fft(pad_alpha)
    return fft_alpha[::2]


def recover_row_negative(k, vert_ppfft, Id):
    """
    Recovers row  -(n//2) < k < 0 of Id.
    Id is modified in place.
    """
    n, m = vert_ppfft.shape[0] - 1, vert_ppfft.shape[1]
    half_n = n // 2
    true_k = k + half_n

    known_ppfft = vert_ppfft[:, 2 * k + n]
    y_ppfft = 8 * np.pi * k * np.arange(-half_n, half_n + 1) / (n * m)

    indices = new_find_closest(k, n)
    known_ppfft = np.take(known_ppfft, indices)
    y_ppfft = np.take(y_ppfft, indices)

    known_I_D_left = Id[true_k, :true_k]
    y_left = -4 * np.pi * np.arange(-half_n, k) / m

    known_I_D_right = Id[true_k, -true_k:][::-1]
    y_right = 4 * np.pi * np.arange(-half_n, k) / m

    known_samples = np.concatenate((known_I_D_left, known_ppfft, known_I_D_right))
    y = np.concatenate((y_left, y_ppfft, y_right))

    alpha = compute_alpha(y, n, known_samples)

    res = resample_row(alpha)

    Id[true_k, true_k:-true_k] = res[true_k:-true_k]


def recover_row_positive(k, vert_ppfft, Id):
    """
    Recovers row 0 < k < n//2 of Id.
    Id is modified in place.
    """
    n, m = vert_ppfft.shape[0] - 1, vert_ppfft.shape[1]
    half_n = n // 2
    true_k = k + half_n

    known_ppfft = vert_ppfft[:, 2 * k + n]
    y_ppfft = 8 * np.pi * k * np.arange(-half_n, half_n + 1) / (n * m)

    indices = new_find_closest(k, n)
    known_ppfft = np.take(known_ppfft, indices)
    y_ppfft = np.take(y_ppfft, indices)

    known_I_D_right = Id[true_k, : (n - true_k)][::-1]
    y_left = -4 * np.pi * np.arange(k + 1, half_n + 1) / m

    known_I_D_left = Id[true_k, (true_k - n) :]
    y_right = 4 * np.pi * np.arange(k + 1, half_n + 1) / m

    known_samples = np.concatenate((known_I_D_left, known_ppfft, known_I_D_right))

    y = np.concatenate((y_left, y_ppfft, y_right))

    alpha = compute_alpha(y, n, known_samples)

    res = resample_row(alpha)

    Id[true_k, (n - true_k) : (true_k - n)] = res[(n - true_k) : (true_k - n)]


def recover_row(k, vert_ppfft, Id):
    """
    Recovers rows k and -k of Id.
    Here, -(n//2) < k < 0
    """
    n = vert_ppfft.shape[0] - 1
    assert -(n // 2) < k < 0
    recover_row_negative(k, vert_ppfft, Id)
    recover_row_positive(-k, vert_ppfft, Id)


def recover_col_negative(k, hori_ppfft, Id):
    """
    Recovers column -(n//2) < k < 0 of Id.
    Id is modified in place.
    """
    n, m = hori_ppfft.shape[0] - 1, hori_ppfft.shape[1]
    half_n = n // 2
    true_k = k + half_n

    known_ppfft = hori_ppfft[:, 2 * k + n]  # n + 1 elements
    y_ppfft = 8 * np.pi * k * np.arange(-half_n, half_n + 1) / (n * m)

    indices = new_find_closest(k, n)
    known_ppfft = np.take(known_ppfft, indices)
    y_ppfft = np.take(y_ppfft, indices)

    known_I_D_left = Id[:true_k, true_k]
    y_left = -4 * np.pi * np.arange(-half_n, k) / m

    known_I_D_right = Id[-true_k:, true_k][::-1]
    y_right = 4 * np.pi * np.arange(-half_n, k) / m

    known_samples = np.concatenate((known_I_D_left, known_ppfft, known_I_D_right))

    y = np.concatenate((y_left, y_ppfft, y_right))

    alpha = compute_alpha(y, n, known_samples)

    res = resample_row(alpha)

    Id[true_k:-true_k, true_k] = res[true_k:-true_k]


def recover_col_positive(k, hori_ppfft, Id):
    """
    Recovers column 0 < k < n//2 Id.
    Id is modified in place.
    """
    n, m = hori_ppfft.shape[0] - 1, hori_ppfft.shape[1]
    half_n = n // 2
    true_k = k + half_n

    known_ppfft = hori_ppfft[:, 2 * k + n]  # n + 1 elements
    y_ppfft = 8 * np.pi * k * np.arange(-half_n, half_n + 1) / (n * m)

    indices = new_find_closest(k, n)
    known_ppfft = np.take(known_ppfft, indices)
    y_ppfft = np.take(y_ppfft, indices)

    known_I_D_right = Id[: (n - true_k), true_k][::-1]
    y_left = -4 * np.pi * np.arange(k + 1, half_n + 1) / m

    known_I_D_left = Id[(true_k - n) :, true_k]
    y_right = 4 * np.pi * np.arange(k + 1, half_n + 1) / m

    known_samples = np.concatenate((known_I_D_left, known_ppfft, known_I_D_right))

    y = np.concatenate((y_left, y_ppfft, y_right))

    alpha = compute_alpha(y, n, known_samples)

    res = resample_row(alpha)

    Id[(n - true_k) : (true_k - n), true_k] = res[(n - true_k) : (true_k - n)]


def recover_col(k, hori_ppfft, Id):
    """
    Recovers columns k and -k of Id.
    Here, -(n//2) < k < 0
    """
    n = hori_ppfft.shape[0] - 1
    assert -(n // 2) < k < 0
    recover_col_negative(k, hori_ppfft, Id)
    recover_col_positive(-k, hori_ppfft, Id)


def onion_peeling(hori_ppfft, vert_ppfft):
    Id = initialize(hori_ppfft, vert_ppfft)
    n = hori_ppfft.shape[0] - 1

    for k in np.arange(-(n // 2) + 1, 0):
        recover_row(k, vert_ppfft, Id)
        recover_col(k, hori_ppfft, Id)

    Id[n // 2, n // 2] = hori_ppfft[0, n]

    return Id
