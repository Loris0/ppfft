import numpy as np

from pad import pad, adj_pad
from new_fft import adj_new_fft
from inverse_toeplitz import InverseToeplitz
from fast_onion_peeling import fast_onion_peeling, precompute_onion_peeling


def adj_F_D(y):
    n = len(y) - 1
    m = 2 * n + 1
    aux = np.zeros(shape=(m,), dtype=complex)
    aux[::2] = y

    if n % 2 == 0:
        return adj_pad(adj_new_fft(aux), original_shape=(n,))

    else:
        return adj_pad(adj_new_fft(np.roll(aux, -1)), original_shape=(n,))


def compute_col(n):
    m = 2 * n + 1
    one = np.ones(n + 1)
    pad_one = pad(one, (m,))

    q_m, r_m = divmod(m, 2)

    indices = (2 * np.arange(0, n)) % m
    indices[indices >= q_m + r_m] -= m
    indices += q_m

    return np.take(adj_new_fft(pad_one), indices)


def precompute_inverse_Id(n):
    c = compute_col(n)

    return InverseToeplitz(col=c)


def fast_inverse_Id(Id, toeplitz: InverseToeplitz):
    n = Id.shape[0] - 1

    A = np.zeros(shape=(n, n + 1), dtype=complex)
    res = np.zeros(shape=(n, n), dtype=complex)

    for l, col in enumerate(Id.T):
        A[:, l] = toeplitz.apply_inverse(adj_F_D(col))

    for u, row in enumerate(A):
        res[u, :] = toeplitz.apply_inverse(adj_F_D(row))

    return res


def precompute_all(n):
    toeplitz_list, nufft_list = precompute_onion_peeling(n)
    toeplitz = precompute_inverse_Id(n)
    return toeplitz_list, nufft_list, toeplitz


def fast_direct_inversion(hori_ppfft, vert_ppfft, precomputations):
    toeplitz_list, nufft_list, toeplitz = precomputations
    Id = fast_onion_peeling(hori_ppfft, vert_ppfft, toeplitz_list, nufft_list)
    sol = fast_inverse_Id(Id, toeplitz)
    return sol
