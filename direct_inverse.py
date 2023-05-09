import numpy as np

from pad import pad, adj_pad
from new_fft import new_fft, adj_new_fft
from fast_resampling import compute_inverse
from onion_peeling import onion_peeling


def D(n):
    q_n, r_n = divmod(n, 2)
    return np.arange(-q_n, q_n + r_n)


def F_D(x):
    n = len(x)
    m = 2 * n + 1
    pad_x = pad(x, (m,))

    fft = new_fft(pad_x)

    if n % 2 == 0:
        return fft[::2]

    else:
        return np.roll(fft, 1)[::2]


def adj_F_D(y):
    n = len(y) - 1
    m = 2 * n + 1
    aux = np.zeros(shape=(m,), dtype=complex)
    aux[::2] = y

    if n % 2 == 0:
        return adj_pad(adj_new_fft(aux), original_shape=(n,))

    else:
        return adj_pad(adj_new_fft(np.roll(aux, -1)), original_shape=(n,))


def compute_FDdaggerFD(n):
    m = 2 * n + 1
    u = D(n + 1)
    k_minus_l = D(n)[:, None] - D(n)[None, :]
    u_times_k_minus_l = np.einsum("u,kl->ukl", u, k_minus_l)
    return np.einsum("ukl->kl", np.exp(4j * np.pi * u_times_k_minus_l / m))


def solve_min(y):
    n = len(y) - 1
    T = compute_FDdaggerFD(n)
    inv_T = compute_inverse(T[:, 0])
    return inv_T @ adj_F_D(y)


def inverse_Id(Id):
    n = Id.shape[0] - 1

    A = np.zeros(shape=(n, n + 1), dtype=complex)
    res = np.zeros(shape=(n, n), dtype=complex)

    for l, col in enumerate(Id.T):
        A[:, l] = solve_min(col)

    for u, row in enumerate(A):
        res[u, :] = solve_min(row)

    return res


def direct_inversion(hori_ppfft, vert_ppfft):
    Id = onion_peeling(hori_ppfft, vert_ppfft)
    sol = inverse_Id(Id)
    return sol
