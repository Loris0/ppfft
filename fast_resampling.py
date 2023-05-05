import numpy as np
from scipy.linalg import toeplitz, solve_toeplitz


def compute_A(y, n):
    half_n = n//2
    k = np.arange(-half_n, half_n)
    return np.exp(1j * np.einsum("j,k->jk", y, k))


def compute_AdaggerA(y, n):
    aux = -1j * np.arange(n)
    aux = np.exp(aux[:, None] * y[None, :])
    c = np.einsum("ij->i", aux)
    return toeplitz(c)


def compute_inverse(c, r=None):
    """
    Computes the Gohberg-Semencul decomposition of T^-1.

    T^-1 = (M1 @ M2 - M3 @ M4) / x0
    """

    e0 = np.zeros_like(c)
    e0[0] = 1

    e1 = np.zeros_like(c)
    e1[-1] = 1

    if r is None:
        x = solve_toeplitz(c, e0)
        y = solve_toeplitz(c, e1)

    else:
        r = np.conjugate(c)
        x = solve_toeplitz((c, r), e0)
        y = solve_toeplitz((c, r), e1)

    x_a = np.zeros_like(x)
    x_a[0] = x[0]

    x_b = np.zeros_like(x)
    x_b[1::] = x[:0:-1]

    M_1 = toeplitz(x, x_a)
    M_4 = toeplitz(np.zeros_like(x), x_b)

    y_a = np.zeros_like(y)
    y_a[0] = y[-1]

    y_b = np.zeros_like(y)
    y_b[1::] = y[:-1]

    M_2 = toeplitz(y_a, y[::-1])
    M_3 = toeplitz(y_b, np.zeros_like(y))

    return (M_1 @ M_2 - M_3 @ M_4) / x[0]


def compute_alpha(y, n, f):
    A = compute_A(y, n)
    T = compute_AdaggerA(y, n)
    inv_T = compute_inverse(T[:, 0])
    alpha = inv_T @ A.conj().T @ f
    return alpha


def compute_alpha_regularized(y, n, f, lambd=1e-6):
    A = compute_A(y, n)
    T = compute_AdaggerA(y, n) + lambd * np.eye(n)
    inv_T = compute_inverse(T[:, 0])
    alpha = inv_T @ A.conj().T @ f
    return alpha
