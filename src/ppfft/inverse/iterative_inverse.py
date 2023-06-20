import numpy as np

from scipy.sparse.linalg import cg, LinearOperator
from ..ppfft.ppfft import ppfft, adj_ppfft


def iterative_inverse(hori_y, vert_y, preconditioned=True, tol=1e-7):
    n, m = hori_y.shape[0] - 1, hori_y.shape[1]

    if preconditioned:
        M = np.empty(shape=(n + 1, m))
        for k in range(m):
            if k - n == 0:
                M[:, k] = 1 / m**2
            else:
                M[:, k] = 2 * (n + 1) * abs(k - n) / (n * m)

        def vec_operator(v):
            a = np.reshape(v, newshape=(n, n))
            hori, vert = ppfft(a)
            b = adj_ppfft(M * hori, M * vert)
            return b.flatten()

        y = adj_ppfft(M * hori_y, M * vert_y).flatten()

    else:

        def vec_operator(v):
            a = np.reshape(v, newshape=(n, n))
            hori, vert = ppfft(a)
            b = adj_ppfft(hori, vert)
            return b.flatten()

        y = adj_ppfft(hori_y, vert_y).flatten()

    operator = LinearOperator(
        shape=(n**2, n**2), matvec=vec_operator, rmatvec=vec_operator
    )

    sol_cg, exit_status = cg(operator, y, x0=np.zeros(n**2), tol=tol)

    return np.reshape(sol_cg, newshape=(n, n)), exit_status
