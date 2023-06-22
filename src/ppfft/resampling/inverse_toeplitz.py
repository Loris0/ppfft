"""
This module defines the InverseToeplitz class, used for 
the fast resampling of trigonometric polynomials.
"""

import numpy as np
from scipy.linalg import solve_toeplitz, matmul_toeplitz


class InverseToeplitz:
    """
    A class for storing a Toeplitz matrix,
    the Gohberg-Semencul decomposition of its inverse,
    and use it to apply the inverse to a vector.
    """

    def __init__(self, col, row=None) -> None:
        """
        A Toeplitz matrix T is represented by:
        - its first column: col
        - its first row: row

        If row is not provided, it is assumed T is Hermitian.
        """
        self.col = col
        self.row = row

        self.x0 = None
        self.m1 = None
        self.m2 = None
        self.m3 = None
        self.m4 = None

        self.gohberg_semencul()

    def gohberg_semencul(self):
        """
        Compute the Gohberg-Semencul decomposition of T^{-1}
        """
        e0 = np.zeros_like(self.col)
        e0[0] = 1

        e1 = np.zeros_like(self.col)
        e1[-1] = 1

        if self.row is None:
            x = solve_toeplitz(self.col, e0)
            y = solve_toeplitz(self.col, e1)

        else:
            x = solve_toeplitz((self.col, self.row), e0)
            y = solve_toeplitz((self.col, self.row), e1)

        x_a = np.zeros_like(x)
        x_a[0] = x[0]

        x_b = np.zeros_like(x)
        x_b[1::] = x[:0:-1]

        y_a = np.zeros_like(y)
        y_a[0] = y[-1]

        y_b = np.zeros_like(y)
        y_b[1::] = y[:-1]

        self.x0 = x[0]
        self.m1 = (x, x_a)
        self.m2 = (y_a, y[::-1])
        self.m3 = (y_b, np.zeros_like(y))
        self.m4 = (np.zeros_like(x), x_b)

    def apply_inverse(self, vec, workers=8):
        """
        Computes T^{-1} @ vec.
        """
        M1M2_v = matmul_toeplitz(
            self.m1, matmul_toeplitz(self.m2, vec, workers), workers
        )

        M3M4_v = matmul_toeplitz(
            self.m3, matmul_toeplitz(self.m4, vec, workers), workers
        )

        return (M1M2_v - M3M4_v) / self.x0
