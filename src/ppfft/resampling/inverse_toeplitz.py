"""
This module defines the InverseToeplitz class, used for 
the fast resampling of trigonometric polynomials.
"""

import numpy as np
import scipy.fft as fft
from scipy.linalg import solve_toeplitz


class InverseToeplitz:
    """A class for storing a Toeplitz matrix,
    the Gohberg-Semencul decomposition of its inverse,
    and use it to apply the inverse to a vector.

    Methods
    -------
    apply_inverse
    """

    def __init__(self, col: np.ndarray, row=None) -> None:
        """Constructor of the class.

        Parameters
        ----------
        col : np.ndarray
            First column of T.
        row : np.ndarray | None
            First row of T. By default None, meaning row = conj(col)

        """
        self.col = col
        self.row = row
        self.n = len(col)

        self.x0 = None
        self.fft_m1 = None
        self.fft_m2 = None
        self.fft_m3 = None
        self.fft_m4 = None

        self.gohberg_semencul()

    def gohberg_semencul(self) -> None:
        """Computes the Gohberg-Semencul decomposition of T^{-1}

        We only store the FFT of the corresponding convolution kernels.
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

        self.fft_m1 = fft.fft(np.concatenate((np.zeros(self.n - 1), x)))

        self.fft_m4 = fft.fft(np.concatenate((x[1:], np.zeros(self.n))))

        col_m2 = np.zeros_like(y)
        col_m2[0] = y[-1]
        self.fft_m2 = fft.fft(np.concatenate((y[:-1], col_m2)))

        col_m3 = np.zeros_like(y)
        col_m3[1::] = y[:-1]
        self.fft_m3 = fft.fft(np.concatenate((np.zeros(self.n - 1), col_m3)))

        self.x0 = x[0]

    def apply_inverse(self, x: np.ndarray) -> np.ndarray:
        """Computes T^{-1} @ vec using the Gohberg-Semencul formula.

        Parameters
        ----------
        x : np.ndarray
            Vector to compute the product T^{-1} @ x.
        workers : int
            Workers parameter for scipy.linalg.matmul_toeplitz, by default 8

        Returns
        -------
        out : np.ndarray
            Value of T^{-1} @ x
        """

        fft_x = fft.fft(x, n=2 * self.n - 1)

        m2_x = fft.ifft(self.fft_m2 * fft_x)[-self.n :]
        m4_x = fft.ifft(self.fft_m4 * fft_x)[-self.n :]

        res = fft.ifft(
            self.fft_m1 * fft.fft(m2_x, n=2 * self.n - 1)
            - self.fft_m3 * fft.fft(m4_x, n=2 * self.n - 1)
        )[-self.n :]

        return res / self.x0
