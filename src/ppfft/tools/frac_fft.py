"""
This module provides the implementation of the 
Fractional Fast Fourier Transform and its adjoint,
when alpha is rational.
"""

import numpy as np
from scipy.signal import czt


def x_right_u_right(x, beta):
    n = len(x)
    q_n, _ = divmod(n, 2)
    x_right = x[q_n:]
    w = np.exp(-2j * np.pi * beta)
    return czt(x_right, w=w)


def x_right_u_left(x, beta):
    n = len(x)
    q_n, _ = divmod(n, 2)
    x_right = x[q_n:]
    w = np.exp(2j * np.pi * beta)
    a = np.exp(-2j * np.pi * beta)
    return czt(x_right, m=q_n, w=w, a=a)[::-1]


def x_left_u_right(x, beta):
    n = len(x)
    q_n, r_n = divmod(n, 2)
    x_left = x[:q_n]
    u = np.arange(q_n + r_n)
    w = np.exp(2j * np.pi * beta)
    return czt(x_left[::-1], m=q_n + r_n, w=w) * np.exp(2j * np.pi * beta * u)


def x_left_u_left(x, beta):
    n = len(x)
    q_n, _ = divmod(n, 2)

    x_left = x[:q_n]
    u = np.arange(-q_n, 0)
    w = np.exp(-2j * np.pi * beta)
    a = np.exp(2j * np.pi * beta)
    return czt(x_left[::-1], w=w, a=a)[::-1] * np.exp(2j * np.pi * beta * u)


def fast_frac_fft(x, beta):
    left_1 = x_left_u_right(x, beta)
    left_2 = x_left_u_left(x, beta)
    right_1 = x_right_u_right(x, beta)
    right_2 = x_right_u_left(x, beta)

    return np.concatenate((left_2 + right_2, left_1 + right_1))


def frac_fft_for_ppfft(x, alpha):
    n = len(x)
    if n % 2 == 0:
        x_pad = np.pad(x, (0, 1))
    else:
        x_pad = np.pad(x, (1, 0))
    return fast_frac_fft(x_pad, alpha / n)


def adj_frac_fft_for_ppfft(y, alpha):
    n = len(y) - 1
    if n % 2 == 0:
        return fast_frac_fft(y, -alpha / n)[:-1]
    else:
        return fast_frac_fft(y, -alpha / n)[1:]
