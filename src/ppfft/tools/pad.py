"""
This module provides the implementation of a zero-padding function
compatible with the alternative definition of the FFT.
"""

import numpy as np


def pad(x: np.ndarray, new_shape: tuple) -> np.ndarray:
    """
    Zero-pads the array x in a way that is compatible with ``new_fft`` and ``new_fft2``.
    """

    res = np.copy(x)
    n_dim = res.ndim

    assert n_dim == len(new_shape)

    pad_width = [(0, 0)] * n_dim

    for i, (n, new_n) in enumerate(zip(x.shape, new_shape)):
        q_n, r_n = divmod(n, 2)
        q_new_n, r_new_n = divmod(new_n, 2)

        if r_n == r_new_n:
            pad_i = (q_new_n - q_n, q_new_n - q_n)
        else:
            pad_i = (q_new_n - q_n, q_new_n - q_n + r_new_n - r_n)

        pad_width[i] = pad_i

    res = np.pad(res, pad_width)

    return res


def adj_pad(x: np.ndarray, original_shape: tuple) -> np.ndarray:
    res = np.copy(x)
    n_dim = res.ndim

    assert n_dim == len(original_shape)

    for i, (n, original_n) in enumerate(zip(x.shape, original_shape)):
        q_n, r_n = divmod(n, 2)
        q_original_n, r_original_n = divmod(original_n, 2)

        if r_n == r_original_n:
            # The padding was: (q_n - q_original_n, q_n - q_original_n)
            indices = np.arange(q_n - q_original_n, n - (q_n - q_original_n))
        else:
            # The padding was: (q_n - q_original_n, q_n - q_original_n + r_n - r_original_n)
            indices = np.arange(
                q_n - q_original_n, n - (q_n - q_original_n + r_n - r_original_n)
            )

        res = np.take(res, indices, axis=i)

    return res
