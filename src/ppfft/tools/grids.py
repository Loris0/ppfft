"""
This module provides functions to compute the coordinates
of the pseudo-polar grids.
"""

import numpy as np


def domain(n: int) -> np.ndarray:
    """Indices of a vector of size n.

    Parameters
    ----------
    n : int
        Domain size.

    Returns
    -------
    out : np.ndarray
        Indices.
    """
    q_n, r_n = divmod(n, 2)
    return np.arange(-q_n, q_n + r_n)


def horizontal_lines(n: int) -> np.ndarray:
    """Positions of the basically horizontal lines of the pseudo-polar grid.

    Parameters
    ----------
    n : int
        Size of the image whose PPFFT we want to compute.

    Returns
    -------
    coords : np.ndarray
        Array of shape (n+1, 2*n+1, 2). coords[..., 0] gives the x coordinates.

    See Also
    --------
    vertical_lines : Positions of the basically vertical lines of the pseudo-polar grid.
    """
    m = 2 * n + 1
    coords = np.empty(shape=(n + 1, m, 2))

    for i_l, l in enumerate(domain(n + 1)):
        for i_k, k in enumerate(domain(2 * n + 1)):
            coords[i_l, i_k, 0] = -2 * l * k / n
            coords[i_l, i_k, 1] = k

    return coords


def vertical_lines(n: int) -> np.ndarray:
    """Positions of the basically vertical lines of the pseudo-polar grid.

    Parameters
    ----------
    n : int
        Size of the image whose PPFFT we want to compute.

    Returns
    -------
    coords : np.ndarray
        Array of shape (n+1, 2*n+1, 2). coords[..., 0] gives the x coordinates.

    See Also
    --------
    horizontal_lines : Positions of the basically horizontal lines of the pseudo-polar grid.
    """
    m = 2 * n + 1
    coords = np.empty(shape=(n + 1, m, 2))

    for i_l, l in enumerate(domain(n + 1)):
        for i_k, k in enumerate(domain(m)):
            coords[i_l, i_k, 0] = k
            coords[i_l, i_k, 1] = -2 * l * k / n

    return coords


def half_horizontal_grid(n: int) -> np.ndarray:
    ks = np.arange(0, n // 2 + 1)
    ls = np.arange(-(n // 2), n // 2 + 1)
    horizontal_x = -2 * ks[:, None] * ls[None, :] / (n * (n + 1))
    horizontal_y = np.tile(ks[:, None], (1, n + 1)) / (n + 1)
    return horizontal_x, horizontal_y


def polar_grid(thetas, n_r):
    """
    n_r should be odd
    """
    rs = domain(n_r) / n_r

    polar_x = np.cos(thetas)[:, None] * rs[None, :]
    polar_y = np.sin(thetas)[:, None] * rs[None, :]

    return polar_x, polar_y
