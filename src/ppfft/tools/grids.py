import numpy as np


def domain(n):
    q_n, r_n = divmod(n, 2)
    return np.arange(-q_n, q_n + r_n)


def horizontal_lines(n: int):
    """
    Computes the positions of the basically horizontal lines of the pseudo-polar grid.

    ## Parameters
    n : int
        Size of the image whose PPFFT we want to compute.

    ## Returns
    coords : np.ndarray
        Array of shape (n+1, 2*n+1, 2). coords[..., 0] gives the x coordinates.

    ## See Also
    vertical_lines : Return the positions of the basically vertical lines of the pseudo-polar grid.
    """
    m = 2 * n + 1
    coords = np.empty(shape=(n + 1, m, 2))

    for i_l, l in enumerate(domain(n + 1)):
        for i_k, k in enumerate(domain(2 * n + 1)):
            coords[i_l, i_k, 0] = -2 * l * k / n
            coords[i_l, i_k, 1] = k

    return coords


def vertical_lines(n: int):
    """
    Computes the positions of the basically vertical lines of the pseudo-polar grid.

    ## Parameters
    n : int
        Size of the image whose PPFFT we want to compute.

    ## Returns
    coords : np.ndarray
        Array of shape (n+1, 2*n+1, 2). coords[..., 0] gives the x coordinates.

    ## See Also
    horizontal_lines : Return the positions of the basically horizontal lines of the pseudo-polar grid.
    """
    m = 2 * n + 1
    coords = np.empty(shape=(n + 1, m, 2))

    for i_l, l in enumerate(domain(n + 1)):
        for i_k, k in enumerate(domain(m)):
            coords[i_l, i_k, 0] = k
            coords[i_l, i_k, 1] = -2 * l * k / n

    return coords
