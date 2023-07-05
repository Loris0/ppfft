import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

from ppfft.tools.grids import domain


def horizontal_grid(n: int) -> np.ndarray:
    coords = np.empty(shape=(n + 1, n + 1, 2))

    for i_l, l in enumerate(domain(n + 1)):
        for i_k, k in enumerate(domain(n + 1)):
            coords[i_k, i_l, 0] = -2 * l * k / (n * (n + 1))
            coords[i_k, i_l, 1] = k / (n + 1)

    return coords


def vertical_grid(n: int) -> np.ndarray:
    coords = np.empty(shape=(n + 1, n + 1, 2))

    for i_l, l in enumerate(domain(n + 1)):
        for i_k, k in enumerate(domain(n + 1)):
            coords[i_k, i_l, 0] = k / (n + 1)
            coords[i_k, i_l, 1] = -2 * l * k / (n * (n + 1))

    return coords


def polar_grid(thetas, n_r):
    n_theta = len(thetas)
    rs = domain(n_r) / n_r
    coords = np.empty(shape=(n_theta, n_r, 2))

    coords[..., 0] = np.cos(thetas)[:, None] * rs[None, :]
    coords[..., 1] = np.sin(thetas)[:, None] * rs[None, :]

    return coords


def new_direct_2d_interp(
    polar_ft, polar_x, polar_y, n, interp_fun=CloughTocher2DInterpolator
):
    """
    2d interpolation from polar gird to pseudo-polar.

    ## Parameters
    polar_ft : np.ndarray
        Samples of the polar Fourier transform. Shape: (n_theta, n_r).
    x : np.ndarray
        x coordinates of the polar grid. Shape: (n_theta, n_r).
    y : np.ndarray
        y coordinates of the polar grid. Shape: (n_theta, n_r).
    n : int
        Size of the original image.
    interp_fun : class, optional
        2d Interpolator used.

    ## Returns
    hori_ppfft : np.ndarray
        Inteprolated horizontal ppfft. Shape: (n+1, n+1).
    vert_ppfft : np.ndarray
        Inteprolated vertical ppfft. Shape: (n+1, n+1).
    """
    points = np.stack((polar_x.flatten(), polar_y.flatten()), axis=-1)
    interpolator = interp_fun(points, polar_ft.flatten(), fill_value=0)

    hori_pos = horizontal_grid(n)
    vert_pos = vertical_grid(n)

    hori_ppfft = interpolator(hori_pos[..., 0], hori_pos[..., 1])
    vert_ppfft = interpolator(vert_pos[..., 0], vert_pos[..., 1])

    return hori_ppfft, vert_ppfft
