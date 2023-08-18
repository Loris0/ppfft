import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

from ppfft.tools.grids import domain


def horizontal_grid(n: int) -> np.ndarray:
    dom = domain(n + 1)
    horizontal_x = -2 * dom[None, :] * dom[:, None] / (n * (n + 1))
    horizontal_y = np.tile(dom[:, None], (1, n + 1)) / (n + 1)
    return horizontal_x, horizontal_y


def vertical_grid(n: int) -> np.ndarray:
    horizontal_x, horizontal_y = horizontal_grid(n)
    return horizontal_y, horizontal_x


def polar_grid(thetas, n_r):
    rs = domain(n_r) / n_r

    polar_x = np.cos(thetas)[:, None] * rs[None, :]
    polar_y = np.sin(thetas)[:, None] * rs[None, :]

    return polar_x, polar_y


def direct_2d_interp(
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

    hori_x, hori_y = horizontal_grid(n)

    hori_ppfft = interpolator(hori_x, hori_y)
    vert_ppfft = interpolator(hori_y, hori_x)

    return hori_ppfft, vert_ppfft
