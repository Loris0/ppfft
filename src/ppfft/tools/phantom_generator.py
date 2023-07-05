import numpy as np


def generate_phantom(n: int) -> np.ndarray:
    """Creates a Shepp-Logan phantom of size (n, n).

    Parameters
    ----------
    n : int
        Size of the image returned.

    Returns
    -------
    out : np.ndarray
        (n, n) Shepp-Logan phantom.
    """
    x, y = np.linspace(-1, 1, n), np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, y)
    res = np.zeros_like(xx)

    A_list = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    a_list = [0.69, 0.6624, 0.11, 0.16, 0.21, 0.046, 0.046, 0.046, 0.023, 0.023]
    b_list = [0.92, 0.874, 0.31, 0.41, 0.25, 0.046, 0.046, 0.023, 0.023, 0.046]
    phi_list = [0, 0, -18 * np.pi / 180, 18 * np.pi / 180, 0, 0, 0, 0, 0, 0]
    x0_list = [0, 0, 0.22, -0.22, 0, 0, 0, -0.08, 0, 0.06]
    y0_list = [0, -0.0184, 0, 0, 0.35, 0.1, -0.1, -0.605, -0.605, -0.605]

    for A, a, b, phi, x0, y0 in zip(A_list, a_list, b_list, phi_list, x0_list, y0_list):
        rot_x = xx * np.cos(phi) + yy * np.sin(phi)
        rot_y = -np.sin(phi) * xx + yy * np.cos(phi)
        mask = (rot_x - x0) ** 2 / a**2 + (rot_y - y0) ** 2 / b**2 <= 1
        res[mask] += A

    return res[::-1]
