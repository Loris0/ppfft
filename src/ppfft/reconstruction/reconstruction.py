import numpy as np

from ..tools.pad import pad
from ..tools.new_fft import new_fft
from ..reconstruction.polar_to_pp import direct_2d_interp
from ..inverse.direct_inverse import direct_inversion
from ..inverse.iterative_inverse import iterative_inverse


def reconstruction(sino, angles=None, tol=1e-3):
    """
    The angles of the projections are supposed to be equispaced in [-pi/2, pi/2[
    """

    n_theta, n = sino.shape

    pad_sino = pad(sino, (n_theta, 2 * n + 1))
    fft_sinogram = new_fft(pad_sino)

    if angles is None:
        angles = np.linspace(-np.pi / 2, np.pi / 2, num=n_theta, endpoint=False)

    p = np.arange(-n, n + 1)

    x_polar = p[None, :] * np.cos(np.pi / 2 + angles)[:, None]
    y_polar = p[None, :] * np.sin(np.pi / 2 + angles)[:, None]

    hori_ppfft, vert_ppfft = direct_2d_interp(fft_sinogram, x_polar, y_polar, n)

    if tol is None:
        return direct_inversion(hori_ppfft, vert_ppfft)

    else:
        res, info = iterative_inverse(hori_ppfft, vert_ppfft, tol)
        if info != 0:
            print("WARNING: convergence to tolerance not achieved")
        return res
