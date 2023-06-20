import numpy as np

from ..tools.pad import pad
from ..reconstruction.polar_to_pp import (
    direct_2d_interp,
    polar_to_pseudopolar,
    new_polar_to_pseudopolar,
)
from ..inverse.fast_direct_inverse import fast_direct_inversion
from ..inverse.iterative_inverse import iterative_inverse


def fast_reconstruction(sino, precomputations, interp_mode="1d", angles=None, tol=1e-3):
    """
    The angles of the projections are supposed to be equispaced in [-pi/2, pi/2[
    """

    n_theta, n = sino.shape

    pad_sino = pad(sino, (n_theta, 2 * n + 1))
    fft_sinogram = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(pad_sino, axes=-1)), axes=-1
    )

    if interp_mode == "1d":
        if n_theta == n:
            hori_ppfft, vert_ppfft = new_polar_to_pseudopolar(fft_sinogram.T)

        elif n_theta == 2 * n:
            hori_ppfft, vert_ppfft = polar_to_pseudopolar(fft_sinogram.T)

        else:
            raise ValueError(
                "The number of projections must be equal to n or 2n in order to use the 1d interpolation mode."
            )

    # else, use direct 2d interpolation (slow)
    else:
        # if angles is not provided, assume they are equispaced in [0, \pi[
        if angles is None:
            angles = np.linspace(-np.pi / 2, np.pi / 2, num=n_theta, endpoint=False)

        p = np.arange(-n, n + 1)

        x_polar = p[None, :] * np.cos(np.pi / 2 + angles)[:, None]
        y_polar = p[None, :] * np.sin(np.pi / 2 + angles)[:, None]

        hori_ppfft, vert_ppfft = direct_2d_interp(fft_sinogram, x_polar, y_polar, n)

    if tol is None:
        return fast_direct_inversion(hori_ppfft, vert_ppfft, precomputations)

    else:
        res, info = iterative_inverse(hori_ppfft, vert_ppfft, tol)
        if info != 0:
            print("WARNING: convergence to tolerance not achieved")
        return res
