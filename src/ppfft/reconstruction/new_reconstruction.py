import numpy as np

from ppfft.reconstruction.new_polar_to_pp import polar_grid, new_direct_2d_interp
from ppfft.inverse.new_direct_inversion import (
    new_direct_inversion,
    precompute_onion_peeling,
)
from ppfft.tools.new_fft import new_fft
from ppfft.tools.pad import pad


def new_reconstruction(
    sino: np.ndarray,
    thetas: np.ndarray,
    precomputations: tuple,
    oversampling: int = None,
) -> np.ndarray:
    """Tomographic reconstruction from sinogram.

    Parameters
    ----------
    sino : np.ndarray
        Sinogram. Shape: (n_theta, n).
    thetas : np.ndarray
        Angles of the projections (argument of `silx.image.Projection`).
    precomputations : tuple
        Output of `precompute_onion_peeling(n)`.
    oversampling : int, optional
        Radial size of zero-padded sinogram, by default None: means 2 * n.

    Returns
    -------
    np.ndarray
        Reconstructed image. Shape: (n, n)
    """
    n_theta, n = sino.shape

    # The sinogram is zero-padded in the radial direction.
    # This improves the interpolation.
    # By default, we pad from size n to size 2 * n.
    if oversampling is None:
        new_n = 2 * n
    else:
        new_n = oversampling

    pad_sino = pad(sino, (n_theta, new_n))
    fft_sinogram = new_fft(pad_sino)  # polar samples

    # Interpolation polar -> pseudo-polar
    polar_x, polar_y = polar_grid(
        np.pi / 2 + thetas, new_n
    )  # pi / 2 is here because of the convention of silx Projection.

    hori, vert = new_direct_2d_interp(fft_sinogram, polar_x, polar_y, n)

    return new_direct_inversion(hori, vert, precomputations)
