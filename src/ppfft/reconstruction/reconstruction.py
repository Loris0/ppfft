import numpy as np

from ppfft.reconstruction.polar_to_pp import polar_grid, direct_2d_interp
from ppfft.inverse.direct_inversion import direct_inversion

from ppfft.tools.new_fft import new_fft
from ppfft.tools.pad import pad


def reconstruction(
    sino: np.ndarray,
    thetas: np.ndarray,
    precomputations: tuple,
    oversampling: int = None,
    workers: int = None,
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
    workers: int, optional
        Maximum number of workers to use for parallel computation. If negative, takes the value `os.cpu_count()`.

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
    fft_sinogram = new_fft(pad_sino, workers=workers)  # polar samples

    # Interpolation polar -> pseudo-polar
    polar_x, polar_y = polar_grid(
        np.pi / 2 + thetas, new_n
    )  # pi / 2 is here because of the convention of silx Projection.

    hori, vert = direct_2d_interp(fft_sinogram, polar_x, polar_y, n)

    return direct_inversion(hori, vert, precomputations, workers=workers)
