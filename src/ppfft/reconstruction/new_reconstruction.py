import numpy as np

from .new_polar_to_pp import new_direct_2d_interp
from ..inverse.new_direct_inversion import new_direct_inversion

from ..tools.grids import domain, polar_grid
from ..tools.new_fft import new_fft
from ..tools.pad import pad


def new_reconstruction(
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
        Radial size of zero-padded sinogram, by default None: means 4 * n.
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
        n_r = 4 * n
    else:
        n_r = oversampling

    angles = np.pi / 2 + thetas

    pad_sino = pad(sino, (n_theta, n_r))
    fft_sinogram = new_fft(pad_sino, workers=workers)  # polar samples
    fft_sinogram *= np.exp(
        -2j
        * np.pi
        * (domain(n_r) / n_r)[None, :]
        * (n // 2)
        * (np.cos(angles) + np.sin(angles))[:, None]
    )

    # Interpolation polar -> pseudo-polar
    polar_x, polar_y = polar_grid(
        angles, n_r
    )  # pi / 2 is here because of the convention of silx Projection.

    hori, vert = new_direct_2d_interp(fft_sinogram, polar_x, polar_y, n)

    return new_direct_inversion(hori, vert, precomputations, workers=workers)
