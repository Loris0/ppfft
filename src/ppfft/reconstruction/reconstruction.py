import numpy as np

from ppfft.reconstruction.polar_to_pp import polar_grid, direct_2d_interp
from ppfft.inverse.direct_inversion import direct_inversion_row
from ..radon_transforms.sinogram_resampling import (
    sinogram_kernel,
    hamming_window,
    compute_b,
    interpolate_sino,
    correct_pp_sinogram,
)
from ..radon_transforms.radon_transform import ppfft_from_radon, radon_from_legacy_radon
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
    Interpolates from polar to pseudo-polar in the Fourier domain.

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

    return direct_inversion_row(hori, vert, *precomputations, workers=workers)


def reconstruction_pp_sino(
    sino: np.ndarray, thetas: np.ndarray, precomputations: tuple
):
    """
    Was only tried with: thetas = np.linspace(-np.pi/4, 3*np.pi/4, num=n_theta)
    """
    n_theta, n = sino.shape
    ts = np.linspace(-0.5, 0.5, num=n)

    # Some parameters used for sinogram resampling.
    # These are the default ones of the paper.
    B = (2 * n + 2) // 8
    R = 1 / (np.sqrt(2) * np.pi)
    W = np.pi * n
    window_size = 8

    # Convolution kernel
    h_full = sinogram_kernel(ts, thetas, B, R, W)
    window = hamming_window(ts, thetas, 2 * window_size / n)
    h_window = window * h_full
    h_window /= np.sum(h_window)

    # Compute abstract coefficients.
    # Regularization can be between 1e-1 and 1e-3.
    b = compute_b(sino, h_window, reg=1e-3)

    # Pseudo-polar sinogram
    pp_sino = interpolate_sino(thetas, window_size, b, n, B, R, W)

    # Correction
    corrected_pp_sino = correct_pp_sinogram(pp_sino)

    # Getting the PPFFT. This uses the new convention.
    hori_ppfft, vert_ppfft = ppfft_from_radon(
        radon_from_legacy_radon(corrected_pp_sino)
    )

    # Inversion of PPFFT
    sol = direct_inversion_row(hori_ppfft, vert_ppfft, *precomputations)

    return sol
