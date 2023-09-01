import numpy as np
import scipy.fft as fft


def hamming_window(ts, thetas, window_size):
    t_grid, theta_grid = np.meshgrid(ts, thetas)
    r = np.sqrt(t_grid**2 + (theta_grid / np.pi) ** 2)
    w = np.zeros_like(r)
    w[r < window_size / 2] = 0.54 + 0.46 * np.cos(
        2 * np.pi * r[r < window_size / 2] / window_size
    )
    return w


def sinogram_kernel(ts, thetas, B, R, W):
    t_grid, theta_grid = np.meshgrid(ts, thetas)
    res = np.zeros_like(t_grid)

    def aux1(mask, t, theta):
        a = (np.cos(W * t - theta * (B + R * W)) - np.cos(B * theta)) / (t - theta * R)
        b = (np.cos(W * t + theta * (B + R * W)) - np.cos(B * theta)) / (t + theta * R)
        res[mask] = (a - b) / (np.pi * theta)

    def aux2(mask, theta):
        res[mask] = (
            2 * theta * R * W * np.sin(theta * B)
            + np.cos(theta * B)
            - np.cos(theta * (B + 2 * R * W))
        ) / (2 * np.pi * R * theta**2)

    def aux3(mask, t):
        res[mask] = (
            2 * t * np.sin(W * t) * (B + R * W) + 2 * R * (np.cos(W * t) - 1)
        ) / (np.pi * t**2)

    def aux4(mask):
        res[mask] = 2 / np.pi * W * (B + (R * W) / 2)

    mask1 = theta_grid == 0
    mask2 = t_grid == 0
    mask3 = (abs(t_grid) - abs(theta_grid * R)) == 0
    aux4(mask1 & mask2)
    aux3(mask1 & (~mask2), t_grid[mask1 & (~mask2)])
    aux2(~mask1 & mask3, theta_grid[~mask1 & mask3])
    aux1(~mask1 & ~mask3, t_grid[~mask1 & ~mask3], theta_grid[~mask1 & ~mask3])

    return res


def compute_b(sinogram, q, reg=1e-6):
    fft2_sino = fft.fft2(sinogram)
    fft2_q = np.abs(fft.fft2(q))
    res = fft.ifft2(fft2_q * fft2_sino / (fft2_q**2 + reg))
    return res.real


def interpolate_sino(thetas, window_size, b, n, B, R, W):
    """
    b is FiltSinogram
    """

    n_theta, n_r = b.shape

    l = np.arange(-(n // 2), n // 2 + 1)
    thetas_pp = np.concatenate(
        (np.arctan(2 * l / n), np.pi / 2 - np.arctan(2 * l / n)[::-1])
    )

    pp_sinogram = np.zeros(shape=(2 * n + 2, 2 * n + 1))  # theta, r
    Mpp, Npp = 2 * n + 1, 2 * n + 2

    # Debugging

    for l in range(Npp):  # theta
        for k in range(Mpp):  # r
            theta_pp = thetas_pp[l]

            if (theta_pp >= -np.pi / 4) and (theta_pp < np.pi / 4):
                K = k - 1 - Mpp / 2
                T = np.abs(np.cos(theta_pp))
            else:
                K = k - 1 - (Mpp + 2 * (theta_pp - np.pi / 4) / (np.pi / 4)) / 2
                T = np.abs(np.sin(theta_pp))

            I = np.arange(1, n_r + 1) - 1 - (n_r + 1) / 2
            d_theta = theta_pp - thetas
            d_t = (K * T - I) / n_r
            d_t -= abs(np.mod(theta_pp + np.pi / 4, np.pi / 2) - np.pi / 4) / (
                np.pi * n_r
            )

            min_theta_idx = np.argmin(np.abs(d_theta))
            min_t_idx = np.argmin(np.abs(d_t))
            delta = int(np.ceil(window_size))

            j = np.arange(-delta, delta + 1) + min_t_idx
            j = j[(j >= 0) & (j < len(d_t))]
            i = np.arange(-delta, delta + 1) + min_theta_idx
            i = i[(i >= 0) & (i < len(d_theta))]

            d_Theta, d_T = np.meshgrid(d_theta[i], d_t[j], indexing="ij")
            Rad = np.sqrt(d_T**2 + (d_Theta / np.pi) ** 2)
            Window = 0.54 + 0.46 * np.cos(2 * np.pi * Rad / window_size * n)
            Window[Rad >= window_size / (2 * n)] = 0

            h = T * Window * sinogram_kernel(d_t[j], d_theta[i], B, R, W) / np.size(b)

            pp_sinogram[l, k] = np.sum(h * b[np.ix_(i, j)])

    return pp_sinogram


def correct_pp_sinogram(pp_sino: np.ndarray) -> np.ndarray:
    """Empirical corrections to resampled pseudo-polar sinogram.

    Parameters
    ----------
    pp_sino : np.ndarray
        Pseudo-polar sinogram, output of `interpolate_sino`.

    Returns
    -------
    np.ndarray
        Corrected sinogram.
    """
    corrected_pp_sino = (
        np.einsum("tr,t->tr", pp_sino, 1 / np.sum(pp_sino, axis=1)) * pp_sino[0].sum()
    )
    return np.roll(corrected_pp_sino.T, -1, axis=0)


def old_sinogram_kernel(ts, thetas, B, R, W):
    def aux(t, theta):
        if theta == 0:
            if t == 0:
                return 2 / np.pi * W * (B + (R * W) / 2)
            else:
                return (
                    2 * t * np.sin(W * t) * (B + R * W) + 2 * R * (np.cos(W * t) - 1)
                ) / (np.pi * t**2)

        elif abs(t) == abs(theta * R):
            return (
                2 * theta * R * W * np.sin(theta * B)
                + np.cos(theta * B)
                - np.cos(theta * (B + 2 * R * W))
            ) / (2 * np.pi * R * theta**2)

        else:
            v1 = (np.cos(W * t - theta * (B + R * W)) - np.cos(B * theta)) / (
                t - theta * R
            )
            v2 = (np.cos(W * t + theta * (B + R * W)) - np.cos(B * theta)) / (
                t + theta * R
            )
            return (v1 - v2) / (np.pi * theta)

    a = np.zeros(shape=(len(thetas), len(ts)))

    for i, theta in enumerate(thetas):
        for j, t in enumerate(ts):
            a[i, j] = aux(t, theta)

    return a
