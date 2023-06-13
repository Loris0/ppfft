import numpy as np
from scipy.interpolate import interp1d, CloughTocher2DInterpolator


"""
Grid definitions.
"""


def horizontal_lines(n: int):
    coords = np.empty(shape=(n + 1, 2 * n + 1, 2))

    for l in range(n + 1):
        for k in range(2 * n + 1):
            coords[l, k, 0] = -2 * (l - n // 2) * (k - n) / n
            coords[l, k, 1] = k - n

    return coords


def vertical_lines(n: int):
    coords = np.empty(shape=(n + 1, 2 * n + 1, 2))

    for l in range(n + 1):
        for k in range(2 * n + 1):
            coords[l, k, 0] = k - n
            coords[l, k, 1] = -2 * (l - n // 2) * (k - n) / n

    return coords


def polar_coordinates(n: int):
    p = np.arange(-n, n + 1)
    q = np.arange(0, 2 * n)

    x = p[:, None] * np.cos(np.pi * q / (2 * n))[None, :]
    y = p[:, None] * np.sin(np.pi * q / (2 * n))[None, :]

    return np.stack((x, y), axis=-1)


def polar_fourier_transform(im):
    n = len(im)
    m = 2 * n + 1
    half_n = n // 2
    p = np.arange(-n, n + 1) / m
    q = np.arange(2 * n)
    u, v = np.arange(-half_n, half_n), np.arange(-half_n, half_n)

    cos = -2j * np.pi * np.einsum("k,u,l->kul", p, u, np.cos(np.pi * q / (2 * n)))
    sin = -2j * np.pi * np.einsum("k,v,l->kvl", p, v, np.sin(np.pi * q / (2 * n)))

    res = np.einsum("uv,kul,kvl->kl", im, np.exp(cos), np.exp(sin))

    return res


"""
Interpolation in two steps, when n_theta = 2n.
"""


def interpolate_vert_ray(q, ray_samples, kind="cubic"):
    """
    Interpolate rays corresponding to the vertical PPFFT.
    0 <= q <= n/2 or n + n/2 <= q <= 2n - 1

    ``ray_samples`` always goes from the bottom of the circle to the top.
    """
    n = (len(ray_samples) - 1) // 2

    ray_pos = np.arange(-n, n + 1)  # positions of the known points along the line
    theta = np.pi * q / (2 * n)
    target_pos = np.arange(-n, n + 1) * np.sqrt(1 + np.tan(theta) ** 2)

    # Interpolate and put zeros outside of the polar disk
    interpolator = interp1d(
        ray_pos, ray_samples, bounds_error=False, fill_value=0, kind=kind
    )

    if 0 <= q <= n // 2:
        res = interpolator(target_pos)
    else:
        res = interpolator(target_pos)[::-1]

    return res


def interpolate_all_vert_rays(polar_ft, kind="cubic"):
    """
    Computes the first step towards the vertical PPFFT.
    """
    n = np.shape(polar_ft)[1] // 2

    res = np.zeros(shape=(2 * n + 1, n + 1), dtype=complex)

    i = 0
    for q in range(n + n // 2, 2 * n):
        res[:, i] = interpolate_vert_ray(q, polar_ft[:, q], kind)
        i += 1

    res[:, i] = polar_ft[:, 0]  # theta = 0, horizontal coordinates
    i += 1

    for q in range(1, n // 2 + 1):
        res[:, i] = interpolate_vert_ray(q, polar_ft[:, q], kind)
        i += 1

    return res[::, ::-1]


def interpolate_hori_ray(q, ray_samples, kind="cubic"):
    """
    Interpolate rays corresponding to the horizontal PPFFT.
    n/2 <= q <= n + n/2
    """
    n = (len(ray_samples) - 1) // 2

    ray_pos = np.arange(-n, n + 1)  # positions of the known points along the line
    theta = np.pi * q / (2 * n)
    target_pos = np.arange(-n, n + 1) * np.sqrt(1 + 1 / np.tan(theta) ** 2)

    # Interpolate and put zeros outside of the polar disk
    interpolator = interp1d(
        ray_pos, ray_samples, bounds_error=False, fill_value=0, kind=kind
    )

    return interpolator(target_pos)


def interpolate_all_hori_rays(polar_ft, kind="cubic"):
    """
    Computes the first step towards the horizontal PPFFT.
    """
    n = np.shape(polar_ft)[1] // 2

    res = np.zeros(shape=(2 * n + 1, n + 1), dtype=complex)

    i = 0

    for q in range(n // 2, n):
        res[:, i] = interpolate_hori_ray(q, polar_ft[:, q], kind)
        i += 1

    res[:, i] = polar_ft[:, n]
    i += 1

    for q in range(n + 1, n + n // 2 + 1):
        res[:, i] = interpolate_hori_ray(q, polar_ft[:, q], kind)
        i += 1

    return res


def interpolate_vert_angle(k, samples, kind="cubic"):
    """
    Reconstruct one column of the horizontal PPFFT.
    -n <= k <= n
    """
    n = len(samples) - 1

    # samples at x = k
    # and angles pi * q / 2n with q = n/2, ... 0, 2n-1, ..., n + n/2
    # meaning the y positions are: x_q = k * tan(theta_q)
    q = np.concatenate(
        (np.arange(0, n // 2 + 1)[::-1], np.arange(n + n // 2, 2 * n)[::-1])
    )
    samples_pos = k * np.tan(np.pi * q / (2 * n))

    # Initialize interpolator
    interpolator = interp1d(samples_pos, samples, kind=kind)

    # The target y positions are: y_l = -2lk / n with -n/2 <= l <= n/2
    target_pos = np.arange(-(n // 2), n // 2 + 1) * (-2 * k / n)

    # Compute result
    res = np.zeros_like(target_pos, dtype=complex)
    res[1:-1] = interpolator(target_pos[1:-1])
    res[0] = samples[0]
    res[-1] = samples[-1]

    return res


def interpolate_hori_angle(k, samples, kind="cubic"):
    """
    Reconstruct one line of the horizontal PPFFT.
    -n <= k <= n
    """
    n = len(samples) - 1

    # samples at y = k
    # and angles pi * q / 2n with n/2 <= q <= n + n/2
    # meaning the x positions are: x_q = k / tan(theta_q)
    samples_pos = k / np.tan(np.pi * np.arange(n // 2, n + n // 2 + 1) / (2 * n))

    # Initialize interpolator
    interpolator = interp1d(samples_pos, samples, kind=kind)

    # The target x positions are: x_l = -2lk / n with -n/2 <= l <= n/2
    target_pos = np.arange(-(n // 2), n // 2 + 1) * (-2 * k / n)

    # Compute result
    res = np.zeros_like(target_pos, dtype=complex)
    res[0] = samples[0]  # the first point is already known
    res[-1] = samples[-1]  # the last one too
    res[1:-1] = interpolator(target_pos[1:-1])

    return res


def interpolate_all_vert_angles(vert_rays, kind="cubic"):
    """
    ``vert_rays`` is the ouput of ``interpolate_all_vert_rays``.
    Its shape is (2n+1, n+1)
    """
    n = np.shape(vert_rays)[1] - 1
    vert_ppfft = np.zeros_like(vert_rays, dtype=complex)

    for k in np.arange(-n, n + 1):
        if k != 0:
            samples = vert_rays[k + n]
            vert_ppfft[k + n] = interpolate_vert_angle(k, samples, kind)
        else:
            vert_ppfft[k + n] = vert_rays[k + n]

    return vert_ppfft.T


def interpolate_all_hori_angles(hori_rays, kind="cubic"):
    """
    ``hori_rays`` is the ouput of ``interpolate_all_hori_rays``.
    Its shape is (2n+1, n+1)
    """
    n = np.shape(hori_rays)[1] - 1
    hori_ppfft = np.zeros_like(hori_rays, dtype=complex)

    for k in np.arange(-n, n + 1):
        if k != 0:
            samples = hori_rays[k + n]
            hori_ppfft[k + n] = interpolate_vert_angle(k, samples, kind)
        else:
            hori_ppfft[k + n] = hori_rays[k + n]

    return hori_ppfft.T


def polar_to_pseudopolar(polar_ft, kind="cubic"):
    vert_rays = interpolate_all_vert_rays(polar_ft, kind)
    hori_rays = interpolate_all_hori_rays(polar_ft, kind)
    vert_ppfft = interpolate_all_vert_angles(vert_rays, kind)
    hori_ppfft = interpolate_all_hori_angles(hori_rays, kind)

    return hori_ppfft, vert_ppfft


"""
Direct 2D interpolation (very slow)
"""


def direct_2d_interp(polar_ft, x, y, n, interp_fun=CloughTocher2DInterpolator):
    """
    2d interpolation from polar gird to pseudo-polar.

    ## Parameters
    polar_ft : np.ndarray
        Samples of the polar Fourier transform. Shape: (n_r, n_theta).
    x : np.ndarray
        x coordinates of the polar grid. Shape: (n_r, n_theta).
    y : np.ndarray
        y coordinates of the polar grid. Shape: (n_r, n_theta).
    n : int
        Size of the original image.
    interp_fun : class, optional
        2d Interpolator used.

    ## Returns
    hori_ppfft : np.ndarray
        Inteprolated horizontal ppfft. Shape: (n+1, 2n+1).
    vert_ppfft : np.ndarray
        Inteprolated vertical ppfft. Shape: (n+1, 2n+1).
    """
    points = np.stack((x.flatten(), y.flatten()), axis=-1)
    interpolator = interp_fun(points, polar_ft.flatten(), fill_value=0)

    hori_pos = horizontal_lines(n)
    vert_pos = vertical_lines(n)

    hori_ppfft = interpolator(hori_pos[..., 0], hori_pos[..., 1])
    vert_ppfft = interpolator(vert_pos[..., 0], vert_pos[..., 1])

    return hori_ppfft, vert_ppfft


"""
Interpolation in two steps, when n_theta = n
"""


def new_interpolate_vert_ray(q, ray_samples, kind="cubic"):
    """
    Interpolate rays corresponding to the vertical PPFFT.
    0 <= q <= n // 4 or n - n // 4 <= q <= n - 1

    ``ray_samples`` always goes from the bottom of the circle to the top.
    """
    n = (len(ray_samples) - 1) // 2

    ray_pos = np.arange(-n, n + 1)  # positions of the known points along the line
    theta = np.pi * q / n
    target_pos = np.arange(-n, n + 1) * np.sqrt(1 + np.tan(theta) ** 2)

    # Interpolate and put zeros outside of the polar disk
    interpolator = interp1d(
        ray_pos, ray_samples, bounds_error=False, fill_value=0, kind=kind
    )

    if 0 <= q <= n // 4:
        res = interpolator(target_pos)
    else:
        res = interpolator(target_pos)[::-1]

    return res


def new_interpolate_all_vert_rays(polar_ft, kind="cubic"):
    """
    Computes the first step towards the vertical PPFFT.

    polar_ft has shape: (2n + 1, n)
    """
    n = np.shape(polar_ft)[1]
    q_n, r_n = divmod(n, 4)

    res = np.zeros(shape=(2 * n + 1, 2 * q_n + 1), dtype=complex)

    i = 0
    for q in range(n - q_n, n):
        res[:, i] = new_interpolate_vert_ray(q, polar_ft[:, q], kind)
        i += 1

    res[:, i] = polar_ft[:, 0]  # theta = 0, horizontal coordinates
    i += 1

    for q in range(1, q_n + 1):
        res[:, i] = new_interpolate_vert_ray(q, polar_ft[:, q], kind)
        i += 1

    return res[::, ::-1]


def new_interpolate_hori_ray(q, ray_samples, kind="cubic"):
    """
    Interpolate rays corresponding to the horizontal PPFFT.
    n // 4 + ((n % 4) != 0) <= q <= n - n // 4 - ((n % 4) != 0)
    """
    n = (len(ray_samples) - 1) // 2

    ray_pos = np.arange(-n, n + 1)  # positions of the known points along the line
    theta = np.pi * q / n
    target_pos = np.arange(-n, n + 1) * np.sqrt(1 + 1 / np.tan(theta) ** 2)

    # Interpolate and put zeros outside of the polar disk
    interpolator = interp1d(
        ray_pos, ray_samples, bounds_error=False, fill_value=0, kind=kind
    )

    return interpolator(target_pos)


def new_interpolate_all_hori_rays(polar_ft, kind="cubic"):
    """
    Computes the first step towards the horizontal PPFFT.
    """
    n = np.shape(polar_ft)[1]
    q_n, r_n = divmod(n, 4)

    res = np.zeros(shape=(2 * n + 1, 2 * q_n + 1), dtype=complex)

    i = 0

    for q in range(n // 4 + (r_n != 0), n // 2):
        res[:, i] = new_interpolate_hori_ray(q, polar_ft[:, q], kind)
        i += 1

    res[:, i] = polar_ft[:, n // 2]
    i += 1

    for q in range(n // 2 + 1, n - n // 4 - (r_n != 0) + 1):
        res[:, i] = new_interpolate_hori_ray(q, polar_ft[:, q], kind)
        i += 1

    return res


def new_interpolate_vert_angle(k, n, samples, kind="cubic"):
    """
    Reconstruct one column of the horizontal PPFFT.
    -n <= k <= n
    """
    q_n, r_n = divmod(n, 4)

    # samples at x = k
    # and angles pi * q / 2n with q = n/2, ... 0, 2n-1, ..., n + n/2
    # meaning the y positions are: x_q = k * tan(theta_q)
    q = np.concatenate((np.arange(0, q_n + 1)[::-1], np.arange(n - q_n, n)[::-1]))
    samples_pos = k * np.tan(np.pi * q / n)

    # Initialize interpolator
    interpolator = interp1d(
        samples_pos, samples, bounds_error=False, fill_value=0, kind=kind
    )

    # The target y positions are: y_l = -2lk / n with -n/2 <= l <= n/2
    target_pos = np.arange(-(n // 2), n // 2 + 1) * (-2 * k / n)

    # Compute result
    res = np.zeros_like(target_pos, dtype=complex)
    res[1:-1] = interpolator(target_pos[1:-1])
    res[0] = samples[0]
    res[-1] = samples[-1]

    return res


def new_interpolate_hori_angle(k, n, samples, kind="cubic"):
    """
    Reconstruct one line of the horizontal PPFFT.
    -n <= k <= n
    """
    q_n, r_n = divmod(n, 4)

    # samples at y = k
    # and angles pi * q / 2n with n/2 <= q <= n + n/2
    # meaning the x positions are: x_q = k / tan(theta_q)
    samples_pos = k / np.tan(
        np.pi * np.arange(q_n + (r_n != 0), n - q_n - (r_n != 0) + 1) / n
    )

    # Initialize interpolator
    interpolator = interp1d(
        samples_pos, samples, bounds_error=False, fill_value=0, kind=kind
    )

    # The target x positions are: x_l = -2lk / n with -n/2 <= l <= n/2
    target_pos = np.arange(-(n // 2), n // 2 + 1) * (-2 * k / n)

    # Compute result
    res = np.zeros_like(target_pos, dtype=complex)
    res[0] = samples[0]  # the first point is already known
    res[-1] = samples[-1]  # the last one too
    res[1:-1] = interpolator(target_pos[1:-1])

    return res


def new_interpolate_all_vert_angles(vert_rays, kind="cubic"):
    """
    ``vert_rays`` is the ouput of ``interpolate_all_vert_rays``.
    Its shape is (2n+1, 2 * (n // 4) + 1)
    """
    n = np.shape(vert_rays)[0] // 2
    vert_ppfft = np.zeros(shape=(2 * n + 1, n + 1), dtype=complex)

    for k in np.arange(-n, n + 1):
        if k != 0:
            samples = vert_rays[k + n]
            vert_ppfft[k + n] = new_interpolate_vert_angle(k, n, samples, kind)
        else:
            vert_ppfft[k + n] = vert_rays[k + n, n // 4]

    return vert_ppfft.T


def new_interpolate_all_hori_angles(hori_rays, kind="cubic"):
    """
    ``hori_rays`` is the ouput of ``interpolate_all_hori_rays``.
    Its shape is (2n+1, 2 * (n // 4) + 1)
    """
    n = np.shape(hori_rays)[0] // 2
    hori_ppfft = np.zeros(shape=(2 * n + 1, n + 1), dtype=complex)

    for k in np.arange(-n, n + 1):
        if k != 0:
            samples = hori_rays[k + n]
            hori_ppfft[k + n] = new_interpolate_vert_angle(k, n, samples, kind)
        else:
            hori_ppfft[k + n] = hori_rays[k + n, n // 4]

    return hori_ppfft.T


def new_polar_to_pseudopolar(polar_ft, kind="cubic"):
    vert_rays = new_interpolate_all_vert_rays(polar_ft, kind)
    hori_rays = new_interpolate_all_hori_rays(polar_ft, kind)
    vert_ppfft = new_interpolate_all_vert_angles(vert_rays, kind)
    hori_ppfft = new_interpolate_all_hori_angles(hori_rays, kind)

    return hori_ppfft, vert_ppfft
