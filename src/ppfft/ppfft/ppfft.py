import numpy as np

from ..tools.pad import pad, adj_pad
from ..tools.new_fft import new_fft2, new_ifft, adj_new_fft2, adj_new_ifft
from ..tools.frac_fft import frac_fft, adj_frac_fft


def ppfft_horizontal(a: np.ndarray) -> np.ndarray:
    """
    Pseudo-Polar Fast Fourier Transform on the basically horizontal lines.

    ## Parameters
    a : np.ndarray
        Input array of shape (n, n) where n is even.

    ## Returns
    y : np.ndarray
        Ouput array of shape (n+1, 2n+1).
    """
    n, _ = a.shape
    m = 2 * n + 1

    res = np.empty((n + 1, m), dtype=np.complex64)
    Id_hat = new_fft2(pad(a, new_shape=(n, m)))
    for k in range(m):
        q = Id_hat[:, k]
        # frac_ft = frac_fft(pad(new_ifft(q), (m,)), 2 * (k - n), n)[n//2:-n//2]
        ifft_q = new_ifft(q)
        pad_ifft_q = pad(ifft_q, (m,))
        frac_pad_ifft_q = frac_fft(pad_ifft_q, 2 * (k - n), n)
        adj_pad_frac_ifft_q = adj_pad(frac_pad_ifft_q, (n + 1,))
        res[:, k] = adj_pad_frac_ifft_q[::-1]

    return res


def ppfft_vertical(a: np.ndarray) -> np.ndarray:
    """
    Pseudo-Polar Fast Fourier Transform on the basically vertical lines.

    ## Parameters
    a : np.ndarray
        Input array of shape (n, n) where n is even.

    ## Returns
    y : np.ndarray
        Ouput array of shape (n+1, 2n+1).

    ## See Also
    ppfft_horizontal : Return the PPFFT on the basically horizontal lines.
    """
    return ppfft_horizontal(a.T)


def ppfft(a: np.ndarray):
    """
    Pseudo-Polar Fast Fourier Transform.

    ## Parameters
    a : np.ndarray
        Input array of shape (n, n) where n is even.

    ## Returns
    hori : np.ndarray
        Horizontal ppfft of shape (n+1, 2n+1)
    vert : np.ndarray
        Vertical ppfft of shape (n+1, 2n+1)
    """
    return ppfft_horizontal(a), ppfft_vertical(a)


def adj_ppfft(hori_ppfft: np.ndarray, vert_ppfft: np.ndarray) -> np.ndarray:
    """
    Adjoint operator of ``ppfft``.
    """
    n, m = hori_ppfft.shape[0] - 1, hori_ppfft.shape[1]
    hori_aux = np.empty(shape=(n, m), dtype=complex)
    vert_aux = np.empty(shape=(n, m), dtype=complex)

    for k in range(m):
        q = hori_ppfft[::-1, k]
        q = pad(q, (m,))
        q = adj_frac_fft(q, 2 * (k - n), n)
        q = adj_pad(q, (n,))
        q = adj_new_ifft(q)
        hori_aux[:, k] = q

        q = vert_ppfft[::-1, k]
        q = pad(q, (m,))
        q = adj_frac_fft(q, 2 * (k - n), n)
        q = adj_pad(q, (n,))
        q = adj_new_ifft(q)
        vert_aux[:, k] = q

    hori_aux = adj_new_fft2(hori_aux)
    hori_a = adj_pad(hori_aux, (n, n))

    vert_aux = adj_new_fft2(vert_aux)
    vert_a = adj_pad(vert_aux, (n, n))
    vert_a = np.transpose(vert_a)

    return hori_a + vert_a


def horizontal_lines(n: int):
    """
    Computes the positions of the basically horizontal lines of the pseudo-polar grid.

    ## Parameters
    n : int
        Size of the image whose PPFFT we want to compute.

    ## Returns
    coords : np.ndarray
        Array of shape (n+1, 2*n+1, 2). coords[..., 0] gives the x coordinates.

    ## See Also
    vertical_lines : Return the positions of the basically vertical lines of the pseudo-polar grid.
    """

    coords = np.empty(shape=(n + 1, 2 * n + 1, 2))

    for l in range(n + 1):
        for k in range(2 * n + 1):
            coords[l, k, 0] = -2 * (l - n // 2) * (k - n) / n
            coords[l, k, 1] = k - n

    return coords


def vertical_lines(n: int):
    """
    Computes the positions of the basically vertical lines of the pseudo-polar grid.

    ## Parameters
    n : int
        Size of the image whose PPFFT we want to compute.

    ## Returns
    coords : np.ndarray
        Array of shape (n+1, 2*n+1, 2). coords[..., 0] gives the x coordinates.

    ## See Also
    horizontal_lines : Return the positions of the basically horizontal lines of the pseudo-polar grid.
    """

    coords = np.empty(shape=(n + 1, 2 * n + 1, 2))

    for l in range(n + 1):
        for k in range(2 * n + 1):
            coords[l, k, 0] = k - n
            coords[l, k, 1] = -2 * (l - n // 2) * (k - n) / n

    return coords
