import numpy as np
import pytest

from ppfft.ppfft.ppfft import ppfft_horizontal, ppfft_vertical
from ppfft.ppfft.ppfft import ppfft, adj_ppfft


def true_ppfft_horizontal(a):
    n, _ = a.shape
    m = 2 * n + 1

    ls = np.arange(-(n // 2), n // 2 + 1)
    ks = np.arange(-n, n + 1)

    u_coords = -2 * ls[:, None] * ks[None, :] / n
    v_coords = np.tile(ks, (n + 1, 1))

    u = np.arange(-(n // 2), n // 2)
    v = np.arange(-(n // 2), n // 2)

    aux_u = np.einsum("u,lk->ulk", u, u_coords)
    aux_v = np.einsum("v,lk->vlk", v, v_coords)

    return np.einsum(
        "uv,ulk,vlk->lk",
        a,
        np.exp(-2j * np.pi * aux_u / m),
        np.exp(-2j * np.pi * aux_v / m),
    )


@pytest.mark.parametrize("n", [100, 102])
def test_ppfft_horizontal(n):
    im = np.random.rand(n, n)
    assert np.allclose(ppfft_horizontal(im), true_ppfft_horizontal(im))


def true_ppfft_vertical(a):
    n, _ = a.shape
    m = 2 * n + 1

    ls = np.arange(-(n // 2), n // 2 + 1)
    ks = np.arange(-n, n + 1)

    v_coords = -2 * ls[:, None] * ks[None, :] / n
    u_coords = np.tile(ks, (n + 1, 1))

    u = np.arange(-(n // 2), n // 2)
    v = np.arange(-(n // 2), n // 2)

    aux_u = np.einsum("u,lk->ulk", u, u_coords)
    aux_v = np.einsum("v,lk->vlk", v, v_coords)

    return np.einsum(
        "uv,ulk,vlk->lk",
        a,
        np.exp(-2j * np.pi * aux_u / m),
        np.exp(-2j * np.pi * aux_v / m),
    )


@pytest.mark.parametrize("n", [100, 102])
def test_ppfft_vertical(n):
    im = np.random.rand(n, n)
    assert np.allclose(ppfft_vertical(im), true_ppfft_vertical(im))


@pytest.mark.parametrize("n", [100, 102])
def test_adjoint_ppfft(n):
    im1 = np.random.rand(n, n)
    im2 = np.random.rand(n + 1, 2 * n + 1)
    im3 = np.random.rand(n + 1, 2 * n + 1)

    h, v = ppfft(im1)

    prod1 = np.vdot(h, im2) + np.vdot(v, im3)
    prod2 = np.vdot(im1, adj_ppfft(im2, im3))
    assert np.isclose(prod1, prod2)
