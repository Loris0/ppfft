import pytest
import numpy as np
from itertools import product

from ppfft.tools.pad import pad, adj_pad
from ppfft.tools.new_fft import new_fft, new_fft2


def true_pad_1d(x, m):
    """
    Ground truth
    """
    n = len(x)

    if n % 2 == 0:
        half_n = n // 2
        # from -(n//2) to n//2 - 1 : 2*(n//2) = n points
        u = np.arange(-half_n, half_n)
    else:
        half_n = n // 2
        # from -(n//2) to n//2 : 2*(n//2) + 1 = n points
        u = np.arange(-half_n, half_n + 1)

    if m % 2 == 0:
        half_m = m // 2
        k = np.arange(-half_m, half_m)
    else:
        half_m = m // 2
        k = np.arange(-half_m, half_m + 1)

    ku = np.exp(-2j * np.pi * np.einsum("k,u->ku", k, u) / m)

    res = np.einsum("u,ku->k", x, ku)

    return res


@pytest.mark.parametrize("n, m", product([10, 11], [15, 16]))
def test_pad_1d(n, m):
    x = np.random.rand(n)
    pad_x = pad(x, (m,))
    assert np.allclose(new_fft(pad_x), true_pad_1d(x, m))


def true_pad_2d(a, new_shape):
    n, m = a.shape
    new_n, new_m = new_shape

    if n % 2 == 0:
        half_n = n // 2
        # from -(n//2) to n//2 - 1 : 2*(n//2) = n points
        u = np.arange(-half_n, half_n)
    else:
        half_n = n // 2
        # from -(n//2) to n//2 : 2*(n//2) + 1 = n points
        u = np.arange(-half_n, half_n + 1)

    if new_n % 2 == 0:
        half_new_n = new_n // 2
        k = np.arange(-half_new_n, half_new_n)
    else:
        half_new_n = new_n // 2
        k = np.arange(-half_new_n, half_new_n + 1)

    ku = np.exp(-2j * np.pi * np.einsum("k,u->ku", k, u) / new_n)

    if m % 2 == 0:
        half_m = m // 2
        # from -(n//2) to n//2 - 1 : 2*(n//2) = n points
        v = np.arange(-half_m, half_m)
    else:
        half_m = m // 2
        # from -(n//2) to n//2 : 2*(n//2) + 1 = n points
        v = np.arange(-half_m, half_m + 1)

    if new_m % 2 == 0:
        half_new_m = new_m // 2
        l = np.arange(-half_new_m, half_new_m)
    else:
        half_new_m = new_m // 2
        l = np.arange(-half_new_m, half_new_m + 1)

    lv = np.exp(-2j * np.pi * np.einsum("l,v->lv", l, v) / new_m)

    res = np.einsum("uv,ku,lv->kl", a, ku, lv)

    return res


@pytest.mark.parametrize("n1, n2, m1, m2", product([4, 5], [4, 5], [10, 11], [10, 11]))
def test_pad_2d(n1, n2, m1, m2):
    shape = (n1, n2)
    new_shape = (m1, m2)
    a = np.random.random(shape)
    pad_a = pad(a, new_shape)
    assert np.allclose(new_fft2(pad_a), true_pad_2d(a, new_shape))


@pytest.mark.parametrize("n, m", product([10, 11], [15, 16]))
def test_adj_pad_1d(n, m):
    shape = (n,)
    new_shape = (m,)
    x = np.random.random(shape)
    y = np.random.random(new_shape)
    assert np.isclose(np.vdot(pad(x, new_shape), y), np.vdot(x, adj_pad(y, shape)))


@pytest.mark.parametrize("n1, n2, m1, m2", product([4, 5], [4, 5], [10, 11], [10, 11]))
def test_adj_pad_2d(n1, n2, m1, m2):
    shape = (n1, n2)
    new_shape = (m1, m2)
    x = np.random.random(shape)
    y = np.random.random(new_shape)
    assert np.isclose(np.vdot(pad(x, new_shape), y), np.vdot(x, adj_pad(y, shape)))
