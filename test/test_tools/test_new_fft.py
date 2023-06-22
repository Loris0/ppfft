import pytest
import numpy as np
from itertools import product

from ppfft.tools.new_fft import *


def true_new_fft(x):
    n = len(x)
    q_n, r_n = divmod(n, 2)

    u = np.arange(-q_n, q_n + r_n)
    k = np.arange(-q_n, q_n + r_n)

    ku = np.exp(-2j * np.pi * np.einsum("k,u->ku", k, u) / n)
    res = np.einsum("u,ku->k", x, ku)

    return res


@pytest.mark.parametrize("n", [10, 11])
def test_new_fft(n):
    x = np.random.rand(n)
    assert np.allclose(new_fft(x), true_new_fft(x))


def true_new_fft2(a):
    n, m = a.shape

    q_n, r_n = divmod(n, 2)
    u = np.arange(-q_n, q_n + r_n)
    l = np.arange(-q_n, q_n + r_n)

    q_m, r_m = divmod(m, 2)
    v = np.arange(-q_m, q_m + r_m)
    k = np.arange(-q_m, q_m + r_m)

    lu = np.exp(-2j * np.pi * np.einsum("l,u->lu", l, u) / n)
    kv = np.exp(-2j * np.pi * np.einsum("k,v->kv", k, v) / m)

    res = np.einsum("uv,ku,lv->kl", a, lu, kv)

    return res


@pytest.mark.parametrize("n, m", product([10, 11], [12, 13]))
def test_new_fft2(n, m):
    a = np.random.rand(n, m)
    assert np.allclose(new_fft2(a), true_new_fft2(a))


@pytest.mark.parametrize("n", [10, 11])
def test_new_ifft(n):
    x = np.random.rand(n)
    assert np.allclose(new_ifft(new_fft(x)), x)


@pytest.mark.parametrize("n, m", product([10, 11], [12, 13]))
def test_new_ifft2(n, m):
    a = np.random.rand(n, m)
    assert np.allclose(new_ifft2(new_fft2(a)), a)


@pytest.mark.parametrize("n", [10, 11])
def test_adj_fft(n):
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    assert np.isclose(np.vdot(new_fft(x1), x2), np.vdot(x1, adj_new_fft(x2)))


@pytest.mark.parametrize("n", [10, 11])
def test_adj_ifft(n):
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    assert np.isclose(np.vdot(new_ifft(x1), x2), np.vdot(x1, adj_new_ifft(x2)))


@pytest.mark.parametrize("n, m", product([10, 11], [12, 13]))
def test_adj_fft2(n, m):
    a = np.random.rand(n, m)
    b = np.random.rand(n, m)
    assert np.isclose(np.vdot(new_fft2(a), b), np.vdot(a, adj_new_fft2(b)))


@pytest.mark.parametrize("n, m", product([10, 11], [12, 13]))
def test_adj_ifft2(n, m):
    a = np.random.rand(n, m)
    b = np.random.rand(n, m)
    assert np.isclose(np.vdot(new_ifft2(a), b), np.vdot(a, adj_new_ifft2(b)))
