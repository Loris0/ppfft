import pytest
import numpy as np

from ppfft.tools.frac_fft import fast_frac_fft
from ppfft.tools.frac_fft import frac_fft_for_ppfft
from ppfft.tools.frac_fft import adj_frac_fft_for_ppfft
from ppfft.tools.grids import domain


def true_fast_frac_fft(x, beta):
    n = len(x)
    us = domain(n)
    js = domain(n)

    res = []
    for u in us:
        res.append(np.sum(x * np.exp(-2j * np.pi * beta * u * js)))

    return np.array(res)


@pytest.mark.parametrize("n", [100, 101])
def test_fast_frac_fft(n):
    x = np.random.rand(n)
    beta = np.random.rand()
    assert np.allclose(fast_frac_fft(x, beta), true_fast_frac_fft(x, beta))


def true_frac_fft_for_ppfft(x, alpha):
    n = len(x)
    us = domain(n + 1)
    js = domain(n)

    res = []
    for u in us:
        res.append(np.sum(x * np.exp(-2j * np.pi * alpha * u * js / n)))

    return np.array(res)


@pytest.mark.parametrize("n", [100, 101])
def test_frac_fft_for_ppfft(n):
    x = np.random.rand(n)
    alpha = np.random.rand()
    assert np.allclose(frac_fft_for_ppfft(x, alpha), true_frac_fft_for_ppfft(x, alpha))


@pytest.mark.parametrize("n", [100, 101])
def test_adj_frac_fft_for_ppfft(n):
    x = np.random.rand(n)
    y = np.random.rand(n + 1)
    alpha = np.random.rand()
    assert np.isclose(
        np.vdot(frac_fft_for_ppfft(x, alpha), y),
        np.vdot(x, adj_frac_fft_for_ppfft(y, alpha)),
    )
