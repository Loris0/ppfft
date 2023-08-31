import pytest
import numpy as np

from ppfft.tools.frac_fft import frac_fft, adj_frac_fft
from ppfft.tools.grids import domain


def true_frac_fft(x, beta, m=None):
    n = len(x)
    if m is None:
        us = domain(n)
    else:
        us = domain(m)

    js = domain(n)

    res = []
    for u in us:
        res.append(np.sum(x * np.exp(-2j * np.pi * beta * u * js)))

    return np.array(res)


@pytest.mark.parametrize("n, m", [(100, None), (100, 101), (101, None), (101, 102)])
def test_frac_fft(n, m):
    x = np.random.rand(n)
    beta = np.random.rand()
    assert np.allclose(frac_fft(x, beta, m), true_frac_fft(x, beta, m))


@pytest.mark.parametrize("n, m", [(100, None), (100, 101), (101, None), (101, 102)])
def test_adj_frac_fft(n, m):
    if m is None:
        m = n
    x = np.random.rand(n)
    y = np.random.rand(m)
    beta = np.random.rand()
    assert np.isclose(
        np.vdot(frac_fft(x, beta, m), y), np.vdot(x, adj_frac_fft(y, beta, n))
    )
