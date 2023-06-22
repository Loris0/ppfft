import pytest
import numpy as np
from itertools import product

from ppfft.tools.frac_fft import frac_fft, adj_frac_fft


def true_frac_fft(x, alpha):
    n = len(x)

    q_n, r_n = divmod(n, 2)
    u = np.arange(-q_n, q_n + r_n)
    k = np.arange(-q_n, q_n + r_n)

    ku = np.exp(-2j * np.pi * alpha * np.einsum("k,u->ku", k, u) / n)
    res = np.einsum("u,ku->k", x, ku)

    return res


@pytest.mark.parametrize("n, a, b", product([10, 11], range(1, 7), range(2, 6)))
def test_frac_fft(n, a, b):
    x = np.random.rand(n)
    assert np.allclose(frac_fft(x, a, b), true_frac_fft(x, a / b))


@pytest.mark.parametrize("n, a, b", product([10, 11], range(1, 7), range(2, 6)))
def test_adj_frac_fft(n, a, b):
    x = np.random.rand(n)
    y = np.random.rand(n)
    assert np.isclose(np.vdot(frac_fft(x, a, b), y), np.vdot(x, adj_frac_fft(y, a, b)))
