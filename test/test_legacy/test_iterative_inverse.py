import pytest
import numpy as np

from ppfft.legacy.ppfft import ppfft
from ppfft.legacy.iterative_inverse import iterative_inverse


@pytest.mark.parametrize("n", [20, 30])
def test_fast_direct_inverse(n):
    im = np.random.rand(n, n)
    h, v = ppfft(im)
    sol, exit_status = iterative_inverse(h, v)

    assert np.allclose(sol, im, atol=1e-7)
