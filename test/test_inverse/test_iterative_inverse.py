import pytest
import numpy as np

from ppfft.ppfft.ppfft import ppfft
from ppfft.inverse.iterative_inverse import iterative_inverse


@pytest.mark.parametrize("n", [20, 30, 40])
def test_fast_direct_inverse(n):
    im = np.random.rand(n, n)
    h, v = ppfft(im)

    assert np.allclose(iterative_inverse(h, v)[0], im, atol=1e-7)
