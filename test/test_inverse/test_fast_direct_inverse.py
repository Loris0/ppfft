import pytest
import numpy as np

from ppfft.ppfft.ppfft import ppfft
from ppfft.inverse.fast_direct_inverse import fast_direct_inversion, precompute_all


@pytest.mark.parametrize("n", [20, 30, 40])
def test_fast_direct_inverse(n):
    precomp = precompute_all(n)
    im = np.random.rand(n, n)
    h, v = ppfft(im)

    assert np.allclose(fast_direct_inversion(h, v, precomp), im, atol=1e-6)
