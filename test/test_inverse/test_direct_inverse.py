import pytest
import numpy as np

from ppfft.ppfft.ppfft import ppfft
from ppfft.inverse.onion_peeling import precompute_onion_peeling
from ppfft.inverse.direct_inversion import direct_inversion_col, direct_inversion_row


@pytest.mark.parametrize("n", [20, 30, 40])
def test_direct_inversion_col(n):
    precomputations = precompute_onion_peeling(n)
    im = np.random.rand(n, n)
    h, v = ppfft(im)
    assert np.allclose(direct_inversion_col(h, v, *precomputations), im, atol=1e-6)


@pytest.mark.parametrize("n", [20, 30, 40])
def test_direct_inversion_row(n):
    precomputations = precompute_onion_peeling(n)
    im = np.random.rand(n, n)
    h, v = ppfft(im)
    assert np.allclose(direct_inversion_row(h, v, *precomputations), im, atol=1e-6)
