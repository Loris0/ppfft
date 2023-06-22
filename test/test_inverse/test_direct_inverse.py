import pytest
import numpy as np

from ppfft.ppfft.ppfft import ppfft
from ppfft.inverse.direct_inverse import direct_inversion


@pytest.mark.parametrize("n", [20, 30, 40])
def test_direct_inverse(n):
    im = np.random.rand(n, n)
    h, v = ppfft(im)

    assert np.allclose(direct_inversion(h, v), im, atol=1e-7)
