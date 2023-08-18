import pytest
import numpy as np

from ppfft.inverse.onion_peeling import (
    new_onion_peeling,
    precompute_onion_peeling,
)
from ppfft.tools.pad import pad
from ppfft.tools.new_fft import new_fft
from ppfft.ppfft.new_ppfft import new_ppfft


@pytest.mark.parametrize("n", [50, 100])
def test_new_onion_peeling(n):
    toeplitz_list, nufft_list = precompute_onion_peeling(n)
    im = np.random.rand(n, n)
    h, v = new_ppfft(im)

    assert np.allclose(
        new_onion_peeling(h, v, toeplitz_list, nufft_list),
        new_fft(pad(im, (n + 1, n)), axis=0),
        atol=1e-6,
    )
