import pytest
import numpy as np
import scipy.fft as fft

from ppfft.inverse.new_onion_peeling import (
    new_onion_peeling,
    precompute_new_onion_peeling,
)
from ppfft.tools.pad import pad
from ppfft.tools.new_fft import new_fft
from ppfft.ppfft.new_ppfft import new_ppfft


@pytest.mark.parametrize("n", [64, 128])
def test_new_onion_peeling(n):
    toeplitz_list, nufft_list = precompute_new_onion_peeling(n)
    im = np.random.rand(n, n)
    h, v = new_ppfft(im)
    fft_row = new_onion_peeling(h, v, toeplitz_list, nufft_list)

    assert np.allclose(fft_row, fft.rfft(im, n=n + 1), atol=1e-5)
