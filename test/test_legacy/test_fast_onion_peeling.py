import pytest
import numpy as np

from ppfft.legacy.fast_onion_peeling import (
    fast_onion_peeling,
    precompute_onion_peeling,
)
from ppfft.tools.pad import pad
from ppfft.tools.new_fft import new_fft2
from ppfft.legacy.ppfft import ppfft


def compute_true_Id(im):
    n = len(im)
    m = 2 * n + 1
    pad_im = pad(im, (m, m))
    return new_fft2(pad_im)[::2, ::2]


@pytest.mark.parametrize("n", [50, 100])
def test_fast_onion_peeling(n):
    toeplitz_list, nufft_list = precompute_onion_peeling(n)
    im = np.random.rand(n, n)
    h, v = ppfft(im)

    assert np.allclose(
        fast_onion_peeling(h, v, toeplitz_list, nufft_list),
        compute_true_Id(im),
        atol=1e-4,
    )
