import pytest
import numpy as np

from ppfft.tools.pad import pad
from ppfft.tools.new_fft import new_fft2
from ppfft.ppfft.ppfft import ppfft
from ppfft.inverse.onion_peeling import onion_peeling


def compute_true_Id(im):
    n = len(im)
    m = 2 * n + 1
    pad_im = pad(im, (m, m))
    return new_fft2(pad_im)[::2, ::2]


@pytest.mark.parametrize("n", [50, 60])
def test_onion_peeling(n):
    im = np.random.rand(n, n)
    h, v = ppfft(im)

    assert np.allclose(onion_peeling(h, v), compute_true_Id(im))
