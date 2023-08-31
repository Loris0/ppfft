import pytest
import numpy as np

from ppfft.tools.pad import pad
from ppfft.tools.new_fft import new_fft
from ppfft.ppfft.ppfft import ppfft
from ppfft.inverse.onion_peeling import (
    precompute_onion_peeling,
    onion_peeling_col,
    onion_peeling_row,
)


@pytest.mark.parametrize("n", [64, 128])
def test_onion_peeling_col(n):
    precomputations = precompute_onion_peeling(n)
    im = np.random.rand(n, n)
    h, v = ppfft(im)
    fft_col = onion_peeling_col(h, v, *precomputations)

    assert np.allclose(fft_col, new_fft(pad(im, (n + 1, n)), axis=0), atol=1e-6)


@pytest.mark.parametrize("n", [64, 128])
def test_onion_peeling_row(n):
    precomputations = precompute_onion_peeling(n)
    im = np.random.rand(n, n)
    h, v = ppfft(im)
    fft_row = onion_peeling_row(h, v, *precomputations)

    assert np.allclose(fft_row, new_fft(pad(im, (n, n + 1)), axis=1), atol=1e-6)
