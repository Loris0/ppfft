import pytest
import numpy as np
from scipy.linalg import solve_toeplitz

from ppfft.resampling.inverse_toeplitz import InverseToeplitz
from pynufft import NUFFT


@pytest.mark.parametrize("n", [100, 101])
def test_inverse_toeplitz(n):
    c = np.random.rand(n) + 1j * np.random.rand(n)
    r = np.random.rand(n) + 1j * np.random.rand(n)
    y = np.random.rand(n) + 1j * np.random.rand(n)
    inv = InverseToeplitz(c, r)

    assert np.allclose(solve_toeplitz((c, r), y), inv.apply_inverse(y))


def trigo_poly(alpha, x):
    n = len(alpha)
    q_n, r_n = divmod(n, 2)
    k = np.arange(-q_n, q_n + r_n)
    kx = np.einsum("k,x->kx", k, x)
    return np.einsum("kx,k->x", np.exp(1j * kx), alpha)


@pytest.mark.parametrize("n", [100, 102])
def test_compute_alpha(n):
    alpha = np.random.rand(n)
    pos = np.linspace(-np.pi, np.pi, n, endpoint=False)
    samples = trigo_poly(alpha, pos)

    c = np.einsum("lj->l", np.exp(-1j * np.arange(0, n)[:, None] * pos[None, :]))
    inv = InverseToeplitz(c)

    NufftObj = NUFFT()
    # We use 5 as oversampling factor, which is default in onion_peeling.
    # Choosing this value seems to give very precise results,
    # (meaning, np.allclose is true without changing atol / rtol)
    # and it does not affect performance too badly.
    NufftObj.plan(om=-pos[:, None], Nd=(n,), Kd=(5 * n,), Jd=(6,))

    assert np.allclose(
        inv.apply_inverse(NufftObj.Kd * NufftObj.adjoint(samples)), alpha
    )
