import numpy as np
import pytest
from amp import utils, c_kernels


def rand_ph(B, F):
    return np.random.rand(B, F)


def rand_dphi(B, F, max_rate=0.01):
    return np.random.rand(B, F) * max_rate


def test_saw_blep_parity():
    np.random.seed(21)
    B, F = 4, 256
    ph = rand_ph(B, F)
    dphi = rand_dphi(B, F)
    out_py = utils.osc_saw_blep(ph, dphi)
    if not getattr(c_kernels, "AVAILABLE", False):
        assert out_py.shape == (B, F)
        return
    out_c = c_kernels.osc_saw_blep_c(ph, dphi)
    assert np.allclose(out_py, out_c, atol=1e-12, rtol=1e-9)


def test_square_blep_parity():
    np.random.seed(22)
    B, F = 3, 200
    ph = rand_ph(B, F)
    dphi = rand_dphi(B, F)
    out_py = utils.osc_square_blep(ph, dphi)
    if not getattr(c_kernels, "AVAILABLE", False):
        assert out_py.shape == (B, F)
        return
    out_c = c_kernels.osc_square_blep_c(ph, dphi, 0.5)
    assert np.allclose(out_py, out_c, atol=1e-12, rtol=1e-9)


def test_triangle_blep_parity():
    np.random.seed(23)
    B, F = 2, 300
    ph = rand_ph(B, F)
    dphi = rand_dphi(B, F)
    out_py = utils.osc_triangle_blep(ph, dphi)
    if not getattr(c_kernels, "AVAILABLE", False):
        assert out_py.shape == (B, F)
        return
    # provide tri_state buffer
    tri_state = np.zeros(B, dtype=float)
    out_c = c_kernels.osc_triangle_blep_c(ph, dphi, tri_state)
    assert np.allclose(out_py, out_c, atol=1e-12, rtol=1e-9)
