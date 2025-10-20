import numpy as np
import pytest
from amp import c_kernels


def _report_counts(name, diff, tol):
    total = diff.size
    exceed = np.count_nonzero(diff > tol)
    return f"{name}: total={total}, exceed>{tol} count={exceed}, max={diff.max():.3e}"


def test_lfo_slew_inversion_sum_error():
    np.random.seed(11)
    B = 6
    F = 512
    x = np.random.randn(B, F).astype(float)
    r = 0.92
    alpha = 0.08
    z0 = np.zeros(B, dtype=float)

    # Need compiled C kernel
    if not getattr(c_kernels, "AVAILABLE", False):
        pytest.skip("C kernels not available; skipping inversion-sum diagnostic")

    out_c = c_kernels.lfo_slew_c(x, r, alpha, z0.copy())
    # vector closed-form
    out_vec = c_kernels.lfo_slew_vector(x, r, alpha, z0.copy())

    diff = np.abs(out_c - out_vec)
    tol = 1e-9
    msg = _report_counts("lfo_slew", diff, tol)
    # Assert maximum absolute difference small
    assert diff.max() < tol, msg


def test_safety_filter_inversion_sum_error():
    np.random.seed(13)
    B = 4
    C = 3
    F = 256
    x = np.random.randn(B, C, F).astype(float)
    a = 0.995
    prev_in = np.zeros((B, C), dtype=float)
    prev_dc = np.zeros((B, C), dtype=float)

    if not getattr(c_kernels, "AVAILABLE", False):
        pytest.skip("C kernels not available; skipping inversion-sum diagnostic")

    out_c = c_kernels.safety_filter_c(x, a, prev_in.copy(), prev_dc.copy())
    out_py = c_kernels.safety_filter_py(x, a, prev_in.copy(), prev_dc.copy())

    diff = np.abs(out_c - out_py)
    tol = 1e-9
    msg = _report_counts("safety_filter", diff, tol)
    assert diff.max() < tol, msg
