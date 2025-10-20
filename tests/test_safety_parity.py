import numpy as np
from amp import c_kernels


def make_input(B=2, C=2, F=128, seed=42):
    rng = np.random.RandomState(seed)
    return (rng.randn(B, C, F).astype('float64') * 0.1)


def test_safety_filter_parity():
    x = make_input()
    a = 0.995
    B, C, F = x.shape
    prev_in = np.zeros((B, C), dtype='float64')
    prev_dc = np.zeros((B, C), dtype='float64')

    py_out = c_kernels.safety_filter_py(x, a, prev_in.copy(), prev_dc.copy())

    # if C available, call it
    if getattr(c_kernels, 'AVAILABLE', False):
        ci_prev_in = prev_in.copy()
        ci_prev_dc = prev_dc.copy()
        c_out = c_kernels.safety_filter_c(x, a, ci_prev_in, ci_prev_dc)
        assert np.allclose(py_out, c_out, rtol=1e-9, atol=1e-12)
    else:
        # sanity: ensure python path returns shape
        assert py_out.shape == x.shape
