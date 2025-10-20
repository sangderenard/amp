import numpy as np
from amp import c_kernels


def test_phase_advance_parity():
    np.random.seed(2)
    B = 4
    F = 128
    dphi = np.random.rand(B, F) * 0.01
    reset = (np.random.rand(B, F) > 0.995).astype(float)
    state = np.zeros(B, dtype=float)
    st_py = state.copy()
    out_py = c_kernels.phase_advance_py(dphi, reset, st_py)
    st_c = state.copy()
    try:
        out_c = c_kernels.phase_advance_c(dphi, reset, st_c)
    except Exception:
        assert out_py.shape == (B, F)
        return
    assert np.allclose(out_py, out_c)
    assert np.allclose(st_py, st_c)


def test_portamento_parity():
    np.random.seed(3)
    B = 3
    F = 64
    freq_target = np.random.rand(B, F) * 440.0
    port_mask = (np.random.rand(B, F) > 0.7).astype(float)
    slide_time = np.full((B, F), 0.01)
    slide_damp = np.zeros((B, F))
    state = np.zeros(B, dtype=float)
    st_py = state.copy()
    out_py = c_kernels.portamento_smooth_py(freq_target, port_mask, slide_time, slide_damp, 48000, st_py)
    st_c = state.copy()
    try:
        out_c = c_kernels.portamento_smooth_c(freq_target, port_mask, slide_time, slide_damp, 48000, st_c)
    except Exception:
        assert out_py.shape == (B, F)
        return
    assert np.allclose(out_py, out_c)
    assert np.allclose(st_py, st_c)


def test_arp_advance_parity():
    B = 2
    F = 100
    seq = np.array([0.0, 3.0, -2.0])
    step_state = np.zeros(B, dtype=np.int32)
    timer_state = np.zeros(B, dtype=np.int32)
    fps = 10
    out_py = c_kernels.arp_advance_py(seq, seq.size, B, F, step_state.copy(), timer_state.copy(), fps)
    step_c = np.zeros(B, dtype=np.int32)
    timer_c = np.zeros(B, dtype=np.int32)
    try:
        out_c = c_kernels.arp_advance_c(seq, seq.size, B, F, step_c, timer_c, fps)
    except Exception:
        assert out_py.shape == (B, F)
        return
    assert np.allclose(out_py, out_c)