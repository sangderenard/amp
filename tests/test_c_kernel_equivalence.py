import numpy as np
import pytest

from amp import c_kernels


def test_subharmonic_process_matches_python_reference():
    """Ensure the C backend matches the Python implementation for a basic frame."""

    if not c_kernels.AVAILABLE:
        pytest.skip(f"C kernels unavailable: {c_kernels.UNAVAILABLE_REASON}")

    rng = np.random.default_rng(seed=1234)
    x = rng.standard_normal((2, 2, 64))

    a_hp_in = 0.25
    a_lp_in = 0.2
    a_sub2 = 0.1
    a_sub4 = 0.05
    a_env_attack = 0.3
    a_env_release = 0.1
    a_hp_out = 0.4
    drive = 1.2
    mix = 0.6
    use_div4 = True

    # Stateful buffers: initialise with deterministic values so both paths start identical.
    def make_state(dtype):
        return {
            "hp_y": np.zeros((2, 2), dtype=dtype),
            "lp_y": np.zeros((2, 2), dtype=dtype),
            "prev": np.zeros((2, 2), dtype=dtype),
            "sign": np.ones((2, 2), dtype=np.int8),
            "ff2": np.ones((2, 2), dtype=np.int8),
            "ff4": np.ones((2, 2), dtype=np.int8),
            "ff4_count": np.zeros((2, 2), dtype=np.int32),
            "sub2_lp": np.zeros((2, 2), dtype=dtype),
            "sub4_lp": np.zeros((2, 2), dtype=dtype),
            "env": np.zeros((2, 2), dtype=dtype),
            "hp_out_y": np.zeros((2, 2), dtype=dtype),
            "hp_out_x": np.zeros((2, 2), dtype=dtype),
        }

    state_c = make_state(np.float64)
    state_py = make_state(np.float64)

    out_c = c_kernels.subharmonic_process_c(
        x,
        a_hp_in,
        a_lp_in,
        a_sub2,
        use_div4,
        a_sub4,
        a_env_attack,
        a_env_release,
        a_hp_out,
        drive,
        mix,
        state_c["hp_y"],
        state_c["lp_y"],
        state_c["prev"],
        state_c["sign"],
        state_c["ff2"],
        state_c["ff4"],
        state_c["ff4_count"],
        state_c["sub2_lp"],
        state_c["sub4_lp"],
        state_c["env"],
        state_c["hp_out_y"],
        state_c["hp_out_x"],
    )

    out_py = c_kernels.subharmonic_process_py(
        x,
        a_hp_in,
        a_lp_in,
        a_sub2,
        use_div4,
        a_sub4,
        a_env_attack,
        a_env_release,
        a_hp_out,
        drive,
        mix,
        state_py["hp_y"],
        state_py["lp_y"],
        state_py["prev"],
        state_py["sign"],
        state_py["ff2"],
        state_py["ff4"],
        state_py["ff4_count"],
        state_py["sub2_lp"],
        state_py["sub4_lp"],
        state_py["env"],
        state_py["hp_out_y"],
        state_py["hp_out_x"],
    )

    np.testing.assert_allclose(out_c, out_py, atol=1e-9, rtol=1e-9)
