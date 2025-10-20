import numpy as np
from amp import c_kernels


def make_input(B=2, C=1, F=128, seed=123):
    rng = np.random.RandomState(seed)
    return (rng.randn(B, C, F).astype('float64') * 0.05)


def test_subharmonic_parity():
    x = make_input(B=2, C=2, F=128)
    B, C, F = x.shape

    # prepare state arrays
    hp_y = np.zeros((B, C), dtype='float64')
    lp_y = np.zeros((B, C), dtype='float64')
    prev = np.zeros((B, C), dtype='float64')
    sign = np.zeros((B, C), dtype=np.int8)
    ff2 = np.ones((B, C), dtype=np.int8)
    ff4 = np.ones((B, C), dtype=np.int8)
    ff4_count = np.zeros((B, C), dtype=np.int32)
    sub2_lp = np.zeros((B, C), dtype='float64')
    sub4_lp = np.zeros((B, C), dtype='float64')
    env = np.zeros((B, C), dtype='float64')
    hp_out_y = np.zeros((B, C), dtype='float64')
    hp_out_x = np.zeros((B, C), dtype='float64')

    params = dict(
        a_hp_in=0.1,
        a_lp_in=0.05,
        a_sub2=0.02,
        use_div4=1,
        a_sub4=0.01,
        a_env_attack=0.01,
        a_env_release=0.005,
        a_hp_out=0.02,
        drive=1.0,
        mix=0.5,
    )

    py_out = c_kernels.subharmonic_process_py(
        x,
        params['a_hp_in'],
        params['a_lp_in'],
        params['a_sub2'],
        bool(params['use_div4']),
        params['a_sub4'],
        params['a_env_attack'],
        params['a_env_release'],
        params['a_hp_out'],
        params['drive'],
        params['mix'],
        hp_y.copy(),
        lp_y.copy(),
        prev.copy(),
        sign.copy(),
        ff2.copy(),
        ff4.copy(),
        ff4_count.copy(),
        sub2_lp.copy(),
        sub4_lp.copy(),
        env.copy(),
        hp_out_y.copy(),
        hp_out_x.copy(),
    )

    if getattr(c_kernels, 'AVAILABLE', False):
        ci_hp_y = hp_y.copy()
        ci_lp_y = lp_y.copy()
        ci_prev = prev.copy()
        ci_sign = sign.copy()
        ci_ff2 = ff2.copy()
        ci_ff4 = ff4.copy()
        ci_ff4_count = ff4_count.copy()
        ci_sub2_lp = sub2_lp.copy()
        ci_sub4_lp = sub4_lp.copy()
        ci_env = env.copy()
        ci_hp_out_y = hp_out_y.copy()
        ci_hp_out_x = hp_out_x.copy()
        c_out = c_kernels.subharmonic_process_c(
            x,
            params['a_hp_in'],
            params['a_lp_in'],
            params['a_sub2'],
            True,
            params['a_sub4'],
            params['a_env_attack'],
            params['a_env_release'],
            params['a_hp_out'],
            params['drive'],
            params['mix'],
            ci_hp_y,
            ci_lp_y,
            ci_prev,
            ci_sign,
            ci_ff2,
            ci_ff4,
            ci_ff4_count,
            ci_sub2_lp,
            ci_sub4_lp,
            ci_env,
            ci_hp_out_y,
            ci_hp_out_x,
        )
        assert np.allclose(py_out, c_out, rtol=1e-9, atol=1e-12)
    else:
        assert py_out.shape == x.shape
