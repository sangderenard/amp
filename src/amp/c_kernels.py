"""Authoritative C-backed kernels for tight loops.

This module builds the canonical native kernels via cffi. Project policy forbids
Python fallbacks for production graph execution, so callers must rely on the
compiled extension on supported platforms. Build diagnostics are captured to aid
environments that still need to surface why the native path failed.
"""
from __future__ import annotations

import traceback
from typing import Optional

import io
import filecmp
import os

import numpy as np
import shutil
import tarfile
import urllib.request

from . import native_build

native_build.ensure_toolchain_env()
_BUILD_CONFIG = native_build.get_build_config()

AVAILABLE = False
_impl = None
UNAVAILABLE_REASON: str | None = None

from pathlib import Path

_EXTRA_COMPILE_ARGS: list[str] = list(_BUILD_CONFIG.compile_args)
_EXTRA_LINK_ARGS: list[str] = list(_BUILD_CONFIG.link_args)

_EIGEN_VERSION = "3.4.0"
_EIGEN_TARBALL_URL = f"https://gitlab.com/libeigen/eigen/-/archive/{_EIGEN_VERSION}/eigen-{_EIGEN_VERSION}.tar.gz"


def _prepare_eigen_headers(third_party_dir: Path) -> tuple[Path | None, Optional[str]]:
    eigen_dir = third_party_dir / "eigen"
    sentinel = eigen_dir / "Eigen" / "Core"
    if sentinel.exists():
        return eigen_dir, None
    try:
        third_party_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_EIGEN_TARBALL_URL) as response:
            archive_data = response.read()
        with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as tar:
            tar.extractall(path=third_party_dir)
        extracted = third_party_dir / f"eigen-{_EIGEN_VERSION}"
        if extracted.exists():
            if eigen_dir.exists():
                shutil.rmtree(eigen_dir)
            extracted.rename(eigen_dir)
        if sentinel.exists():
            return eigen_dir, None
        return None, f"Eigen headers missing after extraction (expected {sentinel})"
    except Exception as exc:
        return None, f"Failed to prepare Eigen headers: {exc}"

def _stage_cffi_source(source: Path, target_name: str) -> Path:
    target = source.with_name(target_name)
    try:
        source_stat = source.stat()
        source_mtime = source_stat.st_mtime
        replicate = not target.exists()
        if not replicate:
            target_stat = target.stat()
            if target_stat.st_size != source_stat.st_size:
                replicate = True
            elif target_stat.st_mtime < source_mtime:
                try:
                    identical = filecmp.cmp(source, target, shallow=False)
                except OSError:
                    identical = False
                replicate = not identical
        if replicate:
            shutil.copy2(source, target)
        return target
    except OSError as exc:
        raise RuntimeError(f"Failed to stage {source.name} for CFFI build: {exc}") from exc


try:
    import cffi
    ffi = cffi.FFI()
    ffi.cdef("""
    void lfo_slew(const double* x, double* out, int B, int F, double r, double alpha, double* z0);
    void safety_filter(const double* x, double* y, int B, int C, int F, double a, double* prev_in, double* prev_dc);
    void dc_block(const double* x, double* out, int B, int C, int F, double a, double* state);
    void subharmonic_process(
        const double* x,
        double* y,
        int B,
        int C,
        int F,
        double a_hp_in,
        double a_lp_in,
        double a_sub2,
        int use_div4,
        double a_sub4,
        double a_env_attack,
        double a_env_release,
        double a_hp_out,
        double drive,
        double mix,
        double* hp_y,
        double* lp_y,
        double* prev,
        int8_t* sign,
        int8_t* ff2,
        int8_t* ff4,
        int32_t* ff4_count,
        double* sub2_lp,
        double* sub4_lp,
        double* env,
        double* hp_out_y,
        double* hp_out_x
    );
    void envelope_process(
        const double* trigger,
        const double* gate,
        const double* drone,
        const double* velocity,
        int B,
        int F,
        int atk_frames,
        int hold_frames,
        int dec_frames,
        int sus_frames,
        int rel_frames,
        double sustain_level,
        int send_resets,
        int* stage,
        double* value,
        double* timer,
        double* vel_state,
        int64_t* activations,
        double* release_start,
        double* amp_out,
        double* reset_out
    );
    void phase_advance(const double* dphi, double* phase_out, int B, int F, double* phase_state, const double* reset);
    void portamento_smooth(const double* freq_target, const double* port_mask, const double* slide_time, const double* slide_damp, int B, int F, int sr, double* freq_state, double* out);
    void arp_advance(const double* seq, int seq_len, double* offsets_out, int B, int F, int* step_state, int* timer_state, int fps);
    void polyblep_arr(const double* t, const double* dt, double* out, int N);
    void osc_saw_blep_c(const double* ph, const double* dphi, double* out, int B, int F);
    void osc_square_blep_c(const double* ph, const double* dphi, double pw, double* out, int B, int F);
    void osc_triangle_blep_c(const double* ph, const double* dphi, double* out, int B, int F, double* tri_state);
    """)
    ffi.cdef("""
    typedef unsigned char uint8_t;
    typedef unsigned int uint32_t;
    typedef struct {
        uint32_t has_audio;
        uint32_t batches;
        uint32_t channels;
        uint32_t frames;
        const double *data;
    } EdgeRunnerAudioView;
    typedef struct {
        const char *name;
        uint32_t batches;
        uint32_t channels;
        uint32_t frames;
        const double *data;
    } EdgeRunnerParamView;
    typedef struct {
        uint32_t count;
        EdgeRunnerParamView *items;
    } EdgeRunnerParamSet;
    typedef struct {
        EdgeRunnerAudioView audio;
        EdgeRunnerParamSet params;
    } EdgeRunnerNodeInputs;
    typedef struct {
        const char *name;
        size_t name_len;
        const char *type_name;
        size_t type_len;
        const char *params_json;
        size_t params_len;
    } EdgeRunnerNodeDescriptor;
    typedef struct {
        char *name;
        uint32_t name_len;
        uint32_t offset;
        uint32_t span;
    } EdgeRunnerCompiledParam;
    typedef struct {
        char *name;
        uint32_t name_len;
        uint32_t function_id;
        uint32_t audio_offset;
        uint32_t audio_span;
        uint32_t param_count;
        EdgeRunnerCompiledParam *params;
    } EdgeRunnerCompiledNode;
    typedef struct {
        uint32_t version;
        uint32_t node_count;
        EdgeRunnerCompiledNode *nodes;
    } EdgeRunnerCompiledPlan;
    typedef struct {
        char *name;
        uint32_t name_len;
        double *values;
        uint32_t value_count;
        double timestamp;
    } EdgeRunnerControlCurve;
    typedef struct {
        uint32_t frames_hint;
        uint32_t curve_count;
        EdgeRunnerControlCurve *curves;
    } EdgeRunnerControlHistory;
    EdgeRunnerCompiledPlan *amp_load_compiled_plan(
        const uint8_t *descriptor_blob,
        size_t descriptor_len,
        const uint8_t *plan_blob,
        size_t plan_len
    );
    void amp_release_compiled_plan(EdgeRunnerCompiledPlan *plan);
    EdgeRunnerControlHistory *amp_load_control_history(
        const uint8_t *blob,
        size_t blob_len,
        int frames_hint
    );
    void amp_release_control_history(EdgeRunnerControlHistory *history);
    int amp_run_node(
        const EdgeRunnerNodeDescriptor *descriptor,
        const EdgeRunnerNodeInputs *inputs,
        int batches,
        int channels,
        int frames,
        double sample_rate,
        double **out_buffer,
        int *out_channels,
        void **state,
        const EdgeRunnerControlHistory *history
    );
    void amp_free(double *buffer);
    void amp_release_state(void *state);
    size_t amp_last_alloc_count_get(void);
    """)
    try:
        native_dir = Path(__file__).resolve().parents[1] / "native"
        include_dir = native_dir / "include"
        third_party_dir = native_dir.parent.parent / "third_party"
        eigen_dir, eigen_error = _prepare_eigen_headers(third_party_dir)
        if eigen_dir is None:
            raise RuntimeError(eigen_error or "Eigen headers unavailable")
        kernels_source = native_dir / "amp_kernels.c"
        kernels_cxx = _stage_cffi_source(kernels_source, "amp_kernels_cffi.cc")
        debug_alloc_source = native_dir / "amp_debug_alloc.c"
        debug_alloc_cxx = _stage_cffi_source(debug_alloc_source, "amp_debug_alloc_cffi.cc")
        ffi.set_source(
            "_amp_ckernels_cffi",
            '#include "amp_native.h"\n',
            sources=[
                str(kernels_cxx),
                str(debug_alloc_cxx),
                str(native_dir / "fft_backend.cpp"),
            ],
            include_dirs=[str(include_dir), str(eigen_dir)],
            extra_compile_args=_EXTRA_COMPILE_ARGS,
            extra_link_args=_EXTRA_LINK_ARGS,
            source_extension=".cc",
        )
        # compile lazy; this will create a module in-place and return its path
        module_path = ffi.compile(verbose=False)
        # DEBUG: expose where cffi wrote the compiled module so post-processing
        # can reliably find and mutate the generated C file.
        try:
            print("[c_kernels] cffi compiled module_path:", module_path)
        except Exception:
            pass
        import importlib.util
        import sys

        compiled_path = Path(module_path)
        target_path = Path(__file__).resolve().parent / compiled_path.name
        if compiled_path.exists() and compiled_path != target_path:
            try:
                target_path.write_bytes(compiled_path.read_bytes())
            except Exception:
                # best-effort copy; continue even if it fails so runtime can still load from module_path
                target_path = compiled_path
        else:
            target_path = compiled_path

        # Post-process the generated C file to insert extra logging at the
        # start of native wrapper entry points. Try several candidate paths
        # where cffi may have emitted the generated C so this works reliably.
        try:
            candidates = [
                compiled_path,  # whatever ffi.compile returned
                target_path,    # the copied location next to this Python file
                Path.cwd() / "_amp_ckernels_cffi.c",  # common filename in cwd
                Path(__file__).resolve().parent / "_amp_ckernels_cffi.c",
            ]
            for gen_c in candidates:
                try:
                    if not gen_c.exists():
                        continue
                    src = gen_c.read_text(encoding="utf-8", errors="ignore")
                    insert_marker = "#include <Python.h>"
                    if insert_marker not in src:
                        continue
                    helper = '''
#if defined(AMP_NATIVE_ENABLE_LOGGING)
// Injected generated-wrapper logger
extern void amp_log_generated(const char *fn, void *py_ts, size_t a, size_t b);
static void _gen_wrapper_log(const char *fn, size_t a, size_t b) {
#ifdef PyThreadState_Get
    void *py_ts = (void *)PyThreadState_Get();
#else
    void *py_ts = (void*)0;
#endif
    amp_log_generated(fn, py_ts, a, b);
}
#define AMP_GEN_WRAPPER_LOG(fn, a, b) _gen_wrapper_log((fn), (a), (b))
#else
#define AMP_GEN_WRAPPER_LOG(fn, a, b) ((void)0)
#endif
'''
                    src = src.replace(insert_marker, insert_marker + "\n" + helper)

                    wrappers = [
                        'amp_load_compiled_plan',
                        'amp_release_compiled_plan',
                        'amp_load_control_history',
                        'amp_release_control_history',
                        'amp_run_node',
                        'amp_free',
                        'amp_release_state'
                    ]
                    for name in wrappers:
                        pattern = name + '('
                        idx = src.find(pattern)
                        if idx == -1:
                            continue
                        brace = src.find('{', idx)
                        if brace == -1:
                            continue
                        injection = f'\n    AMP_GEN_WRAPPER_LOG("{name}", (size_t)0, (size_t)0);\n'
                        src = src[:brace+1] + injection + src[brace+1:]

                    gen_c.write_text(src, encoding="utf-8")
                    # stop after first successful injection
                    break
                except Exception:
                    continue
        except Exception:
            # best-effort only
            pass

        spec = importlib.util.spec_from_file_location("_amp_ckernels_cffi", str(target_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load compiled module from {target_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["_amp_ckernels_cffi"] = module
        spec.loader.exec_module(module)
        _impl = module
        AVAILABLE = True
        UNAVAILABLE_REASON = None
    except Exception as exc:
        # any compile/import error -> disable C backend
        AVAILABLE = False
        detail = traceback.format_exc()
        UNAVAILABLE_REASON = (
            "Failed to compile C kernels via cffi: "
            f"{exc}\n{detail}"
        )
except ModuleNotFoundError as exc:
    AVAILABLE = False
    UNAVAILABLE_REASON = f"cffi is not installed ({exc})"
except Exception as exc:
    AVAILABLE = False
    UNAVAILABLE_REASON = (
        "Unexpected error initialising cffi for C kernels: "
        f"{exc}"
    )


def _require_ctypes_ready(arr: np.ndarray, dtype: np.dtype, *, writable: bool) -> np.ndarray:
    """Validate that ``arr`` can be passed directly to a C kernel."""

    if arr.dtype != dtype:
        raise TypeError(f"expected dtype {dtype}, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        raise ValueError("arrays passed to C kernels must be C-contiguous")
    if writable and not arr.flags.writeable:
        raise ValueError("writable arrays passed to C kernels must be writeable")
    return arr


DTYPE_FLOAT = np.dtype(np.float64)
DTYPE_INT32 = np.dtype(np.int32)
DTYPE_INT64 = np.dtype(np.int64)


def lfo_slew_c(
    x: np.ndarray,
    r: float,
    alpha: float,
    z0: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Call the compiled C kernel to compute exponential smoothing.

    x: (B, F) contiguous C-order array of doubles
    r: feedback coefficient
    alpha: feed coefficient
    z0: optional (B,) array of initial states (modified in-place)

    Returns out (B, F) same dtype.
    Raises RuntimeError if C backend is unavailable.
    """
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    x_buf = _require_ctypes_ready(np.asarray(x), DTYPE_FLOAT, writable=False)
    B, F = x_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for lfo_slew_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if z0 is not None:
        if z0.shape != (B,):
            raise ValueError("z0 must have shape (B,)")
        z_buf = _require_ctypes_ready(z0, DTYPE_FLOAT, writable=True)
        z_ptr = ffi.cast("double *", z_buf.ctypes.data)
    else:
        z_ptr = ffi.cast("double *", ffi.NULL)

    x_ptr = ffi.cast("const double *", x_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.lfo_slew(x_ptr, out_ptr, int(B), int(F), float(r), float(alpha), z_ptr)
    return out


def lfo_slew_py(
    x: np.ndarray,
    r: float,
    alpha: float,
    z0: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Pure-Python sample-sequential fallback (fast with numpy per-row ops).

    Semantics: iterative recurrence z[n] = r*z[n-1] + alpha * x[n].
    """
    x_buf = np.asarray(x, dtype=DTYPE_FLOAT)
    B, F = x_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for lfo_slew_py")
    if z0 is None:
        z = np.zeros(B, dtype=DTYPE_FLOAT)
    else:
        z = np.asarray(z0, dtype=DTYPE_FLOAT)
    for i in range(F):
        xi = x_buf[:, i]
        z = r * z + alpha * xi
        out[:, i] = z
    if z0 is not None:
        z0[:] = z
    return out


def lfo_slew_vector(
    x: np.ndarray,
    r: float,
    alpha: float,
    z0: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized closed-form solution equivalent to iterative recurrence.

    z[n] = r^n * z0 + alpha * r^n * sum_{k=0..n} r^{-k} * x[k]
    Implemented using np.cumsum on axis 1.
    """
    x_buf = np.asarray(x, dtype=DTYPE_FLOAT)
    B, F = x_buf.shape
    idx = np.arange(F, dtype=DTYPE_FLOAT)
    r_pow = r ** idx
    # handle r==0
    with np.errstate(divide='ignore', invalid='ignore'):
        r_inv = np.where(r == 0.0, 0.0, r ** (-idx))
    accum = np.cumsum(x_buf * r_inv[None, :], axis=1)
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for lfo_slew_vector")
    out[:] = r_pow[None, :] * (alpha * accum)
    if z0 is not None:
        out += r_pow[None, :] * z0[:, None]
        z0[:] = out[:, -1]
    return out


def safety_filter_c(
    x: np.ndarray,
    a: float,
    prev_in: Optional[np.ndarray],
    prev_dc: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Call compiled ``safety_filter`` kernel without intermediate copies."""

    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    x_buf = _require_ctypes_ready(np.asarray(x), DTYPE_FLOAT, writable=False)
    B, C, F = x_buf.shape
    if out is None:
        out = np.empty((B, C, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, C, F):
            raise ValueError("out has incorrect shape for safety_filter_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if prev_in is not None:
        if prev_in.shape != (B, C):
            raise ValueError("prev_in must have shape (B, C)")
        prev_in_buf = _require_ctypes_ready(prev_in, DTYPE_FLOAT, writable=True)
        prev_in_ptr = ffi.cast("double *", prev_in_buf.ctypes.data)
    else:
        prev_in_ptr = ffi.cast("double *", ffi.NULL)

    if prev_dc is not None:
        if prev_dc.shape != (B, C):
            raise ValueError("prev_dc must have shape (B, C)")
        prev_dc_buf = _require_ctypes_ready(prev_dc, DTYPE_FLOAT, writable=True)
        prev_dc_ptr = ffi.cast("double *", prev_dc_buf.ctypes.data)
    else:
        prev_dc_ptr = ffi.cast("double *", ffi.NULL)

    x_ptr = ffi.cast("const double *", x_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.safety_filter(x_ptr, out_ptr, int(B), int(C), int(F), float(a), prev_in_ptr, prev_dc_ptr)
    return out


def safety_filter_py(
    x: np.ndarray,
    a: float,
    prev_in: Optional[np.ndarray],
    prev_dc: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    x_buf = np.asarray(x, dtype=DTYPE_FLOAT)
    B, C, F = x_buf.shape
    if out is None:
        out = np.empty((B, C, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, C, F):
            raise ValueError("out has incorrect shape for safety_filter_py")
    pi = np.zeros((B, C), dtype=DTYPE_FLOAT) if prev_in is None else np.asarray(prev_in, dtype=DTYPE_FLOAT)
    pd = np.zeros((B, C), dtype=DTYPE_FLOAT) if prev_dc is None else np.asarray(prev_dc, dtype=DTYPE_FLOAT)
    for b in range(B):
        for c in range(C):
            if F <= 0:
                continue
            # compute diffs
            diffs = np.empty(F, dtype=DTYPE_FLOAT)
            diffs[0] = x_buf[b, c, 0] - pi[b, c]
            if F > 1:
                diffs[1:] = x_buf[b, c, 1:] - x_buf[b, c, :-1]
            powers = a ** np.arange(F, dtype=DTYPE_FLOAT)
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_p = 1.0 / powers
            accum = np.cumsum(diffs * inv_p) + (a * pd[b, c])
            y = accum * powers
            out[b, c, :] = y
            pi[b, c] = x_buf[b, c, -1]
            pd[b, c] = y[-1]
    if prev_in is not None:
        prev_in[:] = pi
    if prev_dc is not None:
        prev_dc[:] = pd
    return out


def dc_block_c(
    x: np.ndarray,
    a: float,
    state: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    x_buf = _require_ctypes_ready(np.asarray(x), DTYPE_FLOAT, writable=False)
    B, C, F = x_buf.shape
    if out is None:
        out = np.empty((B, C, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, C, F):
            raise ValueError("out has incorrect shape for dc_block_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if state is not None:
        if state.shape != (B, C):
            raise ValueError("state must have shape (B, C)")
        state_buf = _require_ctypes_ready(state, DTYPE_FLOAT, writable=True)
        state_ptr = ffi.cast("double *", state_buf.ctypes.data)
    else:
        state_ptr = ffi.cast("double *", ffi.NULL)

    x_ptr = ffi.cast("const double *", x_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.dc_block(x_ptr, out_ptr, int(B), int(C), int(F), float(a), state_ptr)
    return out


def dc_block_py(
    x: np.ndarray,
    a: float,
    state: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    x_buf = np.asarray(x, dtype=DTYPE_FLOAT)
    B, C, F = x_buf.shape
    if out is None:
        out = np.empty((B, C, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, C, F):
            raise ValueError("out has incorrect shape for dc_block_py")
    st = np.zeros((B, C), dtype=DTYPE_FLOAT) if state is None else np.asarray(state, dtype=DTYPE_FLOAT)
    for b in range(B):
        for c in range(C):
            dc = st[b, c]
            for i in range(F):
                xi = x_buf[b, c, i]
                dc = a * dc + (1.0 - a) * xi
                out[b, c, i] = xi - dc
            st[b, c] = dc
    if state is not None:
        state[:] = st
    return out


def subharmonic_process_c(
    x: np.ndarray,
    a_hp_in: float,
    a_lp_in: float,
    a_sub2: float,
    use_div4: bool,
    a_sub4: float,
    a_env_attack: float,
    a_env_release: float,
    a_hp_out: float,
    drive: float,
    mix: float,
    hp_y: np.ndarray,
    lp_y: np.ndarray,
    prev: np.ndarray,
    sign: np.ndarray,
    ff2: np.ndarray,
    ff4: np.ndarray | None,
    ff4_count: np.ndarray | None,
    sub2_lp: np.ndarray,
    sub4_lp: np.ndarray | None,
    env: np.ndarray,
    hp_out_y: np.ndarray,
    hp_out_x: np.ndarray,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, C, F = x.shape
    xb = np.ascontiguousarray(x)
    out = np.empty_like(xb)
    outb = np.ascontiguousarray(out)

    # ensure buffers
    hp_y_b = np.ascontiguousarray(hp_y)
    lp_y_b = np.ascontiguousarray(lp_y)
    prev_b = np.ascontiguousarray(prev)
    sign_b = np.ascontiguousarray(sign.astype(np.int8))
    ff2_b = np.ascontiguousarray(ff2.astype(np.int8))
    ff4_b = np.ascontiguousarray(ff4.astype(np.int8)) if ff4 is not None else ffi.cast("int8_t *", ffi.NULL)
    ff4_count_b = np.ascontiguousarray(ff4_count.astype(np.int32)) if ff4_count is not None else ffi.cast("int32_t *", ffi.NULL)
    sub2_lp_b = np.ascontiguousarray(sub2_lp)
    sub4_lp_b = np.ascontiguousarray(sub4_lp) if sub4_lp is not None else ffi.cast("double *", ffi.NULL)
    env_b = np.ascontiguousarray(env)
    hp_out_y_b = np.ascontiguousarray(hp_out_y)
    hp_out_x_b = np.ascontiguousarray(hp_out_x)

    x_ptr = ffi.cast("const double *", xb.ctypes.data)
    y_ptr = ffi.cast("double *", outb.ctypes.data)
    hp_y_ptr = ffi.cast("double *", hp_y_b.ctypes.data)
    lp_y_ptr = ffi.cast("double *", lp_y_b.ctypes.data)
    prev_ptr = ffi.cast("double *", prev_b.ctypes.data)
    sign_ptr = ffi.cast("int8_t *", sign_b.ctypes.data)
    ff2_ptr = ffi.cast("int8_t *", ff2_b.ctypes.data)
    ff4_ptr = ffi.cast("int8_t *", ff4_b.ctypes.data) if ff4 is not None else ffi.cast("int8_t *", ffi.NULL)
    ff4_count_ptr = ffi.cast("int32_t *", ff4_count_b.ctypes.data) if ff4_count is not None else ffi.cast("int32_t *", ffi.NULL)
    sub2_lp_ptr = ffi.cast("double *", sub2_lp_b.ctypes.data)
    sub4_lp_ptr = ffi.cast("double *", sub4_lp_b.ctypes.data) if sub4_lp is not None else ffi.cast("double *", ffi.NULL)
    env_ptr = ffi.cast("double *", env_b.ctypes.data)
    hp_out_y_ptr = ffi.cast("double *", hp_out_y_b.ctypes.data)
    hp_out_x_ptr = ffi.cast("double *", hp_out_x_b.ctypes.data)

    _impl.lib.subharmonic_process(
        x_ptr,
        y_ptr,
        int(B),
        int(C),
        int(F),
        float(a_hp_in),
        float(a_lp_in),
        float(a_sub2),
        int(1 if use_div4 else 0),
        float(a_sub4),
        float(a_env_attack),
        float(a_env_release),
        float(a_hp_out),
        float(drive),
        float(mix),
        hp_y_ptr,
        lp_y_ptr,
        prev_ptr,
        sign_ptr,
        ff2_ptr,
        ff4_ptr,
        ff4_count_ptr,
        sub2_lp_ptr,
        sub4_lp_ptr,
        env_ptr,
        hp_out_y_ptr,
        hp_out_x_ptr,
    )

    # copy back mutable state
    hp_y[:] = hp_y_b
    lp_y[:] = lp_y_b
    prev[:] = prev_b
    sign[:] = sign_b
    ff2[:] = ff2_b
    if ff4 is not None:
        ff4[:] = ff4_b
    if ff4_count is not None:
        ff4_count[:] = ff4_count_b
    sub2_lp[:] = sub2_lp_b
    if sub4_lp is not None:
        sub4_lp[:] = sub4_lp_b
    env[:] = env_b
    hp_out_y[:] = hp_out_y_b
    hp_out_x[:] = hp_out_x_b

    return outb


def subharmonic_process_py(
    x: np.ndarray,
    a_hp_in: float,
    a_lp_in: float,
    a_sub2: float,
    use_div4: bool,
    a_sub4: float,
    a_env_attack: float,
    a_env_release: float,
    a_hp_out: float,
    drive: float,
    mix: float,
    hp_y: np.ndarray,
    lp_y: np.ndarray,
    prev: np.ndarray,
    sign: np.ndarray,
    ff2: np.ndarray,
    ff4: np.ndarray | None,
    ff4_count: np.ndarray | None,
    sub2_lp: np.ndarray,
    sub4_lp: np.ndarray | None,
    env: np.ndarray,
    hp_out_y: np.ndarray,
    hp_out_x: np.ndarray,
) -> np.ndarray:
    B, C, F = x.shape
    y = np.empty_like(x)
    for t in range(F):
        xt = x[:, :, t]

        # Bandpass driver: simple HP then LP
        hp_y[:] = a_hp_in * (hp_y + xt - prev)
        prev[:] = xt
        bp = lp_y + a_lp_in * (hp_y - lp_y)
        lp_y[:] = bp

        abs_bp = np.abs(bp)
        env[:] = np.where(
            abs_bp > env,
            env + a_env_attack * (abs_bp - env),
            env + a_env_release * (abs_bp - env),
        )

        prev_sign = sign.copy()
        sign_now = (bp > 0.0).astype(np.int8) * 2 - 1
        pos_zc = (prev_sign < 0) & (sign_now > 0)
        sign[:] = sign_now

        ff2[:] = np.where(pos_zc, -ff2, ff2)

        if use_div4 and ff4 is not None and ff4_count is not None:
            ff4_count[:] = np.where(pos_zc, ff4_count + 1, ff4_count)
            toggle4 = pos_zc & (ff4_count >= 2)
            ff4[:] = np.where(toggle4, -ff4, ff4)
            ff4_count[:] = np.where(toggle4, 0, ff4_count)

        sq2 = ff2.astype(x.dtype)
        sub2_lp[:] = sub2_lp + a_sub2 * (sq2 - sub2_lp)
        sub = sub2_lp.copy()

        if use_div4 and sub4_lp is not None and ff4 is not None:
            sq4 = ff4.astype(x.dtype)
            sub4_lp[:] = sub4_lp + a_sub4 * (sq4 - sub4_lp)
            sub = sub + 0.6 * sub4_lp

        sub = np.tanh(drive * sub) * (env + 1e-6)

        dry = xt
        wet = sub
        out_t = (1.0 - mix) * dry + mix * wet

        y_prev = hp_out_y.copy()
        x_prev = hp_out_x.copy()
        hp = a_hp_out * (y_prev + out_t - x_prev)
        hp_out_y[:] = hp
        hp_out_x[:] = out_t
        y[:, :, t] = hp

    return y


def phase_advance_c(
    dphi: np.ndarray,
    reset: np.ndarray | None,
    phase_state: np.ndarray | None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    dphi_buf = _require_ctypes_ready(np.asarray(dphi), DTYPE_FLOAT, writable=False)
    B, F = dphi_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for phase_advance_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if phase_state is not None:
        if phase_state.shape != (B,):
            raise ValueError("phase_state must have shape (B,)")
        phase_buf = _require_ctypes_ready(phase_state, DTYPE_FLOAT, writable=True)
        state_ptr = ffi.cast("double *", phase_buf.ctypes.data)
    else:
        state_ptr = ffi.cast("double *", ffi.NULL)

    if reset is not None:
        reset_buf = _require_ctypes_ready(np.asarray(reset), DTYPE_FLOAT, writable=False)
        if reset_buf.shape != (B, F):
            raise ValueError("reset must have shape (B, F)")
        reset_ptr = ffi.cast("const double *", reset_buf.ctypes.data)
    else:
        reset_ptr = ffi.cast("const double *", ffi.NULL)

    dphi_ptr = ffi.cast("const double *", dphi_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.phase_advance(dphi_ptr, out_ptr, int(B), int(F), state_ptr, reset_ptr)
    return out


def phase_advance_py(
    dphi: np.ndarray,
    reset: np.ndarray | None,
    phase_state: np.ndarray | None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    dphi_buf = np.asarray(dphi, dtype=DTYPE_FLOAT)
    B, F = dphi_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for phase_advance_py")
    if phase_state is None:
        cur = np.zeros(B, dtype=DTYPE_FLOAT)
    else:
        cur = np.asarray(phase_state, dtype=DTYPE_FLOAT)
    if reset is not None:
        reset_buf = np.asarray(reset, dtype=DTYPE_FLOAT)
    else:
        reset_buf = None
    for i in range(F):
        if reset_buf is not None:
            mask = reset_buf[:, i] > 0.5
            if np.any(mask):
                cur = np.where(mask, 0.0, cur)
        cur = (cur + dphi_buf[:, i]) % 1.0
        out[:, i] = cur
    if phase_state is not None:
        phase_state[:] = cur
    return out


def portamento_smooth_c(
    freq_target: np.ndarray,
    port_mask: np.ndarray | None,
    slide_time: np.ndarray | None,
    slide_damp: np.ndarray | None,
    sr: int,
    freq_state: np.ndarray | None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    freq_buf = _require_ctypes_ready(np.asarray(freq_target), DTYPE_FLOAT, writable=False)
    B, F = freq_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for portamento_smooth_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if port_mask is not None:
        port_buf = _require_ctypes_ready(np.asarray(port_mask), DTYPE_FLOAT, writable=False)
        if port_buf.shape != (B, F):
            raise ValueError("port_mask must have shape (B, F)")
        port_ptr = ffi.cast("const double *", port_buf.ctypes.data)
    else:
        port_ptr = ffi.cast("const double *", ffi.NULL)

    if slide_time is not None:
        st_buf = _require_ctypes_ready(np.asarray(slide_time), DTYPE_FLOAT, writable=False)
        if st_buf.shape != (B, F):
            raise ValueError("slide_time must have shape (B, F)")
        st_ptr = ffi.cast("const double *", st_buf.ctypes.data)
    else:
        st_ptr = ffi.cast("const double *", ffi.NULL)

    if slide_damp is not None:
        sd_buf = _require_ctypes_ready(np.asarray(slide_damp), DTYPE_FLOAT, writable=False)
        if sd_buf.shape != (B, F):
            raise ValueError("slide_damp must have shape (B, F)")
        sd_ptr = ffi.cast("const double *", sd_buf.ctypes.data)
    else:
        sd_ptr = ffi.cast("const double *", ffi.NULL)

    if freq_state is not None:
        if freq_state.shape != (B,):
            raise ValueError("freq_state must have shape (B,)")
        state_buf = _require_ctypes_ready(freq_state, DTYPE_FLOAT, writable=True)
        state_ptr = ffi.cast("double *", state_buf.ctypes.data)
    else:
        state_ptr = ffi.cast("double *", ffi.NULL)

    ft_ptr = ffi.cast("const double *", freq_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.portamento_smooth(ft_ptr, port_ptr, st_ptr, sd_ptr, int(B), int(F), int(sr), state_ptr, out_ptr)
    return out


def portamento_smooth_py(
    freq_target: np.ndarray,
    port_mask: np.ndarray | None,
    slide_time: np.ndarray | None,
    slide_damp: np.ndarray | None,
    sr: int,
    freq_state: np.ndarray | None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    freq_buf = np.asarray(freq_target, dtype=DTYPE_FLOAT)
    B, F = freq_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for portamento_smooth_py")
    if freq_state is None:
        cur = np.zeros(B, dtype=DTYPE_FLOAT)
    else:
        cur = np.asarray(freq_state, dtype=DTYPE_FLOAT)
    port_buf = None if port_mask is None else np.asarray(port_mask, dtype=DTYPE_FLOAT)
    st_buf = None if slide_time is None else np.asarray(slide_time, dtype=DTYPE_FLOAT)
    sd_buf = None if slide_damp is None else np.asarray(slide_damp, dtype=DTYPE_FLOAT)
    for i in range(F):
        target = freq_buf[:, i]
        if port_buf is not None:
            active = port_buf[:, i] > 0.5
        else:
            active = np.zeros(B, dtype=bool)
        frames_const = np.maximum(st_buf[:, i] * float(sr) if st_buf is not None else 1.0, 1.0)
        alpha = np.exp(-1.0 / frames_const)
        if sd_buf is not None:
            alpha = alpha ** (1.0 + np.clip(sd_buf[:, i], 0.0, None))
        cur = np.where(active, alpha * cur + (1.0 - alpha) * target, target)
        out[:, i] = cur
    if freq_state is not None:
        freq_state[:] = cur
    return out


def arp_advance_c(
    seq: np.ndarray,
    seq_len: int,
    B: int,
    F: int,
    step_state: np.ndarray,
    timer_state: np.ndarray,
    fps: int,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    seq_buf = _require_ctypes_ready(np.asarray(seq), DTYPE_FLOAT, writable=False)
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for arp_advance_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    step_buf = _require_ctypes_ready(np.asarray(step_state), DTYPE_INT32, writable=True)
    timer_buf = _require_ctypes_ready(np.asarray(timer_state), DTYPE_INT32, writable=True)

    seq_ptr = ffi.cast("const double *", seq_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    step_ptr = ffi.cast("int *", step_buf.ctypes.data)
    timer_ptr = ffi.cast("int *", timer_buf.ctypes.data)
    _impl.lib.arp_advance(seq_ptr, int(seq_len), out_ptr, int(B), int(F), step_ptr, timer_ptr, int(fps))
    return out


def arp_advance_py(
    seq: np.ndarray,
    seq_len: int,
    B: int,
    F: int,
    step_state: np.ndarray,
    timer_state: np.ndarray,
    fps: int,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for arp_advance_py")
        out.fill(0.0)
    seq_list = list(np.asarray(seq, dtype=DTYPE_FLOAT).ravel())
    if len(seq_list) == 0:
        seq_list = [0.0]
    seq_vals = np.asarray(seq_list, dtype=DTYPE_FLOAT)
    step = np.asarray(step_state, dtype=DTYPE_INT32)
    timer = np.asarray(timer_state, dtype=DTYPE_INT32)
    for i in range(F):
        idx = step % seq_vals.size
        out[:, i] = seq_vals[idx]
        timer += 1
        reached = timer >= fps
        if np.any(reached):
            timer[reached] = 0
            step[reached] = (step[reached] + 1) % len(seq_list)
    step_state[:] = step
    timer_state[:] = timer
    return out


def _polyblep_arr_c(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    t_b = np.ascontiguousarray(t)
    dt_b = np.ascontiguousarray(dt)
    out = np.empty_like(t_b)
    _impl.lib.polyblep_arr(ffi.cast("const double *", t_b.ctypes.data), ffi.cast("const double *", dt_b.ctypes.data), ffi.cast("double *", out.ctypes.data), int(out.size))
    return out


def osc_saw_blep_c(ph: np.ndarray, dphi: np.ndarray) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, F = ph.shape
    out = np.empty((B, F), dtype=ph.dtype)
    _impl.lib.osc_saw_blep_c(ffi.cast("const double *", np.ascontiguousarray(ph).ctypes.data), ffi.cast("const double *", np.ascontiguousarray(dphi).ctypes.data), ffi.cast("double *", out.ctypes.data), int(B), int(F))
    return out


def osc_saw_blep_py(ph: np.ndarray, dphi: np.ndarray) -> np.ndarray:
    t = ph
    y = 2.0 * t - 1.0
    # reuse _polyblep_arr implementation
    pb = _polyblep_arr_py(t, dphi)
    return y - pb


def osc_square_blep_c(ph: np.ndarray, dphi: np.ndarray, pw: float = 0.5) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, F = ph.shape
    out = np.empty((B, F), dtype=ph.dtype)
    _impl.lib.osc_square_blep_c(ffi.cast("const double *", np.ascontiguousarray(ph).ctypes.data), ffi.cast("const double *", np.ascontiguousarray(dphi).ctypes.data), float(pw), ffi.cast("double *", out.ctypes.data), int(B), int(F))
    return out


def osc_square_blep_py(ph: np.ndarray, dphi: np.ndarray, pw: float = 0.5) -> np.ndarray:
    t = ph
    y = np.where(t < pw, 1.0, -1.0)
    y = y - _polyblep_arr_py(t, dphi)
    t2 = (t + (1.0 - pw)) % 1.0
    y = y + _polyblep_arr_py(t2, dphi)
    return y


def osc_triangle_blep_c(ph: np.ndarray, dphi: np.ndarray, tri_state: np.ndarray | None = None) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, F = ph.shape
    out = np.empty((B, F), dtype=ph.dtype)
    if tri_state is not None:
        tri_buf = np.ascontiguousarray(tri_state.astype(np.float64))
        tri_ptr = ffi.cast("double *", tri_buf.ctypes.data)
    else:
        tri_buf = None
        tri_ptr = ffi.cast("double *", ffi.NULL)
    _impl.lib.osc_triangle_blep_c(
        ffi.cast("const double *", np.ascontiguousarray(ph).ctypes.data),
        ffi.cast("const double *", np.ascontiguousarray(dphi).ctypes.data),
        ffi.cast("double *", out.ctypes.data),
        int(B),
        int(F),
        tri_ptr,
    )
    if tri_state is not None:
        tri_state[:] = tri_buf[:B]
    return out


def _polyblep_arr_py(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    m = t < dt
    if np.any(m):
        x = t[m] / np.maximum(dt[m], 1e-20)
        out[m] = x + x - x * x - 1.0
    m = t > (1.0 - dt)
    if np.any(m):
        x = (t[m] - 1.0) / np.maximum(dt[m], 1e-20)
        out[m] = x * x + x + x + 1.0
    return out

def envelope_process_c(
    trigger: np.ndarray,
    gate: np.ndarray,
    drone: np.ndarray,
    velocity: np.ndarray,
    atk_frames: int,
    hold_frames: int,
    dec_frames: int,
    sus_frames: int,
    rel_frames: int,
    sustain_level: float,
    send_resets: bool,
    stage: np.ndarray,
    value: np.ndarray,
    timer: np.ndarray,
    vel_state: np.ndarray,
    activations: np.ndarray,
    release_start: np.ndarray,
    *,
    out_amp: np.ndarray | None = None,
    out_reset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B = trigger.shape[0]
    F = trigger.shape[1]
    dtype_float = np.dtype(np.float64)
    dtype_stage = np.dtype(np.int32)
    dtype_acts = np.dtype(np.int64)

    trig_b = _require_ctypes_ready(trigger, dtype_float, writable=False)
    gate_b = _require_ctypes_ready(gate, dtype_float, writable=False)
    drone_b = _require_ctypes_ready(drone, dtype_float, writable=False)
    vel_b = _require_ctypes_ready(velocity, dtype_float, writable=False)
    stage_b = _require_ctypes_ready(stage, dtype_stage, writable=True)
    value_b = _require_ctypes_ready(value, dtype_float, writable=True)
    timer_b = _require_ctypes_ready(timer, dtype_float, writable=True)
    vel_state_b = _require_ctypes_ready(vel_state, dtype_float, writable=True)
    activ_b = _require_ctypes_ready(activations, dtype_acts, writable=True)
    rel_b = _require_ctypes_ready(release_start, dtype_float, writable=True)

    if out_amp is None:
        out_amp = np.empty((B, F), dtype=dtype_float)
    else:
        if out_amp.shape != (B, F):
            raise ValueError("out_amp has incorrect shape")
        _require_ctypes_ready(out_amp, dtype_float, writable=True)
    if out_reset is None:
        out_reset = np.empty((B, F), dtype=dtype_float)
    else:
        if out_reset.shape != (B, F):
            raise ValueError("out_reset has incorrect shape")
        _require_ctypes_ready(out_reset, dtype_float, writable=True)

    amp_ptr = ffi.cast("double *", out_amp.ctypes.data)
    reset_ptr = ffi.cast("double *", out_reset.ctypes.data)
    trig_ptr = ffi.cast("const double *", trig_b.ctypes.data)
    gate_ptr = ffi.cast("const double *", gate_b.ctypes.data)
    drone_ptr = ffi.cast("const double *", drone_b.ctypes.data)
    vel_ptr = ffi.cast("const double *", vel_b.ctypes.data)
    stage_ptr = ffi.cast("int *", stage_b.ctypes.data)
    value_ptr = ffi.cast("double *", value_b.ctypes.data)
    timer_ptr = ffi.cast("double *", timer_b.ctypes.data)
    vel_state_ptr = ffi.cast("double *", vel_state_b.ctypes.data)
    activ_ptr = ffi.cast("int64_t *", activ_b.ctypes.data)
    rel_ptr = ffi.cast("double *", rel_b.ctypes.data)

    _impl.lib.envelope_process(
        trig_ptr,
        gate_ptr,
        drone_ptr,
        vel_ptr,
        int(B),
        int(F),
        int(atk_frames),
        int(hold_frames),
        int(dec_frames),
        int(sus_frames),
        int(rel_frames),
        float(sustain_level),
        int(1 if send_resets else 0),
        stage_ptr,
        value_ptr,
        timer_ptr,
        vel_state_ptr,
        activ_ptr,
        rel_ptr,
        amp_ptr,
        reset_ptr,
    )

    return out_amp, out_reset


def envelope_process_py(
    trigger: np.ndarray,
    gate: np.ndarray,
    drone: np.ndarray,
    velocity: np.ndarray,
    atk_frames: int,
    hold_frames: int,
    dec_frames: int,
    sus_frames: int,
    rel_frames: int,
    sustain_level: float,
    send_resets: bool,
    stage: np.ndarray,
    value: np.ndarray,
    timer: np.ndarray,
    vel_state: np.ndarray,
    activations: np.ndarray,
    release_start: np.ndarray,
    *,
    out_amp: np.ndarray | None = None,
    out_reset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Python fallback envelope processor."""

    trigger_buf = np.asarray(trigger, dtype=DTYPE_FLOAT)
    gate_buf = np.asarray(gate, dtype=DTYPE_FLOAT)
    drone_buf = np.asarray(drone, dtype=DTYPE_FLOAT)
    velocity_buf = np.asarray(velocity, dtype=DTYPE_FLOAT)

    B, F = trigger_buf.shape

    if stage.dtype != DTYPE_INT32:
        raise TypeError("stage must have dtype int32")
    stage_buf = stage
    if value.dtype != DTYPE_FLOAT:
        raise TypeError("value must have dtype float64")
    value_buf = value
    if timer.dtype != DTYPE_FLOAT:
        raise TypeError("timer must have dtype float64")
    timer_buf = timer
    if vel_state.dtype != DTYPE_FLOAT:
        raise TypeError("vel_state must have dtype float64")
    vel_state_buf = vel_state
    if activations.dtype != DTYPE_INT64:
        raise TypeError("activations must have dtype int64")
    activ_buf = activations
    if release_start.dtype != DTYPE_FLOAT:
        raise TypeError("release_start must have dtype float64")
    rel_buf = release_start

    if out_amp is None:
        out_amp = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out_amp.shape != (B, F):
            raise ValueError("out_amp has incorrect shape")
        out_amp = np.asarray(out_amp, dtype=DTYPE_FLOAT)

    if out_reset is None:
        out_reset = np.zeros((B, F), dtype=DTYPE_FLOAT)
    else:
        if out_reset.shape != (B, F):
            raise ValueError("out_reset has incorrect shape")
        out_reset = np.asarray(out_reset, dtype=DTYPE_FLOAT)
        out_reset.fill(0.0)

    for b in range(B):
        st = int(stage_buf[b])
        val = float(value_buf[b])
        tim = float(timer_buf[b])
        vel = float(vel_state_buf[b])
        acts = int(activ_buf[b])
        rel_start_val = float(rel_buf[b])

        trig_line = trigger_buf[b] > 0.5
        gate_line = gate_buf[b] > 0.5
        drone_line = drone_buf[b] > 0.5

        for i in range(F):
            trig = bool(trig_line[i])
            gate_on = bool(gate_line[i])
            drone_on = bool(drone_line[i])

            if trig:
                st = 1
                tim = 0.0
                val = 0.0
                vel = float(velocity_buf[b, i])
                if vel < 0.0:
                    vel = 0.0
                rel_start_val = vel
                acts += 1
                if send_resets:
                    out_reset[b, i] = 1.0
            elif st == 0 and (gate_on or drone_on):
                st = 1
                tim = 0.0
                val = 0.0
                vel = float(velocity_buf[b, i])
                if vel < 0.0:
                    vel = 0.0
                rel_start_val = vel
                acts += 1
                if send_resets:
                    out_reset[b, i] = 1.0

            if st == 1:
                if atk_frames <= 0:
                    val = vel
                    if hold_frames > 0:
                        st = 2
                    elif dec_frames > 0:
                        st = 3
                    else:
                        st = 4
                    tim = 0.0
                else:
                    step = vel / float(atk_frames if atk_frames > 0 else 1)
                    val += step
                    if val > vel:
                        val = vel
                    tim += 1.0
                    if tim >= atk_frames:
                        val = vel
                        if hold_frames > 0:
                            st = 2
                        elif dec_frames > 0:
                            st = 3
                        else:
                            st = 4
                        tim = 0.0
            elif st == 2:
                val = vel
                if hold_frames <= 0:
                    if dec_frames > 0:
                        st = 3
                    else:
                        st = 4
                    tim = 0.0
                else:
                    tim += 1.0
                    if tim >= hold_frames:
                        if dec_frames > 0:
                            st = 3
                        else:
                            st = 4
                        tim = 0.0
            elif st == 3:
                target = vel * sustain_level
                if dec_frames <= 0:
                    val = target
                    st = 4
                    tim = 0.0
                else:
                    delta = (vel - target) / float(dec_frames if dec_frames > 0 else 1)
                    candidate = val - delta
                    if candidate < target:
                        candidate = target
                    val = candidate
                    tim += 1.0
                    if tim >= dec_frames:
                        val = target
                        st = 4
                        tim = 0.0
            elif st == 4:
                val = vel * sustain_level
                if sus_frames > 0:
                    tim += 1.0
                    if tim >= sus_frames:
                        st = 5
                        rel_start_val = val
                        tim = 0.0
                elif not gate_on and not drone_on:
                    st = 5
                    rel_start_val = val
                    tim = 0.0
            elif st == 5:
                if rel_frames <= 0:
                    val = 0.0
                    st = 0
                    tim = 0.0
                else:
                    step = rel_start_val / float(rel_frames if rel_frames > 0 else 1)
                    candidate = val - step
                    if candidate < 0.0:
                        candidate = 0.0
                    val = candidate
                    tim += 1.0
                    if tim >= rel_frames:
                        val = 0.0
                        st = 0
                        tim = 0.0
                if gate_on or drone_on:
                    st = 1
                    tim = 0.0
                    val = 0.0
                    vel = float(velocity_buf[b, i])
                    if vel < 0.0:
                        vel = 0.0
                    rel_start_val = vel
                    acts += 1
                    if send_resets:
                        out_reset[b, i] = 1.0

            if val < 0.0:
                val = 0.0
            out_amp[b, i] = val

        stage_buf[b] = st
        value_buf[b] = val
        timer_buf[b] = tim
        vel_state_buf[b] = vel
        activ_buf[b] = acts
        rel_buf[b] = rel_start_val

    return out_amp, out_reset


