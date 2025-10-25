"""Regression tests exercising advanced oscillator modes and driver node via the native runtime."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cffi")
from cffi import FFI  # type: ignore  # noqa: E402

from amp.native_runtime import AVAILABLE as NATIVE_AVAILABLE, _ensure_library_path  # noqa: E402


pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native runtime unavailable")


def _load_native_interface() -> tuple[FFI, object]:
    ffi = FFI()
    ffi.cdef(
        """
        typedef unsigned int uint32_t;
        typedef unsigned long long size_t;

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

        typedef struct EdgeRunnerControlHistory EdgeRunnerControlHistory;

        typedef struct {
            uint32_t measured_delay_frames;
            float accumulated_heat;
            float reserved[6];
        } AmpNodeMetrics;

        typedef enum {
            AMP_EXECUTION_MODE_FORWARD = 0,
            AMP_EXECUTION_MODE_BACKWARD = 1
        } AmpExecutionMode;

        int amp_run_node_v2(
            const EdgeRunnerNodeDescriptor *descriptor,
            const EdgeRunnerNodeInputs *inputs,
            int batches,
            int channels,
            int frames,
            double sample_rate,
            double **out_buffer,
            int *out_channels,
            void **state,
            const EdgeRunnerControlHistory *history,
            AmpExecutionMode mode,
            AmpNodeMetrics *metrics
        );

        void amp_free(double *buffer);
        void amp_release_state(void *state);
    """
    )
    lib = ffi.dlopen(str(_ensure_library_path()))
    return ffi, lib


def _run_native_node(
    ffi: FFI,
    lib: object,
    *,
    type_name: str,
    descriptor_params: dict[str, object],
    param_arrays: dict[str, np.ndarray],
    frames: int,
    sample_rate: float = 48_000.0,
    audio_input: np.ndarray | None = None,
    mode: int = 0,
) -> np.ndarray:
    descriptor = ffi.new("EdgeRunnerNodeDescriptor *")
    name_bytes = b"test_node"
    type_bytes = type_name.encode("utf-8")
    params_json = (
        "{"
        + ",".join(f"\"{key}\":{_json_encode(value)}" for key, value in sorted(descriptor_params.items()))
        + "}"
    ).encode("utf-8")

    name_buf = ffi.new("char[]", name_bytes)
    type_buf = ffi.new("char[]", type_bytes)
    params_buf = ffi.new("char[]", params_json)
    descriptor.name = name_buf
    descriptor.name_len = len(name_bytes)
    descriptor.type_name = type_buf
    descriptor.type_len = len(type_bytes)
    descriptor.params_json = params_buf
    descriptor.params_len = len(params_json)

    batches = 1
    channels = 1

    if audio_input is not None:
        audio_arr = np.ascontiguousarray(audio_input, dtype=np.float64)
        if audio_arr.ndim != 2:
            raise ValueError("audio_input must be shaped (frames, channels)")
        frames_in, channels_in = audio_arr.shape
        if frames_in != frames:
            raise ValueError(f"audio_input expected {frames} frames, received {frames_in}")
        batches = 1
        channels = channels_in
        audio_ptr = ffi.from_buffer("double[]", audio_arr.ravel())
        audio_view = ffi.new("EdgeRunnerAudioView *")
        audio_view.has_audio = 1
        audio_view.batches = batches
        audio_view.channels = channels_in
        audio_view.frames = frames
        audio_view.data = audio_ptr
        audio_keepalive = [audio_arr, audio_ptr]
    else:
        audio_view = ffi.new("EdgeRunnerAudioView *")
        audio_view.has_audio = 0
        audio_view.batches = 0
        audio_view.channels = 0
        audio_view.frames = 0
        audio_view.data = ffi.NULL
        audio_keepalive = []

    param_items = []
    keepalive = audio_keepalive + [name_buf, type_buf, params_buf]
    for name, array in param_arrays.items():
        arr = np.ascontiguousarray(array, dtype=np.float64)
        if arr.size != frames:
            raise ValueError(f"{name}: expected {frames} samples, received {arr.size}")
        ptr = ffi.from_buffer("double[]", arr)
        name_buf = ffi.new("char[]", name.encode("utf-8"))
        view = ffi.new("EdgeRunnerParamView *")
        view.name = name_buf
        view.batches = batches
        view.channels = 1
        view.frames = frames
        view.data = ptr
        param_items.append(view[0])
        keepalive.extend((arr, ptr, name_buf))

    if param_items:
        param_array = ffi.new("EdgeRunnerParamView[]", param_items)
    else:
        param_array = ffi.NULL

    param_set = ffi.new("EdgeRunnerParamSet *")
    param_set.count = len(param_items)
    param_set.items = param_array if param_array != ffi.NULL else ffi.NULL

    inputs = ffi.new("EdgeRunnerNodeInputs *")
    inputs.audio = audio_view[0]
    inputs.params = param_set[0]

    out_buffer = ffi.new("double **")
    out_channels = ffi.new("int *")
    state_ptr = ffi.new("void **")
    metrics = ffi.new("AmpNodeMetrics *")

    try:
        rc = lib.amp_run_node_v2(
            descriptor,
            inputs,
            batches,
            channels,
            frames,
            float(sample_rate),
            out_buffer,
            out_channels,
            state_ptr,
            ffi.NULL,
            ffi.cast("AmpExecutionMode", mode),
            metrics,
        )
        if int(rc) != 0 or out_buffer[0] == ffi.NULL:
            raise RuntimeError(f"amp_run_node_v2 failed for {type_name} with rc={int(rc)}")
        total = int(out_channels[0]) * frames
        result = np.frombuffer(
            ffi.buffer(out_buffer[0], total * np.dtype(np.float64).itemsize),
            dtype=np.float64,
        ).copy()
    finally:
        if out_buffer[0] != ffi.NULL:
            lib.amp_free(out_buffer[0])
        if state_ptr[0] != ffi.NULL:
            lib.amp_release_state(state_ptr[0])

    return result.reshape(frames, int(out_channels[0]))


def _json_encode(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError("JSON cannot encode NaN or infinity")
        return repr(float(value)) if isinstance(value, float) else str(int(value))
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_json_encode(item) for item in value) + "]"
    raise TypeError(f"Unsupported descriptor parameter type: {type(value)!r}")


def test_polyblep_phase_offset_shifts_waveform():
    ffi, lib = _load_native_interface()
    frames = 16
    sample_rate = 48_000.0
    freq = np.full(frames, 440.0, dtype=np.float64)
    amp = np.ones(frames, dtype=np.float64)
    base = _run_native_node(
        ffi,
        lib,
        type_name="OscNode",
        descriptor_params={"wave": "sine", "mode": "polyblep", "accept_reset": True},
        param_arrays={"freq": freq, "amp": amp},
        frames=frames,
        sample_rate=sample_rate,
    )
    offset = _run_native_node(
        ffi,
        lib,
        type_name="OscNode",
        descriptor_params={"wave": "sine", "mode": "polyblep", "accept_reset": True},
        param_arrays={
            "freq": freq,
            "amp": amp,
            "phase_offset": np.full(frames, 0.25, dtype=np.float64),
        },
        frames=frames,
        sample_rate=sample_rate,
    )
    step = freq[0] / sample_rate
    expected_shift = math.sin(2.0 * math.pi * (step + 0.25))
    assert pytest.approx(float(offset[0, 0]), rel=1e-6, abs=1e-6) == expected_shift
    assert np.mean(np.abs(base - offset)) > 0.1


def test_integrator_mode_smoothes_output():
    ffi, lib = _load_native_interface()
    frames = 128
    sample_rate = 48_000.0
    freq = np.full(frames, 55.0, dtype=np.float64)
    amp = np.ones(frames, dtype=np.float64)
    baseline = _run_native_node(
        ffi,
        lib,
        type_name="OscNode",
        descriptor_params={"wave": "saw", "mode": "polyblep", "accept_reset": True},
        param_arrays={"freq": freq, "amp": amp},
        frames=frames,
        sample_rate=sample_rate,
    )
    integrated = _run_native_node(
        ffi,
        lib,
        type_name="OscNode",
        descriptor_params={
            "wave": "saw",
            "mode": "integrator",
            "integration_leak": 0.995,
            "integration_gain": 0.25,
            "integration_clamp": 1.0,
            "accept_reset": True,
        },
        param_arrays={"freq": freq, "amp": amp},
        frames=frames,
        sample_rate=sample_rate,
    )
    assert integrated.shape == baseline.shape
    assert np.mean(np.abs(baseline - integrated)) > 1e-2


def test_op_amp_mode_applies_slew_limits():
    ffi, lib = _load_native_interface()
    frames = 64
    driver = np.linspace(0.0, 1.0, frames, dtype=np.float64).reshape(frames, 1)
    freq = np.zeros(frames, dtype=np.float64)
    amp = np.ones(frames, dtype=np.float64)
    slew_rate = 200.0
    output = _run_native_node(
        ffi,
        lib,
        type_name="OscNode",
        descriptor_params={
            "wave": "sine",
            "mode": "op_amp",
            "slew_rate": slew_rate,
            "slew_clamp": 1.0,
            "accept_reset": False,
        },
        param_arrays={"freq": freq, "amp": amp},
        frames=frames,
        audio_input=driver,
    )
    per_frame_limit = slew_rate / 48_000.0
    diffs = np.diff(output[:, 0])
    assert np.all(np.abs(diffs) <= per_frame_limit + 1e-9)


def test_parametric_driver_harmonics_affect_waveform():
    ffi, lib = _load_native_interface()
    frames = 128
    sample_rate = 48_000.0
    base = _run_native_node(
        ffi,
        lib,
        type_name="ParametricDriverNode",
        descriptor_params={"mode": "quartz"},
        param_arrays={"frequency": np.full(frames, 220.0), "amplitude": np.ones(frames)},
        frames=frames,
        sample_rate=sample_rate,
    )
    custom = _run_native_node(
        ffi,
        lib,
        type_name="ParametricDriverNode",
        descriptor_params={"mode": "custom", "harmonics": "1.0,0.5"},
        param_arrays={"frequency": np.full(frames, 220.0), "amplitude": np.ones(frames)},
        frames=frames,
        sample_rate=sample_rate,
    )
    assert base.shape == custom.shape
    assert np.mean(np.abs(base - custom)) > 1e-2
