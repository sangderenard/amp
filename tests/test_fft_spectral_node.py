"""High-resolution FFT spectral node regression exercising gated band reprints."""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cffi")
from cffi import FFI

from amp.native_runtime import AVAILABLE as NATIVE_AVAILABLE
from amp.native_runtime import _ensure_library_path


NATIVE_ONLY = pytest.mark.skipif(
    not NATIVE_AVAILABLE, reason="Native runtime unavailable"
)


def _smooth_step(x: np.ndarray, slope: float) -> np.ndarray:
    """Return a numerically-stable smooth step for ``x``."""

    return 0.5 * (1.0 + np.tanh(slope * x))


def _write_grayscale_png(path: Path, image: np.ndarray) -> None:
    """Persist ``image`` (H, W) as an 8-bit grayscale PNG."""

    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    height, width = image.shape
    header = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    idat = zlib.compress(scanlines, level=6)

    with path.open("wb") as stream:
        stream.write(header)
        stream.write(chunk(b"IHDR", ihdr))
        stream.write(chunk(b"IDAT", idat))
        stream.write(chunk(b"IEND", b""))


def _load_fft_interface() -> tuple[FFI, object]:
    """Return a cffi handle for interacting with ``amp_run_node_v2``."""

    ffi = FFI()
    ffi.cdef(
        """
        typedef unsigned int uint32_t;
        typedef unsigned char uint8_t;
        typedef unsigned long size_t;

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
    library_path = _ensure_library_path()
    lib = ffi.dlopen(str(library_path))
    return ffi, lib


def _generate_spectral_instruction_set(
    frames: int,
    batches: int,
    channels: int,
    window_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    """Return timeline, release schedule, gate, audio, and per-band parameter curves."""

    time_axis = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    time_grid = time_axis[:, None, None]
    batch_norm = (np.arange(batches, dtype=np.float64) + 0.5) / batches
    channel_norm = (np.arange(channels, dtype=np.float64) + 0.5) / channels
    batch_grid = batch_norm[None, :, None]
    channel_grid = channel_norm[None, None, :]
    slot_batch_flat = np.repeat(batch_norm, channels)
    slot_channel_flat = np.tile(channel_norm, batches)

    slot_mix = 0.42 * batch_grid + 0.58 * channel_grid
    release_grid = 0.015 + 0.78 * slot_mix
    stage_offsets = np.array([0.0, 0.045, 0.09, 0.135, 0.18], dtype=np.float64)
    stage_weights = np.array([0.19, 0.21, 0.22, 0.21, 0.17], dtype=np.float64)
    gate = np.zeros((frames, batches, channels), dtype=np.float64)
    for weight, offset in zip(stage_weights, stage_offsets):
        gate += weight * _smooth_step(time_grid - (release_grid + offset), slope=72.0)
    gate = np.clip(gate, 0.0, 1.0)

    micro_pulses = np.zeros_like(gate)
    for idx, offset in enumerate(stage_offsets):
        sigma = 0.003 + 0.004 * (idx + 1)
        micro_pulses += np.exp(-((time_grid - (release_grid + offset)) ** 2) / (2.0 * sigma**2))
    micro_pulses /= micro_pulses.max(initial=1.0)

    base_center = 0.05 + 0.9 * slot_mix
    center_drift = 0.04 * np.sin(
        2.0 * math.pi * (time_grid * (channel_grid * 6.0 + 1.25) + batch_grid * 0.65)
    )
    center = np.clip(base_center + center_drift * (0.45 + 0.55 * gate), 0.0, 1.0)

    width_base = (
        0.02
        + 0.2 * gate
        + 0.05 * micro_pulses
        + 0.03 * np.sin(2.0 * math.pi * time_grid * (batch_grid * 1.5 + channel_grid * 3.5))
    )
    width = np.clip(width_base, 0.01, 0.7)
    lower = np.clip(center - 0.5 * width, 0.0, 0.999)
    upper = np.clip(center + 0.5 * width, 0.001, 1.0)
    upper = np.maximum(upper, lower + 5.0e-4)

    intensity = np.clip(
        (gate**1.8)
        * (0.35 + 0.65 * micro_pulses)
        * (0.65 + 0.35 * np.cos(2.0 * math.pi * time_grid * (channel_grid * 2.5 + batch_grid * 0.75)) ** 2),
        0.0,
        1.0,
    )

    phase_velocity = 2.0 * math.pi * (0.25 + 0.5 * gate + 0.3 * micro_pulses)
    phase_velocity *= 1.0 + 0.18 * np.sin(
        2.0 * math.pi * time_grid * (channel_grid * 3.0 + batch_grid) + channel_grid * 8.0
    )
    phase_velocity = np.abs(phase_velocity) + 1.0e-6
    phase_offset = np.cumsum(phase_velocity, axis=0)

    divisor = 0.62 + 0.28 * gate + 0.08 * np.sin(2.0 * math.pi * time_grid * (channel_grid * 4.0 + batch_grid * 0.5))
    divisor = np.clip(divisor, 0.18, None)
    divisor_imag = 0.17 * gate * np.cos(2.0 * math.pi * time_grid * (channel_grid * 5.0 + batch_grid * 0.3))
    stabilizer = (1.0e-9 + gate * 3.5e-9) * (1.0 + 0.15 * micro_pulses)

    slot_count = batches * channels
    rng = np.random.default_rng(0x5A_FEED)
    noise = rng.normal(0.0, 1.0, size=(frames, slot_count))
    alpha = math.exp(-6.0 / max(1, window_size))
    audio_matrix = np.empty((frames, slot_count), dtype=np.float64)
    state = np.zeros(slot_count, dtype=np.float64)
    for frame_idx in range(frames):
        state = alpha * state + (1.0 - alpha) * noise[frame_idx]
        audio_matrix[frame_idx] = state
    state.fill(0.0)
    for frame_idx in range(frames - 1, -1, -1):
        state = alpha * state + (1.0 - alpha) * audio_matrix[frame_idx]
        audio_matrix[frame_idx] = state

    envelope = (0.42 + 0.58 * gate.reshape(frames, -1)) * (0.33 + 0.67 * micro_pulses.reshape(frames, -1))
    audio_matrix *= envelope
    sin_term = np.sin(
        2.0
        * math.pi
        * (
            time_axis[:, None] * (slot_channel_flat[None, :] * 12.0 + slot_batch_flat[None, :] * 2.0)
            + 0.5 * phase_offset.reshape(frames, -1)
        )
    )
    audio_matrix += 0.1 * gate.reshape(frames, -1) * sin_term

    audio = audio_matrix.reshape(frames, batches, channels)

    release_schedule = np.ascontiguousarray(release_grid[0], dtype=np.float64)
    curves = {
        "divisor": divisor,
        "divisor_imag": divisor_imag,
        "phase_offset": phase_offset,
        "lower_band": lower,
        "upper_band": upper,
        "filter_intensity": intensity,
        "stabilizer": stabilizer,
    }

    return time_axis, release_schedule, gate, audio, curves


def _flatten_frameslots(array: np.ndarray) -> np.ndarray:
    """Return ``array`` flattened in (frame, slot) order."""

    if array.ndim == 3:
        frames = array.shape[0]
        reshaped = array.reshape(frames, -1)
    elif array.ndim == 2:
        reshaped = array
    else:
        raise ValueError("array must have two or three dimensions")
    return np.ascontiguousarray(reshaped.reshape(-1), dtype=np.float64)


@NATIVE_ONLY
def test_fft_spectral_node_generates_high_resolution_spectrogram(tmp_path: Path) -> None:
    ffi, lib = _load_fft_interface()

    frames = 512
    window_size = 256
    oversample_ratio = 4
    batches = 32
    channels = 384
    slot_count = batches * channels
    sample_rate = 48_000.0

    (
        time_axis,
        release_schedule,
        gate,
        audio_slots,
        curves,
    ) = _generate_spectral_instruction_set(frames, batches, channels, window_size)

    instructions = frames * slot_count
    assert instructions >= 10_000
    assert slot_count >= 10_000

    for batch_idx in range(batches):
        for channel_idx in range(channels):
            release_point = float(release_schedule[batch_idx, channel_idx])
            before_mask = time_axis < max(0.0, release_point - 0.01)
            after_mask = time_axis > min(1.0, release_point + 0.05)
            intensity_slice = curves["filter_intensity"][:, batch_idx, channel_idx]
            if np.any(before_mask):
                assert float(intensity_slice[before_mask].max()) < 0.035
            if np.any(after_mask):
                assert float(intensity_slice[after_mask].mean()) > 0.06
            phase_slice = curves["phase_offset"][:, batch_idx, channel_idx]
            assert np.all(np.diff(phase_slice) > 0.0)
            lower_slice = curves["lower_band"][:, batch_idx, channel_idx]
            upper_slice = curves["upper_band"][:, batch_idx, channel_idx]
            assert np.all(lower_slice + 5.0e-4 <= upper_slice)

    descriptor = ffi.new("EdgeRunnerNodeDescriptor *")
    name_buf = ffi.new("char[]", b"fft_spectral")
    type_buf = ffi.new("char[]", b"FFTDivisionNode")
    params_json = (
        "{" "\"window_size\":"
        + str(window_size)
        + ",\"stabilizer\":1e-9,\"epsilon\":1e-12,\"declared_delay\":"
        + str(window_size - 1)
        + ",\"oversample_ratio\":"
        + str(oversample_ratio)
        + ",\"supports_v2\":true}"
    ).encode("utf-8")
    params_buf = ffi.new("char[]", params_json)
    descriptor.name = name_buf
    descriptor.name_len = len(b"fft_spectral")
    descriptor.type_name = type_buf
    descriptor.type_len = len(b"FFTDivisionNode")
    descriptor.params_json = params_buf
    descriptor.params_len = len(params_json)

    keepalive: list[object] = [name_buf, type_buf, params_buf]

    audio_flat = _flatten_frameslots(audio_slots)
    audio_ptr = ffi.from_buffer("double[]", audio_flat)
    audio_view = ffi.new("EdgeRunnerAudioView *")
    audio_view.has_audio = 1
    audio_view.batches = batches
    audio_view.channels = channels
    audio_view.frames = frames
    audio_view.data = audio_ptr
    keepalive.append(audio_ptr)

    param_names = (
        "divisor",
        "divisor_imag",
        "phase_offset",
        "lower_band",
        "upper_band",
        "filter_intensity",
        "stabilizer",
    )

    param_views = ffi.new("EdgeRunnerParamView[]", len(param_names))
    for idx, name in enumerate(param_names):
        array = _flatten_frameslots(curves[name])
        param_name_buf = ffi.new("char[]", name.encode("utf-8"))
        keepalive.append(param_name_buf)
        buf_ptr = ffi.from_buffer("double[]", array)
        view = param_views[idx]
        view.name = param_name_buf
        view.batches = batches
        view.channels = channels
        view.frames = frames
        view.data = buf_ptr
        keepalive.extend([array, buf_ptr])

    param_set = ffi.new("EdgeRunnerParamSet *")
    param_set.count = len(param_names)
    param_set.items = param_views

    inputs = ffi.new("EdgeRunnerNodeInputs *")
    inputs.audio = audio_view[0]
    inputs.params = param_set[0]

    out_buffer = ffi.new("double **")
    out_channels = ffi.new("int *")
    state_ptr = ffi.new("void **")
    metrics = ffi.new("AmpNodeMetrics *")

    rc = lib.amp_run_node_v2(
        descriptor,
        inputs,
        batches,
        channels,
        frames,
        sample_rate,
        out_buffer,
        out_channels,
        state_ptr,
        ffi.NULL,
        ffi.cast("AmpExecutionMode", 1),
        metrics,
    )

    try:
        assert rc == 0, f"amp_run_node_v2 failed with rc={rc}"
        assert out_buffer[0] != ffi.NULL
        assert int(out_channels[0]) == channels
        total = slot_count * frames
        pcm = np.frombuffer(
            ffi.buffer(out_buffer[0], total * np.dtype(np.float64).itemsize),
            dtype=np.float64,
        ).copy()
    finally:
        if out_buffer[0] != ffi.NULL:
            lib.amp_free(out_buffer[0])
        if state_ptr[0] != ffi.NULL:
            lib.amp_release_state(state_ptr[0])

    pcm_matrix = pcm.reshape(frames, slot_count)
    rms = np.sqrt(np.mean(pcm_matrix**2, axis=0))
    assert np.all(rms > 1.0e-4)
    assert rms.max() / rms.min() < 6.0

    assert metrics.measured_delay_frames == window_size - 1
    phase_flat = curves["phase_offset"].reshape(frames, -1)
    lower_flat = curves["lower_band"].reshape(frames, -1)
    upper_flat = curves["upper_band"].reshape(frames, -1)
    intensity_flat = curves["filter_intensity"].reshape(frames, -1)
    assert pytest.approx(metrics.reserved[0], rel=1e-6) == float(phase_flat[-1, -1])
    assert pytest.approx(metrics.reserved[1], rel=1e-6) == float(lower_flat[-1, -1])
    assert pytest.approx(metrics.reserved[2], rel=1e-6) == float(upper_flat[-1, -1])
    assert pytest.approx(metrics.reserved[3], rel=1e-6) == float(intensity_flat[-1, -1])
    assert pytest.approx(metrics.reserved[4], rel=1e-6) == float(window_size)

    hop = window_size // 16
    window = np.hanning(window_size)
    segment_count = 1 + (frames - window_size) // hop
    spectra = np.empty((segment_count, window_size // 2 + 1), dtype=np.float64)
    for idx in range(segment_count):
        start = idx * hop
        segment = pcm_matrix[start : start + window_size]
        tapered = segment * window[:, None]
        fft = np.fft.rfft(tapered, axis=0)
        magnitude = np.sqrt(np.mean(np.abs(fft) ** 2, axis=1))
        spectra[idx] = magnitude

    assert spectra.shape[1] >= window_size // 2 + 1
    assert float(np.max(spectra)) > 0.0

    with np.errstate(divide="ignore"):
        log_spectra = 20.0 * np.log10(np.maximum(spectra, 1.0e-12))
    log_spectra -= log_spectra.max()
    min_val = float(log_spectra.min())
    if math.isclose(min_val, 0.0, abs_tol=1.0e-12):
        min_val = -1.0
    scaled = np.clip(log_spectra / min_val, 0.0, 1.0)
    normalised = 1.0 - scaled
    image = np.flipud((normalised.T * 255.0).astype(np.uint8))

    target = tmp_path / "fft_spectral_superposition.png"
    _write_grayscale_png(target, image)

    assert target.exists()
    assert target.stat().st_size > 0

    with target.open("rb") as stream:
        signature = stream.read(8)
    assert signature == b"\x89PNG\r\n\x1a\n"
