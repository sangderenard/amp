"""Generate a high-resolution FFT spectrogram PNG (and optional WAV) via the native runtime.

This utility mirrors the regression harness in ``tests/test_fft_spectral_node.py`` but exposes
command-line knobs so you can render an example spectrogram outside the test suite.  It:

* synthesises per-frame / per-band metadata (batch × channel × band × frame order)
* invokes ``amp_run_node_v2`` in backward mode to reconstruct the time-domain signal
* writes a spectrogram PNG with >= 1024 vertical frequency pixels
* optionally writes a float32 WAV containing a mono mixdown of the synthesised audio

Examples
--------

  python tools/generate_fft_png.py
  python tools/generate_fft_png.py --bands 333 --frames 4096 --window 2048 --png output/fft_demo.png --wav output/fft_demo.wav
"""
from __future__ import annotations

import argparse
import importlib.util
import wave
from pathlib import Path

import numpy as np


def load_test_module():
    spec = importlib.util.spec_from_file_location(
        "test_fft_spectral_node",
        Path(__file__).resolve().parents[1] / "tests" / "test_fft_spectral_node.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--png",
        default="output/fft_spectral_superposition.png",
        help="Destination spectrogram PNG path (default: output/fft_spectral_superposition.png)",
    )
    p.add_argument(
        "--wav",
        default=None,
        help="Optional destination WAV path (float32 mono mixdown). If omitted, no WAV is written.",
    )
    p.add_argument("--bands", type=int, default=333, help="Number of spectral bands / channels (default: 333)")
    p.add_argument("--batches", type=int, default=1, help="Batch dimension for the node invocation (default: 1)")
    p.add_argument("--frames", type=int, default=4096, help="Number of frames to generate (default: 4096)")
    p.add_argument(
        "--window",
        type=int,
        default=2048,
        help="STFT window size used by the node metadata (default: 2048 → >=1024 bins)",
    )
    p.add_argument(
        "--oversample",
        type=int,
        default=16,
        help="Oversample ratio recorded in the descriptor metadata (default: 16)",
    )
    p.add_argument(
        "--width-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the synthetic band widths (<1 → thinner, >1 → wider; default: 1.0)",
    )
    p.add_argument(
        "--sample-rate",
        type=float,
        default=48_000.0,
        help="Sample rate for synthesis + WAV output (default: 48000)",
    )
    p.add_argument(
        "--hop",
        type=int,
        default=None,
        help="Optional STFT hop size. Defaults to window//16 if unspecified.",
    )
    args = p.parse_args()

    png_path = Path(args.png)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    wav_path: Path | None = None
    if args.wav:
        wav_path = Path(args.wav)
        wav_path.parent.mkdir(parents=True, exist_ok=True)

    mod = load_test_module()
    ffi, lib = mod._load_fft_interface()

    frames = int(args.frames)
    window_size = int(args.window)
    oversample_ratio = int(args.oversample)
    batches = int(args.batches)
    channels = int(args.bands)
    slot_count = batches * channels
    sample_rate = float(args.sample_rate)
    hop = int(args.hop) if args.hop is not None else max(1, window_size // 16)

    (
        time_axis,
        release_schedule,
        gate,
        audio_slots,
        curves,
    ) = mod._generate_spectral_instruction_set(
        frames,
        batches,
        channels,
        window_size,
        float(args.width_scale),
    )

    # Build descriptor + inputs (copied from the test)
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

    audio_flat = mod._flatten_frameslots(audio_slots)
    audio_ptr = ffi.from_buffer("double[]", audio_flat)
    audio_view = ffi.new("EdgeRunnerAudioView *")
    audio_view.has_audio = 1
    audio_view.batches = batches
    audio_view.channels = channels
    audio_view.frames = frames
    audio_view.data = audio_ptr

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
    keepalive = [name_buf, type_buf, params_buf, audio_ptr]
    for idx, name in enumerate(param_names):
        array = mod._flatten_frameslots(curves[name])
        param_name_buf = ffi.new("char[]", name.encode("utf-8"))
        buf_ptr = ffi.from_buffer("double[]", array)
        view = param_views[idx]
        view.name = param_name_buf
        view.batches = batches
        view.channels = channels
        view.frames = frames
        view.data = buf_ptr
        keepalive.extend([param_name_buf, array, buf_ptr])

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

    if int(rc) != 0:
        raise RuntimeError(f"amp_run_node_v2 failed with rc={rc}")

    if out_buffer[0] == ffi.NULL:
        raise RuntimeError("native node returned no buffer")

    total = slot_count * frames
    pcm = np.frombuffer(
        ffi.buffer(out_buffer[0], total * np.dtype(np.float64).itemsize),
        dtype=np.float64,
    ).copy()

    try:
        pcm_matrix = pcm.reshape(frames, slot_count)
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

        with np.errstate(divide="ignore"):
            log_spectra = 20.0 * np.log10(np.maximum(spectra, 1.0e-12))
        log_spectra -= log_spectra.max()
        min_val = float(log_spectra.min())
        if np.isclose(min_val, 0.0, atol=1.0e-12):
            min_val = -1.0
        scaled = np.clip(log_spectra / min_val, 0.0, 1.0)
        normalised = 1.0 - scaled
        image = np.flipud((normalised.T * 255.0).astype(np.uint8))

        # write output PNG using helper from the test module
        mod._write_grayscale_png(png_path, image)
        print("Wrote PNG to:", png_path)

        if wav_path is not None:
            mono = np.mean(pcm_matrix, axis=1)
            peak = np.max(np.abs(mono))
            if peak > 0:
                mono = mono / peak * 0.98  # headroom
            mono_f32 = mono.astype(np.float32)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(4)  # float32
                wf.setframerate(int(sample_rate))
                wf.writeframes(mono_f32.tobytes())
            print("Wrote WAV to:", wav_path)

    finally:
        if out_buffer[0] != ffi.NULL:
            lib.amp_free(out_buffer[0])
        if state_ptr[0] != ffi.NULL:
            lib.amp_release_state(state_ptr[0])


if __name__ == "__main__":
    main()
