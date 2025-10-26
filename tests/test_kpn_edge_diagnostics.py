from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from amp.native_runtime import AVAILABLE as NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native runtime unavailable")


def _load_demo_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "demo_kpn_kpn_native_correct.py"
    spec = importlib.util.spec_from_file_location("amp_demo_kpn", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_block(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        return np.require(arr[np.newaxis, np.newaxis, :], requirements=("C",))
    if arr.ndim == 2:
        return np.require(arr[:, np.newaxis, :], requirements=("C",))
    return np.require(arr, requirements=("C",))


def _build_base_params(module, frames: int, sample_rate: float) -> Dict[str, object]:
    time = np.linspace(0.0, (frames - 1) / sample_rate, frames, dtype=np.float64)
    pitch_curve = 220.0 + 110.0 * np.sin(2.0 * np.pi * time / max(time[-1] or 1.0, 1.0))
    pitch_slew = np.full(frames, 0.0, dtype=np.float64)

    driver_frequency = pitch_curve * 1.01
    driver_amplitude = 0.7 + 0.2 * np.sin(2.0 * np.pi * time * 0.25)
    driver_blend = np.linspace(0.1, 0.9, frames, dtype=np.float64)

    oscillator_freq = pitch_curve
    oscillator_amp = np.full(frames, 0.6, dtype=np.float64)
    oscillator_slew = np.full(frames, 6000.0, dtype=np.float64)

    ratio_curve = 1.0 + 0.02 * np.sin(2.0 * np.pi * time * 0.5)

    fft_defaults = {
        "divisor": np.ones(frames, dtype=np.float64),
        "divisor_imag": np.zeros(frames, dtype=np.float64),
        "phase_offset": np.zeros(frames, dtype=np.float64),
        "lower_band": np.zeros(frames, dtype=np.float64),
        "upper_band": np.ones(frames, dtype=np.float64),
        "filter_intensity": np.ones(frames, dtype=np.float64),
        "stabilizer": np.full(frames, 1.0e-9, dtype=np.float64),
    }

    params: Dict[str, object] = {
        "_B": 1,
        "_C": 1,
        module.pitch.name: {
            "pitch_hz": _as_block(pitch_curve),
            "slew_hz_per_s": _as_block(pitch_slew),
        },
        module.driver.name: {
            "frequency": _as_block(driver_frequency),
            "amplitude": _as_block(driver_amplitude),
            "render_mode": _as_block(driver_blend),
        },
        module.oscillator.name: {
            "freq": _as_block(oscillator_freq),
            "amp": _as_block(oscillator_amp),
            "slew": _as_block(oscillator_slew),
        },
        module.pitch_shift.name: {
            "ratio": _as_block(ratio_curve),
        },
        "fft": {key: _as_block(val) for key, val in fft_defaults.items()},
    }
    return params


def _collect_edges(graph) -> Iterable[Tuple[str, str]]:
    edges: list[Tuple[str, str]] = []
    audio_inputs = getattr(graph, "_audio_inputs", {})
    for target, sources in audio_inputs.items():
        for source in sources:
            edges.append((str(source), str(target)))
    return edges


def test_kpn_edge_waveforms_and_spectrograms(tmp_path: Path) -> None:
    sample_rate = 48_000.0
    frames = 256

    if "CC" not in os.environ:
        gcc = shutil.which("gcc")
        if gcc:
            os.environ["CC"] = gcc

    demo = _load_demo_module()

    base_graph, _ = demo.build_graph(int(sample_rate))
    edges = list(_collect_edges(base_graph))
    assert edges, "Expected at least one audio edge in the KPN graph"

    reports: Dict[str, Dict[str, float]] = {}
    expected_files: set[str] = set()

    for source, target in edges:
        graph, module = demo.build_graph(int(sample_rate))
        graph.set_sink(source)
        params = _build_base_params(module, frames, sample_rate)
        block = graph.render_block(frames, sample_rate=int(sample_rate), base_params=params)
        assert block.shape[-1] == frames

        waveform = np.asarray(block[0, 0], dtype=np.float64)
        peak = float(np.max(np.abs(waveform)))
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        assert peak > 1.0e-6
        assert rms > 1.0e-6

        spec_image = demo.compute_spectrogram(waveform, sample_rate, window_size=64, hop=16)
        assert spec_image.ndim == 2
        assert spec_image.shape[0] == 33
        assert spec_image.shape[1] > 0

        edge_key = f"{source}->{target}"
        reports[edge_key] = {"peak": peak, "rms": rms}

        wave_path = tmp_path / f"{edge_key}_waveform.npy"
        spec_path = tmp_path / f"{edge_key}_spectrogram.npy"
        np.save(wave_path, waveform)
        np.save(spec_path, spec_image)
        expected_files.update({wave_path.name, spec_path.name})

    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})

    stats_path = tmp_path / "edge_stats.json"
    stats_path.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    assert stats_path.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv))
