"""Builtin node implementations for the shared audio graph."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np

DEFAULT_DTYPE = np.float64


def _ensure_bcf(audio_in: np.ndarray | None, frames: int) -> tuple[np.ndarray | None, int]:
    """Normalise *audio_in* to ``(B, C, F)`` for downstream processing."""

    if audio_in is None:
        return None, 1
    arr = np.asarray(audio_in, dtype=DEFAULT_DTYPE)
    if arr.ndim == 1:
        arr = arr[None, None, :]
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Expected audio input to have rank 1, 2, or 3; got {arr.ndim}")
    if arr.shape[2] != frames:
        raise ValueError(f"Audio input frame mismatch: got {arr.shape[2]}, expected {frames}")
    return arr, arr.shape[0]


class AudioNode:
    """Base class for graph nodes that operate on ``(B, C, F)`` buffers."""

    def __init__(self, name: str, params: Mapping[str, float] | None = None) -> None:
        self.name = name
        self.params = dict(params or {})

    def process(
        self,
        frames: int,
        sample_rate: int,
        audio_in: np.ndarray | None,
        mods: Mapping[str, Sequence[tuple[np.ndarray, float, str]]],
        params: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError


class SilenceNode(AudioNode):
    """Produce a silent buffer with the configured channel count."""

    def __init__(self, name: str, params: Mapping[str, float] | None = None) -> None:
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 1))

    def process(self, frames: int, sample_rate: int, audio_in, mods, params):  # noqa: D401 - see base class
        _, batches = _ensure_bcf(audio_in, frames)
        return np.zeros((batches, self.channels, frames), dtype=DEFAULT_DTYPE)


class ConstantNode(AudioNode):
    """Emit a constant signal for each requested channel."""

    def __init__(self, name: str, params: Mapping[str, float] | None = None) -> None:
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 1))
        self.value = float(self.params.get("value", 0.0))

    def process(self, frames: int, sample_rate: int, audio_in, mods, params):  # noqa: D401 - see base class
        _, batches = _ensure_bcf(audio_in, frames)
        buffer = np.full((batches, self.channels, frames), self.value, dtype=DEFAULT_DTYPE)
        return buffer


class SineOscillatorNode(AudioNode):
    """Simple sine oscillator with optional overrides via ``params``."""

    def __init__(self, name: str, params: Mapping[str, float] | None = None) -> None:
        super().__init__(name, params)
        self.frequency = float(self.params.get("frequency", 440.0))
        self.amplitude = float(self.params.get("amplitude", 0.5))
        self._phase = float(self.params.get("phase", 0.0)) % 1.0

    def process(self, frames: int, sample_rate: int, audio_in, mods, params):  # noqa: D401 - see base class
        _, batches = _ensure_bcf(audio_in, frames)
        batch_size = batches
        freq = params.get("frequency")
        amp = params.get("amplitude")
        if freq is None:
            freq = np.full((batch_size, 1, frames), self.frequency, dtype=DEFAULT_DTYPE)
        if amp is None:
            amp = np.full((batch_size, 1, frames), self.amplitude, dtype=DEFAULT_DTYPE)
        freq = np.asarray(freq, dtype=DEFAULT_DTYPE)[:, 0, :]
        amp = np.asarray(amp, dtype=DEFAULT_DTYPE)[:, 0, :]

        dphi = freq / float(sample_rate)
        phase = (self._phase + np.cumsum(dphi, axis=1)) % 1.0
        self._phase = float((phase[0, -1] + dphi[0, -1]) % 1.0)
        wave = np.sin(2.0 * math.pi * phase, dtype=DEFAULT_DTYPE)
        return (wave * amp)[:, None, :]


def _match_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.shape[1] == channels:
        return data
    if data.shape[1] == 1:
        return np.repeat(data, channels, axis=1)
    if data.shape[1] > channels:
        return data[:, :channels, :]
    pad = np.zeros((data.shape[0], channels - data.shape[1], data.shape[2]), dtype=data.dtype)
    return np.concatenate([data, pad], axis=1)


class MixNode(AudioNode):
    """Sum all inputs into a target channel count."""

    def __init__(self, name: str, params: Mapping[str, float] | None = None) -> None:
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 2))
        self.gain = float(self.params.get("gain", 1.0))

    def process(self, frames: int, sample_rate: int, audio_in, mods, params):  # noqa: D401 - see base class
        data, batches = _ensure_bcf(audio_in, frames)
        if data is None:
            return np.zeros((batches, self.channels, frames), dtype=DEFAULT_DTYPE)
        summed = np.sum(data, axis=1, keepdims=True)
        if self.channels > 1:
            summed = np.repeat(summed, self.channels, axis=1)
        return summed * self.gain


class SafetyNode(AudioNode):
    """Final node that applies a simple DC blocker per channel."""

    def __init__(self, name: str, params: Mapping[str, float] | None = None) -> None:
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 2))
        self.dc_alpha = float(self.params.get("dc_alpha", 0.995))
        self._state: Dict[int, np.ndarray] = {}

    def process(self, frames: int, sample_rate: int, audio_in, mods, params):  # noqa: D401 - see base class
        data, batches = _ensure_bcf(audio_in, frames)
        if data is None:
            data = np.zeros((batches, self.channels, frames), dtype=DEFAULT_DTYPE)
        else:
            data = _match_channels(data, self.channels)

        out = np.empty_like(data)
        for batch in range(batches):
            state = self._state.setdefault(batch, np.zeros(self.channels, dtype=DEFAULT_DTYPE))
            for ch in range(self.channels):
                prev = state[ch]
                for i in range(frames):
                    prev = self.dc_alpha * prev + (1.0 - self.dc_alpha) * data[batch, ch, i]
                    out[batch, ch, i] = data[batch, ch, i] - prev
                state[ch] = prev
        np.clip(out, -1.0, 1.0, out=out)
        return out


NODE_TYPES: Dict[str, type[AudioNode]] = {
    "silence": SilenceNode,
    "constant": ConstantNode,
    "sine": SineOscillatorNode,
    "sine_oscillator": SineOscillatorNode,
    "mix": MixNode,
    "safety": SafetyNode,
}


__all__ = ["AudioNode", "NODE_TYPES"]
