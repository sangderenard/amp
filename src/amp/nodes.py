"""Builtin audio node implementations."""

from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

import numpy as np

DEFAULT_DTYPE = np.float64


class AudioNode:
    """Base class for audio graph nodes."""

    def __init__(self, name: str) -> None:
        self.name = name

    def render(self, frames: int, sample_rate: int, inputs: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class SilenceNode(AudioNode):
    """Produce a silent buffer."""

    def __init__(self, name: str, params: Dict[str, float] | None = None) -> None:
        super().__init__(name)
        self.channels = int((params or {}).get("channels", 1))

    def render(self, frames: int, sample_rate: int, inputs: Sequence[np.ndarray]) -> np.ndarray:  # noqa: D401 - see base class
        return np.zeros((self.channels, frames), dtype=DEFAULT_DTYPE)


class SineOscillatorNode(AudioNode):
    """Simple sine oscillator."""

    def __init__(self, name: str, params: Dict[str, float] | None = None) -> None:
        super().__init__(name)
        params = params or {}
        self.frequency = float(params.get("frequency", 440.0))
        self.amplitude = float(params.get("amplitude", 0.5))
        self._phase = float(params.get("phase", 0.0)) % 1.0

    def render(self, frames: int, sample_rate: int, inputs: Sequence[np.ndarray]) -> np.ndarray:  # noqa: D401 - see base class
        phase_inc = self.frequency / float(sample_rate)
        steps = np.arange(frames, dtype=DEFAULT_DTYPE)
        phases = (self._phase + steps * phase_inc) % 1.0
        self._phase = float((phases[-1] + phase_inc) % 1.0)
        wave = np.sin(2.0 * math.pi * phases).astype(DEFAULT_DTYPE, copy=False)
        return (wave * self.amplitude)[None, :]


class ConstantNode(AudioNode):
    """Emit a constant signal."""

    def __init__(self, name: str, params: Dict[str, float] | None = None) -> None:
        super().__init__(name)
        params = params or {}
        self.value = float(params.get("value", 0.0))
        self.channels = int(params.get("channels", 1))

    def render(self, frames: int, sample_rate: int, inputs: Sequence[np.ndarray]) -> np.ndarray:  # noqa: D401 - see base class
        buffer = np.full((self.channels, frames), self.value, dtype=DEFAULT_DTYPE)
        return buffer


def _match_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.shape[0] == channels:
        return data
    if data.shape[0] == 1:
        return np.repeat(data, channels, axis=0)
    if data.shape[0] > channels:
        return data[:channels]
    pad = np.zeros((channels - data.shape[0], data.shape[1]), dtype=data.dtype)
    return np.concatenate([data, pad], axis=0)


class MixNode(AudioNode):
    """Sum inputs into a fixed number of channels."""

    def __init__(self, name: str, params: Dict[str, float] | None = None) -> None:
        super().__init__(name)
        params = params or {}
        self.channels = int(params.get("channels", 2))
        self.gain = float(params.get("gain", 1.0))

    def render(self, frames: int, sample_rate: int, inputs: Sequence[np.ndarray]) -> np.ndarray:  # noqa: D401 - see base class
        out = np.zeros((self.channels, frames), dtype=DEFAULT_DTYPE)
        for buf in inputs:
            if buf.ndim != 2:
                raise ValueError(f"Input to {self.name} must be 2D (channels, frames), got {buf.shape}")
            out += _match_channels(buf, self.channels)
        out *= self.gain
        return out


class SafetyNode(AudioNode):
    """Last stage that ensures stability and channel count."""

    def __init__(self, name: str, params: Dict[str, float] | None = None) -> None:
        super().__init__(name)
        params = params or {}
        self.channels = int(params.get("channels", 2))
        self.dc_alpha = float(params.get("dc_alpha", 0.995))
        self._state = np.zeros(self.channels, dtype=DEFAULT_DTYPE)

    def render(self, frames: int, sample_rate: int, inputs: Sequence[np.ndarray]) -> np.ndarray:  # noqa: D401 - see base class
        if inputs:
            data = _match_channels(inputs[0], self.channels)
        else:
            data = np.zeros((self.channels, frames), dtype=DEFAULT_DTYPE)
        # simple DC blocking per channel
        out = np.empty_like(data)
        for ch in range(self.channels):
            prev = self._state[ch]
            for i in range(frames):
                prev = self.dc_alpha * prev + (1.0 - self.dc_alpha) * data[ch, i]
                out[ch, i] = data[ch, i] - prev
            self._state[ch] = prev
        np.clip(out, -1.0, 1.0, out=out)
        return out


NODE_TYPES: Dict[str, Callable[[str, Dict[str, float] | None], AudioNode]] = {
    "silence": SilenceNode,
    "constant": ConstantNode,
    "sine_oscillator": SineOscillatorNode,
    "sine": SineOscillatorNode,
    "mix": MixNode,
    "safety": SafetyNode,
}


__all__ = ["AudioNode", "NODE_TYPES"]
