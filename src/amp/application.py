"""High level application orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import AppConfig, load_configuration
from .graph import AudioGraph
from .app import build_runtime_graph
from .joystick import JoystickController, JoystickState, JoystickUnavailableError


@dataclass(slots=True)
class SynthApplication:
    """Runtime container for the synthesiser graph and C-ready node buffers.

    The application loads a configuration, builds the audio graph, and
    exposes a small API for rendering C-ready output buffers in-process.
    No external controllers or audio devices are required, which keeps tests and
    command line usage deterministic and ensures all per-node data is C-ready.
    """

    config: AppConfig
    graph: AudioGraph
    joystick: Optional[JoystickController] = None
    joystick_error: Optional[str] = None

    @classmethod
    def from_config(cls, config: AppConfig) -> "SynthApplication":
        # Prefer the interactive/default graph construction when possible.
        # Many configs (including the default distributed config) are a
        # lightweight headless graph; the interactive graph is the canonical
        # runtime layout. If the config provides a custom graph explicitly
        # we will still respect it, otherwise construct the interactive
        # runtime graph using `build_runtime_graph` so headless outputs
        # match interactive behaviour.
        graph: AudioGraph
        try:
            cfg_nodes = getattr(config.graph, "nodes", None) or []
            runtime_state = {
                "root_midi": 60,
                "free_variant": "continuous",
                "base_token": "12tet/full",
                "waves": ["sine", "square", "saw"],
                "wave_idx": 0,
                "filter_type": "lowpass",
                "mod_wave_types": ["sine", "square", "saw"],
                "mod_wave_idx": 0,
                "mod_rate_hz": 4.0,
                "mod_depth": 0.5,
                "mod_use_input": False,
                "polyphony_mode": "strings",
                "polyphony_voices": 3,
            }
            if getattr(config.graph, "use_runtime_graph", False) or not cfg_nodes:
                graph, _, _ = build_runtime_graph(config.sample_rate, runtime_state)
            else:
                graph = AudioGraph.from_config(
                    config.graph, config.sample_rate, config.runtime.output_channels
                )
        except Exception:
            # Fallback to config-driven construction on any error
            graph = AudioGraph.from_config(config.graph, config.sample_rate, config.runtime.output_channels)
        joystick: Optional[JoystickController] = None
        joystick_error: Optional[str] = None
        if config.runtime.joystick.enabled:
            try:
                joystick = JoystickController.create(config.runtime.joystick)
            except JoystickUnavailableError as exc:
                joystick_error = str(exc)
        return cls(config=config, graph=graph, joystick=joystick, joystick_error=joystick_error)

    @classmethod
    def from_file(cls, path: str) -> "SynthApplication":
        return cls.from_config(load_configuration(path))

    def render(self, frames: Optional[int] = None) -> np.ndarray:
        """
        Render a C-ready output buffer for the given number of frames from the configured graph.

        Parameters
        ----------
        frames:
            Optional frame count. When omitted, the runtime default from the
            configuration is used.
        Returns
        -------
        np.ndarray
            C-ready output buffer (node buffer) for the rendered audio block.
        """
        frame_count = frames or self.config.runtime.frames_per_chunk
        # The graph.render method must return a C-ready output buffer (node buffer)
        return self.graph.render(frame_count)

    def poll_joystick(self) -> Optional[JoystickState]:
        """Return the latest joystick state when a controller is available."""

        if not self.joystick:
            return None
        return self.joystick.poll()

    def summary(self) -> str:
        """
        Return a human-readable description of the graph and C-ready node buffers.
        """
        lines = [
            f"Sample rate: {self.config.sample_rate} Hz",
            f"Output channels: {self.config.runtime.output_channels}",
            f"Frames per chunk: {self.config.runtime.frames_per_chunk}",
        ]
        if self.config.runtime.joystick.enabled:
            if self.joystick:
                lines.append("Joystick: connected")
            else:
                lines.append(f"Joystick: unavailable ({self.joystick_error or 'not detected'})")
        else:
            lines.append("Joystick: disabled")
        lines.append("Nodes (C-ready node buffers):")
        for node in self.graph.ordered_nodes:
            lines.append(f"  - {node.name} ({node.__class__.__name__})")
        return "\n".join(lines)


__all__ = ["SynthApplication"]
