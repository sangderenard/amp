"""High level application orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import AppConfig, load_configuration
from .graph import AudioGraph


@dataclass(slots=True)
class SynthApplication:
    """Runtime container for the synthesiser graph.

    The application loads a configuration, builds the audio graph and
    exposes a small API for rendering buffers in-process.  No external
    controllers or audio devices are required which keeps tests and
    command line usage deterministic.
    """

    config: AppConfig
    graph: AudioGraph

    @classmethod
    def from_config(cls, config: AppConfig) -> "SynthApplication":
        graph = AudioGraph.from_config(config.graph, config.sample_rate, config.runtime.output_channels)
        return cls(config=config, graph=graph)

    @classmethod
    def from_file(cls, path: str) -> "SynthApplication":
        return cls.from_config(load_configuration(path))

    def render(self, frames: Optional[int] = None) -> np.ndarray:
        """Render *frames* samples from the configured graph.

        Parameters
        ----------
        frames:
            Optional frame count.  When omitted the runtime default from the
            configuration is used.
        """

        frame_count = frames or self.config.runtime.frames_per_chunk
        return self.graph.render(frame_count)

    def summary(self) -> str:
        """Return a human readable description of the graph."""

        lines = [
            f"Sample rate: {self.config.sample_rate} Hz",
            f"Output channels: {self.config.runtime.output_channels}",
            f"Frames per chunk: {self.config.runtime.frames_per_chunk}",
            "Nodes:",
        ]
        for node in self.graph.ordered_nodes:
            lines.append(f"  - {node.name} ({node.__class__.__name__})")
        return "\n".join(lines)


__all__ = ["SynthApplication"]
