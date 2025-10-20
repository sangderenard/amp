"""Configuration loading for the synthesiser."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping
import json

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.json"


@dataclass(slots=True)
class JoystickConfig:
    """Optional joystick mapping configuration."""

    enabled: bool = False
    axes: Mapping[str, int] = field(default_factory=dict)
    buttons: Mapping[str, int] = field(default_factory=dict)


DEFAULT_FRAMES_PER_CHUNK = 512
DEFAULT_OUTPUT_CHANNELS = 2


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime parameters that are independent of the graph layout."""

    frames_per_chunk: int = DEFAULT_FRAMES_PER_CHUNK
    output_channels: int = DEFAULT_OUTPUT_CHANNELS
    joystick: JoystickConfig = field(default_factory=JoystickConfig)
    log_summary: bool = False


@dataclass(slots=True)
class NodeConfig:
    name: str
    type: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConnectionConfig:
    source: str
    target: str
    kind: str = "audio"  # future expansion


@dataclass(slots=True)
class GraphConfig:
    nodes: List[NodeConfig]
    connections: List[ConnectionConfig]
    sink: str


@dataclass(slots=True)
class AppConfig:
    sample_rate: int
    runtime: RuntimeConfig
    graph: GraphConfig


def _normalise_runtime(data: MutableMapping[str, Any]) -> RuntimeConfig:
    joystick_data = data.get("joystick", {}) or {}
    joystick = JoystickConfig(
        enabled=bool(joystick_data.get("enabled", False)),
        axes=dict(joystick_data.get("axes", {}) or {}),
        buttons=dict(joystick_data.get("buttons", {}) or {}),
    )
    return RuntimeConfig(
        frames_per_chunk=int(data.get("frames_per_chunk", DEFAULT_FRAMES_PER_CHUNK)),
        output_channels=int(data.get("output_channels", DEFAULT_OUTPUT_CHANNELS)),
        joystick=joystick,
        log_summary=bool(data.get("log_summary", False)),
    )


def _normalise_graph(data: Mapping[str, Any]) -> GraphConfig:
    node_items = data.get("nodes", [])
    if not node_items:
        raise ValueError("graph.nodes must contain at least one node definition")
    nodes = [
        NodeConfig(
            name=str(item["name"]),
            type=str(item["type"]),
            params=dict(item.get("params", {}) or {}),
        )
        for item in node_items
    ]
    connections = [
        ConnectionConfig(
            source=str(item["source"]),
            target=str(item["target"]),
            kind=str(item.get("kind", "audio")),
        )
        for item in data.get("connections", [])
    ]
    sink = str(data.get("sink"))
    if not sink:
        raise ValueError("graph.sink must be provided")
    return GraphConfig(nodes=nodes, connections=connections, sink=sink)


def load_configuration(path: str | Path) -> AppConfig:
    """Load an :class:`AppConfig` from ``path``."""

    with open(path, "r", encoding="utf8") as fh:
        raw = json.load(fh)
    runtime = _normalise_runtime(dict(raw.get("runtime", {}) or {}))
    graph = _normalise_graph(raw.get("graph", {}))
    sample_rate = int(raw.get("sample_rate", 44100))
    return AppConfig(sample_rate=sample_rate, runtime=runtime, graph=graph)


__all__ = [
    "AppConfig",
    "ConnectionConfig",
    "DEFAULT_CONFIG_PATH",
    "GraphConfig",
    "JoystickConfig",
    "NodeConfig",
    "RuntimeConfig",
    "load_configuration",
]
