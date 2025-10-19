"""Audio graph execution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .config import GraphConfig
from .nodes import NODE_TYPES, AudioNode


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    kind: str = "audio"


class AudioGraph:
    """Executable directed acyclic graph of :class:`AudioNode` objects."""

    def __init__(self, nodes: Dict[str, AudioNode], edges: Sequence[GraphEdge], sink: str, sample_rate: int, output_channels: int):
        self._nodes = nodes
        self._edges = list(edges)
        self.sink = sink
        self.sample_rate = int(sample_rate)
        self.output_channels = int(output_channels)
        self._order = self._topological_order()

    @classmethod
    def from_config(cls, config: GraphConfig, sample_rate: int, output_channels: int) -> "AudioGraph":
        nodes = {}
        for node_cfg in config.nodes:
            try:
                node_factory = NODE_TYPES[node_cfg.type.lower()]
            except KeyError as exc:
                raise KeyError(f"Unknown node type '{node_cfg.type}'") from exc
            nodes[node_cfg.name] = node_factory(node_cfg.name, node_cfg.params)
        edges = [GraphEdge(e.source, e.target, e.kind) for e in config.connections]
        if config.sink not in nodes:
            raise ValueError(f"Sink node '{config.sink}' is not defined in graph.nodes")
        return cls(nodes=nodes, edges=edges, sink=config.sink, sample_rate=sample_rate, output_channels=output_channels)

    def _topological_order(self) -> List[AudioNode]:
        incoming: Dict[str, int] = {name: 0 for name in self._nodes}
        adjacency: Dict[str, List[str]] = {name: [] for name in self._nodes}
        for edge in self._edges:
            incoming[edge.target] = incoming.get(edge.target, 0) + 1
            adjacency.setdefault(edge.source, []).append(edge.target)
        queue = [name for name, count in incoming.items() if count == 0]
        order: List[str] = []
        while queue:
            name = queue.pop(0)
            order.append(name)
            for succ in adjacency.get(name, []):
                incoming[succ] -= 1
                if incoming[succ] == 0:
                    queue.append(succ)
        if len(order) != len(self._nodes):
            raise ValueError("Graph contains cycles or undefined nodes")
        return [self._nodes[name] for name in order]

    @property
    def ordered_nodes(self) -> List[AudioNode]:
        return list(self._order)

    def render(self, frames: int) -> np.ndarray:
        cache: Dict[str, np.ndarray] = {}
        for node in self._order:
            inputs = [cache[src.source] for src in self._edges if src.target == node.name and src.kind == "audio"]
            cache[node.name] = node.render(frames, self.sample_rate, inputs)
        if self.sink not in cache:
            raise RuntimeError(f"Sink node '{self.sink}' was not rendered")
        data = cache[self.sink]
        if data.ndim != 2:
            raise ValueError(f"Node '{self.sink}' produced unexpected shape {data.shape}; expected (channels, frames)")
        channels, _ = data.shape
        if channels < self.output_channels:
            pad = np.zeros((self.output_channels - channels, data.shape[1]), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)
        elif channels > self.output_channels:
            data = data[: self.output_channels]
        return data


__all__ = ["AudioGraph"]
