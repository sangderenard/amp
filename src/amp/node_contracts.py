"""Runtime contracts describing node IO expectations for the C edge runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class NodeContract:
    """Captures the static IO contract of a node implementation."""

    type_name: str
    channel_attributes: Tuple[str, ...] = ("channels", "out_channels")
    channel_params: Tuple[str, ...] = ("channels", "out_channels")
    stereo_params: Tuple[str, ...] = ()
    default_channels: int | None = None
    allow_python_fallback: bool = False
    notes: str | None = None

    def known_channel_keys(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(self.channel_attributes + self.channel_params))


class NodeContractRegistry:
    """Registry of node contracts keyed by descriptor type name."""

    def __init__(self) -> None:
        self._contracts: Dict[str, NodeContract] = {}

    def register(self, contract: NodeContract) -> None:
        if contract.type_name in self._contracts:
            raise ValueError(f"Duplicate contract registration for {contract.type_name}")
        self._contracts[contract.type_name] = contract

    def get(self, type_name: str) -> NodeContract | None:
        return self._contracts.get(type_name)

    def contracts(self) -> Iterable[NodeContract]:
        return tuple(self._contracts.values())


_REGISTRY = NodeContractRegistry()


def get_node_contract(type_name: str) -> NodeContract | None:
    """Return the registered contract for ``type_name`` (if any)."""

    return _REGISTRY.get(type_name)


def register_node_contract(contract: NodeContract) -> None:
    """Register ``contract`` for the associated descriptor type."""

    _REGISTRY.register(contract)


def _bootstrap() -> None:
    """Populate the registry with built-in runtime contracts."""

    register_node_contract(
        NodeContract(
            type_name="ConstantNode",
            default_channels=1,
            allow_python_fallback=False,
        )
    )
    register_node_contract(
        NodeContract(
            type_name="GainNode",
            allow_python_fallback=False,
        )
    )
    register_node_contract(
        NodeContract(
            type_name="MixNode",
            channel_attributes=("out_channels", "channels"),
            allow_python_fallback=False,
        )
    )
    register_node_contract(
        NodeContract(
            type_name="SafetyNode",
            channel_attributes=("channels",),
            allow_python_fallback=False,
            notes="SafetyNode must execute via the C backend to ensure realtime clipping.",
        )
    )
    register_node_contract(
        NodeContract(
            type_name="SineOscillatorNode",
            channel_attributes=("channels",),
            channel_params=("channels",),
            allow_python_fallback=False,
        )
    )
    register_node_contract(
        NodeContract(
            type_name="OscNode",
            channel_attributes=("channels", "out_channels"),
            channel_params=("channels", "out_channels"),
            stereo_params=("pan",),
            allow_python_fallback=False,
            notes="OscNode promotes to stereo whenever pan data is supplied.",
        )
    )
    register_node_contract(
        NodeContract(
            type_name="ControllerNode",
            channel_attributes=("channels",),
            allow_python_fallback=False,
        )
    )
    register_node_contract(
        NodeContract(
            type_name="LFONode",
            allow_python_fallback=False,
        )
    )
    register_node_contract(
        NodeContract(
            type_name="EnvelopeModulatorNode",
            allow_python_fallback=False,
        )
    )
    register_node_contract(
        NodeContract(
            type_name="PitchQuantizerNode",
            allow_python_fallback=False,
        )
    )
    register_node_contract(
        NodeContract(
            type_name="SubharmonicLowLifterNode",
            allow_python_fallback=False,
        )
    )


_bootstrap()
