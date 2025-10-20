"""Optional joystick support built on top of :mod:`pygame`."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Dict, Mapping

from .config import JoystickConfig


class JoystickUnavailableError(RuntimeError):
    """Raised when joystick support is requested but not available."""


@dataclass(slots=True)
class JoystickState:
    """Snapshot of the current joystick inputs."""

    axes: Dict[str, float]
    buttons: Dict[str, bool]


def _load_pygame() -> ModuleType:
    try:
        module = import_module("pygame")
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise JoystickUnavailableError("pygame is not installed") from exc
    if module is None:  # pragma: no cover - defensive
        raise JoystickUnavailableError("pygame import returned None")
    return module


class JoystickController:
    """Thin wrapper around ``pygame.joystick`` with named mappings."""

    def __init__(self, pygame_module: ModuleType, device: object, axes: Mapping[str, int], buttons: Mapping[str, int]) -> None:
        self._pygame = pygame_module
        self._device = device
        self._axes = dict(axes)
        self._buttons = dict(buttons)

    @classmethod
    def create(cls, config: JoystickConfig) -> "JoystickController":
        pygame = _load_pygame()
        if not pygame.get_init():  # pragma: no cover - depends on pygame behaviour
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count == 0:
            raise JoystickUnavailableError("No joystick devices detected")
        device = pygame.joystick.Joystick(0)
        if hasattr(device, "init"):
            device.init()
        return cls(pygame, device, config.axes, config.buttons)

    def poll(self) -> JoystickState:
        """Return the current state of all mapped axes and buttons."""

        if hasattr(self._pygame, "event") and hasattr(self._pygame.event, "pump"):
            self._pygame.event.pump()
        axes = {name: float(self._device.get_axis(index)) for name, index in self._axes.items()}
        buttons = {name: bool(self._device.get_button(index)) for name, index in self._buttons.items()}
        return JoystickState(axes=axes, buttons=buttons)

    def close(self) -> None:
        if hasattr(self._device, "quit"):
            self._device.quit()


__all__ = ["JoystickController", "JoystickState", "JoystickUnavailableError"]

