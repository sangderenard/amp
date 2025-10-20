"""Interactive state defaults and constants."""

from __future__ import annotations

import os
from types import ModuleType
from typing import Any, Dict

# =========================
# Settings / fidelity
# =========================
RAW_DTYPE = "float64"
MAX_FRAMES = 4096

# =========================
# Persistence
# =========================
MAPPINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mappings.json")
MAX_UNDO = 10

# =========================
# Tuning/Mode token helpers
# =========================
FREE_VARIANTS = ("continuous", "weighted", "stepped")

_DEFAULT_WAVES = ["sine", "square", "saw", "triangle"]
_DEFAULT_FILTER_TYPES = ["lowpass", "highpass", "bandpass", "notch", "peaking"]
_DEFAULT_MOD_WAVES = ["sine", "square", "saw", "triangle"]


def _axes_index_or_default(joy: Any, candidate: int, default: int) -> int:
    """Return the candidate axis index when available, otherwise the fallback."""

    if joy is None:
        return default
    try:
        axes = joy.get_numaxes()
    except Exception:  # pragma: no cover - defensive
        return default
    if axes is None:
        return default
    try:
        axes = int(axes)
    except (TypeError, ValueError):
        return default
    if axes > candidate:
        return min(candidate, axes - 1)
    return default


def build_default_state(*, joy: Any, pygame: ModuleType) -> Dict[str, Any]:
    """Return the default application state.

    Parameters
    ----------
    joy:
        The joystick instance (real or virtual).  Axis counts are used to
        determine sensible defaults for filter controls.
    pygame:
        The pygame module, used so callers can supply the initialized module
        without this helper importing pygame during module import.
    """

    sample_file = os.path.join(os.path.dirname(__file__), "sample.wav")

    return {
        "base_token": "12tet/full",
        "root_midi": 60,
        "free_variant": "continuous",
        "waves": list(_DEFAULT_WAVES),
        "wave_idx": 0,
        "filter_types": list(_DEFAULT_FILTER_TYPES),
        "filter_type": "lowpass",
        "filter_axis_cutoff": _axes_index_or_default(joy, 4, 3),
        "filter_axis_q": _axes_index_or_default(joy, 5, 4),
        "peaking_gain_db": 6.0,
        "source_type": "osc",
        "sample_file": sample_file,
        "mod_wave_types": list(_DEFAULT_MOD_WAVES),
        "mod_wave_idx": 0,
        "mod_rate_hz": 4.0,
        "mod_depth": 0.5,
        "mod_route": "both",
        "mod_use_input": False,
        "mod_slew_ms": 5.0,
        "keymap": {
            "toggle_menu": pygame.K_m,
            "open_keymap": pygame.K_k,
            "wave_next": pygame.K_x,
            "mode_next": pygame.K_y,
            "drone_toggle": pygame.K_b,
            "toggle_source": pygame.K_n,
            "free_variant_next": pygame.K_z,
            "root_up": pygame.K_PERIOD,
            "root_down": pygame.K_COMMA,
            "root_reset": pygame.K_SLASH,
        },
        "buttonmap": {
            4: {"token": "12tet/full"},
            5: {"token": "FREE"},
        },
        "bumper_priority": [4, 5],
        "double_tap_window": 0.33,
        "free_variant_button": 6,
        "polyphony_mode": "strings",
        "polyphony_voices": 3,
        "envelope_params": {
            "attack_ms": 12.0,
            "hold_ms": 8.0,
            "decay_ms": 90.0,
            "sustain_level": 0.65,
            "sustain_ms": 0.0,
            "release_ms": 220.0,
            "send_resets": True,
        },
    }


__all__ = [
    "RAW_DTYPE",
    "MAX_FRAMES",
    "MAPPINGS_FILE",
    "MAX_UNDO",
    "FREE_VARIANTS",
    "build_default_state",
]
