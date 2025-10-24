"""Runtime diagnostics helpers for optional Python/C bridge logging."""
from __future__ import annotations

import threading
from pathlib import Path

__all__ = ["enable_py_c_logging", "py_c_logging_enabled", "log_py_c_call"]


_LOG_PY_C_CALLS = False
_LOG_PATH = Path("logs/py_c_calls.log")
_LOG_LOCK = threading.Lock()


def enable_py_c_logging(enabled: bool) -> None:
    """Enable or disable logging of Python/C bridge activity."""

    global _LOG_PY_C_CALLS
    flag = bool(enabled)
    _LOG_PY_C_CALLS = flag
    try:
        from . import native_runtime
    except Exception:
        return
    try:
        native_runtime.set_native_logging_enabled(flag)
    except Exception:
        return


def py_c_logging_enabled() -> bool:
    """Return ``True`` when Python/C bridge logging is enabled."""

    return _LOG_PY_C_CALLS


def log_py_c_call(message: str) -> None:
    """Append ``message`` to the bridge log when logging is enabled."""

    if not _LOG_PY_C_CALLS:
        return
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    try:
        with _LOG_LOCK:
            with _LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(f"{message}\n")
    except Exception:
        return
