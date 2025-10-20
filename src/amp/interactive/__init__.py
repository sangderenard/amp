"""Joystick-driven interactive application components."""


def run_app(*args, **kwargs):
    """Invoke the interactive joystick-controlled application."""

    from .app import run

    return run(*args, **kwargs)


__all__ = ["run_app"]
