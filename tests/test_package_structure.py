"""Package surface tests ensuring single application layout."""

import importlib

import pytest


def test_interactive_namespace_missing():
    """The interactive subpackage should no longer exist."""

    spec = importlib.util.find_spec("amp.interactive")
    assert spec is None


def test_run_app_exposed():
    """The package should expose the run_app helper at top level."""

    amp = importlib.import_module("amp")
    assert hasattr(amp, "run_app")
    assert callable(amp.run_app)


@pytest.mark.parametrize("module", ["app", "cli", "nodes", "graph"])
def test_modules_reside_in_amp(module: str):
    """Modules should resolve directly from the amp package."""

    spec = importlib.util.find_spec(f"amp.{module}")
    assert spec is not None, f"amp.{module} should be importable"
