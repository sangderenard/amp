from pathlib import Path

from amp.config import DEFAULT_CONFIG_PATH, AppConfig, load_configuration


def test_default_configuration_loads(tmp_path: Path) -> None:
    config = load_configuration(DEFAULT_CONFIG_PATH)
    assert isinstance(config, AppConfig)
    assert config.sample_rate > 0
    assert config.runtime.output_channels == 2
    assert config.graph.sink


def test_configuration_requires_nodes(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"sample_rate": 44100, "graph": {"nodes": [], "sink": "out"}}')
    try:
        load_configuration(bad)
    except ValueError as exc:
        assert "graph.nodes" in str(exc)
    else:
        raise AssertionError("expected ValueError")
