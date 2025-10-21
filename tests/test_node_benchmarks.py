from __future__ import annotations

from amp import node_benchmarks


def test_run_node_benchmarks_smoke():
    results = node_benchmarks.run_node_benchmarks(
        [1],
        frames=32,
        sample_rate=48_000.0,
        iterations=1,
        node_names=["silence"],
        seed=123,
    )
    assert "silence" in results
    stats = results["silence"][1]
    assert stats.mean_seconds >= 0.0
    assert stats.max_seconds >= stats.min_seconds


def test_main_lists_nodes(capsys):
    exit_code = node_benchmarks.main(["--list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "silence" in out
