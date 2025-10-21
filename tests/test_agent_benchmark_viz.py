import numpy as np
import pandas as pd

from amp.agent_benchmark_viz import create_timeline_figure, prepare_timeline_dataframe


def _sample_timeline(rows: int = 8) -> pd.DataFrame:
    data = []
    realised_start = 0.0
    for idx in range(rows):
        block_ms = 5.8 + idx * 0.1
        render_ms = 4.0 + (idx % 3) * 1.1
        realised_end = realised_start + block_ms
        wall_start = realised_start * 0.9
        wall_end = wall_start + render_ms
        data.append(
            {
                "iteration": idx,
                "is_warmup": idx < 2,
                "scheduled_start_ms": realised_start,
                "scheduled_end_ms": realised_end,
                "realised_start_ms": realised_start,
                "realised_end_ms": realised_end,
                "wall_start_ms": wall_start,
                "wall_end_ms": wall_end,
                "render_ms": render_ms,
                "block_ms": block_ms,
                "buffer_ahead_ms": max(0.0, block_ms - render_ms),
                "underrun_gap_ms": 0.0,
                "cumulative_gap_ms": 0.0,
                "audio_peak": 0.6 + 0.02 * idx,
                "audio_rms": 0.4 + 0.01 * idx,
                "momentary_active": idx % 2 == 0,
                "drone_active": False,
                "velocity_mean": 0.3 + 0.05 * idx,
                "pitch_mean": 0.1,
                "start_sample": idx * 256,
                "end_sample": (idx + 1) * 256,
            }
        )
        realised_start = realised_end
    return pd.DataFrame.from_records(data)


def test_prepare_timeline_dataframe_adds_expected_columns():
    prepared = prepare_timeline_dataframe(_sample_timeline())
    frame = prepared.frame

    assert np.allclose(frame["timeline_duration_ms"], frame["block_ms"])
    assert np.all(frame["availability_ratio"] > 0.0)
    assert np.all(frame["throughput_ma"] > 0.0)
    assert "burst_score" in frame.columns
    assert prepared.midpoints.shape[0] == len(frame)
    assert prepared.widths.shape[0] == len(frame)


def test_create_timeline_figure_has_expected_traces_and_slider():
    df = _sample_timeline()
    fig = create_timeline_figure(df, burst_thresholds=[0.2, 0.4, 0.6])

    assert len(fig.data) == 3
    bar, throughput_line, cumulative_line = fig.data
    assert bar.type == "bar"
    assert throughput_line.name == "Throughput (MA)"
    assert cumulative_line.name == "Cumulative produced"

    assert fig.layout.sliders, "Expected a burst threshold slider"
    slider = fig.layout.sliders[0]
    assert len(slider.steps) == 3

