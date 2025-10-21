"""Visualisation utilities for agent benchmark timeline data.

This module accepts the :class:`pandas.DataFrame` returned from
``scripts.agent_benchmark.benchmark_default_graph`` and builds an interactive
timeline figure exposing the features that were previously described in prose:

* A time-aligned throughput ribbon that encodes how much audio was produced
  relative to the render cost of each block.
* Highlight overlays that flag "bursty" windows where the synthesiser is
  especially active.
* A cumulative production line that makes it easy to compare the realised
  schedule with the amount of audio that should have been emitted.
* Interactive controls that allow the viewer to tune the burst detection
  threshold without having to recompute the summary data.

The functions are intentionally isolated from any CLI so that they can be used
directly from notebooks, tests, or future tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots


_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "iteration",
        "is_warmup",
        "scheduled_start_ms",
        "scheduled_end_ms",
        "realised_start_ms",
        "realised_end_ms",
        "wall_start_ms",
        "wall_end_ms",
        "render_ms",
        "block_ms",
        "buffer_ahead_ms",
        "underrun_gap_ms",
        "cumulative_gap_ms",
        "audio_peak",
        "audio_rms",
        "momentary_active",
        "drone_active",
        "velocity_mean",
        "pitch_mean",
        "start_sample",
        "end_sample",
    }
)


@dataclass(frozen=True)
class PreparedTimeline:
    """Container for timeline data that is ready to plot.

    Attributes
    ----------
    frame:
        Copy of the source data sorted by realised start time with convenience
        columns added.
    midpoints:
        1-D array containing the centre of each render block in milliseconds.
    widths:
        Duration of each render block in milliseconds; used for the bar widths
        of the throughput ribbon.
    """

    frame: pd.DataFrame
    midpoints: np.ndarray
    widths: np.ndarray


def prepare_timeline_dataframe(timeline_df: pd.DataFrame) -> PreparedTimeline:
    """Validate and enrich the timeline dataframe.

    Parameters
    ----------
    timeline_df:
        DataFrame produced by :func:`scripts.agent_benchmark.benchmark_default_graph`.

    Returns
    -------
    PreparedTimeline
        Wrapper containing a sorted copy of the dataframe and derived columns
        used by :func:`create_timeline_figure`.
    """

    missing = _REQUIRED_COLUMNS.difference(timeline_df.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(
            "Timeline dataframe is missing required columns: " f"{missing_columns}"
        )

    frame = timeline_df.copy().sort_values("realised_start_ms").reset_index(drop=True)

    start = frame["realised_start_ms"].to_numpy(dtype=np.float64)
    end = frame["realised_end_ms"].to_numpy(dtype=np.float64)
    duration = end - start

    # Guard against zero or negative durations by falling back to the block
    # duration (which should match the scheduled duration).
    fallback_duration = frame["block_ms"].to_numpy(dtype=np.float64)
    duration = np.where(duration > 0.0, duration, fallback_duration)

    render = frame["render_ms"].to_numpy(dtype=np.float64)
    block = frame["block_ms"].to_numpy(dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        availability_ratio = np.divide(block, render, where=render > 0.0)
        efficiency_ratio = np.divide(render, block, where=block > 0.0)

    availability_ratio = np.nan_to_num(availability_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    efficiency_ratio = np.nan_to_num(efficiency_ratio, nan=0.0, posinf=0.0, neginf=0.0)

    # The burst score emphasises moments where the performer is actively
    # manipulating the instrument. Velocity gives us a measure of intensity,
    # while the gate ensures we ignore latent controller noise.
    gate = frame["momentary_active"].astype(float)
    velocity = frame["velocity_mean"].to_numpy(dtype=np.float64)
    burst_score = np.clip(gate * np.maximum(velocity, 0.0), 0.0, None)

    midpoints = (start + end) / 2.0
    cumulative_produced = np.cumsum(block)
    rolling_window = max(1, min(16, int(np.ceil(len(frame) * 0.05)) or 1))
    throughput_ma = pd.Series(availability_ratio).rolling(rolling_window, min_periods=1).mean()

    frame["timeline_duration_ms"] = duration
    frame["timeline_midpoint_ms"] = midpoints
    frame["availability_ratio"] = availability_ratio
    frame["efficiency_ratio"] = efficiency_ratio
    frame["burst_score"] = burst_score
    frame["cumulative_produced_ms"] = cumulative_produced
    frame["throughput_ma"] = throughput_ma.to_numpy(dtype=np.float64)

    return PreparedTimeline(frame=frame, midpoints=midpoints, widths=duration)


def _burst_intervals(frame: pd.DataFrame, *, threshold: float) -> List[Tuple[float, float]]:
    """Collapse the dataframe into contiguous intervals that exceed *threshold*."""

    starts: List[float] = []
    ends: List[float] = []

    active = False
    current_start = 0.0
    current_end = 0.0

    for row in frame.itertuples(index=False):
        score = float(getattr(row, "burst_score"))
        start_ms = float(getattr(row, "realised_start_ms"))
        end_ms = float(getattr(row, "realised_end_ms"))
        if score >= threshold and not active:
            current_start = start_ms
            active = True
        if active:
            current_end = end_ms
        if active and score < threshold:
            starts.append(current_start)
            ends.append(start_ms)
            active = False
    if active:
        starts.append(current_start)
        ends.append(current_end)

    return list(zip(starts, ends))


def _build_slider_steps(frame: pd.DataFrame, thresholds: Sequence[float]) -> List[dict]:
    """Construct slider steps that swap the highlight rectangles."""

    steps: List[dict] = []
    for value in thresholds:
        shapes = _interval_shapes(_burst_intervals(frame, threshold=value))
        text = f"Burst threshold ≥ {value:.2f}"
        steps.append(
            {
                "method": "relayout",
                "label": f"{value:.2f}",
                "args": [
                    {
                        "shapes": shapes,
                        "annotations": [
                            {
                                "xref": "paper",
                                "yref": "paper",
                                "x": 0.99,
                                "y": 1.05,
                                "showarrow": False,
                                "text": text,
                                "font": {"size": 12},
                            }
                        ],
                    }
                ],
            }
        )
    return steps


def _interval_shapes(intervals: Iterable[Tuple[float, float]]) -> List[dict]:
    shapes: List[dict] = []
    for start, end in intervals:
        if end <= start:
            continue
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": start,
                "x1": end,
                "y0": 0.0,
                "y1": 1.0,
                "fillcolor": "rgba(255, 127, 80, 0.25)",
                "line": {"width": 0},
            }
        )
    return shapes


def create_timeline_figure(
    timeline_df: pd.DataFrame,
    *,
    burst_thresholds: Sequence[float] | None = None,
    show_warmup: bool = False,
) -> go.Figure:
    """Create an interactive figure describing the benchmark timeline.

    Parameters
    ----------
    timeline_df:
        Raw dataframe returned by :func:`benchmark_default_graph`.
    burst_thresholds:
        Optional collection of thresholds to expose in the interactive slider.
        The default picks sensible breakpoints based on the observed burst
        scores.
    show_warmup:
        When ``True`` the warmup iterations are rendered with reduced opacity
        instead of being hidden.
    """

    prepared = prepare_timeline_dataframe(timeline_df)
    frame = prepared.frame

    if not show_warmup:
        frame = frame.loc[~frame["is_warmup"]].reset_index(drop=True)
        if frame.empty:
            frame = prepared.frame

    midpoints = frame["timeline_midpoint_ms"].to_numpy(dtype=np.float64)
    widths = frame["timeline_duration_ms"].to_numpy(dtype=np.float64)
    throughput = frame["availability_ratio"].to_numpy(dtype=np.float64)
    throughput_ma = frame["throughput_ma"].to_numpy(dtype=np.float64)
    cumulative = frame["cumulative_produced_ms"].to_numpy(dtype=np.float64)

    hover_text = [
        "<b>Iteration</b>: {iteration}<br>"
        "<b>Realised</b>: {start:.1f} – {end:.1f} ms<br>"
        "<b>Render</b>: {render:.2f} ms<br>"
        "<b>Block</b>: {block:.2f} ms<br>"
        "<b>Availability</b>: {availability:.2f}×<br>"
        "<b>Buffer ahead</b>: {buffer:.2f} ms<br>"
        "<b>Audio peak</b>: {peak:.3f}<br>"
        "<b>Audio RMS</b>: {rms:.3f}"
        .format(
            iteration=row.iteration,
            start=row.realised_start_ms,
            end=row.realised_end_ms,
            render=row.render_ms,
            block=row.block_ms,
            availability=row.availability_ratio,
            buffer=row.buffer_ahead_ms,
            peak=row.audio_peak,
            rms=row.audio_rms,
        )
        for row in frame.itertuples(index=False)
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            name="Availability",
            x=midpoints,
            y=throughput,
            width=widths,
            marker={
                "color": throughput,
                "colorscale": "Blues",
                "cmin": 0.0,
                "cmax": max(1.0, float(np.percentile(throughput, 95)) or 1.0),
                "colorbar": {"title": "Block / Render"},
            },
            hoverinfo="text",
            hovertext=hover_text,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            name="Throughput (MA)",
            x=midpoints,
            y=throughput_ma,
            mode="lines",
            line={"width": 2, "color": "#ef553b"},
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            name="Cumulative produced",
            x=midpoints,
            y=cumulative,
            mode="lines",
            line={"width": 2, "color": "#2ca02c"},
            hovertemplate="<b>Cumulative</b>: %{y:.1f} ms<extra></extra>",
        ),
        secondary_y=True,
    )

    warmup_overlay: List[dict] = []
    if show_warmup and prepared.frame["is_warmup"].any():
        warm = prepared.frame.loc[prepared.frame["is_warmup"]]
        warmup_overlay = _interval_shapes(
            zip(warm["realised_start_ms"], warm["realised_end_ms"])
        )

    if burst_thresholds is None:
        min_score = float(frame["burst_score"].min())
        max_score = float(frame["burst_score"].max())
        if max_score <= min_score:
            burst_thresholds = [max_score]
        else:
            span = max_score - min_score
            step = span / 3.0
            burst_thresholds = [min_score + step * i for i in range(1, 4)]

    slider_steps = _build_slider_steps(frame, burst_thresholds)
    initial_shapes = (_interval_shapes(_burst_intervals(frame, threshold=burst_thresholds[0]))
        if burst_thresholds
        else []
    )

    fig.update_layout(
        title="Real-time benchmark timeline",
        bargap=0.05,
        xaxis={"title": "Realised time (ms)"},
        yaxis={"title": "Availability (× real-time)", "rangemode": "tozero"},
        yaxis2={"title": "Cumulative produced (ms)", "rangemode": "tozero"},
        legend={"orientation": "h", "y": -0.2, "x": 0.0},
        hovermode="x unified",
        template="plotly_white",
        shapes=initial_shapes + warmup_overlay,
        sliders=[
            {
                "active": 0,
                "y": -0.15,
                "len": 0.6,
                "pad": {"t": 30},
                "currentvalue": {"prefix": "Burst threshold: "},
                "steps": slider_steps,
            }
        ]
        if burst_thresholds
        else [],
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.99,
                "y": 1.05,
                "showarrow": False,
                "text": f"Burst threshold ≥ {burst_thresholds[0]:.2f}",
                "font": {"size": 12},
            }
        ]
        if burst_thresholds
        else [],
    )

    if warmup_overlay:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=1.05,
            text="Warmup iterations",
            showarrow=False,
            font={"size": 12},
        )

    return fig


__all__ = ["PreparedTimeline", "prepare_timeline_dataframe", "create_timeline_figure"]

