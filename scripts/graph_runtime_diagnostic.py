#!/usr/bin/env python3
"""Deterministic graph renderer for CFFI diagnostics.

This helper instantiates the same interactive graph used by the AMP UI and drives
it with neutral controller curves so the CFFI edge-runner can be compiled and
exercised without pygame or joystick hardware.  It records the generated C source,
compiled plan, and timing metadata to make native crashes reproducible.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.amp import app as amp_app
from src.amp import nodes
from src.amp import state as app_state
from src.amp.application import SynthApplication
from src.amp.config import load_configuration
from src.amp.graph import AudioGraph
from src.amp.runner import render_audio_block

_CONTROL_KEYS: Tuple[str, ...] = (
    "trigger",
    "gate",
    "drone",
    "velocity",
    "cutoff",
    "q",
    "pitch_input",
    "pitch_span",
    "pitch_root",
)


class _StubPygame:
    """Minimal pygame shim so build_default_state does not import pygame."""

    K_m = ord("m")
    K_k = ord("k")
    K_x = ord("x")
    K_y = ord("y")
    K_b = ord("b")
    K_n = ord("n")
    K_z = ord("z")
    K_PERIOD = ord(".")
    K_COMMA = ord(",")
    K_SLASH = ord("/")


def _neutral_curves(frames: int, state: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Return a mapping of controller keys to neutral per-frame curves."""

    zeros = np.zeros(frames, dtype=np.float64)
    curves: Dict[str, np.ndarray] = {key: zeros.copy() for key in _CONTROL_KEYS}
    curves["pitch_span"][:] = float(state.get("free_span_oct", 2.0))
    curves["pitch_root"][:] = float(state.get("root_midi", 60))
    return curves


def _discover_envelopes(graph: AudioGraph) -> Tuple[str, ...]:
    return tuple(
        node.name
        for node in graph.ordered_nodes
        if isinstance(node, nodes.EnvelopeModulatorNode)
    )


def _build_interactive_graph(sample_rate: int) -> Tuple[AudioGraph, Dict[str, object], Tuple[str, ...], Tuple[str, ...]]:
    state = app_state.build_default_state(joy=None, pygame=_StubPygame())
    graph, envelope_names, amp_mod_names = amp_app.build_runtime_graph(sample_rate, state)
    return graph, state, tuple(envelope_names), tuple(amp_mod_names)


def _build_from_config(path: Path) -> Tuple[AudioGraph, Dict[str, object], int, int, Tuple[str, ...], Tuple[str, ...]]:
    cfg = load_configuration(path)
    app = SynthApplication.from_config(cfg)
    state = app_state.build_default_state(joy=None, pygame=_StubPygame())
    envelope_names = _discover_envelopes(app.graph)
    return (
        app.graph,
        state,
        cfg.sample_rate,
        cfg.runtime.frames_per_chunk,
        envelope_names,
        tuple(),
    )


def _materialise_artifacts(graph: AudioGraph, dump_dir: Path) -> Dict[str, object]:
    """Write CFFI artifacts (C source, descriptors, plan) into dump_dir."""

    dump_dir.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, object] = {}

    descriptors = graph.serialize_node_descriptors()
    descriptor_path = dump_dir / "node_descriptors.bin"
    descriptor_path.write_bytes(descriptors)
    metadata["descriptor_len"] = len(descriptors)

    runner = graph._ensure_edge_runner()
    compiled_plan = getattr(runner, "_compiled_plan", None)
    if compiled_plan:
        plan_path = dump_dir / "compiled_plan.bin"
        plan_path.write_bytes(compiled_plan)
        metadata["compiled_plan_len"] = len(compiled_plan)
        plan_desc = runner.describe_compiled_plan()
        (dump_dir / "compiled_plan.json").write_text(
            json.dumps(plan_desc, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metadata["compiled_plan"] = plan_desc
    else:
        metadata["compiled_plan_len"] = 0

    generated_sources: list[str] = []
    for candidate in (
        Path.cwd() / "_amp_ckernels_cffi.c",
        Path(__file__).resolve().parent.parent / "src" / "amp" / "_amp_ckernels_cffi.c",
    ):
        if candidate.exists():
            target = dump_dir / candidate.name
            if candidate.resolve() != target.resolve():
                shutil.copy2(candidate, target)
            else:
                # Already in place; ensure timestamp captured in metadata.
                target = candidate
            generated_sources.append(str(target))
    metadata["generated_sources"] = tuple(generated_sources)

    return metadata


def _render_blocks(
    graph: AudioGraph,
    state: Dict[str, object],
    *,
    frames: int,
    blocks: int,
    sample_rate: int,
    envelope_names: Iterable[str],
    amp_mod_names: Iterable[str],
    dump_dir: Path,
) -> Dict[str, object]:
    """Render ``blocks`` sequential blocks and persist diagnostics."""

    cache: Dict[str, np.ndarray] = {}
    summary: Dict[str, object] = {"blocks": []}
    neutral_template = _neutral_curves(frames, state)
    block_duration = frames / float(sample_rate)

    for block_index in range(blocks):
        event_time = block_index * block_duration
        neutral = {key: value.copy() for key, value in neutral_template.items()}
        graph.record_control_event(
            event_time,
            pitch=np.zeros(1, dtype=np.float64),
            envelope=np.zeros(1, dtype=np.float64),
            extras=neutral,
        )
        render_start = time.perf_counter()
        audio_block, meta = render_audio_block(
            graph,
            event_time,
            frames,
            sample_rate,
            neutral,
            state,
            list(envelope_names),
            list(amp_mod_names),
            cache,
        )
        render_end = time.perf_counter()
        if meta.get("render_duration") is None:
            meta["render_duration"] = render_end - render_start
        summary_entry = {
            "block_index": block_index,
            "render_duration": meta.get("render_duration"),
            "frames": frames,
            "node_timings": meta.get("node_timings", {}),
            "peak": float(np.abs(audio_block).max()) if audio_block.size else 0.0,
        }
        summary["blocks"].append(summary_entry)

        block_path = dump_dir / f"block_{block_index:03d}.npy"
        np.save(block_path, audio_block)
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the AMP interactive graph via the CFFI edge runner without pygame.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional configuration file. When omitted the interactive three-oscillator graph is used.",
    )
    parser.add_argument("--frames", type=int, default=None, help="Frames per render block (defaults to config or 1024).")
    parser.add_argument("--blocks", type=int, default=1, help="Number of sequential blocks to render.")
    parser.add_argument("--sample-rate", type=int, default=None, help="Override sample rate used for rendering.")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("logs/graph_diagnostic"),
        help="Directory for captured diagnostics (C sources, plans, PCM blocks).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path to write aggregated metadata JSON.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.config is not None:
        graph, state, cfg_sample_rate, cfg_frames, envelope_names, amp_mod_names = _build_from_config(args.config)
        sample_rate = args.sample_rate or cfg_sample_rate
        frames = args.frames or cfg_frames
    else:
        sample_rate = args.sample_rate or 44100
        graph, state, envelope_names, amp_mod_names = _build_interactive_graph(sample_rate)
        frames = args.frames or 1024

    dump_dir = args.dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    render_summary = _render_blocks(
        graph,
        state,
        frames=frames,
        blocks=max(1, int(args.blocks)),
        sample_rate=sample_rate,
        envelope_names=envelope_names,
        amp_mod_names=amp_mod_names,
        dump_dir=dump_dir,
    )

    artifact_meta = _materialise_artifacts(graph, dump_dir)
    metadata = {
        "sample_rate": sample_rate,
        "frames": frames,
        "dump_dir": str(dump_dir),
        "block_count": len(render_summary.get("blocks", [])),
        "artifacts": artifact_meta,
        "render_summary": render_summary,
        "timestamp": time.time(),
    }

    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
