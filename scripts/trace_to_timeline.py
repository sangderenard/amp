#!/usr/bin/env python3
"""
Parse `test_fft_division_node` trace logs and plot timeline graphs by worker
cycle (chunk), subdivided by stage (log tag) and order (event order within
cycle).

This script targets the real diagnostic output produced by
`test_fft_division_node.cpp` (examples: `[stream-trace]`, `[TEST-DIAG]`,
`[fft_division_node][diag]`, `[MAILBOX-DUMP]`, and metric lines like
`[forward_metrics] observed delay=...`). It extracts numeric fields and
produces one plot per metric showing values vs. worker cycle, with a grid
of subplots partitioned by `stage` and `order`.

Usage:
  python scripts/trace_to_timeline.py path/to/trace.log --metrics processing_time_seconds,measured_delay --output out.png

If `--metrics` is omitted the script will auto-detect numeric metrics.
"""

from __future__ import annotations
import argparse
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import sys

import matplotlib.pyplot as plt
import pandas as pd


RE_CHUNK = re.compile(r"chunk=(?P<chunk>\d+)")
RE_KEYVAL = re.compile(r"(?P<k>[a-zA-Z_][a-zA-Z0-9_]*)=(?P<v>-?[0-9]+(?:\.[0-9eE+-]+)?)")
RE_METRIC_TAG = re.compile(r"\[(?P<tag>[^\]]+)\].*?(?P<body>.*)")
RE_TEST_DIAG_AFTER_POP = re.compile(r"after_populate .* frames_committed=(?P<frames>\d+)")
RE_PCM_SAMPLE = re.compile(r"pcm\[(?P<idx>\d+)\]=\s*(?P<val>-?[0-9]+(?:\.[0-9eE+-]+)?)")
RE_OBSERVED_DELAY = re.compile(r"observed delay=(?P<delay>\d+)")


def try_parse_number(s: str):
    try:
        if '.' in s or 'e' in s or 'E' in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def extract_keyvals(text: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for m in RE_KEYVAL.finditer(text):
        k = m.group('k')
        v = try_parse_number(m.group('v'))
        d[k] = v
    return d


def parse_trace_lines(lines: List[str]) -> pd.DataFrame:
    events = []
    current_event_index = 0
    for lineno, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        # Try to extract tag and body inside brackets
        tag_match = RE_METRIC_TAG.search(line)
        tag = None
        body = line
        if tag_match:
            tag = tag_match.group('tag')
            body = tag_match.group('body').strip()

        # Determine cycle (chunk) if present; cycle grouping may be overridden later
        chunk_m = RE_CHUNK.search(line)
        if chunk_m:
            cycle = int(chunk_m.group('chunk'))
        else:
            # fallback: use a monotonically increasing event index as cycle
            cycle = current_event_index

        # order: count of events seen for this cycle
        events_for_cycle = sum(1 for e in events if e['cycle'] == cycle)
        order = events_for_cycle

        # Extract key=val numeric pairs
        kv = extract_keyvals(body)

        # Detect PCM sample lines like: pcm[0001]= 0.123456
        pcm_m = RE_PCM_SAMPLE.search(body)
        if pcm_m:
            try:
                kv['pcm_index'] = int(pcm_m.group('idx'))
                kv['pcm_sample'] = try_parse_number(pcm_m.group('val'))
            except Exception:
                pass

        # Special-case observed delay
        od = RE_OBSERVED_DELAY.search(body)
        if od:
            kv['measured_delay'] = int(od.group('delay'))

        # Special-case TEST-DIAG after_populate frames_committed
        m_after = RE_TEST_DIAG_AFTER_POP.search(body)
        if m_after:
            frames = int(m_after.group('frames'))
            # disambiguate based on body content: pcm vs spectral
            low = body.lower()
            if 'pcm' in low or 'pcm_read' in low:
                kv['pcm_frames_committed'] = frames
            elif 'spect' in low or 'spectral' in low:
                kv['spectral_frames_committed'] = frames
            else:
                kv['frames_committed'] = frames

        event = {
            'lineno': lineno + 1,
            'raw': line,
            'cycle': cycle,
            'stage': tag if tag is not None else 'log',
            'order': order,
        }
        # merge kv into event
        event.update(kv)

        events.append(event)
        current_event_index += 1

    if not events:
        raise SystemExit('No events parsed from trace')

    df = pd.DataFrame(events)
    # ensure numeric columns are numeric
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c], errors='ignore')
            except Exception:
                pass
    return df


def plot_grid(df: pd.DataFrame, metric: str, out_path: Optional[str], max_cols: int = 3):
    stages = list(df['stage'].unique())
    orders = sorted(list(df['order'].unique()))
    nplots = len(stages) * len(orders)
    cols = min(max_cols, len(orders)) if orders else 1
    rows = math.ceil(nplots / cols) if nplots else 1

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
    fig.suptitle(f'{metric} by cycle (subdivided by stage / order)')
    ax_idx = 0
    for s in stages:
        for o in orders:
            r = ax_idx // cols
            c = ax_idx % cols
            ax = axes[r][c]
            sel = df[(df['stage'] == s) & (df['order'] == o) & (df[metric].notna())]
            if sel.empty:
                ax.set_visible(False)
            else:
                sel_sorted = sel.sort_values('cycle')
                ax.plot(sel_sorted['cycle'], sel_sorted[metric], '-o', markersize=3)
                ax.set_xlabel('cycle')
                ax.set_ylabel(metric)
                ax.grid(True, linestyle=':', linewidth=0.4)
            ax.set_title(f'stage={s} order={o}')
            ax_idx += 1

    # hide leftover axes
    for leftover in range(ax_idx, rows * cols):
        r = leftover // cols
        c = leftover % cols
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_path:
        # append metric name to output filename
        if out_path.lower().endswith('.png'):
            out_file = out_path.replace('.png', f'_{metric}.png')
        else:
            out_file = f'{out_path}_{metric}.png'
        plt.savefig(out_file, dpi=150)
        print('Wrote', out_file)
        plt.close(fig)
    else:
        plt.show()


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description='Parse test_fft_division_node traces and plot timelines')
    p.add_argument('trace', nargs='?', help='Trace log file (use - or omit to read from stdin)')
    p.add_argument('--metrics', help='Comma-separated metric keys to plot (default: autodetect)')
    p.add_argument('--output', help='Output PNG base path (files named <base>_<metric>.png). If omitted opens interactive plot')
    p.add_argument('--max-cols', type=int, default=3, help='Max columns in subplot grid')
    args = p.parse_args(argv)

    # Read trace lines: prefer stdin pipe when no filename or '-' is passed.
    if not args.trace or args.trace == '-':
        # Read entire stdin until EOF (works when piping from the test binary):
        data = sys.stdin.read()
        lines = data.splitlines()
    else:
        with open(args.trace, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

    df = parse_trace_lines(lines)

    # Decide cycle grouping auto-heuristic: prefer parsed 'chunk' unless it's effectively constant
    unique_chunks = df['cycle'].nunique()
    total_events = len(df)
    if unique_chunks <= 1:
        print(f"Auto: 'chunk' appears constant (unique={unique_chunks}); grouping by event index")
        df['cycle'] = df['lineno']
    else:
        print(f"Auto: using 'chunk' as cycle grouping (unique chunks={unique_chunks}, events={total_events})")

    # Determine metrics to plot
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ('lineno', 'cycle', 'order')]
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    else:
        metrics = numeric_cols

    if not metrics:
        raise SystemExit('No numeric metrics detected; pass --metrics explicitly')

    for metric in metrics:
        if metric not in df.columns:
            print('Warning: metric', metric, 'not found in parsed fields')
            continue
        plot_grid(df, metric, args.output, max_cols=args.max_cols)


if __name__ == '__main__':
    main()
