from pathlib import Path
import json
import time
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import importlib.util
from amp.native_runtime import NativeGraphExecutor, UNAVAILABLE_REASON

def _load_demo_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "demo_kpn_kpn_native_correct.py"
    spec = importlib.util.spec_from_file_location("amp_demo_kpn", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect(duration=0.5, sr=48000, block_size=256, out_dir=Path('output')/ 'collect_metrics'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demo = _load_demo_module()
    graph, module = demo.build_graph(int(sr))
    try:
        executor = NativeGraphExecutor(graph)
    except Exception as exc:
        print('Native runtime unavailable or failed to init:', exc)
        return 2
    with executor:
        total_frames = int(round(duration * sr))
        streamer = executor.create_streamer(
            total_frames=total_frames,
            sample_rate=float(sr),
            base_params=None,
            control_history_blob=b"",
            ring_frames=total_frames,
            block_frames=block_size,
        )
        with streamer:
            streamer.start()
            while True:
                produced, consumed, status = streamer.status()
                if status != 0:
                    raise RuntimeError('streamer error', status)
                if produced >= total_frames:
                    break
                time.sleep(0.01)
            streamer.stop()
            pcm = streamer.collect(total_frames)
        # Now collect node summaries
        summaries = {}
        for node in graph.ordered_nodes:
            summary = executor.ffi.new('AmpGraphNodeSummary *')
            rc = executor.lib.amp_graph_runtime_describe_node(executor._runtime, node.name.encode('utf-8'), summary)
            if int(rc) != 0:
                continue
            metrics = None
            if summary.has_metrics:
                m = summary.metrics
                metrics = {
                    'measured_delay_frames': int(m.measured_delay_frames),
                    'accumulated_heat': float(m.accumulated_heat),
                    'processing_time_seconds': float(getattr(m, 'processing_time_seconds', 0.0)),
                    'logging_time_seconds': float(getattr(m, 'logging_time_seconds', 0.0)),
                }
            summaries[node.name] = {
                'declared_delay_frames': int(summary.declared_delay_frames),
                'oversample_ratio': int(summary.oversample_ratio),
                'supports_v2': bool(summary.supports_v2),
                'has_metrics': bool(summary.has_metrics),
                'metrics': metrics,
                'total_heat_accumulated': float(summary.total_heat_accumulated),
            }
        out_path = out_dir / 'node_summaries.json'
        out_path.write_text(json.dumps(summaries, indent=2), encoding='utf-8')
        print('Wrote node summaries to', out_path)
    return 0


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=0.5)
    args = p.parse_args()
    rc = collect(duration=args.duration)
    sys.exit(rc)
