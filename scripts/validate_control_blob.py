#!/usr/bin/env python3
"""Validate a control-history binary blob written by AudioGraph/ControlDelay.

Usage: python scripts/validate_control_blob.py [path]

Exits with code 0 on success. Prints human-readable diagnostics and exits
with non-zero when a malformed blob is detected (unexpected EOF, inconsistent
lengths, or trailing bytes).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def validate(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: blob not found: {path}")
        return 2
    data = path.read_bytes()
    n = len(data)
    ptr = 0
    def eof(need: int) -> bool:
        return ptr + need > n

    if n < 8:
        print(f"ERROR: blob too small ({n} bytes); need at least 8 for header")
        return 3

    try:
        event_count, key_count = struct.unpack_from('<II', data, ptr)
    except struct.error as exc:
        print(f"ERROR: failed to unpack header: {exc}")
        return 4
    ptr += 8
    print(f"event_count={event_count}, key_count={key_count}, total_bytes={n}")

    key_lengths = []
    for i in range(key_count):
        if eof(4):
            print(f"ERROR: unexpected EOF reading length of key {i} (need 4 bytes)")
            return 5
        (l,) = struct.unpack_from('<I', data, ptr)
        ptr += 4
        key_lengths.append(l)
    print(f"key_lengths={key_lengths}")

    keys = []
    for i, l in enumerate(key_lengths):
        if eof(l):
            print(f"ERROR: unexpected EOF reading key {i} (need {l} bytes)")
            return 6
        key = data[ptr:ptr+l].decode('utf-8', errors='replace')
        ptr += l
        keys.append(key)
    print(f"keys={keys}")

    # Also print JSON metadata if available
    meta_path = Path('logs/last_control_blob.json')
    if meta_path.exists():
        try:
            import json

            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            print(f"metadata={meta}")
        except Exception as exc:
            print(f"warning: failed to read JSON metadata: {exc}")

    # For each event, read timestamp (double) + per-key array sizes and data
    for ei in range(event_count):
        if eof(8):
            print(f"ERROR: unexpected EOF reading timestamp for event {ei} (need 8 bytes)")
            return 7
        (ts,) = struct.unpack_from('<d', data, ptr)
        ptr += 8
        print(f"event {ei} timestamp={ts}")
        for ki, key in enumerate(keys):
            if eof(4):
                print(f"ERROR: unexpected EOF reading array size for key '{key}' on event {ei}")
                return 8
            (size,) = struct.unpack_from('<I', data, ptr)
            ptr += 4
            print(f"  key '{key}' size={size}")
            if size:
                bytes_needed = size * 8
                if eof(bytes_needed):
                    print(f"ERROR: unexpected EOF reading {bytes_needed} bytes for key '{key}' on event {ei}")
                    return 9
                # attempt to unpack first few values for sanity
                max_preview = min(5, size)
                fmt = '<' + 'd'*max_preview
                vals = struct.unpack_from(fmt, data, ptr)
                print(f"    first_vals={vals}{' ...' if size>max_preview else ''}")
                ptr += bytes_needed

    if ptr != n:
        print(f"ERROR: trailing/extra bytes: parsed {ptr} bytes but file is {n} bytes")
        # still print how many bytes remain
        print(f"  trailing_bytes={n-ptr}")
        return 10

    print("OK: blob parsed successfully and matches expected format")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs='?', default='logs/last_control_blob.bin')
    args = p.parse_args(argv)
    return validate(Path(args.path))


if __name__ == '__main__':
    raise SystemExit(main())
