"""Correlate native_mem_ops.log BAD_MEM* events with native_alloc_trace.log

Produces a summarized CSV to stdout and a detailed JSON per BAD event.
"""
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs'

mem_ops = LOGS / 'native_mem_ops.log'
allocs = LOGS / 'native_alloc_trace.log'

# Regexes
re_bad = re.compile(r'^(BAD_MEM(?:CPY|SET))\s+([^\s]+):?(\d+)?\s+([^\s]+)\s+dest=([0-9A-Fa-f]+)\s+n=(\d+)')
re_precopy = re.compile(r'^PRECOPY\s+([^\s]+)\s+dest=([0-9A-Fa-f]+)\s+dest_cap=(\d+)\s+req_len=(\d+)')
re_alloc = re.compile(r'^(CALLOC|MALLOC|REALLOC|FREE|REGISTER|UNREGISTER)\s+([^\s]+):?(\d+)?\s+([^\s]+)')
re_alloc_details = re.compile(r'.*ptr=([0-9A-Fa-f]+)(?:\s+old=([0-9A-Fa-f]+))?(?:\s+new=([0-9A-Fa-f]+))?.*size=(\d+)')
re_register = re.compile(r'^REGISTER\s+([0-9A-Fa-fxX]+)\s+size=(\d+)')
re_unregister = re.compile(r'^UNREGISTER\s+([0-9A-Fa-fxX]+)')

# Load alloc timeline
alloc_lines = []
with allocs.open('r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.rstrip('\n')
        mreg = re_register.match(line)
        if mreg:
            ptr, size = mreg.groups()
            alloc_lines.append({'idx': i, 'kind': 'REGISTER', 'file': None, 'line': None, 'ptr': ptr.replace('0x','').replace('0X',''), 'size': int(size), 'raw': line})
            continue
        munreg = re_unregister.match(line)
        if munreg:
            ptr = munreg.group(1)
            alloc_lines.append({'idx': i, 'kind': 'UNREGISTER', 'file': None, 'line': None, 'ptr': ptr.replace('0x','').replace('0X',''), 'size': None, 'raw': line})
            continue
        m = re_alloc.match(line)
        if not m:
            continue
        kind, file, lineno, rest = m.groups()
        m2 = re_alloc_details.search(line)
        if not m2:
            continue
        ptr, old, new, size = m2.groups()
        alloc_lines.append({'idx': i, 'kind': kind, 'file': file, 'line': int(lineno) if lineno else None, 'ptr': ptr, 'size': int(size), 'raw': line})

# Build index by ptr for fast lookup
alloc_index = {}
for a in alloc_lines:
    p = int(a['ptr'], 16)
    alloc_index.setdefault(p, []).append(a)

# Parse mem ops and correlate
bad_events = []
with mem_ops.open('r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        m = re_bad.search(line)
        if m:
            kind, file, lineno, func, dest_hex, n = m.groups()
            dest = int(dest_hex, 16)
            n = int(n)
            bad_events.append({'idx': i, 'line': line.rstrip(), 'kind': kind, 'file': file, 'lineno': int(lineno) if lineno else None, 'func': func, 'dest': dest, 'n': n})

# Build allocation ranges with lifetimes (start_idx..end_idx). Only include entries with size.
alloc_ranges = []
ptr_to_ranges = {}
for a in alloc_lines:
    if a.get('size') is None:
        continue
    start_addr = int(a['ptr'], 16)
    start_idx = a['idx']
    # initialize end_idx to a very large number (meaning not freed yet)
    end_idx = 10**12
    # we will fill end_idx when we see an UNREGISTER/FREE for the same ptr later
    r = {'start': start_addr, 'end': start_addr + a['size'], 'size': a['size'], 'start_idx': start_idx, 'end_idx': end_idx, 'meta': a}
    alloc_ranges.append((r['start'], r['end'], r))
    ptr_to_ranges.setdefault(a['ptr'].lower(), []).append(r)

# Now walk alloc_lines to find UNREGISTER/FREE events and set end_idx for matching ranges
for a in alloc_lines:
    if a['kind'] in ('UNREGISTER', 'FREE'):
        p = a['ptr'].lower()
        if p in ptr_to_ranges:
            # find the most-recent range for this ptr that doesn't yet have an end_idx set
            ranges = ptr_to_ranges[p]
            # look for the last range whose end_idx is still the sentinel
            for rr in reversed(ranges):
                if rr['end_idx'] == 10**12:
                    rr['end_idx'] = a['idx']
                    break

# Sort by start address
alloc_ranges.sort(key=lambda t: t[0])

# helper: find containing alloc
import bisect
starts = [r[0] for r in alloc_ranges]

def find_containing(dest):
    i = bisect.bisect_right(starts, dest) - 1
    if i >= 0:
        # There may be multiple ranges that start before dest; search backward until start <= dest < end
        for j in range(i, -1, -1):
            start, end, a = alloc_ranges[j]
            if start <= dest < end:
                return a
            # if dest is greater than end, we can stop searching earlier entries
            if dest >= end:
                break
    return None

# correlate
reports = []
for be in bad_events:
    a = find_containing(be['dest'])
    if a:
        # check lifetime: was allocation live at the time of the bad event?
        if a['start_idx'] <= be['idx'] < a['end_idx']:
            cls = 'contained_live'
        else:
            cls = 'use_after_free' if be['idx'] >= a['end_idx'] else 'before_alloc'
        reports.append({'bad_idx': be['idx'], 'bad_line': be['line'], 'dest': hex(be['dest']), 'n': be['n'], 'alloc_ptr': format(a['start'], '016X'), 'alloc_size': a['size'], 'alloc_raw': a['meta']['raw'], 'class': cls, 'alloc_start_idx': a['start_idx'], 'alloc_end_idx': (a['end_idx'] if a['end_idx'] != 10**12 else None)})
    else:
        # try to find nearest alloc by start address
        i = bisect.bisect_left(starts, be['dest'])
        cand = None
        if i < len(alloc_ranges):
            cand = alloc_ranges[i][2]
        if i-1 >= 0 and (cand is None or abs(alloc_ranges[i-1][0]-be['dest']) < abs((cand['start'] if cand else 0)-be['dest'])):
            cand = alloc_ranges[i-1][2]
        reports.append({'bad_idx': be['idx'], 'bad_line': be['line'], 'dest': hex(be['dest']), 'n': be['n'], 'alloc_ptr': format(cand['start'], '016X') if cand else None, 'alloc_size': cand['size'] if cand else None, 'alloc_raw': cand['meta']['raw'] if cand else None, 'class': 'no_contain'})

# Summarize clusters by dest pointer
from collections import Counter, defaultdict
ctr = Counter([r['alloc_ptr'] for r in reports])

# Print top clusters
print('top_alloc_ptr, count')
for ptr, cnt in ctr.most_common(20):
    print(f'{ptr},{cnt}')

# write detailed
out = ROOT / 'logs' / 'correlation_report.json'
with out.open('w', encoding='utf-8') as f:
    json.dump(reports, f, indent=2)

print('\nWrote detailed report to', out)
