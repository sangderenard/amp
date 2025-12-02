#!/usr/bin/env python3
"""

This script is intentionally small and prints concise diagnostic output so we can determine whether
sampler->FFT->dump flow is producing frames.
"""
import os
import sys
import time
from cffi import FFI
import numpy as np

# Try to locate amp_native.dll in build output
candidates = []
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_debug = os.path.join(root, 'build', 'Debug', 'amp_native.dll')
build_release = os.path.join(root, 'build', 'Release', 'amp_native.dll')
if os.path.exists(build_debug):
    candidates.append(build_debug)
if os.path.exists(build_release):
    candidates.append(build_release)
# fallback to PATH
candidates.append(None)

ffi = FFI()
ffi.cdef('''
    typedef struct KpnStreamSession KpnStreamSession;
    KpnStreamSession *amp_kpn_session_create_from_blobs(const uint8_t *descriptor_blob, size_t descriptor_len, const uint8_t *plan_blob, size_t plan_len, int frames_hint, double sample_rate, uint32_t ring_frames, uint32_t block_frames);
    int amp_kpn_session_start(KpnStreamSession *session);
    void amp_kpn_session_stop(KpnStreamSession *session);
    void amp_kpn_session_destroy(KpnStreamSession *session);
    int amp_kpn_session_stage_sampler_buffer(KpnStreamSession *session, const double *samples, size_t frames, uint32_t channels, const char *node_name);
    int amp_kpn_session_available(KpnStreamSession *session, unsigned long long *out_frames);
    int amp_kpn_session_read(KpnStreamSession *session, double *destination, size_t max_frames, uint32_t *out_frames, uint32_t *out_channels, unsigned long long *out_sequence);
    int amp_kpn_session_status(KpnStreamSession *session, unsigned long long *out_produced_frames, unsigned long long *out_consumed_frames);
    int amp_sampler_peek(const char *node_name, const double **out_samples, size_t *out_frames, uint32_t *out_channels, size_t *out_read_pos);
    int amp_kpn_session_dump_count(KpnStreamSession *session, uint32_t *out_count);
    int amp_kpn_session_pop_dump(KpnStreamSession *session, double *destination, size_t max_frames, uint32_t *out_frames, uint32_t *out_channels, unsigned long long *out_sequence);
    int amp_sampler_unregister(const char *node_name);
    
    typedef struct {
        char name[32];
        uint32_t ring_capacity;
        uint32_t ring_size;
        uint32_t reader_count;
        uint32_t head_position;
        uint32_t tail_position;
        uint64_t produced_total;
    } AmpGraphNodeTapDebugEntry;

    typedef struct {
        char name[64];
        uint32_t ring_capacity;
        uint32_t ring_size;
        uint32_t reader_count;
        uint32_t declared_delay_frames;
        uint32_t oversample_ratio;
        uint8_t supports_v2;
        uint8_t prefill_only;
        float last_heat;
        double last_processing_time_seconds;
        double last_total_time_seconds;
        double total_heat_accumulated;
        uint64_t debug_sequence;
        uint64_t debug_sample_count;
        uint64_t debug_total_frames;
        uint64_t debug_total_batches;
        uint64_t debug_total_channels;
        uint64_t debug_metrics_samples;
        uint64_t debug_last_timestamp_millis;
        uint64_t debug_execute_count;
        uint64_t debug_ready_count;
        uint64_t debug_failed_execute_count;
        double debug_sum_processing_seconds;
        double debug_sum_logging_seconds;
        double debug_sum_total_seconds;
        double debug_sum_thread_cpu_seconds;
        uint32_t debug_last_frames;
        uint32_t debug_last_batches;
        uint32_t debug_last_channels;
        uint32_t debug_min_frames;
        uint32_t debug_preferred_frames;
        uint32_t debug_max_frames;
        double debug_priority_weight;
        uint32_t debug_channel_expand;
        uint8_t fifo_simultaneous_availability;
        uint8_t fifo_release_policy;
        uint32_t fifo_primary_consumer;
        uint32_t tap_count;
        AmpGraphNodeTapDebugEntry taps[8];
    } AmpGraphNodeDebugEntry;

    typedef struct {
        uint32_t version;
        uint32_t node_count;
        uint32_t sink_index;
        double sample_rate;
        uint32_t scheduler_mode;
        uint64_t produced_frames;
        uint64_t consumed_frames;
        uint32_t ring_capacity;
        uint32_t ring_size;
        uint32_t dump_queue_depth;
    uint64_t streamer_loop_count;
    int64_t last_error_code;
    char last_error_stage[32];
    char last_error_node[64];
    char last_error_detail[128];
    } AmpGraphDebugSnapshot;

    int amp_kpn_session_debug_snapshot(
        KpnStreamSession *session,
        AmpGraphNodeDebugEntry *node_entries,
        uint32_t node_capacity,
        AmpGraphDebugSnapshot *snapshot
    );
''')

lib = None
last_exc = None
for p in candidates:
    try:
        lib = ffi.dlopen(p)
        break
    except Exception as e:
        last_exc = e
if lib is None:
    print('Failed to load amp_native.dll; tried:', candidates)
    print('Last error:', last_exc)
    sys.exit(2)

print('Loaded amp_native:', getattr(lib, '__name__', str(p)))

# Prepare a short test tone
sr = 48000
N = 64000
t = np.arange(N, dtype=np.float64) / float(sr)
sig = 0.25 * np.sin(2 * np.pi * 440.0 * t)
arr = np.ascontiguousarray(sig, dtype=np.float64)


def create_and_run(stage_before_start=False, label=''):
    print('\n--- run:', label, 'stage_before_start=', stage_before_start, '---')
    session = lib.amp_kpn_session_create_from_blobs(ffi.NULL, 0, ffi.NULL, 0, int(arr.size), float(sr), 65536, 1024)
    if session == ffi.NULL or session is None:
        print('Failed to create session')
        return False

    ptr = ffi.cast('const double *', arr.ctypes.data)
    if stage_before_start:
        try:
            r = lib.amp_kpn_session_stage_sampler_buffer(session, ptr, arr.size, 1, ffi.new('char[]', b'sampler'))
            print('stage (before start) rc=', r)
        except Exception as e:
            print('stage exception', e)

    rc = lib.amp_kpn_session_start(session)
    if rc != 0:
        print('Failed to start session rc=', rc)
        lib.amp_kpn_session_destroy(session)
        return False

    if not stage_before_start:
        print('Session started; staging sampler buffer as name "sampler"')
        try:
            r = lib.amp_kpn_session_stage_sampler_buffer(session, ptr, arr.size, 1, ffi.new('char[]', b'sampler'))
            print('stage (after start) rc=', r)
        except Exception as e:
            print('stage exception', e)

    # Diagnostic: check sampler registry visibility
    try:
        samples_p = ffi.new('const double **')
        frames_p = ffi.new('size_t *')
        ch_p = ffi.new('uint32_t *')
        readpos_p = ffi.new('size_t *')
        rc_peek = lib.amp_sampler_peek(ffi.new('char[]', b'sampler'), samples_p, frames_p, ch_p, readpos_p)
        print('amp_sampler_peek rc=', rc_peek, 'frames=', int(frames_p[0]) if rc_peek==0 else None, 'channels=', int(ch_p[0]) if rc_peek==0 else None, 'read_pos=', int(readpos_p[0]) if rc_peek==0 else None)
    except Exception:
        pass

    # Diagnostic: check produced/consumed counters
    try:
        prod_p = ffi.new('unsigned long long *')
        cons_p = ffi.new('unsigned long long *')
        lib.amp_kpn_session_status(session, prod_p, cons_p)
        print('status produced=', int(prod_p[0]), 'consumed=', int(cons_p[0]))
    except Exception:
        pass

    def try_debug_snapshot():
        try:
            node_cap = 16
            node_entries = ffi.new('AmpGraphNodeDebugEntry[]', node_cap)
            snapshot = ffi.new('AmpGraphDebugSnapshot *')
            rc_snap = lib.amp_kpn_session_debug_snapshot(session, node_entries, node_cap, snapshot)
            if rc_snap < 0:
                print('[debug-snapshot] rc=', rc_snap)
                return rc_snap
            node_count = int(rc_snap)
            sn = snapshot[0]
            print('[debug-snapshot] nodes=%d produced=%d consumed=%d ring_capacity=%d ring_size=%d dump_depth=%d' % (
                node_count, int(sn.produced_frames), int(sn.consumed_frames), int(sn.ring_capacity), int(sn.ring_size), int(sn.dump_queue_depth)
            ))
            if int(sn.last_error_code) != 0:
                try:
                    stage = ffi.string(sn.last_error_stage).decode('utf-8', errors='ignore')
                except Exception:
                    stage = ''
                try:
                    node_name = ffi.string(sn.last_error_node).decode('utf-8', errors='ignore')
                except Exception:
                    node_name = ''
                try:
                    detail = ffi.string(sn.last_error_detail).decode('utf-8', errors='ignore')
                except Exception:
                    detail = ''
                print('  last_error code=%d stage=%s node=%s detail=%s' % (int(sn.last_error_code), stage, node_name, detail))
            max_show = min(8, node_count)
            for i in range(max_show):
                ne = node_entries[i]
                try:
                    name = ffi.string(ne.name).decode('utf-8', errors='ignore')
                except Exception:
                    name = '<invalid>'
                exec_count = int(getattr(ne, 'debug_execute_count', 0))
                ready_count = int(getattr(ne, 'debug_ready_count', 0))
                failed_count = int(getattr(ne, 'debug_failed_execute_count', 0))
                produced_frames = int(getattr(ne, 'debug_total_frames', 0))
                print('  node[%d] name=%s ring_size=%d ring_capacity=%d produced_frames=%d tap_count=%d exec_count=%d ready_count=%d failed_execs=%d' % (
                        i, name, int(ne.ring_size), int(ne.ring_capacity), produced_frames, int(ne.tap_count), exec_count, ready_count, failed_count
                    ))
            return node_count
        except AttributeError:
            # symbol not present in the loaded native lib; skip
            return -2
        except Exception as e:
            print('Exception while requesting debug snapshot:', e)
            return -3

    # Try snapshot before polling
    try_debug_snapshot()

    # Polling read loop
    print('Polling for output via amp_kpn_session_read (timeout 6s)')
    start = time.time()
    found = False
    last_status_print = 0.0
    while time.time() - start < 60.0:
        # check available frames
        avail_p = ffi.new('unsigned long long *')
        lib.amp_kpn_session_available(session, avail_p)
        avail = int(avail_p[0])
        # periodic status + sampler peek diagnostics
        now = time.time()
        if now - last_status_print > 0.5:
            try:
                prod_p = ffi.new('unsigned long long *')
                cons_p = ffi.new('unsigned long long *')
                lib.amp_kpn_session_status(session, prod_p, cons_p)
                dc_p = ffi.new('uint32_t *')
                lib.amp_kpn_session_dump_count(session, dc_p)
                print('[status] t=%.3f produced=%d consumed=%d avail=%d dump_count=%d' % (now - start, int(prod_p[0]), int(cons_p[0]), avail, int(dc_p[0])))
            except Exception:
                pass
            # Also request a debug snapshot periodically to catch transient writes
            try:
                try_debug_snapshot()
            except Exception:
                pass
            try:
                samples_p = ffi.new('const double **')
                frames_p = ffi.new('size_t *')
                ch_p = ffi.new('uint32_t *')
                readpos_p = ffi.new('size_t *')
                rc_peek = lib.amp_sampler_peek(ffi.new('char[]', b'sampler'), samples_p, frames_p, ch_p, readpos_p)
                if rc_peek == 0:
                    print('[sampler peek] frames=%d channels=%d read_pos=%d' % (int(frames_p[0]), int(ch_p[0]), int(readpos_p[0])))
                else:
                    print('[sampler peek] rc=%d' % (int(rc_peek),))
            except Exception:
                pass
            last_status_print = now
        if avail > 0:
            # read up to avail, but cap per-read size for safety
            to_read = min(avail, 4096)
            out_frames_p = ffi.new('uint32_t *')
            out_ch_p = ffi.new('uint32_t *')
            seq_p = ffi.new('unsigned long long *')
            # allocate a buffer; assume channels <= 128 for safety
            max_channels = 128
            buf = ffi.new('double[]', int(to_read) * max_channels)
            r = lib.amp_kpn_session_read(session, buf, int(to_read), out_frames_p, out_ch_p, seq_p)
            if r == 0:
                nframes = int(out_frames_p[0]); nch = int(out_ch_p[0]); seq = int(seq_p[0])
                print('read rc=0 frames=', nframes, 'channels=', nch, 'seq=', seq)
                if nframes > 0 and nch > 0:
                    data = np.frombuffer(ffi.buffer(buf, nframes * nch * 8), dtype=np.float64).copy()
                    print('data shape:', (nframes, nch))
                    print('first row sample:', data.reshape((nframes, nch))[0, :min(8, nch)])
                    found = True
                    break
            else:
                print('read rc=', r)
        time.sleep(0.05)

    if not found:
        # final status
        avail_p = ffi.new('unsigned long long *')
        lib.amp_kpn_session_available(session, avail_p)
        dc_p = ffi.new('uint32_t *')
        lib.amp_kpn_session_dump_count(session, dc_p)
        print('No output found within timeout; avail=', int(avail_p[0]), 'dump_count=', int(dc_p[0]))

        # Also try a debug snapshot at timeout before forcing stop
        print('Requesting debug snapshot at timeout...')
        try_debug_snapshot()

        # Force stop -> flush -> pop_dump to surface any buffered dumps
        print('Forcing session stop to trigger flush and checking dump queue...')
        try:
            lib.amp_kpn_session_stop(session)
        except Exception:
            pass
        # give runtime a moment to flush
        time.sleep(0.25)
        try:
            dc_p = ffi.new('uint32_t *')
            lib.amp_kpn_session_dump_count(session, dc_p)
            dumps = int(dc_p[0])
            print('dump_count after stop=', dumps)
            dump_index = 0
            while dumps > 0:
                # pop each dump
                out_frames_p = ffi.new('uint32_t *')
                out_ch_p = ffi.new('uint32_t *')
                seq_p = ffi.new('unsigned long long *')
                # generous buffer: 65536 frames * 8 bytes * 2 channels
                max_channels = 128
                max_frames = 65536
                buf = ffi.new('double[]', int(max_frames) * max_channels)
                r = lib.amp_kpn_session_pop_dump(session, buf, int(max_frames), out_frames_p, out_ch_p, seq_p)
                if r != 0:
                    print('pop_dump rc=', r)
                    break
                nframes = int(out_frames_p[0]); nch = int(out_ch_p[0]); seq = int(seq_p[0])
                print('pop_dump[%d] frames=%d channels=%d seq=%d' % (dump_index, nframes, nch, seq))
                if nframes > 0 and nch > 0:
                    data = np.frombuffer(ffi.buffer(buf, nframes * nch * 8), dtype=np.float64).copy()
                    print('dump data shape:', (nframes, nch))
                    print('first row dump sample:', data.reshape((nframes, nch))[0, :min(8, nch)])
                dump_index += 1
                # re-check dump_count
                lib.amp_kpn_session_dump_count(session, dc_p)
                dumps = int(dc_p[0])
        except Exception as e:
            print('Exception while popping dumps:', e)

    # Cleanup
    try:
        lib.amp_sampler_unregister(ffi.new('char[]', b'sampler'))
    except Exception:
        pass
    try:
        lib.amp_kpn_session_stop(session)
    except Exception:
        pass
    try:
        lib.amp_kpn_session_destroy(session)
    except Exception:
        pass

    if found:
        print('test (', label, '): SUCCESS')
        return True
    else:
        print('test (', label, '): NO DATA')
        return False


# Run two variants: stage before start, and stage after start
res1 = create_and_run(stage_before_start=True, label='before_start')
res2 = create_and_run(stage_before_start=False, label='after_start')

if res1 or res2:
    sys.exit(0)
else:
    sys.exit(5)

# Diagnostic: check sampler registry visibility
try:
    samples_p = ffi.new('const double **')
    frames_p = ffi.new('size_t *')
    ch_p = ffi.new('uint32_t *')
    readpos_p = ffi.new('size_t *')
    rc_peek = lib.amp_sampler_peek(ffi.new('char[]', b'sampler'), samples_p, frames_p, ch_p, readpos_p)
    print('amp_sampler_peek rc=', rc_peek, 'frames=', int(frames_p[0]) if rc_peek==0 else None, 'channels=', int(ch_p[0]) if rc_peek==0 else None, 'read_pos=', int(readpos_p[0]) if rc_peek==0 else None)
except Exception:
    pass

# Diagnostic: check produced/consumed counters
try:
    prod_p = ffi.new('unsigned long long *')
    cons_p = ffi.new('unsigned long long *')
    lib.amp_kpn_session_status(session, prod_p, cons_p)
    print('status produced=', int(prod_p[0]), 'consumed=', int(cons_p[0]))
except Exception:
    pass

print('Polling for output via amp_kpn_session_read (timeout 6s)')
start = time.time()
found = False
last_status_print = 0.0
while time.time() - start < 6.0:
    # check available frames
    avail_p = ffi.new('unsigned long long *')
    lib.amp_kpn_session_available(session, avail_p)
    avail = int(avail_p[0])
    # periodic status + sampler peek diagnostics
    now = time.time()
    if now - last_status_print > 0.5:
        try:
            prod_p = ffi.new('unsigned long long *')
            cons_p = ffi.new('unsigned long long *')
            lib.amp_kpn_session_status(session, prod_p, cons_p)
            dc_p = ffi.new('uint32_t *')
            lib.amp_kpn_session_dump_count(session, dc_p)
            print('[status] t=%.3f produced=%d consumed=%d avail=%d dump_count=%d' % (now - start, int(prod_p[0]), int(cons_p[0]), avail, int(dc_p[0])))
        except Exception:
            pass
        try:
            samples_p = ffi.new('const double **')
            frames_p = ffi.new('size_t *')
            ch_p = ffi.new('uint32_t *')
            readpos_p = ffi.new('size_t *')
            rc_peek = lib.amp_sampler_peek(ffi.new('char[]', b'sampler'), samples_p, frames_p, ch_p, readpos_p)
            if rc_peek == 0:
                print('[sampler peek] frames=%d channels=%d read_pos=%d' % (int(frames_p[0]), int(ch_p[0]), int(readpos_p[0])))
            else:
                print('[sampler peek] rc=%d' % (int(rc_peek),))
        except Exception:
            pass
        last_status_print = now
    if avail > 0:
        # read up to avail, but cap per-read size for safety
        to_read = min(avail, 4096)
        out_frames_p = ffi.new('uint32_t *')
        out_ch_p = ffi.new('uint32_t *')
        seq_p = ffi.new('unsigned long long *')
        # allocate a buffer; assume channels <= 128 for safety
        max_channels = 128
        buf = ffi.new('double[]', int(to_read) * max_channels)
        r = lib.amp_kpn_session_read(session, buf, int(to_read), out_frames_p, out_ch_p, seq_p)
        if r == 0:
            nframes = int(out_frames_p[0]); nch = int(out_ch_p[0]); seq = int(seq_p[0])
            print('read rc=0 frames=', nframes, 'channels=', nch, 'seq=', seq)
            if nframes > 0 and nch > 0:
                data = np.frombuffer(ffi.buffer(buf, nframes * nch * 8), dtype=np.float64).copy()
                print('data shape:', (nframes, nch))
                print('first row sample:', data.reshape((nframes, nch))[0, :min(8, nch)])
                found = True
                break
        else:
            print('read rc=', r)
    time.sleep(0.05)

if not found:
    # final status
    avail_p = ffi.new('unsigned long long *')
    lib.amp_kpn_session_available(session, avail_p)
    dc_p = ffi.new('uint32_t *')
    lib.amp_kpn_session_dump_count(session, dc_p)
    print('No output found within timeout; avail=', int(avail_p[0]), 'dump_count=', int(dc_p[0]))

    # Force stop -> flush -> pop_dump to surface any buffered dumps
    print('Forcing session stop to trigger flush and checking dump queue...')
    try:
        lib.amp_kpn_session_stop(session)
    except Exception:
        pass
    # give runtime a moment to flush
    time.sleep(0.25)
    try:
        dc_p = ffi.new('uint32_t *')
        lib.amp_kpn_session_dump_count(session, dc_p)
        dumps = int(dc_p[0])
        print('dump_count after stop=', dumps)
        dump_index = 0
        while dumps > 0:
            # pop each dump
            out_frames_p = ffi.new('uint32_t *')
            out_ch_p = ffi.new('uint32_t *')
            seq_p = ffi.new('unsigned long long *')
            # generous buffer: 65536 frames * 8 bytes * 2 channels
            max_channels = 128
            max_frames = 65536
            buf = ffi.new('double[]', int(max_frames) * max_channels)
            r = lib.amp_kpn_session_pop_dump(session, buf, int(max_frames), out_frames_p, out_ch_p, seq_p)
            if r != 0:
                print('pop_dump rc=', r)
                break
            nframes = int(out_frames_p[0]); nch = int(out_ch_p[0]); seq = int(seq_p[0])
            print('pop_dump[%d] frames=%d channels=%d seq=%d' % (dump_index, nframes, nch, seq))
            if nframes > 0 and nch > 0:
                data = np.frombuffer(ffi.buffer(buf, nframes * nch * 8), dtype=np.float64).copy()
                print('dump data shape:', (nframes, nch))
                print('first row dump sample:', data.reshape((nframes, nch))[0, :min(8, nch)])
            dump_index += 1
            # re-check dump_count
            lib.amp_kpn_session_dump_count(session, dc_p)
            dumps = int(dc_p[0])
    except Exception as e:
        print('Exception while popping dumps:', e)

# Cleanup
try:
    lib.amp_sampler_unregister(ffi.new('char[]', b'sampler'))
except Exception:
    pass
try:
    lib.amp_kpn_session_stop(session)
except Exception:
    pass
try:
    lib.amp_kpn_session_destroy(session)
except Exception:
    pass

if found:
    print('SUCCESS')
    sys.exit(0)
else:
    print('NO DATA')
    sys.exit(5)
