#!/usr/bin/env python3
"""
Headless smoke test for KPN streaming session.
Creates a session via cffi, stages a short sine signal into the sampler
registry under node name 'sampler', starts the session, polls for
available frames and dump_count, pops dumps and prints shapes.
"""
import time
import math
import threading
import numpy as np
import os
from cffi import FFI

ffi = FFI()
ffi.cdef("""
typedef struct KpnStreamSession KpnStreamSession;
KpnStreamSession *amp_kpn_session_create_from_blobs(const uint8_t *descriptor_blob, size_t descriptor_len, const uint8_t *plan_blob, size_t plan_len, int frames_hint, double sample_rate, uint32_t ring_frames, uint32_t block_frames);
int amp_kpn_session_start(KpnStreamSession *session);
void amp_kpn_session_stop(KpnStreamSession *session);
void amp_kpn_session_destroy(KpnStreamSession *session);
int amp_kpn_session_available(KpnStreamSession *session, unsigned long long *out_frames);
int amp_kpn_session_read(KpnStreamSession *session, double *destination, size_t max_frames, uint32_t *out_frames, uint32_t *out_channels, unsigned long long *out_sequence);
int amp_kpn_session_dump_count(KpnStreamSession *session, uint32_t *out_count);
int amp_kpn_session_pop_dump(KpnStreamSession *session, double *destination, size_t max_frames, uint32_t *out_frames, uint32_t *out_channels, unsigned long long *out_sequence);
int amp_kpn_session_stage_sampler_buffer(KpnStreamSession *session, const double *samples, size_t frames, uint32_t channels, const char *node_name);
int amp_sampler_unregister(const char *node_name);
""")

candidates = []
# global timeout (seconds) for the whole script; can be overridden with E2E_TIMEOUT env var
GLOBAL_TIMEOUT = int(os.environ.get('E2E_TIMEOUT', '8'))


# watchdog: if main logic hasn't finished by GLOBAL_TIMEOUT, try cleanup then exit
def _watchdog(timeout, cleanup_cb=None):
    time.sleep(timeout)
    print(f"Global timeout ({timeout}s) reached. Attempting cleanup and exiting.")
    try:
        if cleanup_cb:
            cleanup_cb()
    except Exception as e:
        print('watchdog cleanup error', e)
    # Forcefully terminate to avoid hanging the runner
    os._exit(124)


# try to load amp_native from build/Debug or PATH
candidates = []
cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_debug = os.path.join(cwd, 'build', 'Debug', 'amp_native.dll')
build_release = os.path.join(cwd, 'build', 'Release', 'amp_native.dll')
candidates.extend([build_debug, build_release])
last_exc = None
lib = None
for p in candidates:
    if os.path.exists(p):
        try:
            lib = ffi.dlopen(p)
            break
        except Exception as e:
            last_exc = e
            lib = None

if lib is None:
    try:
        lib = ffi.dlopen(None)
    except Exception as e:
        last_exc = e

if lib is None:
    raise RuntimeError("Failed to load amp_native.dll: %r" % (last_exc,))

# generate a short test signal (mono) - 0.2s of a sinusoid at 440Hz @48k
sr = 48000.0
dur_s = 0.2
N = int(sr * dur_s)
t = np.arange(N, dtype=np.float64) / sr
sig = 0.5 * np.sin(2.0 * math.pi * 440.0 * t)

# create a session with null blobs (demo graph fallback)
session = ffi.NULL
session = lib.amp_kpn_session_create_from_blobs(ffi.NULL, 0, ffi.NULL, 0, int(N), float(sr), max(65536, 4096), 1024)
if session == ffi.NULL or session is None:
    raise RuntimeError('Failed to create session')

rc = lib.amp_kpn_session_start(session)
if rc != 0:
    print('amp_kpn_session_start rc=', rc)
    lib.amp_kpn_session_destroy(session)
    raise SystemExit(1)

# start watchdog thread that will attempt cleanup and exit if global timeout reached
def _cleanup_cb():
    try:
        if session != ffi.NULL:
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
    except Exception:
        pass

wd = threading.Thread(target=_watchdog, args=(GLOBAL_TIMEOUT, _cleanup_cb), daemon=True)
wd.start()

# stage the signal into sampler registry under name 'sampler'
arr_c = np.ascontiguousarray(sig, dtype=np.float64)
ptr = ffi.cast('const double *', arr_c.ctypes.data)
try:
    srcc = lib.amp_kpn_session_stage_sampler_buffer(session, ptr, arr_c.size, 1, ffi.new('char[]', b'sampler'))
    print('stage rc', srcc)
except Exception as e:
    print('stage exception', e)

# poll loop: wait for dumps up to timeout
start = time.time()
timeout = 6.0
poll_interval = 0.05
collected = []
collected_pcm = []

try:
    while True:
        # check available frames
        avail_p = ffi.new('unsigned long long *')
        lib.amp_kpn_session_available(session, avail_p)
        avail = int(avail_p[0])
        if avail > 0:
            to_read = min(avail, 1024)
            out_frames_p = ffi.new('uint32_t *')
            out_ch_p = ffi.new('uint32_t *')
            seq_p = ffi.new('unsigned long long *')
            # guess channels for read buffer: use 128 channels as worst-case
            buf = ffi.new('double[]', to_read * 128)
            r = lib.amp_kpn_session_read(session, buf, to_read, out_frames_p, out_ch_p, seq_p)
            if r == 0:
                nframes = int(out_frames_p[0])
                nch = int(out_ch_p[0])
                if nframes > 0 and nch > 0:
                    raw = np.frombuffer(ffi.buffer(buf, nframes * nch * 8), dtype=np.float64).copy()
                    arr = raw.reshape((nframes, nch))
                    print('READ frames', nframes, 'channels', nch)
                    if nch == 1:
                        collected_pcm.append(arr[:, 0])
                    else:
                        collected.append(arr)

        # check dump queue
        dump_count_p = ffi.new('uint32_t *')
        lib.amp_kpn_session_dump_count(session, dump_count_p)
        dump_count = int(dump_count_p[0])
        while dump_count > 0:
            out_frames_p = ffi.new('uint32_t *')
            out_ch_p = ffi.new('uint32_t *')
            seq_p = ffi.new('unsigned long long *')
            buf = ffi.new('double[]', max(1, 4096 * 128))
            r = lib.amp_kpn_session_pop_dump(session, buf, 4096, out_frames_p, out_ch_p, seq_p)
            if r == 0:
                nframes = int(out_frames_p[0])
                nch = int(out_ch_p[0])
                if nframes > 0 and nch > 0:
                    raw = np.frombuffer(ffi.buffer(buf, nframes * nch * 8), dtype=np.float64).copy()
                    arr = raw.reshape((nframes, nch))
                    print('DUMP frames', nframes, 'channels', nch)
                    collected.append(arr)
            dump_count_p = ffi.new('uint32_t *')
            lib.amp_kpn_session_dump_count(session, dump_count_p)
            dump_count = int(dump_count_p[0])

        if time.time() - start > timeout:
            print('timeout')
            break
        time.sleep(poll_interval)
finally:
        try:
            lib.amp_sampler_unregister(ffi.new('char[]', b'sampler'))
        except Exception:
            pass
        try:
            if session != ffi.NULL:
                lib.amp_kpn_session_stop(session)
        except Exception:
            pass
        try:
            if session != ffi.NULL:
                lib.amp_kpn_session_destroy(session)
        except Exception:
            pass

print('collected dumps:', len(collected))
print('collected pcm chunks:', len(collected_pcm))
if len(collected) > 0:
    shapes = [a.shape for a in collected]
    print('dump shapes:', shapes)
if len(collected_pcm) > 0:
    total_pcm = np.concatenate(collected_pcm, axis=0)
    print('total pcm frames:', total_pcm.shape)

print('done')
