
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#include <ctype.h>
#include <errno.h>
#include <float.h>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__cplusplus)
#include <Eigen/Dense>
#endif
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif
#if defined(__GNUC__) && !defined(_WIN32) && !defined(_WIN64)
#include <execinfo.h>
#endif
/* POSIX mkdir() */
#include <sys/stat.h>
#include <sys/types.h>

#include "amp_native.h"
#include "amp_fft_backend.h"

#ifndef M_LN2
#define M_LN2 0.693147180559945309417232121458176568
#endif

typedef struct {
    int *boundaries;
    int *trig_indices;
    int8_t *gate_bool;
    int8_t *drone_bool;
    size_t boundary_cap;
    size_t trig_cap;
    size_t bool_cap;
} envelope_scratch_t;

static envelope_scratch_t envelope_scratch = { NULL, NULL, NULL, NULL, 0, 0, 0 };
/* Debug: track last allocation element count for diagnostics. */
static size_t amp_last_alloc_count = 0;

AMP_CAPI size_t amp_last_alloc_count_get(void) {
    return amp_last_alloc_count;
}

/*
 * Edge runner contract (mirrors `_EDGE_RUNNER_CDEF` in Python).
 *
 * The runtime passes node descriptors/inputs to `amp_run_node`, which may
 * allocate per-node state (returned via `state`) and a heap-owned audio buffer
 * (`out_buffer`).
 *
 * Return codes:
 *   0   -> success
 *  -1   -> allocation failure / invalid contract usage
 *  -3   -> unsupported node kind (caller should fall back to Python)
 */
/* Persistent log file handles. Lazily opened on first use so builds that run
   in read-only or log-less environments can continue without crashing. */
#if defined(AMP_NATIVE_ENABLE_LOGGING)
static FILE *log_f_alloc = NULL;
static FILE *log_f_memops = NULL;
static FILE *log_f_ccalls = NULL;
static FILE *log_f_cgenerated = NULL;

#if defined(_WIN32) || defined(_WIN64)
static CRITICAL_SECTION log_lock;
static int log_lock_initialized = 0;
#define LOG_LOCK_INIT() do { if (!log_lock_initialized) { InitializeCriticalSection(&log_lock); log_lock_initialized = 1; } } while(0)
#define LOG_LOCK() EnterCriticalSection(&log_lock)
#define LOG_UNLOCK() LeaveCriticalSection(&log_lock)
#else
#include <pthread.h>
static pthread_mutex_t log_lock;
static int log_lock_initialized = 0;
#define LOG_LOCK_INIT() do { if (!log_lock_initialized) { pthread_mutex_init(&log_lock, NULL); log_lock_initialized = 1; } } while(0)
#define LOG_LOCK() pthread_mutex_lock(&log_lock)
#define LOG_UNLOCK() pthread_mutex_unlock(&log_lock)
#endif

static void close_all_logs(void);

#if defined(__GNUC__)
#define AMP_LIKELY(x) __builtin_expect(!!(x), 1)
#define AMP_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define AMP_LIKELY(x) (x)
#define AMP_UNLIKELY(x) (x)
#endif

/* Logging is opt-in: diagnostics remain disabled until explicitly enabled. */
static int logging_mode_enabled = 0;

static int amp_native_logging_enabled_internal(void) {
    return logging_mode_enabled;
}

AMP_CAPI int amp_native_logging_enabled(void) {
    return amp_native_logging_enabled_internal();
}

static void ensure_log_files_open(void) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    if (!log_lock_initialized) LOG_LOCK_INIT();
    LOG_LOCK();
    /* Create logs directory if it doesn't exist. On Windows CreateDirectoryA
       is a no-op if the directory already exists; on POSIX use mkdir(). */
#if defined(_WIN32) || defined(_WIN64)
    CreateDirectoryA("logs", NULL);
#else
    /* ignore errors: directory may already exist */
    mkdir("logs", 0775);
#endif

    if (log_f_alloc == NULL) log_f_alloc = fopen("logs/native_alloc_trace.log", "a");
    if (log_f_memops == NULL) log_f_memops = fopen("logs/native_mem_ops.log", "a");
    if (log_f_ccalls == NULL) log_f_ccalls = fopen("logs/native_c_calls.log", "a");
    if (log_f_cgenerated == NULL) log_f_cgenerated = fopen("logs/native_c_generated.log", "a");
    LOG_UNLOCK();

    /* Register close handler once (safe to call repeatedly). We do this
       outside the lock to avoid re-entrancy issues on some platforms. */
    static int atexit_registered = 0;
    if (!atexit_registered) {
        atexit(close_all_logs);
        atexit_registered = 1;
    }
}

/* Lightweight native-entry logger. Appends one-line records to logs/native_c_calls.log
   Format: <timestamp> <py_thread_state_ptr_or_tid> <function> <arg1> <arg2>\n
   Keep this function minimal and tolerant of failures (best-effort logging only).
*/
static void _log_native_call(const char *fn, size_t a, size_t b) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    ensure_log_files_open();
    if (log_f_ccalls == NULL) {
        return;
    }
    double t = (double)time(NULL);
#ifdef PyThreadState_Get
    void *py_ts = (void *)PyThreadState_Get();
    fprintf(log_f_ccalls, "%.3f %p %s %zu %zu\n", t, py_ts, fn, a, b);
#else
#if defined(_WIN32) || defined(_WIN64)
    unsigned long tid = (unsigned long) GetCurrentThreadId();
    fprintf(log_f_ccalls, "%.3f %lu %s %zu %zu\n", t, tid, fn, a, b);
#else
    unsigned long tid = (unsigned long) pthread_self();
    fprintf(log_f_ccalls, "%.3f %lu %s %zu %zu\n", t, tid, fn, a, b);
#endif
#endif
    fflush(log_f_ccalls);
}

/* Generated-wrapper logger: record wrapper entry and a couple numeric args. */
static void _gen_wrapper_log(const char *fn, size_t a, size_t b) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    ensure_log_files_open();
    if (log_f_cgenerated == NULL) return;
#ifdef PyThreadState_Get
    void *py_ts = (void *)PyThreadState_Get();
    fprintf(log_f_cgenerated, "%s %p %zu %zu\n", fn, py_ts, a, b);
#else
    fprintf(log_f_cgenerated, "%s %p %zu %zu\n", fn, (void*)0, a, b);
#endif
    fflush(log_f_cgenerated);
}

/*
 * Debug allocation wrappers that capture caller file/line/function.
 * To ensure the wrappers call the real libc allocation functions we
 * temporarily undef the common allocation macros, declare wrappers that
 * accept caller metadata, then re-map the allocation names to pass
 * __FILE__/__LINE__/__func__ automatically.
 */
#undef malloc
#undef calloc
#undef realloc
#undef free

#undef memcpy
#undef memset

static void *(*real_malloc_fn)(size_t) = malloc;
static void *(*real_calloc_fn)(size_t, size_t) = calloc;
static void *(*real_realloc_fn)(void *, size_t) = realloc;
static void (*real_free_fn)(void *) = free;
static void *(*real_memcpy_fn)(void *, const void *, size_t) = memcpy;
static void *(*real_memset_fn)(void *, int, size_t) = memset;

/* Simple in-memory allocation registry to detect writes into freed or
   undersized buffers. This is intentionally lightweight and only used
   in debug runs. We maintain a singly-linked list of allocation records
   updated by the allocation wrappers and consulted by the mem-op wrappers. */
typedef struct alloc_rec {
    void *ptr;
    size_t size;
    struct alloc_rec *next;
    /* recorded backtrace captured at registration time */
    void *bt[32];
    unsigned short bt_count;
} alloc_rec;

static alloc_rec *alloc_list = NULL;

/* forward declaration for backtrace dumper (defined below) */
static void dump_backtrace(FILE *g);
/* forward declarations for allocation backtrace helpers */
static void capture_stack_frames(void **out_frames, unsigned short *out_count);
static void dump_alloc_backtrace(FILE *g, struct alloc_rec *r);

static void close_all_logs(void) {
    if (!log_lock_initialized) LOG_LOCK_INIT();
    LOG_LOCK();
    FILE *alloc_handle = log_f_alloc;
    FILE *memops_handle = log_f_memops;
    FILE *ccalls_handle = log_f_ccalls;
    FILE *cgen_handle = log_f_cgenerated;
    log_f_alloc = NULL;
    log_f_memops = NULL;
    log_f_ccalls = NULL;
    log_f_cgenerated = NULL;
    LOG_UNLOCK();
    if (alloc_handle) { fflush(alloc_handle); fclose(alloc_handle); }
    if (memops_handle) { fflush(memops_handle); fclose(memops_handle); }
    if (ccalls_handle) { fflush(ccalls_handle); fclose(ccalls_handle); }
    if (cgen_handle) { fflush(cgen_handle); fclose(cgen_handle); }
}

/* Exported helper for other compilation units to emit generated-wrapper logs
   using the same cached-file backing store. This avoids repeated fopen()/fclose()
   in additional C files. */
AMP_CAPI void amp_log_generated(const char *fn, void *py_ts, size_t a, size_t b) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    ensure_log_files_open();
    if (!log_f_cgenerated) return;
    double _start = _now_clock_seconds();
    fprintf(log_f_cgenerated, "%s %p %zu %zu\n", fn, py_ts, a, b);
    fflush(log_f_cgenerated);
    double _end = _now_clock_seconds();
    if (_tl_current_node != NULL) {
        _tl_logging_accum += (_end - _start);
    }
}

/* Exported helper for other C files to log native-entry calls into the
   cached native_c_calls log. Mirrors the previous one-line format. */
AMP_CAPI void amp_log_native_call_external(const char *fn, size_t a, size_t b) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    ensure_log_files_open();
    if (!log_f_ccalls) return;
    double _start = _now_clock_seconds();
    double t = (double)time(NULL);
#if defined(_WIN32) || defined(_WIN64)
    unsigned long tid = (unsigned long) GetCurrentThreadId();
    fprintf(log_f_ccalls, "%.3f %lu %s %zu %zu\n", t, tid, fn, a, b);
#else
    unsigned long tid = (unsigned long) pthread_self();
    fprintf(log_f_ccalls, "%.3f %lu %s %zu %zu\n", t, tid, fn, a, b);
#endif
    fflush(log_f_ccalls);
    double _end = _now_clock_seconds();
    if (_tl_current_node != NULL) {
        _tl_logging_accum += (_end - _start);
    }
}

AMP_CAPI void amp_native_logging_set(int enabled) {
    int normalised = enabled ? 1 : 0;
    if (!normalised) {
        close_all_logs();
    }
    if (!log_lock_initialized) LOG_LOCK_INIT();
    LOG_LOCK();
    logging_mode_enabled = normalised;
    LOG_UNLOCK();
    if (normalised) {
        ensure_log_files_open();
    }
}

/* Dump a compact snapshot of the live allocation registry to the given file.
 * This is intended to be called when we observe unexpected unregisters so
 * the offline correlator can inspect the live registry at the moment of the
 * event. Keep the output compact but include pointer, size and (if present)
 * the recorded registration backtrace id.
 */
static void dump_alloc_snapshot(FILE *g) {
    if (!amp_native_logging_enabled_internal()) return;
    alloc_rec *it = alloc_list;
    fprintf(g, "ALLOC_SNAPSHOT_BEGIN\n");
    while (it) {
        fprintf(g, "ALLOC_ENTRY ptr=%p size=%zu bt_count=%u\n", it->ptr, it->size, (unsigned)it->bt_count);
        /* print the recorded registration backtrace (if available) inline */
        for (unsigned i = 0; i < it->bt_count; ++i) {
            fprintf(g, "RBT %p\n", it->bt[i]);
        }
        it = it->next;
    }
    fprintf(g, "ALLOC_SNAPSHOT_END\n");
}

static void register_alloc(void *ptr, size_t size) {
    if (!amp_native_logging_enabled_internal()) return;
    if (ptr == NULL) return;
    /* If already registered, update size and record an update log instead
       of creating duplicate entries. This avoids confusing duplicate
       REGISTER lines and makes unregistering deterministic. */
    alloc_rec *it = alloc_list;
    while (it) {
        if (it->ptr == ptr) {
            /* update size if changed */
            if (it->size != size) {
                ensure_log_files_open();
                if (log_f_alloc) {
                    fprintf(log_f_alloc, "REGISTER_UPDATE %p old=%zu new=%zu\n", ptr, it->size, size);
                    dump_alloc_snapshot(log_f_alloc);
                    dump_backtrace(log_f_alloc);
                    fflush(log_f_alloc);
                }
                it->size = size;
            }
            return;
        }
        it = it->next;
    }
    alloc_rec *r = (alloc_rec *)malloc(sizeof(alloc_rec));
    if (r == NULL) return;
    r->ptr = ptr;
    r->size = size;
    r->next = alloc_list;
    /* record allocation backtrace for correlation */
    r->bt_count = 0;
    capture_stack_frames(r->bt, &r->bt_count);
    alloc_list = r;
    /* Durable log so post-processing can see registrations that
       originate from stack-local buffers or manual calls to register_alloc. */
    ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "REGISTER %p size=%zu\n", ptr, size);
        /* print recorded registration backtrace for easier offline mapping */
        dump_alloc_backtrace(log_f_alloc, r);
        fflush(log_f_alloc);
    }
}

static void unregister_alloc(void *ptr) {
    if (!amp_native_logging_enabled_internal()) return;
    if (ptr == NULL) return;
    /* Remove all matching entries for ptr (shouldn't be duplicates if
       register_alloc prevents them, but be robust). Record if no entry
       was found to help detect double-free or mismatched lifetime. */
    alloc_rec **pp = &alloc_list;
    int removed = 0;
    while (*pp) {
        if ((*pp)->ptr == ptr) {
            alloc_rec *remove = *pp;
            *pp = remove->next;
            ensure_log_files_open();
            if (log_f_alloc) {
                fprintf(log_f_alloc, "UNREGISTER %p\n", ptr);
                /* dump the registry and the recorded registration backtrace */
                dump_alloc_snapshot(log_f_alloc);
                dump_alloc_backtrace(log_f_alloc, remove);
                dump_backtrace(log_f_alloc);
                fflush(log_f_alloc);
            }
            free(remove);
            removed = 1;
            /* continue scanning to remove duplicates */
            continue;
        }
        pp = &(*pp)->next;
    }
    if (!removed) {
        ensure_log_files_open();
        if (log_f_alloc) {
            fprintf(log_f_alloc, "UNREGISTER_NOTFOUND %p\n", ptr);
            dump_alloc_snapshot(log_f_alloc);
            dump_backtrace(log_f_alloc);
            fflush(log_f_alloc);
        }
    }
}

/* Check whether [addr, addr+len) is fully contained inside a known
   allocated record. Returns 1 if contained, 0 otherwise. */
static int range_within_alloc(void *addr, size_t len) {
    if (addr == NULL) return 0;
    unsigned char *a = (unsigned char *)addr;
    alloc_rec *r = alloc_list;
    while (r) {
        unsigned char *base = (unsigned char *)r->ptr;
        if (a >= base && (a + len) <= (base + r->size)) return 1;
        r = r->next;
    }
    return 0;
}

/* Portable backtrace dumper: prints a compact list of return addresses or
   symbolified strings (where available) to the provided file stream. This
   is intentionally small and best-effort: the presence of addresses in the
   BAD_* logs will significantly speed offline correlation even without
   symbol resolution. */
static void dump_backtrace(FILE *g) {
    if (!amp_native_logging_enabled_internal()) return;
    if (g == NULL) return;
#if defined(_WIN32) || defined(_WIN64)
    /* Use CaptureStackBackTrace which is available on Windows. We print
       raw return addresses; symbol resolution can be done offline if
       required. */
    void *frames[64];
    USHORT count = CaptureStackBackTrace(0, (ULONG) (sizeof(frames)/sizeof(frames[0])), frames, NULL);
    fprintf(g, "BACKTRACE_FRAMES %u\n", (unsigned)count);
    for (USHORT i = 0; i < count; ++i) {
        fprintf(g, "BT %p\n", frames[i]);
    }
#else
    /* POSIX: use backtrace/backtrace_symbols where available. */
#ifdef __GNUC__
    void *frames[64];
    int count = backtrace(frames, (int)(sizeof(frames)/sizeof(frames[0])));
    char **symbols = backtrace_symbols(frames, count);
    if (symbols != NULL) {
        fprintf(g, "BACKTRACE_FRAMES %d\n", count);
        for (int i = 0; i < count; ++i) {
            fprintf(g, "BT %s\n", symbols[i]);
        }
        free(symbols);
    } else {
        fprintf(g, "BACKTRACE_FRAMES %d\n", count);
        for (int i = 0; i < count; ++i) fprintf(g, "BT %p\n", frames[i]);
    }
#else
    (void)g; /* no-op if backtrace APIs unavailable */
#endif
#endif
}

/* Capture stack frames into the provided buffer and set count. This is
   used to record a backtrace at allocation time for later correlation. */
static void capture_stack_frames(void **out_frames, unsigned short *out_count) {
    if (!amp_native_logging_enabled_internal()) return;
    if (out_frames == NULL || out_count == NULL) return;
#if defined(_WIN32) || defined(_WIN64)
    void *frames[64];
    USHORT count = CaptureStackBackTrace(0, (ULONG)(sizeof(frames)/sizeof(frames[0])), frames, NULL);
    unsigned short copy_count = (unsigned short)((count > 32) ? 32 : count);
    for (unsigned short i = 0; i < copy_count; ++i) out_frames[i] = frames[i];
    *out_count = copy_count;
#else
#ifdef __GNUC__
    void *frames[64];
    int count = backtrace(frames, (int)(sizeof(frames)/sizeof(frames[0])));
    int copy_count = (count > 32) ? 32 : count;
    for (int i = 0; i < copy_count; ++i) out_frames[i] = frames[i];
    *out_count = (unsigned short)copy_count;
#else
    *out_count = 0;
#endif
#endif
}

/* Print the recorded allocation backtrace stored in alloc_rec (if any). */
static void dump_alloc_backtrace(FILE *g, alloc_rec *r) {
    if (!amp_native_logging_enabled_internal()) return;
    if (g == NULL || r == NULL) return;
    fprintf(g, "REGISTER_BACKTRACE %u\n", (unsigned)r->bt_count);
    for (unsigned i = 0; i < r->bt_count; ++i) {
        fprintf(g, "RBT %p\n", r->bt[i]);
    }
}

static void *_dbg_malloc(size_t s, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_malloc_fn(s);
    }
    void *p = malloc(s); /* calls real malloc because we undef'd macro above */
    /* log: op=malloc, size, ptr, caller */
    _log_native_call("malloc", (size_t)s, (size_t)(uintptr_t)p);
    ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "MALLOC %s:%d %s size=%zu ptr=%p\n", file, line, func, s, p);
        fflush(log_f_alloc);
    }
    if (p != NULL) register_alloc(p, s);
    return p;
}

static void *_dbg_calloc(size_t n, size_t size, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_calloc_fn(n, size);
    }
    void *p = calloc(n, size);
    _log_native_call("calloc", (size_t)(n * size), (size_t)(uintptr_t)p);
    ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "CALLOC %s:%d %s nmemb=%zu size=%zu ptr=%p\n", file, line, func, n, size, p);
        fflush(log_f_alloc);
    }
    if (p != NULL) register_alloc(p, n * size);
    return p;
}

static void *_dbg_realloc(void *ptr, size_t s, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_realloc_fn(ptr, s);
    }
    void *p = realloc(ptr, s);
    _log_native_call("realloc_old", (size_t)(uintptr_t)ptr, (size_t)(uintptr_t)p);
    _log_native_call("realloc_new", (size_t)s, (size_t)(uintptr_t)p);
    ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "REALLOC %s:%d %s old=%p new=%p size=%zu\n", file, line, func, ptr, p, s);
        dump_backtrace(log_f_alloc);
        fflush(log_f_alloc);
    }
    /* update registry: remove old entry and register new pointer */
    if (ptr != NULL) unregister_alloc(ptr);
    if (p != NULL) register_alloc(p, s);
    return p;
}

static void _dbg_free(void *ptr, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        if (ptr != NULL) real_free_fn(ptr);
        return;
    }
    _log_native_call("free", (size_t)(uintptr_t)ptr, 0);
    ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "FREE %s:%d %s ptr=%p\n", file, line, func, ptr);
        dump_backtrace(log_f_alloc);
        fflush(log_f_alloc);
    }
    if (ptr != NULL) {
        unregister_alloc(ptr);
        free(ptr);
    }
}

/* Debug wrappers for memcpy/memset to log memory writes/copies. These use
   simple byte loops to avoid calling the (possibly macro-redirected)
   libc functions and to keep the logging minimal and self-contained. */
static void *_dbg_memcpy(void *dest, const void *src, size_t n, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_memcpy_fn(dest, src, n);
    }
    ensure_log_files_open();
    if (log_f_memops) fprintf(log_f_memops, "MEMCPY %s:%d %s dest=%p src=%p n=%zu\n", file, line, func, dest, src, n);
    if (dest == NULL || src == NULL || n == 0) {
        return dest;
    }
    /* Check destination range against known allocations. If not found,
       allow destinations that are likely on the current stack to avoid
       false positives for local buffers. We use a heuristic window around
       the current stack pointer. */
        if (!range_within_alloc(dest, n)) {
        uintptr_t stack_probe = (uintptr_t)&stack_probe;
        uintptr_t d = (uintptr_t)dest;
        const uintptr_t STACK_WINDOW = 0x100000; /* 1MB */
        if (!(d >= stack_probe - STACK_WINDOW && d <= stack_probe + STACK_WINDOW)) {
            ensure_log_files_open();
            if (log_f_memops) {
                fprintf(log_f_memops, "BAD_MEMCPY %s:%d %s dest=%p n=%zu stack_probe=%p (no matching alloc or too small)\n", file, line, func, dest, n, (void*)stack_probe);
                /* dump live allocation registry snapshot for post-mortem correlation */
                alloc_rec *it = alloc_list;
                while (it) {
                    fprintf(log_f_memops, "ALLOC_SNAPSHOT ptr=%p size=%zu\n", it->ptr, it->size);
                    it = it->next;
                }
                /* Append a backtrace; best-effort and compact. */
                dump_backtrace(log_f_memops);
                fflush(log_f_memops);
            }
        }
    }
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;
    for (size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
    return dest;
}

static void *_dbg_memset(void *s_ptr, int c, size_t n, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_memset_fn(s_ptr, c, n);
    }
    ensure_log_files_open();
    if (log_f_memops) fprintf(log_f_memops, "MEMSET %s:%d %s ptr=%p val=%d n=%zu\n", file, line, func, s_ptr, c, n);
    if (s_ptr == NULL || n == 0) return s_ptr;
    if (!range_within_alloc(s_ptr, n)) {
        uintptr_t stack_probe = (uintptr_t)&stack_probe;
        uintptr_t d = (uintptr_t)s_ptr;
        const uintptr_t STACK_WINDOW = 0x100000; /* 1MB */
        if (!(d >= stack_probe - STACK_WINDOW && d <= stack_probe + STACK_WINDOW)) {
            ensure_log_files_open();
            if (log_f_memops) {
                fprintf(log_f_memops, "BAD_MEMSET %s:%d %s ptr=%p n=%zu stack_probe=%p (no matching alloc or too small)\n", file, line, func, s_ptr, n, (void*)stack_probe);
                dump_backtrace(log_f_memops);
                fflush(log_f_memops);
            }
        }
    }
    unsigned char *p = (unsigned char *)s_ptr;
    unsigned char v = (unsigned char)c;
    for (size_t i = 0; i < n; ++i) p[i] = v;
    return s_ptr;
}

/* Re-map the allocation names so callers automatically pass caller metadata. */
#define malloc(s) _dbg_malloc((s), __FILE__, __LINE__, __func__)
#define calloc(n,s) _dbg_calloc((n),(s), __FILE__, __LINE__, __func__)
#define realloc(p,s) _dbg_realloc((p),(s), __FILE__, __LINE__, __func__)
#define free(p) _dbg_free((p), __FILE__, __LINE__, __func__)

/* Redirect memcpy/memset to debug wrappers so we capture large buffer writes */
#define memcpy(d,s,n) _dbg_memcpy((d),(s),(n), __FILE__, __LINE__, __func__)
#define memset(p,c,n) _dbg_memset((p),(c),(n), __FILE__, __LINE__, __func__)

#define AMP_LOG_NATIVE_CALL(fn, a, b) _log_native_call((fn), (a), (b))
#define AMP_LOG_GENERATED(fn, a, b) _gen_wrapper_log((fn), (a), (b))

#else  /* !AMP_NATIVE_ENABLE_LOGGING */

static inline int amp_native_logging_enabled_internal(void) {
    return 0;
}

AMP_CAPI int amp_native_logging_enabled(void) {
    return 0;
}

static inline void ensure_log_files_open(void) {
    (void)0;
}

static inline void close_all_logs(void) {
    (void)0;
}

static inline void _log_native_call(const char *fn, size_t a, size_t b) {
    (void)fn;
    (void)a;
    (void)b;
}

static inline void _gen_wrapper_log(const char *fn, size_t a, size_t b) {
    (void)fn;
    (void)a;
    (void)b;
}

AMP_CAPI void amp_log_generated(const char *fn, void *py_ts, size_t a, size_t b) {
    (void)fn;
    (void)py_ts;
    (void)a;
    (void)b;
}

#define log_f_alloc ((FILE *)0)
#define log_f_memops ((FILE *)0)
#define log_f_ccalls ((FILE *)0)
#define log_f_cgenerated ((FILE *)0)

static inline void register_alloc(void *ptr, size_t size) {
    (void)ptr;
    (void)size;
}

static inline void unregister_alloc(void *ptr) {
    (void)ptr;
}

AMP_CAPI void amp_log_native_call_external(const char *fn, size_t a, size_t b) {
    (void)fn;
    (void)a;
    (void)b;
}

AMP_CAPI void amp_native_logging_set(int enabled) {
    (void)enabled;
}

#define AMP_LOG_NATIVE_CALL(fn, a, b) ((void)0)
#define AMP_LOG_GENERATED(fn, a, b) ((void)0)

#endif /* AMP_NATIVE_ENABLE_LOGGING */

static void destroy_compiled_plan(EdgeRunnerCompiledPlan *plan) {
    if (plan == NULL) {
        return;
    }
    if (plan->nodes != NULL) {
        for (uint32_t i = 0; i < plan->node_count; ++i) {
            EdgeRunnerCompiledNode *node = &plan->nodes[i];
            if (node->params != NULL) {
                for (uint32_t j = 0; j < node->param_count; ++j) {
                    EdgeRunnerCompiledParam *param = &node->params[j];
                    if (param->name != NULL) {
                        free(param->name);
                        param->name = NULL;
                    }
                }
                free(node->params);
                node->params = NULL;
            }
            if (node->name != NULL) {
                free(node->name);
                node->name = NULL;
            }
        }
        free(plan->nodes);
        plan->nodes = NULL;
    }
    free(plan);
}

static int read_u32_le(const uint8_t **cursor, size_t *remaining, uint32_t *out_value) {
    if (cursor == NULL || remaining == NULL || out_value == NULL) {
        return 0;
    }
    if (*remaining < 4) {
        return 0;
    }
    const uint8_t *ptr = *cursor;
    *out_value = (uint32_t)ptr[0]
        | ((uint32_t)ptr[1] << 8)
        | ((uint32_t)ptr[2] << 16)
        | ((uint32_t)ptr[3] << 24);
    *cursor += 4;
    *remaining -= 4;
    return 1;
}

AMP_CAPI EdgeRunnerCompiledPlan *amp_load_compiled_plan(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len
) {
    AMP_LOG_NATIVE_CALL("amp_load_compiled_plan", descriptor_len, plan_len);
    AMP_LOG_GENERATED("amp_load_compiled_plan", (size_t)descriptor_blob, (size_t)plan_blob);
    if (descriptor_blob == NULL || plan_blob == NULL) {
        return NULL;
    }
    if (descriptor_len < 4 || plan_len < 12) {
        return NULL;
    }

    const uint8_t *descriptor_cursor = descriptor_blob;
    size_t descriptor_remaining = descriptor_len;
    uint32_t descriptor_count = 0;
    if (!read_u32_le(&descriptor_cursor, &descriptor_remaining, &descriptor_count)) {
        return NULL;
    }

    const uint8_t *cursor = plan_blob;
    size_t remaining = plan_len;
    if (remaining < 4) {
        return NULL;
    }
    if (cursor[0] != 'A' || cursor[1] != 'M' || cursor[2] != 'P' || cursor[3] != 'L') {
        return NULL;
    }
    cursor += 4;
    remaining -= 4;

    uint32_t version = 0;
    uint32_t node_count = 0;
    if (!read_u32_le(&cursor, &remaining, &version) || !read_u32_le(&cursor, &remaining, &node_count)) {
        return NULL;
    }
    if (descriptor_count != node_count) {
        return NULL;
    }

    EdgeRunnerCompiledPlan *plan = (EdgeRunnerCompiledPlan *)calloc(1, sizeof(EdgeRunnerCompiledPlan));
    if (plan == NULL) {
        return NULL;
    }
    plan->version = version;
    plan->node_count = node_count;

    if (node_count == 0) {
        if (remaining != 0) {
            destroy_compiled_plan(plan);
            return NULL;
        }
        return plan;
    }

    plan->nodes = (EdgeRunnerCompiledNode *)calloc(node_count, sizeof(EdgeRunnerCompiledNode));
    if (plan->nodes == NULL) {
        destroy_compiled_plan(plan);
        return NULL;
    }

    for (uint32_t idx = 0; idx < node_count; ++idx) {
        EdgeRunnerCompiledNode *node = &plan->nodes[idx];
        uint32_t function_id = 0;
        uint32_t name_len = 0;
        uint32_t audio_offset = 0;
        uint32_t audio_span = 0;
        uint32_t param_count = 0;
        if (!read_u32_le(&cursor, &remaining, &function_id)
            || !read_u32_le(&cursor, &remaining, &name_len)
            || !read_u32_le(&cursor, &remaining, &audio_offset)
            || !read_u32_le(&cursor, &remaining, &audio_span)
            || !read_u32_le(&cursor, &remaining, &param_count)) {
            destroy_compiled_plan(plan);
            return NULL;
        }
        if (remaining < name_len) {
            destroy_compiled_plan(plan);
            return NULL;
        }
        node->name = (char *)malloc((size_t)name_len + 1);
        if (node->name == NULL) {
            destroy_compiled_plan(plan);
            return NULL;
        }
        memcpy(node->name, cursor, name_len);
        node->name[name_len] = '\0';
        node->name_len = name_len;
        cursor += name_len;
        remaining -= name_len;
        node->function_id = function_id;
        node->audio_offset = audio_offset;
        node->audio_span = audio_span;
        node->param_count = param_count;
        if (param_count > 0) {
            node->params = (EdgeRunnerCompiledParam *)calloc(param_count, sizeof(EdgeRunnerCompiledParam));
            if (node->params == NULL) {
                destroy_compiled_plan(plan);
                return NULL;
            }
        }
        for (uint32_t param_idx = 0; param_idx < param_count; ++param_idx) {
            EdgeRunnerCompiledParam *param = &node->params[param_idx];
            uint32_t param_name_len = 0;
            uint32_t param_offset = 0;
            uint32_t param_span = 0;
            if (!read_u32_le(&cursor, &remaining, &param_name_len)
                || !read_u32_le(&cursor, &remaining, &param_offset)
                || !read_u32_le(&cursor, &remaining, &param_span)) {
                destroy_compiled_plan(plan);
                return NULL;
            }
            if (remaining < param_name_len) {
                destroy_compiled_plan(plan);
                return NULL;
            }
            param->name = (char *)malloc((size_t)param_name_len + 1);
            if (param->name == NULL) {
                destroy_compiled_plan(plan);
                return NULL;
            }
            memcpy(param->name, cursor, param_name_len);
            param->name[param_name_len] = '\0';
            param->name_len = param_name_len;
            param->offset = param_offset;
            param->span = param_span;
            cursor += param_name_len;
            remaining -= param_name_len;
        }
    }

    if (remaining != 0) {
        destroy_compiled_plan(plan);
        return NULL;
    }

    return plan;
}

AMP_CAPI void amp_release_compiled_plan(EdgeRunnerCompiledPlan *plan) {
    AMP_LOG_NATIVE_CALL("amp_release_compiled_plan", (size_t)(plan != NULL), 0);
    AMP_LOG_GENERATED("amp_release_compiled_plan", (size_t)plan, 0);
    destroy_compiled_plan(plan);
}

static void destroy_control_history(EdgeRunnerControlHistory *history) {
    if (history == NULL) {
        return;
    }
    if (history->curves != NULL) {
        for (uint32_t i = 0; i < history->curve_count; ++i) {
            EdgeRunnerControlCurve *curve = &history->curves[i];
            if (curve->name != NULL) {
                free(curve->name);
                curve->name = NULL;
            }
            if (curve->values != NULL) {
                free(curve->values);
                curve->values = NULL;
            }
            curve->value_count = 0;
        }
        free(history->curves);
        history->curves = NULL;
    }
    free(history);
}

static const EdgeRunnerControlCurve *find_history_curve(
    const EdgeRunnerControlHistory *history,
    const char *name,
    size_t name_len
) {
    if (history == NULL || name == NULL || name_len == 0) {
        return NULL;
    }
    for (uint32_t i = 0; i < history->curve_count; ++i) {
        const EdgeRunnerControlCurve *curve = &history->curves[i];
        if (curve->name_len == name_len && curve->name != NULL && strncmp(curve->name, name, name_len) == 0) {
            return curve;
        }
    }
    return NULL;
}

static void apply_history_curve(
    double *dest,
    int batches,
    int frames,
    const EdgeRunnerControlCurve *curve
) {
    if (dest == NULL || curve == NULL || curve->values == NULL || curve->value_count == 0) {
        return;
    }
    int count = (int)curve->value_count;
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    for (int b = 0; b < batches; ++b) {
        for (int f = 0; f < frames; ++f) {
            double value = 0.0;
            if (count >= frames) {
                if (f < count) {
                    value = curve->values[f];
                } else {
                    value = curve->values[count - 1];
                }
            } else if (count == 1) {
                value = curve->values[0];
            } else {
                if (f < count) {
                    value = curve->values[f];
                } else {
                    value = curve->values[count - 1];
                }
            }
            dest[((size_t)b * (size_t)frames) + (size_t)f] = value;
        }
    }
}

AMP_CAPI EdgeRunnerControlHistory *amp_load_control_history(
    const uint8_t *blob,
    size_t blob_len,
    int frames_hint
) {
    AMP_LOG_NATIVE_CALL("amp_load_control_history", blob_len, (size_t)frames_hint);
    AMP_LOG_GENERATED("amp_load_control_history", (size_t)blob, (size_t)frames_hint);
    if (blob == NULL || blob_len < 8) {
        return NULL;
    }
    const uint8_t *cursor = blob;
    size_t remaining = blob_len;
    uint32_t event_count = 0;
    uint32_t key_count = 0;
    if (!read_u32_le(&cursor, &remaining, &event_count) || !read_u32_le(&cursor, &remaining, &key_count)) {
        return NULL;
    }
    EdgeRunnerControlHistory *history = (EdgeRunnerControlHistory *)calloc(1, sizeof(EdgeRunnerControlHistory));
    if (history == NULL) {
        return NULL;
    }
    history->frames_hint = frames_hint > 0 ? (uint32_t)frames_hint : 0U;
    history->curve_count = key_count;
    if (key_count > 0) {
        history->curves = (EdgeRunnerControlCurve *)calloc(key_count, sizeof(EdgeRunnerControlCurve));
        if (history->curves == NULL) {
            destroy_control_history(history);
            return NULL;
        }
    }
    if (key_count == 0) {
        return history;
    }
    uint32_t *name_lengths = (uint32_t *)calloc(key_count, sizeof(uint32_t));
    if (name_lengths == NULL) {
        destroy_control_history(history);
        return NULL;
    }
    for (uint32_t i = 0; i < key_count; ++i) {
        if (!read_u32_le(&cursor, &remaining, &name_lengths[i])) {
            free(name_lengths);
            destroy_control_history(history);
            return NULL;
        }
    }
    for (uint32_t i = 0; i < key_count; ++i) {
        uint32_t name_len = name_lengths[i];
        if (remaining < name_len) {
            free(name_lengths);
            destroy_control_history(history);
            return NULL;
        }
        EdgeRunnerControlCurve *curve = &history->curves[i];
        curve->name = (char *)malloc((size_t)name_len + 1);
        if (curve->name == NULL) {
            free(name_lengths);
            destroy_control_history(history);
            return NULL;
        }
        memcpy(curve->name, cursor, name_len);
        curve->name[name_len] = '\0';
        curve->name_len = name_len;
        curve->value_count = 0;
        curve->values = NULL;
        curve->timestamp = -DBL_MAX;
        cursor += name_len;
        remaining -= name_len;
    }
    free(name_lengths);
    for (uint32_t event_idx = 0; event_idx < event_count; ++event_idx) {
        if (remaining < sizeof(double)) {
            destroy_control_history(history);
            return NULL;
        }
        double timestamp = 0.0;
        memcpy(&timestamp, cursor, sizeof(double));
        cursor += sizeof(double);
        remaining -= sizeof(double);
        for (uint32_t key_idx = 0; key_idx < key_count; ++key_idx) {
            uint32_t value_count = 0;
            if (!read_u32_le(&cursor, &remaining, &value_count)) {
                destroy_control_history(history);
                return NULL;
            }
            double *values_copy = NULL;
            if (value_count > 0) {
                size_t bytes = (size_t)value_count * sizeof(double);
                if (remaining < bytes) {
                    destroy_control_history(history);
                    return NULL;
                }
                values_copy = (double *)malloc(bytes);
                if (values_copy == NULL) {
                    destroy_control_history(history);
                    return NULL;
                }
                memcpy(values_copy, cursor, bytes);
                cursor += bytes;
                remaining -= bytes;
            }
            EdgeRunnerControlCurve *curve = &history->curves[key_idx];
            if (value_count > 0 && (curve->values == NULL || timestamp >= curve->timestamp)) {
                if (curve->values != NULL) {
                    free(curve->values);
                }
                curve->values = values_copy;
                curve->value_count = value_count;
                curve->timestamp = timestamp;
                values_copy = NULL;
            }
            if (values_copy != NULL) {
                free(values_copy);
            }
        }
    }
    return history;
}

AMP_CAPI void amp_release_control_history(EdgeRunnerControlHistory *history) {
    AMP_LOG_NATIVE_CALL("amp_release_control_history", (size_t)(history != NULL), 0);
    AMP_LOG_GENERATED("amp_release_control_history", (size_t)history, 0);
    destroy_control_history(history);
}

static int envelope_reserve_scratch(int F) {
    size_t needed_boundaries = (size_t)(4 * F + 4);
    size_t needed_trig = (size_t)(F > 0 ? F : 1);
    size_t needed_bool = (size_t)(F > 0 ? F : 1);

    if (envelope_scratch.boundary_cap < needed_boundaries) {
        int *new_boundaries = (int *)realloc(envelope_scratch.boundaries, needed_boundaries * sizeof(int));
        if (new_boundaries == NULL) {
            return 0;
        }
        envelope_scratch.boundaries = new_boundaries;
        envelope_scratch.boundary_cap = needed_boundaries;
    }

    if (envelope_scratch.trig_cap < needed_trig) {
        int *new_trig = (int *)realloc(envelope_scratch.trig_indices, needed_trig * sizeof(int));
        if (new_trig == NULL) {
            return 0;
        }
        envelope_scratch.trig_indices = new_trig;
        envelope_scratch.trig_cap = needed_trig;
    }

    if (envelope_scratch.bool_cap < needed_bool) {
        int8_t *gate_ptr = envelope_scratch.gate_bool;
        int8_t *drone_ptr = envelope_scratch.drone_bool;
        int8_t *new_gate = (int8_t *)realloc(gate_ptr, needed_bool * sizeof(int8_t));
        int8_t *new_drone = (int8_t *)realloc(drone_ptr, needed_bool * sizeof(int8_t));
        if (new_gate == NULL || new_drone == NULL) {
            if (new_gate != NULL) {
                envelope_scratch.gate_bool = new_gate;
            }
            if (new_drone != NULL) {
                envelope_scratch.drone_bool = new_drone;
            }
            return 0;
        }
        envelope_scratch.gate_bool = new_gate;
        envelope_scratch.drone_bool = new_drone;
        envelope_scratch.bool_cap = needed_bool;
    }

    return 1;
}

void lfo_slew(const double* x, double* out, int B, int F, double r, double alpha, double* z0) {
    for (int b = 0; b < B; ++b) {
        double state = 0.0;
        if (z0 != NULL) state = z0[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            double xi = x[base + i];
            state = r * state + alpha * xi;
            out[base + i] = state;
        }
        if (z0 != NULL) z0[b] = state;
    }
}

void safety_filter(const double* x, double* y, int B, int C, int F, double a, double* prev_in, double* prev_dc) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            int chan = b * C + c;
            double pi = 0.0;
            double pd = 0.0;
            if (prev_in != NULL) pi = prev_in[chan];
            if (prev_dc != NULL) pd = prev_dc[chan];
            int base = chan * F;
            for (int i = 0; i < F; ++i) {
                double xin = x[base + i];
                double diff;
                if (i == 0) diff = xin - pi;
                else diff = xin - x[base + i - 1];
                pd = a * pd + diff;
                y[base + i] = pd;
            }
            if (prev_in != NULL) prev_in[chan] = x[base + F - 1];
            if (prev_dc != NULL) prev_dc[chan] = y[base + F - 1];
        }
    }
}

void dc_block(const double* x, double* out, int B, int C, int F, double a, double* state) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            int chan = b * C + c;
            double dc = 0.0;
            if (state != NULL) dc = state[chan];
            int base = chan * F;
            for (int i = 0; i < F; ++i) {
                double xin = x[base + i];
                dc = a * dc + (1.0 - a) * xin;
                out[base + i] = xin - dc;
            }
            if (state != NULL) state[chan] = dc;
        }
    }
}

void subharmonic_process(
    const double* x,
    double* y,
    int B,
    int C,
    int F,
    double a_hp_in,
    double a_lp_in,
    double a_sub2,
    int use_div4,
    double a_sub4,
    double a_env_attack,
    double a_env_release,
    double a_hp_out,
    double drive,
    double mix,
    double* hp_y,
    double* lp_y,
    double* prev,
    int8_t* sign,
    int8_t* ff2,
    int8_t* ff4,
    int32_t* ff4_count,
    double* sub2_lp,
    double* sub4_lp,
    double* env,
    double* hp_out_y,
    double* hp_out_x
) {
    // Layout: arrays are flattened per-channel: index = (b*C + c) * F + t
    for (int t = 0; t < F; ++t) {
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                int chan = b * C + c;
                int base = chan * F;
                double xt = x[base + t];

                // Bandpass driver: simple HP then LP
                double hp_y_val = hp_y[chan];
                double prev_val = prev[chan];
                double lp_y_val = lp_y[chan];
                double hp_in = a_hp_in * (hp_y_val + xt - prev_val);
                hp_y[chan] = hp_in;
                prev[chan] = xt;
                double bp = lp_y_val + a_lp_in * (hp_in - lp_y_val);
                lp_y[chan] = bp;

                // env
                double abs_bp = fabs(bp);
                double env_val = env[chan];
                if (abs_bp > env_val) env_val = env_val + a_env_attack * (abs_bp - env_val);
                else env_val = env_val + a_env_release * (abs_bp - env_val);
                env[chan] = env_val;

                // sign and flip-flops
                int8_t prev_sign = sign[chan];
                int8_t sign_now = (bp > 0.0) ? 1 : -1;
                int pos_zc = (prev_sign < 0) && (sign_now > 0);
                sign[chan] = sign_now;

                if (pos_zc) ff2[chan] = -ff2[chan];

                if (use_div4) {
                    if (pos_zc) ff4_count[chan] = ff4_count[chan] + 1;
                    int toggle4 = (pos_zc && (ff4_count[chan] >= 2));
                    if (toggle4) ff4[chan] = -ff4[chan];
                    if (toggle4) ff4_count[chan] = 0;
                }

                double sq2 = (double) ff2[chan];
                double sub2_lp_val = sub2_lp[chan];
                sub2_lp_val = sub2_lp_val + a_sub2 * (sq2 - sub2_lp_val);
                sub2_lp[chan] = sub2_lp_val;
                double sub_val = sub2_lp_val;

                if (use_div4) {
                    double sq4 = (double) ff4[chan];
                    double sub4_lp_val = sub4_lp[chan];
                    sub4_lp_val = sub4_lp_val + a_sub4 * (sq4 - sub4_lp_val);
                    sub4_lp[chan] = sub4_lp_val;
                    sub_val = sub_val + 0.6 * sub4_lp_val;
                }

                double sub = tanh(drive * sub_val) * (env_val + 1e-6);

                double dry = xt;
                double wet = sub;
                double out_t = (1.0 - mix) * dry + mix * wet;

                double y_prev = hp_out_y[chan];
                double x_prev = hp_out_x[chan];
                double hp = a_hp_out * (y_prev + out_t - x_prev);
                hp_out_y[chan] = hp;
                hp_out_x[chan] = out_t;
                y[base + t] = hp;
            }
        }
    }
}

static void envelope_start_attack(
    int index,
    const double* velocity,
    int send_resets,
    double* reset_line,
    int* stage,
    double* timer,
    double* value,
    double* vel_state,
    double* release_start,
    int64_t* activations
) {
    double vel = velocity[index];
    if (vel < 0.0) vel = 0.0;
    *stage = 1;
    *timer = 0.0;
    *value = 0.0;
    *vel_state = vel;
    *release_start = vel;
    *activations += 1;
    if (send_resets && reset_line != NULL) {
        reset_line[index] = 1.0;
    }
}

static void envelope_process_simple(
    const double* trigger,
    const double* gate,
    const double* drone,
    const double* velocity,
    int B,
    int F,
    int atk_frames,
    int hold_frames,
    int dec_frames,
    int sus_frames,
    int rel_frames,
    double sustain_level,
    int send_resets,
    int* stage,
    double* value,
    double* timer,
    double* vel_state,
    int64_t* activations,
    double* release_start,
    double* amp_out,
    double* reset_out
) {
    for (int b = 0; b < B; ++b) {
        int st = stage[b];
        double val = value[b];
        double tim = timer[b];
        double vel = vel_state[b];
        int64_t acts = activations[b];
        double rel_start = release_start[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            int idx = base + i;
            int trig = trigger[idx] > 0.5 ? 1 : 0;
            int gate_on = gate[idx] > 0.5 ? 1 : 0;
            int drone_on = drone[idx] > 0.5 ? 1 : 0;

            if (trig) {
                envelope_start_attack(i, velocity + base, send_resets, reset_out != NULL ? reset_out + base : NULL, &st, &tim, &val, &vel, &rel_start, &acts);
            } else if (st == 0 && (gate_on || drone_on)) {
                envelope_start_attack(i, velocity + base, send_resets, reset_out != NULL ? reset_out + base : NULL, &st, &tim, &val, &vel, &rel_start, &acts);
            }

            if (st == 1) {
                if (atk_frames <= 0) {
                    val = vel;
                    if (hold_frames > 0) st = 2;
                    else if (dec_frames > 0) st = 3;
                    else st = 4;
                    tim = 0.0;
                } else {
                    val += vel / (double)(atk_frames > 0 ? atk_frames : 1);
                    if (val > vel) val = vel;
                    tim += 1.0;
                    if (tim >= atk_frames) {
                        val = vel;
                        if (hold_frames > 0) st = 2;
                        else if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                    }
                }
            } else if (st == 2) {
                val = vel;
                if (hold_frames <= 0) {
                    if (dec_frames > 0) st = 3;
                    else st = 4;
                    tim = 0.0;
                } else {
                    tim += 1.0;
                    if (tim >= hold_frames) {
                        if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                    }
                }
            } else if (st == 3) {
                double target = vel * sustain_level;
                if (dec_frames <= 0) {
                    val = target;
                    st = 4;
                    tim = 0.0;
                } else {
                    double delta = (vel - target) / (double)(dec_frames > 0 ? dec_frames : 1);
                    double candidate = val - delta;
                    if (candidate < target) candidate = target;
                    val = candidate;
                    tim += 1.0;
                    if (tim >= dec_frames) {
                        val = target;
                        st = 4;
                        tim = 0.0;
                    }
                }
            } else if (st == 4) {
                val = vel * sustain_level;
                if (sus_frames > 0) {
                    tim += 1.0;
                    if (tim >= sus_frames) {
                        st = 5;
                        rel_start = val;
                        tim = 0.0;
                    }
                } else if (!gate_on && !drone_on) {
                    st = 5;
                    rel_start = val;
                    tim = 0.0;
                }
            } else if (st == 5) {
                if (rel_frames <= 0) {
                    val = 0.0;
                    st = 0;
                    tim = 0.0;
                } else {
                    double step = rel_start / (double)(rel_frames > 0 ? rel_frames : 1);
                    double candidate = val - step;
                    if (candidate < 0.0) candidate = 0.0;
                    val = candidate;
                    tim += 1.0;
                    if (tim >= rel_frames) {
                        val = 0.0;
                        st = 0;
                        tim = 0.0;
                    }
                }
                if (gate_on || drone_on) {
                    envelope_start_attack(i, velocity + base, send_resets, reset_out != NULL ? reset_out + base : NULL, &st, &tim, &val, &vel, &rel_start, &acts);
                }
            }

            if (val < 0.0) val = 0.0;
            amp_out[idx] = val;
        }
        stage[b] = st;
        value[b] = val;
        timer[b] = tim;
        vel_state[b] = vel;
        activations[b] = acts;
        release_start[b] = rel_start;
    }
}

void envelope_process(
    const double* trigger,
    const double* gate,
    const double* drone,
    const double* velocity,
    int B,
    int F,
    int atk_frames,
    int hold_frames,
    int dec_frames,
    int sus_frames,
    int rel_frames,
    double sustain_level,
    int send_resets,
    int* stage,
    double* value,
    double* timer,
    double* vel_state,
    int64_t* activations,
    double* release_start,
    double* amp_out,
    double* reset_out
) {
    if (reset_out != NULL) {
        size_t total = (size_t)B * (size_t)F;
        memset(reset_out, 0, total * sizeof(double));
    }
    if (B <= 0 || F <= 0) {
        return;
    }

    if (!envelope_reserve_scratch(F)) {
        envelope_process_simple(
            trigger,
            gate,
            drone,
            velocity,
            B,
            F,
            atk_frames,
            hold_frames,
            dec_frames,
            sus_frames,
            rel_frames,
            sustain_level,
            send_resets,
            stage,
            value,
            timer,
            vel_state,
            activations,
            release_start,
            amp_out,
            reset_out
        );
        return;
    }

    int* boundaries = envelope_scratch.boundaries;
    int* trig_indices = envelope_scratch.trig_indices;
    int8_t* gate_bool = envelope_scratch.gate_bool;
    int8_t* drone_bool = envelope_scratch.drone_bool;

    for (int b = 0; b < B; ++b) {
        int st = stage[b];
        double val = value[b];
        double tim = timer[b];
        double vel = vel_state[b];
        int64_t acts = activations[b];
        double rel_start = release_start[b];

        const double* trig_line = trigger + b * F;
        const double* gate_line = gate + b * F;
        const double* drone_line = drone + b * F;
        const double* vel_line = velocity + b * F;
        double* amp_line = amp_out + b * F;
        double* reset_line = reset_out != NULL ? reset_out + b * F : NULL;

        int trig_count = 0;
        for (int i = 0; i < F; ++i) {
            if (trig_line[i] > 0.5) {
                trig_indices[trig_count++] = i;
            }
            gate_bool[i] = gate_line[i] > 0.5 ? 1 : 0;
            drone_bool[i] = drone_line[i] > 0.5 ? 1 : 0;
        }

        int boundary_count = 0;
        boundaries[boundary_count++] = 0;
        boundaries[boundary_count++] = F;
        for (int i = 0; i < trig_count; ++i) {
            boundaries[boundary_count++] = trig_indices[i];
        }
        for (int i = 1; i < F; ++i) {
            if (gate_bool[i] != gate_bool[i - 1]) {
                boundaries[boundary_count++] = i;
            }
            if (drone_bool[i] != drone_bool[i - 1]) {
                boundaries[boundary_count++] = i;
            }
        }

        for (int i = 1; i < boundary_count; ++i) {
            int key = boundaries[i];
            int j = i - 1;
            while (j >= 0 && boundaries[j] > key) {
                boundaries[j + 1] = boundaries[j];
                --j;
            }
            boundaries[j + 1] = key;
        }

        int unique_count = 0;
        for (int i = 0; i < boundary_count; ++i) {
            int val_b = boundaries[i];
            if (val_b < 0) val_b = 0;
            if (val_b > F) val_b = F;
            if (unique_count == 0 || boundaries[unique_count - 1] != val_b) {
                boundaries[unique_count++] = val_b;
            }
        }
        if (unique_count < 2) {
            boundaries[0] = 0;
            boundaries[1] = F;
            unique_count = 2;
        }

        int trig_ptr = 0;
        for (int seg = 0; seg < unique_count - 1; ++seg) {
            int start = boundaries[seg];
            int stop = boundaries[seg + 1];
            if (start >= F) {
                break;
            }
            if (stop > F) {
                stop = F;
            }
            if (stop <= start) {
                continue;
            }

            while (trig_ptr < trig_count && trig_indices[trig_ptr] == start) {
                envelope_start_attack(
                    start,
                    vel_line,
                    send_resets,
                    reset_line,
                    &st,
                    &tim,
                    &val,
                    &vel,
                    &rel_start,
                    &acts
                );
                trig_ptr++;
            }

            int t = start;
            while (t < stop) {
                int gate_on = (gate_bool[t] != 0) || (drone_bool[t] != 0);

                int changed = 1;
                while (changed) {
                    changed = 0;
                    if (st == 1 && atk_frames <= 0) {
                        val = vel;
                        if (hold_frames > 0) st = 2;
                        else if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                        changed = 1;
                        continue;
                    }
                    if (st == 2 && hold_frames <= 0) {
                        if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                        changed = 1;
                        continue;
                    }
                    if (st == 3 && dec_frames <= 0) {
                        val = vel * sustain_level;
                        st = 4;
                        tim = 0.0;
                        changed = 1;
                        continue;
                    }
                    if (st == 5 && rel_frames <= 0) {
                        val = 0.0;
                        st = 0;
                        tim = 0.0;
                        changed = 1;
                        continue;
                    }
                }

                if (st == 0) {
                    if (gate_on) {
                        envelope_start_attack(
                            t,
                            vel_line,
                            send_resets,
                            reset_line,
                            &st,
                            &tim,
                            &val,
                            &vel,
                            &rel_start,
                            &acts
                        );
                        continue;
                    }
                    int seg_len = stop - t;
                    for (int k = 0; k < seg_len; ++k) {
                        amp_line[t + k] = 0.0;
                    }
                    val = 0.0;
                    tim = 0.0;
                    t = stop;
                    continue;
                }

                if (st == 1) {
                    if (atk_frames <= 0) {
                        continue;
                    }
                    int remaining = atk_frames - (int)tim;
                    if (remaining <= 0) remaining = 1;
                    int seg_len = stop - t;
                    if (seg_len > remaining) seg_len = remaining;
                    if (seg_len <= 0) {
                        t = stop;
                        continue;
                    }
                    double step = vel / (atk_frames > 0 ? (double)atk_frames : 1.0);
                    for (int k = 0; k < seg_len; ++k) {
                        double sample = val + step * (double)(k + 1);
                        if (vel >= 0.0 && sample > vel) sample = vel;
                        if (sample < 0.0) sample = 0.0;
                        amp_line[t + k] = sample;
                    }
                    val = amp_line[t + seg_len - 1];
                    tim += (double)seg_len;
                    if (atk_frames > 0 && tim >= atk_frames) {
                        val = vel;
                        if (hold_frames > 0) st = 2;
                        else if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                    }
                    t += seg_len;
                    continue;
                }

                if (st == 2) {
                    if (hold_frames <= 0) {
                        continue;
                    }
                    int remaining = hold_frames - (int)tim;
                    if (remaining <= 0) remaining = 1;
                    int seg_len = stop - t;
                    if (seg_len > remaining) seg_len = remaining;
                    if (seg_len <= 0) {
                        t = stop;
                        continue;
                    }
                    for (int k = 0; k < seg_len; ++k) {
                        amp_line[t + k] = vel;
                    }
                    val = vel;
                    tim += (double)seg_len;
                    if (tim >= hold_frames) {
                        if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                    }
                    t += seg_len;
                    continue;
                }

                if (st == 3) {
                    if (dec_frames <= 0) {
                        continue;
                    }
                    int remaining = dec_frames - (int)tim;
                    if (remaining <= 0) remaining = 1;
                    int seg_len = stop - t;
                    if (seg_len > remaining) seg_len = remaining;
                    if (seg_len <= 0) {
                        t = stop;
                        continue;
                    }
                    double target = vel * sustain_level;
                    double delta = (vel - target) / (dec_frames > 0 ? (double)dec_frames : 1.0);
                    for (int k = 0; k < seg_len; ++k) {
                        double sample = val - delta * (double)(k + 1);
                        if (sample < target) sample = target;
                        if (sample < 0.0) sample = 0.0;
                        amp_line[t + k] = sample;
                    }
                    val = amp_line[t + seg_len - 1];
                    tim += (double)seg_len;
                    if (tim >= dec_frames) {
                        val = target;
                        st = 4;
                        tim = 0.0;
                    }
                    t += seg_len;
                    continue;
                }

                if (st == 4) {
                    double sustain_val = vel * sustain_level;
                    if (sus_frames > 0) {
                        int remaining = sus_frames - (int)tim;
                        if (remaining <= 0) remaining = 1;
                        int seg_len = stop - t;
                        if (seg_len > remaining) seg_len = remaining;
                        if (seg_len <= 0) {
                            t = stop;
                            continue;
                        }
                        for (int k = 0; k < seg_len; ++k) {
                            amp_line[t + k] = sustain_val;
                        }
                        val = sustain_val;
                        tim += (double)seg_len;
                        if (tim >= sus_frames) {
                            st = 5;
                            rel_start = val;
                            tim = 0.0;
                        }
                        t += seg_len;
                        continue;
                    } else {
                        int seg_len = stop - t;
                        if (!gate_on && seg_len > 1) {
                            seg_len = 1;
                        }
                        if (seg_len <= 0) {
                            seg_len = 1;
                            if (t + seg_len > stop) seg_len = stop - t;
                        }
                        if (seg_len <= 0) {
                            break;
                        }
                        for (int k = 0; k < seg_len; ++k) {
                            amp_line[t + k] = sustain_val;
                        }
                        val = sustain_val;
                        if (!gate_on) {
                            st = 5;
                            rel_start = val;
                            tim = 0.0;
                        } else {
                            tim = 0.0;
                        }
                        t += seg_len;
                        continue;
                    }
                }

                if (st == 5) {
                    if (gate_on) {
                        amp_line[t] = 0.0;
                        envelope_start_attack(
                            t,
                            vel_line,
                            send_resets,
                            reset_line,
                            &st,
                            &tim,
                            &val,
                            &vel,
                            &rel_start,
                            &acts
                        );
                        t += 1;
                        continue;
                    }
                    if (rel_frames <= 0) {
                        continue;
                    }
                    int remaining = rel_frames - (int)tim;
                    if (remaining <= 0) remaining = 1;
                    int seg_len = stop - t;
                    if (seg_len > remaining) seg_len = remaining;
                    if (seg_len <= 0) {
                        seg_len = remaining;
                        if (seg_len <= 0) seg_len = 1;
                        if (t + seg_len > stop) seg_len = stop - t;
                        if (seg_len <= 0) {
                            break;
                        }
                    }
                    double step = rel_start / (rel_frames > 0 ? (double)rel_frames : 1.0);
                    for (int k = 0; k < seg_len; ++k) {
                        double sample = val - step * (double)(k + 1);
                        if (sample < 0.0) sample = 0.0;
                        amp_line[t + k] = sample;
                    }
                    val = amp_line[t + seg_len - 1];
                    tim += (double)seg_len;
                    if (tim >= rel_frames) {
                        val = 0.0;
                        st = 0;
                        tim = 0.0;
                    }
                    t += seg_len;
                    continue;
                }

                // Unknown stage -> silence and exit segment.
                for (int k = t; k < stop; ++k) {
                    amp_line[k] = 0.0;
                }
                val = 0.0;
                tim = 0.0;
                st = 0;
                t = stop;
            }
        }

        if (val < 0.0) val = 0.0;
        stage[b] = st;
        value[b] = val;
        timer[b] = tim;
        vel_state[b] = vel;
        activations[b] = acts;
        release_start[b] = rel_start;
    }
}

// Advance phase per frame with optional reset line. dphi and phase_state are arrays of length B*F and B respectively
void phase_advance(const double* dphi, double* phase_out, int B, int F, double* phase_state, const double* reset) {
    for (int b = 0; b < B; ++b) {
        double cur = 0.0;
        if (phase_state != NULL) cur = phase_state[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            if (reset != NULL && reset[base + i] > 0.5) cur = 0.0;
            cur = cur + dphi[base + i];
            // wrap into [0,1)
            cur = cur - floor(cur);
            phase_out[base + i] = cur;
        }
        if (phase_state != NULL) phase_state[b] = cur;
    }
}

// Portamento smoothing: per-frame smoothing with alpha derived from slide_time and slide_damp
void portamento_smooth(const double* freq_target, const double* port_mask, const double* slide_time, const double* slide_damp, int B, int F, int sr, double* freq_state, double* out) {
    for (int b = 0; b < B; ++b) {
        double cur = 0.0;
        if (freq_state != NULL) cur = freq_state[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            double target = freq_target[base + i];
            int active = port_mask != NULL && port_mask[base + i] > 0.5 ? 1 : 0;
            double frames_const = slide_time != NULL ? slide_time[base + i] * (double)sr : 1.0;
            if (frames_const < 1.0) frames_const = 1.0;
            double alpha = exp(-1.0 / frames_const);
            if (slide_damp != NULL) alpha = pow(alpha, 1.0 + fmax(0.0, slide_damp[base + i]));
            if (active) cur = alpha * cur + (1.0 - alpha) * target;
            else cur = target;
            out[base + i] = cur;
        }
        if (freq_state != NULL) freq_state[b] = cur;
    }
}

// Arp advance: write offsets per frame, update step/timer states
void arp_advance(const double* seq, int seq_len, double* offsets_out, int B, int F, int* step_state, int* timer_state, int fps) {
    for (int b = 0; b < B; ++b) {
        int step = 0;
        int timer = 0;
        if (step_state != NULL) step = step_state[b];
        if (timer_state != NULL) timer = timer_state[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            int idx = step % (seq_len > 0 ? seq_len : 1);
            offsets_out[base + i] = seq[idx];
            timer += 1;
            if (timer >= fps) {
                timer = 0;
                step = (step + 1) % (seq_len > 0 ? seq_len : 1);
            }
        }
        if (step_state != NULL) step_state[b] = step;
        if (timer_state != NULL) timer_state[b] = timer;
    }
}

void polyblep_arr(const double* t, const double* dt, double* out, int N) {
    for (int i = 0; i < N; ++i) {
        out[i] = 0.0;
    }
    for (int i = 0; i < N; ++i) {
        double ti = t[i];
        double dti = dt[i];
        if (ti < dti) {
            double x = ti / (dti > 0.0 ? dti : 1e-20);
            out[i] = x + x - x * x - 1.0;
        } else if (ti > 1.0 - dti) {
            double x = (ti - 1.0) / (dti > 0.0 ? dti : 1e-20);
            out[i] = x * x + x + x + 1.0;
        } else {
            out[i] = 0.0;
        }
    }
}

void osc_saw_blep_c(const double* ph, const double* dphi, double* out, int B, int F) {
    int N = B * F;
    for (int i = 0; i < N; ++i) {
        double t = ph[i];
        double y = 2.0 * t - 1.0;
        double pb = 0.0;
        double dti = dphi[i];
        if (t < dti) {
            double x = t / (dti > 0.0 ? dti : 1e-20);
            pb = x + x - x * x - 1.0;
        } else if (t > 1.0 - dti) {
            double x = (t - 1.0) / (dti > 0.0 ? dti : 1e-20);
            pb = x * x + x + x + 1.0;
        }
        out[i] = y - pb;
    }
}

void osc_square_blep_c(const double* ph, const double* dphi, double pw, double* out, int B, int F) {
    int N = B * F;
    for (int i = 0; i < N; ++i) {
        double t = ph[i];
        double y = (t < pw) ? 1.0 : -1.0;
        // subtract polyblep at rising edge
        double pb1 = 0.0;
        double dti = dphi[i];
        if (t < dti) {
            double x = t / (dti > 0.0 ? dti : 1e-20);
            pb1 = x + x - x * x - 1.0;
        } else if (t > 1.0 - dti) {
            double x = (t - 1.0) / (dti > 0.0 ? dti : 1e-20);
            pb1 = x * x + x + x + 1.0;
        }
        // add polyblep at falling edge (t + (1-pw))%1
        double t2 = t + (1.0 - pw);
        if (t2 >= 1.0) t2 -= 1.0;
        double pb2 = 0.0;
        if (t2 < dti) {
            double x = t2 / (dti > 0.0 ? dti : 1e-20);
            pb2 = x + x - x * x - 1.0;
        } else if (t2 > 1.0 - dti) {
            double x = (t2 - 1.0) / (dti > 0.0 ? dti : 1e-20);
            pb2 = x * x + x + x + 1.0;
        }
        out[i] = y - pb1 + pb2;
    }
}

void osc_triangle_blep_c(const double* ph, const double* dphi, double* out, int B, int F, double* tri_state) {
    // Use square -> leaky integrator per-batch sequence
    for (int b = 0; b < B; ++b) {
        double s = 0.0;
        if (tri_state != NULL) s = tri_state[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            double t = ph[base + i];
            // square
            double y = (t < 0.5) ? 1.0 : -1.0;
            // blep corrections around edges
            double dti = dphi[base + i];
            double pb1 = 0.0;
            if (t < dti) {
                double x = t / (dti > 0.0 ? dti : 1e-20);
                pb1 = x + x - x * x - 1.0;
            } else if (t > 1.0 - dti) {
                double x = (t - 1.0) / (dti > 0.0 ? dti : 1e-20);
                pb1 = x * x + x + x + 1.0;
            }
            double t2 = t + 0.5; if (t2 >= 1.0) t2 -= 1.0;
            double pb2 = 0.0;
            if (t2 < dti) {
                double x = t2 / (dti > 0.0 ? dti : 1e-20);
                pb2 = x + x - x * x - 1.0;
            } else if (t2 > 1.0 - dti) {
                double x = (t2 - 1.0) / (dti > 0.0 ? dti : 1e-20);
                pb2 = x * x + x + x + 1.0;
            }
            double sq = y - pb1 + pb2;
            double leak = 0.9995;
            s = leak * s + (1.0 - leak) * sq;
            out[base + i] = s;
        }
        if (tri_state != NULL) tri_state[b] = s;
    }
}

typedef enum {
    NODE_KIND_UNKNOWN = 0,
    NODE_KIND_CONSTANT,
    NODE_KIND_GAIN,
    NODE_KIND_MIX,
    NODE_KIND_SAFETY,
    NODE_KIND_SINE_OSC,
    NODE_KIND_CONTROLLER,
    NODE_KIND_LFO,
    NODE_KIND_ENVELOPE,
    NODE_KIND_PITCH,
    NODE_KIND_PITCH_SHIFT,
    NODE_KIND_OSC,
    NODE_KIND_OSC_PITCH,
    NODE_KIND_DRIVER,
    NODE_KIND_SUBHARM,
    NODE_KIND_FFT_DIV,
    NODE_KIND_SPECTRAL_DRIVE,
} node_kind_t;

typedef struct {
    node_kind_t kind;
    union {
        struct {
            double value;
            int channels;
        } constant;
        struct {
            int out_channels;
        } mix;
        struct {
            double *state;
            int batches;
            int channels;
            double alpha;
        } safety;
        struct {
            double *phase;
            int batches;
            int channels;
            double base_phase;
        } sine;
        struct {
            double *phase;
            double *phase_buffer;
            double *wave_buffer;
            double *dphi_buffer;
            double *tri_state;
            double *integrator_state;
            double *op_amp_state;
            int batches;
            int channels;
            double base_phase;
            int stereo;
            int driver_channels;
            int mode;
        } osc;
        struct {
            double *phase;
            double *harmonics;
            int harmonic_count;
            int batches;
            int mode;
        } driver;
        struct {
            double *analysis_window;
            double *synthesis_window;
            double *prev_phase;
            double *phase_accum;
            int window_size;
            int hop_size;
            int resynthesis_hop;
        } pitch_shift;
        struct {
            double *slew_state;
            int batches;
            double phase;
        } lfo;
        struct {
            int *stage;
            double *value;
            double *timer;
            double *velocity;
            int64_t *activations;
            double *release_start;
            int batches;
        } envelope;
        struct {
            double *last_freq;
            int batches;
        } pitch;
        struct {
            double *last_value;
            int batches;
        } osc_pitch;
        struct {
            double *hp_y;
            double *lp_y;
            double *prev;
            int8_t *sign;
            int8_t *ff2;
            int8_t *ff4;
            int32_t *ff4_count;
            double *sub2_lp;
            double *sub4_lp;
            double *env;
            double *hp_out_y;
            double *hp_out_x;
            int batches;
            int channels;
            int use_div4;
        } subharm;
        struct {
            double *input_buffer;
            double *divisor_buffer;
            double *divisor_imag_buffer;
            double *phase_buffer;
            double *lower_buffer;
            double *upper_buffer;
            double *filter_buffer;
            double *window;
            double *work_real;
            double *work_imag;
            double *div_real;
            double *div_imag;
            double *ifft_real;
            double *ifft_imag;
            double *result_buffer;
            double *div_fft_real;
            double *div_fft_imag;
            double *recomb_buffer;
            int window_size;
            int algorithm;
            int window_kind;
            int filled;
            int position;
            double epsilon;
            int batches;
            int channels;
            int slots;
            int recomb_filled;
            double last_phase;
            double last_lower;
            double last_upper;
            double last_filter;
            double total_heat;
            int dynamic_carrier_band_count;
            double dynamic_carrier_last_sum;
            double *dynamic_phase;
            double *dynamic_step_re;
            double *dynamic_step_im;
            int dynamic_k_active;
            int enable_remainder;
            double remainder_energy;
        } fftdiv;
        struct {
            int mode;
        } spectral_drive;
    } u;
} node_state_t;

typedef enum {
    OSC_MODE_POLYBLEP = 0,
    OSC_MODE_INTEGRATOR = 1,
    OSC_MODE_OP_AMP = 2
} osc_mode_t;

static int json_copy_string(const char *json, size_t json_len, const char *key, char *out, size_t out_len);

static int parse_osc_mode(const char *json, size_t json_len, int default_mode) {
    char buffer[32];
    if (!json_copy_string(json, json_len, "mode", buffer, sizeof(buffer))) {
        return default_mode;
    }
    for (char *p = buffer; *p != '\0'; ++p) {
        if (*p >= 'A' && *p <= 'Z') {
            *p = (char)(*p - 'A' + 'a');
        }
    }
    if (strcmp(buffer, "integrator") == 0 || strcmp(buffer, "blep_integrator") == 0) {
        return OSC_MODE_INTEGRATOR;
    }
    if (strcmp(buffer, "op_amp") == 0 || strcmp(buffer, "opamp") == 0 || strcmp(buffer, "slew_opamp") == 0) {
        return OSC_MODE_OP_AMP;
    }
    return default_mode;
}

typedef enum {
    DRIVER_MODE_QUARTZ = 0,
    DRIVER_MODE_PIEZO = 1,
    DRIVER_MODE_CUSTOM = 2
} driver_mode_t;

static int parse_driver_mode(const char *json, size_t json_len, int default_mode) {
    char buffer[32];
    if (!json_copy_string(json, json_len, "mode", buffer, sizeof(buffer))) {
        return default_mode;
    }
    for (char *p = buffer; *p != '\0'; ++p) {
        if (*p >= 'A' && *p <= 'Z') {
            *p = (char)(*p - 'A' + 'a');
        }
    }
    if (strcmp(buffer, "piezo") == 0 || strcmp(buffer, "piezoelectric") == 0) {
        return DRIVER_MODE_PIEZO;
    }
    if (strcmp(buffer, "custom") == 0 || strcmp(buffer, "harmonic") == 0) {
        return DRIVER_MODE_CUSTOM;
    }
    return DRIVER_MODE_QUARTZ;
}

static void fft_state_free_buffers(node_state_t *state);

static void release_node_state(node_state_t *state) {
    if (state == NULL) {
        return;
    }
    if (state->kind == NODE_KIND_SAFETY && state->u.safety.state != NULL) {
        free(state->u.safety.state);
        state->u.safety.state = NULL;
        state->u.safety.batches = 0;
        state->u.safety.channels = 0;
        state->u.safety.alpha = 0.0;
    }
    if (state->kind == NODE_KIND_SINE_OSC && state->u.sine.phase != NULL) {
        free(state->u.sine.phase);
        state->u.sine.phase = NULL;
        state->u.sine.batches = 0;
        state->u.sine.channels = 0;
        state->u.sine.base_phase = 0.0;
    }
    if (state->kind == NODE_KIND_OSC) {
        free(state->u.osc.phase);
        free(state->u.osc.phase_buffer);
        free(state->u.osc.wave_buffer);
        free(state->u.osc.dphi_buffer);
        free(state->u.osc.tri_state);
        free(state->u.osc.integrator_state);
        free(state->u.osc.op_amp_state);
        state->u.osc.phase = NULL;
        state->u.osc.phase_buffer = NULL;
        state->u.osc.wave_buffer = NULL;
        state->u.osc.dphi_buffer = NULL;
        state->u.osc.tri_state = NULL;
        state->u.osc.integrator_state = NULL;
        state->u.osc.op_amp_state = NULL;
        state->u.osc.batches = 0;
        state->u.osc.channels = 0;
        state->u.osc.stereo = 0;
        state->u.osc.driver_channels = 0;
        state->u.osc.mode = OSC_MODE_POLYBLEP;
    }
    if (state->kind == NODE_KIND_DRIVER) {
        free(state->u.driver.phase);
        free(state->u.driver.harmonics);
        state->u.driver.phase = NULL;
        state->u.driver.harmonics = NULL;
        state->u.driver.harmonic_count = 0;
        state->u.driver.batches = 0;
        state->u.driver.mode = DRIVER_MODE_QUARTZ;
    }
    if (state->kind == NODE_KIND_LFO) {
        free(state->u.lfo.slew_state);
        state->u.lfo.slew_state = NULL;
        state->u.lfo.batches = 0;
        state->u.lfo.phase = 0.0;
    }
    if (state->kind == NODE_KIND_ENVELOPE) {
        free(state->u.envelope.stage);
        free(state->u.envelope.value);
        free(state->u.envelope.timer);
        free(state->u.envelope.velocity);
        free(state->u.envelope.activations);
        free(state->u.envelope.release_start);
        state->u.envelope.stage = NULL;
        state->u.envelope.value = NULL;
        state->u.envelope.timer = NULL;
        state->u.envelope.velocity = NULL;
        state->u.envelope.activations = NULL;
        state->u.envelope.release_start = NULL;
        state->u.envelope.batches = 0;
    }
    if (state->kind == NODE_KIND_PITCH) {
        free(state->u.pitch.last_freq);
        state->u.pitch.last_freq = NULL;
        state->u.pitch.batches = 0;
    }
    if (state->kind == NODE_KIND_PITCH_SHIFT) {
        free(state->u.pitch_shift.analysis_window);
        free(state->u.pitch_shift.synthesis_window);
        free(state->u.pitch_shift.prev_phase);
        free(state->u.pitch_shift.phase_accum);
        state->u.pitch_shift.analysis_window = NULL;
        state->u.pitch_shift.synthesis_window = NULL;
        state->u.pitch_shift.prev_phase = NULL;
        state->u.pitch_shift.phase_accum = NULL;
        state->u.pitch_shift.window_size = 0;
        state->u.pitch_shift.hop_size = 0;
        state->u.pitch_shift.resynthesis_hop = 0;
    }
    if (state->kind == NODE_KIND_OSC_PITCH) {
        free(state->u.osc_pitch.last_value);
        state->u.osc_pitch.last_value = NULL;
        state->u.osc_pitch.batches = 0;
    }
    if (state->kind == NODE_KIND_SUBHARM) {
        free(state->u.subharm.hp_y);
        free(state->u.subharm.lp_y);
        free(state->u.subharm.prev);
        free(state->u.subharm.sign);
        free(state->u.subharm.ff2);
        free(state->u.subharm.ff4);
        free(state->u.subharm.ff4_count);
        free(state->u.subharm.sub2_lp);
        free(state->u.subharm.sub4_lp);
        free(state->u.subharm.env);
        free(state->u.subharm.hp_out_y);
        free(state->u.subharm.hp_out_x);
        state->u.subharm.hp_y = NULL;
        state->u.subharm.lp_y = NULL;
        state->u.subharm.prev = NULL;
        state->u.subharm.sign = NULL;
        state->u.subharm.ff2 = NULL;
        state->u.subharm.ff4 = NULL;
        state->u.subharm.ff4_count = NULL;
        state->u.subharm.sub2_lp = NULL;
        state->u.subharm.sub4_lp = NULL;
        state->u.subharm.env = NULL;
        state->u.subharm.hp_out_y = NULL;
        state->u.subharm.hp_out_x = NULL;
        state->u.subharm.batches = 0;
        state->u.subharm.channels = 0;
        state->u.subharm.use_div4 = 0;
    }
    if (state->kind == NODE_KIND_FFT_DIV) {
        fft_state_free_buffers(state);
        state->u.fftdiv.total_heat = 0.0;
    }
    free(state);
}

static node_kind_t determine_node_kind(const EdgeRunnerNodeDescriptor *descriptor) {
    if (descriptor == NULL || descriptor->type_name == NULL) {
        return NODE_KIND_UNKNOWN;
    }
    if (strcmp(descriptor->type_name, "ConstantNode") == 0) {
        return NODE_KIND_CONSTANT;
    }
    if (strcmp(descriptor->type_name, "GainNode") == 0) {
        return NODE_KIND_GAIN;
    }
    if (strcmp(descriptor->type_name, "MixNode") == 0) {
        return NODE_KIND_MIX;
    }
    if (strcmp(descriptor->type_name, "SafetyNode") == 0) {
        return NODE_KIND_SAFETY;
    }
    if (strcmp(descriptor->type_name, "SineOscillatorNode") == 0) {
        return NODE_KIND_SINE_OSC;
    }
    if (strcmp(descriptor->type_name, "ControllerNode") == 0) {
        return NODE_KIND_CONTROLLER;
    }
    if (strcmp(descriptor->type_name, "LFONode") == 0) {
        return NODE_KIND_LFO;
    }
    if (strcmp(descriptor->type_name, "EnvelopeModulatorNode") == 0) {
        return NODE_KIND_ENVELOPE;
    }
    if (strcmp(descriptor->type_name, "PitchQuantizerNode") == 0) {
        return NODE_KIND_PITCH;
    }
    if (strcmp(descriptor->type_name, "PitchShiftNode") == 0
        || strcmp(descriptor->type_name, "pitch_shift") == 0) {
        return NODE_KIND_PITCH_SHIFT;
    }
    if (strcmp(descriptor->type_name, "OscillatorPitchNode") == 0
        || strcmp(descriptor->type_name, "oscillator_pitch") == 0) {
        return NODE_KIND_OSC_PITCH;
    }
    if (strcmp(descriptor->type_name, "OscNode") == 0) {
        return NODE_KIND_OSC;
    }
    if (strcmp(descriptor->type_name, "ParametricDriverNode") == 0
        || strcmp(descriptor->type_name, "parametric_driver") == 0) {
        return NODE_KIND_DRIVER;
    }
    if (strcmp(descriptor->type_name, "SubharmonicLowLifterNode") == 0) {
        return NODE_KIND_SUBHARM;
    }
    if (strcmp(descriptor->type_name, "FFTDivisionNode") == 0) {
        return NODE_KIND_FFT_DIV;
    }
    if (strcmp(descriptor->type_name, "SpectralDriveNode") == 0
        || strcmp(descriptor->type_name, "spectral_drive") == 0) {
        return NODE_KIND_SPECTRAL_DRIVE;
    }
    return NODE_KIND_UNKNOWN;
}

static double json_get_double(const char *json, size_t json_len, const char *key, double default_value) {
    (void)json_len;
    if (json == NULL || key == NULL) {
        return default_value;
    }
    size_t key_len = strlen(key);
    if (key_len == 0) {
        return default_value;
    }
    char pattern[128];
    if (key_len + 3 >= sizeof(pattern)) {
        return default_value;
    }
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = json;
    size_t pattern_len = strlen(pattern);
    while ((cursor = strstr(cursor, pattern)) != NULL) {
        cursor += pattern_len;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
            cursor++;
        }
        if (*cursor != ':') {
            continue;
        }
        cursor++;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }
        char *endptr = NULL;
        double value = strtod(cursor, &endptr);
        if (endptr == cursor) {
            cursor = endptr;
            continue;
        }
        return value;
    }
    return default_value;
}

static int json_get_int(const char *json, size_t json_len, const char *key, int default_value) {
    double value = json_get_double(json, json_len, key, (double)default_value);
    if (value >= 0.0) {
        return (int)(value + 0.5);
    }
    return (int)(value - 0.5);
}

static int json_get_bool(const char *json, size_t json_len, const char *key, int default_value) {
    double value = json_get_double(json, json_len, key, default_value ? 1.0 : 0.0);
    return value >= 0.5 ? 1 : 0;
}

static int json_copy_string(const char *json, size_t json_len, const char *key, char *out, size_t out_len) {
    (void)json_len;
    if (out == NULL || out_len == 0) {
        return 0;
    }
    /* Register destination buffer so mem-op logging can correlate writes. */
    register_alloc(out, out_len);
    out[0] = '\0';
    if (json == NULL || key == NULL) {
        return 0;
    }
    size_t key_len = strlen(key);
    if (key_len == 0) {
        return 0;
    }
    char pattern[128];
    if (key_len + 3 >= sizeof(pattern)) {
        return 0;
    }
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = json;
    size_t pattern_len = strlen(pattern);
    while ((cursor = strstr(cursor, pattern)) != NULL) {
        cursor += pattern_len;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
            cursor++;
        }
        if (*cursor != ':') {
            continue;
        }
        cursor++;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
            cursor++;
        }
        if (*cursor != '"') {
            continue;
        }
        cursor++;
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != '"') {
            cursor++;
        }
        size_t length = (size_t)(cursor - start);
        if (length >= out_len) {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            /* log attempted oversize copy */
            ensure_log_files_open();
            if (log_f_memops) {
                void *stack_probe = (void *)&pattern;
                /* log the destination and a stack probe so we can detect stack-derived buffers */
                fprintf(log_f_memops, "PRECOPY json_copy_string base=%p dest=%p dest_cap=%zu req_len=%zu stack=%p\n", out, out, out_len, length, stack_probe);
            }
#endif
            length = out_len > 0 ? out_len - 1 : 0;
        } else {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                void *stack_probe = (void *)&pattern;
                fprintf(log_f_memops, "PRECOPY json_copy_string base=%p dest=%p dest_cap=%zu req_len=%zu stack=%p\n", out, out, out_len, length, stack_probe);
            }
#endif
        }
        /* use safe copy that respects the provided destination capacity */
        if (out_len > 0 && length > 0) {
            memcpy(out, start, length);
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            /* POSTCOPY: record that we actually wrote to 'out' */
            ensure_log_files_open();
            if (log_f_memops) {
                fprintf(log_f_memops, "POSTCOPY json_copy_string dest=%p wrote=%zu\n", out, length);
            }
#endif
        }
        out[length] = '\0';
        unregister_alloc(out);
        return 1;
    }
    unregister_alloc(out);
    return 0;
}

static const EdgeRunnerParamView *find_param(const EdgeRunnerNodeInputs *inputs, const char *name) {
    if (inputs == NULL || name == NULL) {
        return NULL;
    }
    uint32_t count = inputs->params.count;
    EdgeRunnerParamView *items = inputs->params.items;
    for (uint32_t i = 0; i < count; ++i) {
        const EdgeRunnerParamView *view = &items[i];
        if (view->name != NULL && strcmp(view->name, name) == 0) {
            return view;
        }
    }
    return NULL;
}

typedef struct {
    char output[64];
    char source[64];
} controller_source_t;

static int parse_csv_tokens(const char *csv, char tokens[][64], int max_tokens) {
    if (csv == NULL || tokens == NULL || max_tokens <= 0) {
        return 0;
    }
    /* Register tokens buffer for correlation of PRECOPY/MEMCPY events */
    register_alloc(tokens, (size_t)max_tokens * 64);
    int count = 0;
    const char *cursor = csv;
    while (*cursor != '\0' && count < max_tokens) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == ',') {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != ',') {
            cursor++;
        }
        size_t len = (size_t)(cursor - start);
        if (len >= 63) {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                void *base = (void *)tokens;
                /* log both the tokens base and per-token dest so we can map to allocations */
                fprintf(log_f_memops, "PRECOPY parse_csv_tokens base=%p dest=%p dest_cap=%d req_len=%zu\n", base, tokens[count], 64, len);
            }
#endif
            len = 63;
        } else {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                void *base = (void *)tokens;
                fprintf(log_f_memops, "PRECOPY parse_csv_tokens base=%p dest=%p dest_cap=%d req_len=%zu\n", base, tokens[count], 64, len);
            }
#endif
        }
        if (len > 0) {
            /* write with postcopy logging and bounds assertion */
            memcpy(tokens[count], start, len);
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                fprintf(log_f_memops, "POSTCOPY parse_csv_tokens dest=%p wrote=%zu token_idx=%d\n", tokens[count], len, count);
            }
#endif
        }
        tokens[count][len] = '\0';
        count++;
    }
    unregister_alloc(tokens);
    return count;
}

static int parse_controller_sources(const char *csv, controller_source_t *items, int max_items) {
    if (csv == NULL || items == NULL || max_items <= 0) {
        return 0;
    }
    /* Register items buffer so writes to items->output/source are tracked */
    register_alloc(items, (size_t)max_items * sizeof(controller_source_t));
    int count = 0;
    const char *cursor = csv;
    while (*cursor != '\0' && count < max_items) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == ',') {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }
        const char *eq = strchr(cursor, '=');
        if (eq == NULL) {
            break;
        }
        size_t key_len = (size_t)(eq - cursor);
        if (key_len >= sizeof(items[count].output)) {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                void *base = (void *)items;
                fprintf(log_f_memops, "PRECOPY parse_controller_sources.output base=%p dest=%p dest_cap=%zu req_len=%zu\n", base, items[count].output, (size_t)sizeof(items[count].output), key_len);
            }
#endif
            key_len = sizeof(items[count].output) - 1;
        } else {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                void *base = (void *)items;
                fprintf(log_f_memops, "PRECOPY parse_controller_sources.output base=%p dest=%p dest_cap=%zu req_len=%zu\n", base, items[count].output, (size_t)sizeof(items[count].output), key_len);
            }
#endif
        }
        if (key_len > 0) {
            memcpy(items[count].output, cursor, key_len);
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                fprintf(log_f_memops, "POSTCOPY parse_controller_sources.output dest=%p wrote=%zu idx=%d\n", items[count].output, key_len, count);
            }
#endif
        }
        items[count].output[key_len] = '\0';
        cursor = eq + 1;
        const char *end = strchr(cursor, ',');
        if (end == NULL) {
            end = cursor + strlen(cursor);
        }
        size_t value_len = (size_t)(end - cursor);
        if (value_len >= sizeof(items[count].source)) {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                void *base = (void *)items;
                fprintf(log_f_memops, "PRECOPY parse_controller_sources.source base=%p dest=%p dest_cap=%zu req_len=%zu\n", base, items[count].source, (size_t)sizeof(items[count].source), value_len);
            }
#endif
            value_len = sizeof(items[count].source) - 1;
        } else {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                void *base = (void *)items;
                fprintf(log_f_memops, "PRECOPY parse_controller_sources.source base=%p dest=%p dest_cap=%zu req_len=%zu\n", base, items[count].source, (size_t)sizeof(items[count].source), value_len);
            }
#endif
        }
        if (value_len > 0) {
            memcpy(items[count].source, cursor, value_len);
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            ensure_log_files_open();
            if (log_f_memops) {
                fprintf(log_f_memops, "POSTCOPY parse_controller_sources.source dest=%p wrote=%zu idx=%d\n", items[count].source, value_len, count);
            }
#endif
        }
        items[count].source[value_len] = '\0';
        cursor = end;
        count++;
    }
    unregister_alloc(items);
    return count;
}

static int parse_csv_doubles(const char *csv, double *values, int max_values) {
    if (csv == NULL || values == NULL || max_values <= 0) {
        return 0;
    }
    int count = 0;
    const char *cursor = csv;
    while (*cursor != '\0' && count < max_values) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == ',') {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }
        char *endptr = NULL;
        double value = strtod(cursor, &endptr);
        if (endptr == cursor) {
            break;
        }
        values[count++] = value;
        cursor = endptr;
    }
    return count;
}

static const double *ensure_param_plane(
    const EdgeRunnerParamView *view,
    int batches,
    int frames,
    double default_value,
    double **owned_out
) {
    if (owned_out != NULL) {
        *owned_out = NULL;
    }
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    size_t total = (size_t)batches * (size_t)frames;
    if (view == NULL || view->data == NULL) {
        if (owned_out == NULL) {
            return NULL;
        }
        double *buf = (double *)malloc(total * sizeof(double));
        if (buf == NULL) {
            return NULL;
        }
        for (size_t i = 0; i < total; ++i) {
            buf[i] = default_value;
        }
        *owned_out = buf;
        return buf;
    }
    int vb = view->batches > 0 ? (int)view->batches : batches;
    int vc = view->channels > 0 ? (int)view->channels : 1;
    int vf = view->frames > 0 ? (int)view->frames : frames;
    if (vb == batches && vc == 1 && vf == frames) {
        return view->data;
    }
    if (owned_out == NULL) {
        return NULL;
    }
    double *buf = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    amp_last_alloc_count = total;
    if (buf == NULL) {
        return NULL;
    }
    for (int b = 0; b < batches; ++b) {
        for (int f = 0; f < frames; ++f) {
            size_t idx = (size_t)b * (size_t)frames + (size_t)f;
            double value = default_value;
            if (b < vb && f < vf) {
                size_t src_idx = ((size_t)b * (size_t)vc) * (size_t)vf + (size_t)f;
                if (vc > 0) {
                    src_idx = ((size_t)b * (size_t)vc + 0) * (size_t)vf + (size_t)f;
                }
                size_t span = (size_t)vb * (size_t)vc * (size_t)vf;
                if (src_idx < span) {
                    value = view->data[src_idx];
                }
            }
            buf[idx] = value;
        }
    }
    *owned_out = buf;
    return buf;
}

static double read_scalar_param(const EdgeRunnerParamView *view, double default_value) {
    if (view == NULL || view->data == NULL) {
        return default_value;
    }
    size_t total = (size_t)(view->batches ? view->batches : 1)
        * (size_t)(view->channels ? view->channels : 1)
        * (size_t)(view->frames ? view->frames : 1);
    if (total == 0) {
        return default_value;
    }
    return view->data[total - 1];
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) {
        return -1;
    }
    if (da > db) {
        return 1;
    }
    return 0;
}

static int build_sorted_grid(const double *values, int count, double *sorted, double *ext) {
    if (values == NULL || sorted == NULL || ext == NULL || count <= 0) {
        return 0;
    }
    int n = count;
    if (n < 2) {
        n = 12;
        for (int i = 0; i < n; ++i) {
            sorted[i] = (double)i * 100.0;
        }
    } else {
        memcpy(sorted, values, (size_t)n * sizeof(double));
        qsort(sorted, (size_t)n, sizeof(double), compare_double);
    }
    for (int i = 0; i < n; ++i) {
        ext[i] = sorted[i];
    }
    ext[n] = sorted[0] + 1200.0;
    return n;
}

static double grid_warp_forward_value(double cents, const double *grid, const double *grid_ext, int N) {
    double octs = floor(cents / 1200.0);
    double c_mod = fmod(cents, 1200.0);
    if (c_mod < 0.0) {
        c_mod += 1200.0;
    }
    int idx = 0;
    for (int i = 0; i < N; ++i) {
        if (c_mod >= grid_ext[i] && c_mod < grid_ext[i + 1]) {
            idx = i;
            break;
        }
        if (i == N - 1) {
            idx = N - 1;
        }
    }
    double lower = grid_ext[idx];
    double upper = grid_ext[idx + 1];
    double denom = upper - lower;
    if (fabs(denom) < 1e-9) {
        denom = 1e-9;
    }
    double t = (c_mod - lower) / denom;
    double u_mod = (double)idx + t;
    return octs * (double)N + u_mod;
}

static double grid_warp_inverse_value(double u, const double *grid, const double *grid_ext, int N) {
    double octs = floor(u / (double)N);
    double u_mod = u - octs * (double)N;
    int idx = (int)floor(u_mod);
    if (idx < 0) {
        idx = 0;
    }
    if (idx >= N) {
        idx = N - 1;
    }
    double frac = u_mod - (double)idx;
    double lower = grid_ext[idx];
    double upper = grid_ext[idx + 1];
    double cents = lower + frac * (upper - lower);
    return octs * 1200.0 + cents;
}

#define FFT_ALGORITHM_EIGEN 0
#define FFT_ALGORITHM_DFT 1
#define FFT_ALGORITHM_DYNAMIC_OSCILLATORS 2
#define FFT_ALGORITHM_HOOK 3

#define PITCH_SHIFT_DEFAULT_WINDOW 1024
#define PITCH_SHIFT_DEFAULT_HOP 256
#define PITCH_SHIFT_DEFAULT_RESYNTH_HOP 256

#define FFT_WINDOW_RECTANGULAR 0
#define FFT_WINDOW_HANN 1
#define FFT_WINDOW_HAMMING 2

#define FFT_DYNAMIC_CARRIER_LIMIT 64U

static int round_to_int(double value) {
    if (value >= 0.0) {
        return (int)(value + 0.5);
    }
    return (int)(value - 0.5);
}

static int clamp_algorithm_kind(int kind) {
    switch (kind) {
        case FFT_ALGORITHM_EIGEN:
        case FFT_ALGORITHM_DFT:
        case FFT_ALGORITHM_DYNAMIC_OSCILLATORS:
        case FFT_ALGORITHM_HOOK:
            return kind;
        default:
            break;
    }
    return FFT_ALGORITHM_EIGEN;
}

static int clamp_window_kind(int kind) {
    switch (kind) {
        case FFT_WINDOW_RECTANGULAR:
        case FFT_WINDOW_HANN:
        case FFT_WINDOW_HAMMING:
            return kind;
        default:
            break;
    }
    return FFT_WINDOW_HANN;
}

static double clamp_unit_double(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static double wrap_phase_two_pi(double phase) {
    double wrapped = fmod(phase, 2.0 * M_PI);
    if (wrapped < 0.0) {
        wrapped += 2.0 * M_PI;
    }
    return wrapped;
}

static double compute_band_gain(double ratio, double lower, double upper, double intensity) {
    /* The minimum inside/outside gain is clamped to 1e-6 to avoid hard
       muting. Documented here so callers know we intentionally leave a floor
       for numerical stability. */
    double lower_clamped = clamp_unit_double(lower);
    double upper_clamped = clamp_unit_double(upper);
    if (upper_clamped < lower_clamped) {
        double tmp = lower_clamped;
        lower_clamped = upper_clamped;
        upper_clamped = tmp;
    }
    double intensity_clamped = clamp_unit_double(intensity);
    double inside_gain = intensity_clamped;
    if (inside_gain < 1e-6) {
        inside_gain = 1e-6;
    }
    double outside_gain = 1.0 - intensity_clamped;
    if (outside_gain < 1e-6) {
        outside_gain = 1e-6;
    }
    if (ratio >= lower_clamped && ratio <= upper_clamped) {
        return inside_gain;
    }
    return outside_gain;
}

static int solve_linear_system(double *matrix, double *rhs, int dim) {
    if (matrix == NULL || rhs == NULL || dim <= 0) {
        return -1;
    }
    for (int col = 0; col < dim; ++col) {
        int pivot = col;
        double max_val = fabs(matrix[col * dim + col]);
        for (int row = col + 1; row < dim; ++row) {
            double candidate = fabs(matrix[row * dim + col]);
            if (candidate > max_val) {
                max_val = candidate;
                pivot = row;
            }
        }
        if (max_val < 1e-18) {
            return -1;
        }
        if (pivot != col) {
            for (int k = col; k < dim; ++k) {
                double tmp = matrix[col * dim + k];
                matrix[col * dim + k] = matrix[pivot * dim + k];
                matrix[pivot * dim + k] = tmp;
            }
            double rhs_tmp = rhs[col];
            rhs[col] = rhs[pivot];
            rhs[pivot] = rhs_tmp;
        }
        double diag = matrix[col * dim + col];
        for (int row = col + 1; row < dim; ++row) {
            double factor = matrix[row * dim + col] / diag;
            matrix[row * dim + col] = 0.0;
            for (int k = col + 1; k < dim; ++k) {
                matrix[row * dim + k] -= factor * matrix[col * dim + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for (int row = dim - 1; row >= 0; --row) {
        double accum = rhs[row];
        for (int k = row + 1; k < dim; ++k) {
            accum -= matrix[row * dim + k] * rhs[k];
        }
        double diag = matrix[row * dim + row];
        if (fabs(diag) < 1e-18) {
            return -1;
        }
        rhs[row] = accum / diag;
    }
    return 0;
}

static size_t param_total_count(const EdgeRunnerParamView *view) {
    if (view == NULL) {
        return 0;
    }
    size_t batches = view->batches > 0U ? view->batches : 1U;
    size_t channels = view->channels > 0U ? view->channels : 1U;
    size_t frames = view->frames > 0U ? view->frames : 1U;
    return batches * channels * frames;
}

static double read_param_value(const EdgeRunnerParamView *view, size_t index, double default_value) {
    if (view == NULL || view->data == NULL) {
        return default_value;
    }
    size_t total = param_total_count(view);
    if (total == 0) {
        return default_value;
    }
    if (index >= total) {
        index = total - 1;
    }
    return view->data[index];
}

typedef struct {
    uint32_t band_count;
    double last_sum;
} fft_dynamic_carrier_summary_t;

static int parse_dynamic_carrier_index(const char *name, uint32_t *index_out) {
    if (name == NULL || index_out == NULL) {
        return 0;
    }
    const char *prefix = "carrier_band";
    size_t prefix_len = strlen(prefix);
    if (strncmp(name, prefix, prefix_len) != 0) {
        return 0;
    }
    const char *cursor = name + prefix_len;
    if (*cursor == '_' || *cursor == '-') {
        cursor++;
    }
    if (*cursor == '\0') {
        return 0;
    }
    for (const char *probe = cursor; *probe != '\0'; ++probe) {
        if (!isdigit((unsigned char)*probe)) {
            return 0;
        }
    }
    unsigned long parsed = strtoul(cursor, NULL, 10);
    if (parsed >= FFT_DYNAMIC_CARRIER_LIMIT) {
        return 0;
    }
    *index_out = (uint32_t)parsed;
    return 1;
}

static fft_dynamic_carrier_summary_t summarize_dynamic_carriers(const EdgeRunnerNodeInputs *inputs) {
    fft_dynamic_carrier_summary_t summary;
    summary.band_count = 0U;
    summary.last_sum = 0.0;
    if (inputs == NULL) {
        return summary;
    }
    uint32_t param_count = inputs->params.count;
    EdgeRunnerParamView *items = inputs->params.items;
    for (uint32_t i = 0; i < param_count; ++i) {
        const EdgeRunnerParamView *view = &items[i];
        uint32_t index = 0U;
        if (!parse_dynamic_carrier_index(view->name, &index)) {
            continue;
        }
        if (index + 1U > summary.band_count) {
            summary.band_count = index + 1U;
        }
        size_t total = param_total_count(view);
        if (total > 0U) {
            double last_value = read_param_value(view, total - 1U, 0.0);
            summary.last_sum += last_value;
        }
    }
    return summary;
}

static uint32_t collect_dynamic_carrier_views(
    const EdgeRunnerNodeInputs *inputs,
    const EdgeRunnerParamView **views,
    uint32_t limit
) {
    if (views == NULL || limit == 0U) {
        return 0U;
    }
    for (uint32_t i = 0; i < limit; ++i) {
        views[i] = NULL;
    }
    if (inputs == NULL) {
        return 0U;
    }
    uint32_t max_index = 0U;
    uint32_t param_count = inputs->params.count;
    EdgeRunnerParamView *items = inputs->params.items;
    for (uint32_t i = 0; i < param_count; ++i) {
        EdgeRunnerParamView *view = &items[i];
        uint32_t index = 0U;
        if (!parse_dynamic_carrier_index(view->name, &index)) {
            continue;
        }
        if (index >= limit) {
            continue;
        }
        views[index] = view;
        if (index + 1U > max_index) {
            max_index = index + 1U;
        }
    }
    return max_index;
}

static int parse_algorithm_string(const char *json, size_t json_len, int default_value) {
    char buffer[32];
    if (!json_copy_string(json, json_len, "algorithm", buffer, sizeof(buffer))) {
        return default_value;
    }
    for (size_t i = 0; buffer[i] != '\0'; ++i) {
        buffer[i] = (char)tolower((unsigned char)buffer[i]);
    }
    if (
        strcmp(buffer, "fft") == 0
        || strcmp(buffer, "eigen") == 0
        || strcmp(buffer, "radix2") == 0
        || strcmp(buffer, "cooleytukey") == 0
        || strcmp(buffer, "nufft") == 0
        || strcmp(buffer, "nonuniform") == 0
        || strcmp(buffer, "czt") == 0
        || strcmp(buffer, "chirpz") == 0
        || strcmp(buffer, "chirpzt") == 0
    ) {
        return FFT_ALGORITHM_EIGEN;
    }
    if (strcmp(buffer, "hook") == 0 || strcmp(buffer, "custom_fft") == 0) {
        return FFT_ALGORITHM_HOOK;
    }
    if (strcmp(buffer, "dft") == 0 || strcmp(buffer, "direct") == 0 || strcmp(buffer, "slow") == 0) {
        return FFT_ALGORITHM_DFT;
    }
    if (
        strcmp(buffer, "dynamic") == 0
        || strcmp(buffer, "dynamic_oscillators") == 0
        || strcmp(buffer, "dynamicoscillators") == 0
    ) {
        return FFT_ALGORITHM_DYNAMIC_OSCILLATORS;
    }
    return default_value;
}

static int parse_window_string(const char *json, size_t json_len, int default_value) {
    char buffer[32];
    if (!json_copy_string(json, json_len, "window", buffer, sizeof(buffer))) {
        return default_value;
    }
    for (size_t i = 0; buffer[i] != '\0'; ++i) {
        buffer[i] = (char)tolower((unsigned char)buffer[i]);
    }
    if (strcmp(buffer, "rect") == 0 || strcmp(buffer, "rectangular") == 0) {
        return FFT_WINDOW_RECTANGULAR;
    }
    if (strcmp(buffer, "hann") == 0 || strcmp(buffer, "hanning") == 0) {
        return FFT_WINDOW_HANN;
    }
    if (strcmp(buffer, "hamming") == 0) {
        return FFT_WINDOW_HAMMING;
    }
    return default_value;
}

static int is_power_of_two_int(int value) {
    if (value <= 0) {
        return 0;
    }
    return (value & (value - 1)) == 0;
}

static void fft_backend_transform(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
) {
    amp_fft_backend_transform(in_real, in_imag, out_real, out_imag, n, inverse);
}

static void compute_dft(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
) {
    if (n <= 0 || out_real == NULL || out_imag == NULL) {
        return;
    }
    double sign = inverse != 0 ? 1.0 : -1.0;
    for (int k = 0; k < n; ++k) {
        double sum_real = 0.0;
        double sum_imag = 0.0;
        for (int t = 0; t < n; ++t) {
            double real = in_real != NULL ? in_real[t] : 0.0;
            double imag = in_imag != NULL ? in_imag[t] : 0.0;
            double angle = sign * 2.0 * M_PI * (double)k * (double)t / (double)n;
            double c = cos(angle);
            double s = sin(angle);
            sum_real += real * c - imag * s;
            sum_imag += real * s + imag * c;
        }
        if (inverse != 0) {
            sum_real /= (double)n;
            sum_imag /= (double)n;
        }
        out_real[k] = sum_real;
        out_imag[k] = sum_imag;
    }
}

typedef void (*fft_algorithm_impl_fn)(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
);

typedef struct {
    int kind;
    const char *label;
    fft_algorithm_impl_fn forward;
    fft_algorithm_impl_fn inverse;
    int requires_power_of_two;
    int supports_dynamic_carriers;
    int requires_hook;
} fft_algorithm_class_t;

static void fft_algorithm_backend_forward(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    fft_backend_transform(in_real, in_imag, out_real, out_imag, n, 0);
}

static void fft_algorithm_backend_inverse(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    fft_backend_transform(in_real, in_imag, out_real, out_imag, n, 1);
}

static void fft_algorithm_dft_forward(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    compute_dft(in_real, in_imag, out_real, out_imag, n, 0);
}

static void fft_algorithm_dft_inverse(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    compute_dft(in_real, in_imag, out_real, out_imag, n, 1);
}

static void fft_algorithm_dynamic_forward(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    fft_backend_transform(in_real, in_imag, out_real, out_imag, n, 0);
}

static void fft_algorithm_dynamic_inverse(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    fft_backend_transform(in_real, in_imag, out_real, out_imag, n, 1);
}

static const fft_algorithm_class_t FFT_ALGORITHM_CLASSES[] = {
    {
        FFT_ALGORITHM_EIGEN,
        "fft",
        fft_algorithm_backend_forward,
        fft_algorithm_backend_inverse,
        0,
        0,
        0,
    },
    {
        FFT_ALGORITHM_HOOK,
        "hook",
        fft_algorithm_backend_forward,
        fft_algorithm_backend_inverse,
        0,
        0,
        1,
    },
    {
        FFT_ALGORITHM_DFT,
        "dft",
        fft_algorithm_dft_forward,
        fft_algorithm_dft_inverse,
        0,
        0,
        0,
    },
    {
        FFT_ALGORITHM_DYNAMIC_OSCILLATORS,
        "dynamic_oscillators",
        fft_algorithm_dynamic_forward,
        fft_algorithm_dynamic_inverse,
        0,
        1,
        0,
    },
};

static const fft_algorithm_class_t *select_fft_algorithm(int kind) {
    size_t count = sizeof(FFT_ALGORITHM_CLASSES) / sizeof(FFT_ALGORITHM_CLASSES[0]);
    for (size_t i = 0; i < count; ++i) {
        if (FFT_ALGORITHM_CLASSES[i].kind == kind) {
            return &FFT_ALGORITHM_CLASSES[i];
        }
    }
    return NULL;
}

static void fill_window_weights(double *window, int window_size, int window_kind) {
    if (window == NULL || window_size <= 0) {
        return;
    }
    if (window_size == 1) {
        window[0] = 1.0;
        return;
    }
    for (int i = 0; i < window_size; ++i) {
        double value = 1.0;
        double phase = (double)i / (double)(window_size - 1);
        switch (window_kind) {
            case FFT_WINDOW_RECTANGULAR:
                value = 1.0;
                break;
            case FFT_WINDOW_HANN:
                value = 0.5 * (1.0 - cos(2.0 * M_PI * phase));
                break;
            case FFT_WINDOW_HAMMING:
                value = 0.54 - 0.46 * cos(2.0 * M_PI * phase);
                break;
            default:
                value = 1.0;
                break;
        }
        window[i] = value;
    }
}

static void fft_state_free_buffers(node_state_t *state) {
    if (state == NULL) {
        return;
    }
    free(state->u.fftdiv.input_buffer);
    free(state->u.fftdiv.divisor_buffer);
    free(state->u.fftdiv.divisor_imag_buffer);
    free(state->u.fftdiv.phase_buffer);
    free(state->u.fftdiv.lower_buffer);
    free(state->u.fftdiv.upper_buffer);
    free(state->u.fftdiv.filter_buffer);
    free(state->u.fftdiv.window);
    free(state->u.fftdiv.work_real);
    free(state->u.fftdiv.work_imag);
    free(state->u.fftdiv.div_real);
    free(state->u.fftdiv.div_imag);
    free(state->u.fftdiv.ifft_real);
    free(state->u.fftdiv.ifft_imag);
    free(state->u.fftdiv.result_buffer);
    free(state->u.fftdiv.div_fft_real);
    free(state->u.fftdiv.div_fft_imag);
    free(state->u.fftdiv.recomb_buffer);
    free(state->u.fftdiv.dynamic_phase);
    free(state->u.fftdiv.dynamic_step_re);
    free(state->u.fftdiv.dynamic_step_im);
    state->u.fftdiv.input_buffer = NULL;
    state->u.fftdiv.divisor_buffer = NULL;
    state->u.fftdiv.divisor_imag_buffer = NULL;
    state->u.fftdiv.phase_buffer = NULL;
    state->u.fftdiv.lower_buffer = NULL;
    state->u.fftdiv.upper_buffer = NULL;
    state->u.fftdiv.filter_buffer = NULL;
    state->u.fftdiv.window = NULL;
    state->u.fftdiv.work_real = NULL;
    state->u.fftdiv.work_imag = NULL;
    state->u.fftdiv.div_real = NULL;
    state->u.fftdiv.div_imag = NULL;
    state->u.fftdiv.ifft_real = NULL;
    state->u.fftdiv.ifft_imag = NULL;
    state->u.fftdiv.result_buffer = NULL;
    state->u.fftdiv.div_fft_real = NULL;
    state->u.fftdiv.div_fft_imag = NULL;
    state->u.fftdiv.recomb_buffer = NULL;
    state->u.fftdiv.dynamic_phase = NULL;
    state->u.fftdiv.dynamic_step_re = NULL;
    state->u.fftdiv.dynamic_step_im = NULL;
    state->u.fftdiv.window_size = 0;
    state->u.fftdiv.algorithm = FFT_ALGORITHM_EIGEN;
    state->u.fftdiv.window_kind = -1;
    state->u.fftdiv.filled = 0;
    state->u.fftdiv.position = 0;
    state->u.fftdiv.batches = 0;
    state->u.fftdiv.channels = 0;
    state->u.fftdiv.slots = 0;
    state->u.fftdiv.epsilon = 0.0;
    state->u.fftdiv.recomb_filled = 0;
    state->u.fftdiv.last_phase = 0.0;
    state->u.fftdiv.last_lower = 0.0;
    state->u.fftdiv.last_upper = 1.0;
    state->u.fftdiv.last_filter = 1.0;
    state->u.fftdiv.dynamic_carrier_band_count = 0;
    state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
    state->u.fftdiv.dynamic_k_active = 0;
    state->u.fftdiv.enable_remainder = 1;
    state->u.fftdiv.remainder_energy = 0.0;
}

static int ensure_fft_state_buffers(node_state_t *state, int slots, int window_size) {
    if (state == NULL) {
        return -1;
    }
    if (slots <= 0) {
        slots = 1;
    }
    if (window_size <= 0) {
        window_size = 1;
    }
    if (state->u.fftdiv.input_buffer != NULL && state->u.fftdiv.window_size == window_size && state->u.fftdiv.slots == slots) {
        return 0;
    }
    double preserved_heat = state->u.fftdiv.total_heat;
    fft_state_free_buffers(state);
    size_t total = (size_t)window_size * (size_t)slots;
    if (total == 0) {
        state->u.fftdiv.total_heat = preserved_heat;
        return -1;
    }
    state->u.fftdiv.input_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.divisor_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.divisor_imag_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.phase_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.lower_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.upper_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.filter_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.window = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.work_real = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.work_imag = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.div_real = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.div_imag = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.ifft_real = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.ifft_imag = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.result_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.div_fft_real = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.div_fft_imag = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.recomb_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.dynamic_phase = (double *)calloc((size_t)slots * FFT_DYNAMIC_CARRIER_LIMIT, sizeof(double));
    state->u.fftdiv.dynamic_step_re = (double *)calloc((size_t)slots * FFT_DYNAMIC_CARRIER_LIMIT, sizeof(double));
    state->u.fftdiv.dynamic_step_im = (double *)calloc((size_t)slots * FFT_DYNAMIC_CARRIER_LIMIT, sizeof(double));
    if (state->u.fftdiv.input_buffer == NULL || state->u.fftdiv.divisor_buffer == NULL || state->u.fftdiv.divisor_imag_buffer == NULL
        || state->u.fftdiv.phase_buffer == NULL || state->u.fftdiv.lower_buffer == NULL || state->u.fftdiv.upper_buffer == NULL
        || state->u.fftdiv.filter_buffer == NULL || state->u.fftdiv.window == NULL || state->u.fftdiv.work_real == NULL
        || state->u.fftdiv.work_imag == NULL || state->u.fftdiv.div_real == NULL || state->u.fftdiv.div_imag == NULL
        || state->u.fftdiv.ifft_real == NULL || state->u.fftdiv.ifft_imag == NULL || state->u.fftdiv.result_buffer == NULL
        || state->u.fftdiv.div_fft_real == NULL || state->u.fftdiv.div_fft_imag == NULL || state->u.fftdiv.recomb_buffer == NULL
        || state->u.fftdiv.dynamic_phase == NULL || state->u.fftdiv.dynamic_step_re == NULL || state->u.fftdiv.dynamic_step_im == NULL) {
        fft_state_free_buffers(state);
        state->u.fftdiv.total_heat = preserved_heat;
        return -1;
    }
    state->u.fftdiv.window_size = window_size;
    state->u.fftdiv.slots = slots;
    state->u.fftdiv.filled = 0;
    state->u.fftdiv.position = window_size > 0 ? window_size - 1 : 0;
    state->u.fftdiv.algorithm = FFT_ALGORITHM_EIGEN;
    state->u.fftdiv.window_kind = -1;
    state->u.fftdiv.batches = 0;
    state->u.fftdiv.channels = 0;
    state->u.fftdiv.epsilon = 1e-9;
    state->u.fftdiv.recomb_filled = 0;
    state->u.fftdiv.last_phase = 0.0;
    state->u.fftdiv.last_lower = 0.0;
    state->u.fftdiv.last_upper = 1.0;
    state->u.fftdiv.last_filter = 1.0;
    state->u.fftdiv.total_heat = preserved_heat;
    state->u.fftdiv.dynamic_carrier_band_count = 0;
    state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
    state->u.fftdiv.dynamic_k_active = 0;
    state->u.fftdiv.enable_remainder = 1;
    state->u.fftdiv.remainder_energy = 0.0;
    return 0;
}

static int run_constant_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    int channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", 1);
    if (channels <= 0) {
        channels = 1;
    }
    double value = json_get_double(descriptor->params_json, descriptor->params_len, "value", 0.0);
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        return -1;
    }
    for (size_t i = 0; i < total; ++i) {
        buffer[i] = value;
    }
    if (state != NULL) {
        state->u.constant.value = value;
        state->u.constant.channels = channels;
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

static double render_lfo_wave(const char *wave, double phase) {
    if (wave != NULL) {
        if (strcmp(wave, "square") == 0) {
            return phase < 0.5 ? 1.0 : -1.0;
        }
        if (strcmp(wave, "saw") == 0) {
            double t = phase - floor(phase);
            return 2.0 * t - 1.0;
        }
        if (strcmp(wave, "triangle") == 0) {
            double t = phase - floor(phase);
            return 2.0 * fabs(2.0 * t - 1.0) - 1.0;
        }
    }
    return sin(phase * 2.0 * M_PI);
}

static int run_controller_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels,
    const EdgeRunnerControlHistory *history
) {
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    /* allocate parsing buffers on the heap so they can be registered
       and reliably observed by the debug registry (avoids untracked
       stack writes and lifetime confusion). Sizes are chosen from the
       descriptor length with safe defaults and caps. */
    char *outputs_csv = NULL;
    char *sources_csv = NULL;
    char (*output_names)[64] = NULL;
    controller_source_t *mappings = NULL;
    int output_count = 0;
    int mapping_count = 0;
    int resolved_channels = 0;
    size_t total = 0;
    double *buffer = NULL;
    int _rc = -1;
    size_t outputs_cap = 256;
    size_t sources_cap = 512;
    if (descriptor != NULL && descriptor->params_len > 0) {
        size_t p = descriptor->params_len + 1;
        if (p > outputs_cap) outputs_cap = p;
        if (p > sources_cap) sources_cap = p;
    }
    /* clamp to reasonable bounds */
    if (outputs_cap < 256) outputs_cap = 256;
    if (sources_cap < 512) sources_cap = 512;
    if (outputs_cap > 65536) outputs_cap = 65536;
    if (sources_cap > 65536) sources_cap = 65536;
    outputs_csv = (char *)malloc(outputs_cap);
    if (outputs_csv == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    register_alloc(outputs_csv, outputs_cap);
    sources_csv = (char *)malloc(sources_cap);
    if (sources_csv == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    register_alloc(sources_csv, sources_cap);
    output_names = (char (*)[64])malloc((size_t)32 * 64);
    if (output_names == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    register_alloc(output_names, (size_t)32 * 64);
    mappings = (controller_source_t *)malloc((size_t)32 * sizeof(controller_source_t));
    if (mappings == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    register_alloc(mappings, (size_t)32 * sizeof(controller_source_t));
    if (json_copy_string(descriptor->params_json, descriptor->params_len, "__controller_outputs__", outputs_csv, outputs_cap)) {
        output_count = parse_csv_tokens(outputs_csv, output_names, 32);
    }
    if (output_count <= 0 && inputs != NULL) {
        uint32_t count = inputs->params.count;
        EdgeRunnerParamView *items = inputs->params.items;
        for (uint32_t i = 0; i < count && i < 32U; ++i) {
            if (items[i].name != NULL) {
                strncpy(output_names[output_count], items[i].name, sizeof(output_names[output_count]) - 1);
                output_names[output_count][sizeof(output_names[output_count]) - 1] = '\0';
                output_count++;
            }
        }
    }
    if (output_count <= 0) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    if (json_copy_string(descriptor->params_json, descriptor->params_len, "__controller_sources__", sources_csv, sources_cap)) {
        mapping_count = parse_controller_sources(sources_csv, mappings, 32);
    }
    resolved_channels = output_count;
    if (inputs != NULL && inputs->params.count > 0) {
        const EdgeRunnerParamView *view = &inputs->params.items[0];
        if (batches <= 0 && view->batches > 0) {
            batches = (int)view->batches;
        }
        if (frames <= 0 && view->frames > 0) {
            frames = (int)view->frames;
        }
    }
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    total = (size_t)batches * (size_t)resolved_channels * (size_t)frames;
    buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    for (int c = 0; c < resolved_channels; ++c) {
        const char *source_name = output_names[c];
        for (int m = 0; m < mapping_count; ++m) {
            if (strcmp(mappings[m].output, output_names[c]) == 0) {
                source_name = mappings[m].source;
                break;
            }
        }
        const EdgeRunnerParamView *view = find_param(inputs, source_name);
        int view_missing = (view == NULL || view->data == NULL);
        double *owned = NULL;
        const double *data = ensure_param_plane(view, batches, frames, 0.0, &owned);
        if (data == NULL) {
            free(buffer);
            _rc = -1;
            goto cleanup_run_controller_node;
        }
        if (view_missing && owned != NULL && history != NULL) {
            const EdgeRunnerControlCurve *curve = find_history_curve(history, source_name, strlen(source_name));
            if (curve != NULL) {
                apply_history_curve(owned, batches, frames, curve);
            }
            data = owned;
        }
        for (int b = 0; b < batches; ++b) {
            for (int f = 0; f < frames; ++f) {
                size_t src_idx = (size_t)b * (size_t)frames + (size_t)f;
                size_t dst_idx = ((size_t)b * (size_t)resolved_channels + (size_t)c) * (size_t)frames + (size_t)f;
                /* bounds checks and durable logging for write operations */
                size_t total_len = total; /* copied to local for clarity */
                if (dst_idx >= total_len) {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
                    ensure_log_files_open();
                    if (log_f_memops) {
                        fprintf(log_f_memops, "BAD_WRITE run_controller_node dst_idx=%zu total=%zu src_idx=%zu c=%d b=%d f=%d resolved_channels=%d frames=%d\n", dst_idx, total_len, src_idx, c, b, f, resolved_channels, frames);
                        fflush(log_f_memops);
                    }
#endif
                    /* attempt to continue safely */
                    continue;
                }
                /* ensure src_idx is in range of the data plane */
                /* we don't know 'data' length statically; we will conservatively attempt to log if src_idx seems large */
                buffer[dst_idx] = data[src_idx];
#if defined(AMP_NATIVE_ENABLE_LOGGING)
                ensure_log_files_open();
                if (log_f_memops) {
                    fprintf(log_f_memops, "POSTWRITE run_controller_node dest=%p dst_idx=%zu wrote=1 src_idx=%zu node=%s\n", (void *)buffer, dst_idx, src_idx, descriptor->name ? descriptor->name : "<noname>");
                    fflush(log_f_memops);
                }
#endif
            }
        }
        if (owned != NULL) {
            free(owned);
        }
    }
    *out_buffer = buffer;
    *out_channels = resolved_channels;
    _rc = 0;

cleanup_run_controller_node:
    /* Free and unregister any heap buffers allocated above. Keep this idempotent. */
    if (outputs_csv != NULL) {
        unregister_alloc(outputs_csv);
        free(outputs_csv);
        outputs_csv = NULL;
    }
    if (sources_csv != NULL) {
        unregister_alloc(sources_csv);
        free(sources_csv);
        sources_csv = NULL;
    }
    if (output_names != NULL) {
        unregister_alloc(output_names);
        free(output_names);
        output_names = NULL;
    }
    if (mappings != NULL) {
        unregister_alloc(mappings);
        free(mappings);
        mappings = NULL;
    }
    return _rc;
}

static int run_lfo_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    char wave_buf[32];
    if (!json_copy_string(descriptor->params_json, descriptor->params_len, "wave", wave_buf, sizeof(wave_buf))) {
        strcpy(wave_buf, "sine");
    }
    double rate_hz = json_get_double(descriptor->params_json, descriptor->params_len, "rate_hz", 1.0);
    double depth = json_get_double(descriptor->params_json, descriptor->params_len, "depth", 0.5);
    double slew_ms = json_get_double(descriptor->params_json, descriptor->params_len, "slew_ms", 0.0);
    int use_input = json_get_bool(descriptor->params_json, descriptor->params_len, "use_input", 0);
    int B = batches > 0 ? batches : 1;
    if (inputs->audio.batches > 0) {
        B = (int)inputs->audio.batches;
    }
    int F = frames > 0 ? frames : 1;
    int audio_channels = 0;
    const double *audio_data = NULL;
    if (use_input && inputs != NULL && inputs->audio.has_audio && inputs->audio.data != NULL) {
        B = inputs->audio.batches > 0 ? (int)inputs->audio.batches : B;
        F = inputs->audio.frames > 0 ? (int)inputs->audio.frames : F;
        audio_channels = inputs->audio.channels > 0 ? (int)inputs->audio.channels : 1;
        audio_data = inputs->audio.data;
    }
    size_t total = (size_t)B * (size_t)F;
    double *buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        return -1;
    }
    if (use_input && audio_data != NULL) {
        for (int b = 0; b < B; ++b) {
            double max_abs = 0.0;
            for (int c = 0; c < audio_channels; ++c) {
                for (int f = 0; f < F; ++f) {
                    size_t idx = ((size_t)b * (size_t)audio_channels + (size_t)c) * (size_t)F + (size_t)f;
                    double val = fabs(audio_data[idx]);
                    if (val > max_abs) {
                        max_abs = val;
                    }
                }
            }
            if (max_abs < 1e-12) {
                max_abs = 1.0;
            }
            for (int f = 0; f < F; ++f) {
                size_t src_idx = ((size_t)b * (size_t)audio_channels) * (size_t)F + (size_t)f;
                double sample = audio_data[src_idx];
                buffer[(size_t)b * (size_t)F + (size_t)f] = (sample / max_abs) * depth;
            }
        }
    } else {
        if (sample_rate <= 0.0) {
            sample_rate = 48000.0;
        }
        double step = rate_hz / sample_rate;
        double phase = 0.0;
        if (state != NULL) {
            phase = state->u.lfo.phase;
        }
        for (int b = 0; b < B; ++b) {
            double local_phase = phase;
            for (int f = 0; f < F; ++f) {
                double value = render_lfo_wave(wave_buf, local_phase) * depth;
                buffer[(size_t)b * (size_t)F + (size_t)f] = value;
                local_phase += step;
                local_phase -= floor(local_phase);
            }
            if (state != NULL) {
                phase = local_phase;
            }
        }
        if (state != NULL) {
            state->u.lfo.phase = phase;
        }
    }
    if (slew_ms > 0.0 && state != NULL) {
        if (sample_rate <= 0.0) {
            sample_rate = 48000.0;
        }
        double alpha = 1.0 - exp(-1.0 / (sample_rate * (slew_ms / 1000.0)));
        if (alpha < 1.0 - 1e-15) {
            double r = 1.0 - alpha;
            if (state->u.lfo.slew_state == NULL || state->u.lfo.batches != B) {
                free(state->u.lfo.slew_state);
                state->u.lfo.slew_state = (double *)calloc((size_t)B, sizeof(double));
                state->u.lfo.batches = B;
            }
            if (state->u.lfo.slew_state != NULL) {
                lfo_slew(buffer, buffer, B, F, r, alpha, state->u.lfo.slew_state);
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = 1;
    return 0;
}

static int run_envelope_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    const EdgeRunnerParamView *trigger_view = find_param(inputs, "trigger");
    const EdgeRunnerParamView *gate_view = find_param(inputs, "gate");
    const EdgeRunnerParamView *drone_view = find_param(inputs, "drone");
    const EdgeRunnerParamView *velocity_view = find_param(inputs, "velocity");
    const EdgeRunnerParamView *send_reset_view = find_param(inputs, "send_reset");
    if (trigger_view != NULL && trigger_view->batches > 0) {
        B = (int)trigger_view->batches;
    }
    if (trigger_view != NULL && trigger_view->frames > 0) {
        F = (int)trigger_view->frames;
    }
    if (B <= 0) {
        B = 1;
    }
    if (F <= 0) {
        F = 1;
    }
    double attack_ms = json_get_double(descriptor->params_json, descriptor->params_len, "attack_ms", 12.0);
    double hold_ms = json_get_double(descriptor->params_json, descriptor->params_len, "hold_ms", 8.0);
    double decay_ms = json_get_double(descriptor->params_json, descriptor->params_len, "decay_ms", 90.0);
    double sustain_level = json_get_double(descriptor->params_json, descriptor->params_len, "sustain_level", 0.7);
    double sustain_ms = json_get_double(descriptor->params_json, descriptor->params_len, "sustain_ms", 0.0);
    double release_ms = json_get_double(descriptor->params_json, descriptor->params_len, "release_ms", 220.0);
    int send_resets_default = json_get_bool(descriptor->params_json, descriptor->params_len, "send_resets", 1);
    int atk_frames = (int)lrint((attack_ms / 1000.0) * sample_rate);
    int hold_frames = (int)lrint((hold_ms / 1000.0) * sample_rate);
    int dec_frames = (int)lrint((decay_ms / 1000.0) * sample_rate);
    int sus_frames = (int)lrint((sustain_ms / 1000.0) * sample_rate);
    int rel_frames = (int)lrint((release_ms / 1000.0) * sample_rate);
    if (atk_frames < 0) atk_frames = 0;
    if (hold_frames < 0) hold_frames = 0;
    if (dec_frames < 0) dec_frames = 0;
    if (sus_frames < 0) sus_frames = 0;
    if (rel_frames < 0) rel_frames = 0;
    double *owned_trigger = NULL;
    double *owned_gate = NULL;
    double *owned_drone = NULL;
    double *owned_velocity = NULL;
    const double *trigger = ensure_param_plane(trigger_view, B, F, 0.0, &owned_trigger);
    const double *gate = ensure_param_plane(gate_view, B, F, 0.0, &owned_gate);
    const double *drone = ensure_param_plane(drone_view, B, F, 0.0, &owned_drone);
    const double *velocity = ensure_param_plane(velocity_view, B, F, 1.0, &owned_velocity);
    if (trigger == NULL || gate == NULL || drone == NULL || velocity == NULL) {
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    double send_reset_value = read_scalar_param(send_reset_view, (double)send_resets_default);
    int send_reset_flag = send_reset_value >= 0.5 ? 1 : 0;
    size_t total = (size_t)B * (size_t)F * 2;
    double *buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    double *amp_plane = buffer;
    double *reset_plane = buffer + (size_t)B * (size_t)F;
    if (state == NULL) {
        free(buffer);
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    if (state->u.envelope.stage == NULL || state->u.envelope.batches != B) {
        free(state->u.envelope.stage);
        free(state->u.envelope.value);
        free(state->u.envelope.timer);
        free(state->u.envelope.velocity);
        free(state->u.envelope.activations);
        free(state->u.envelope.release_start);
        state->u.envelope.stage = (int *)calloc((size_t)B, sizeof(int));
        state->u.envelope.value = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.timer = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.velocity = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.activations = (int64_t *)calloc((size_t)B, sizeof(int64_t));
        state->u.envelope.release_start = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.batches = B;
    }
    if (state->u.envelope.stage == NULL || state->u.envelope.value == NULL || state->u.envelope.timer == NULL) {
        free(buffer);
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    envelope_process(
        trigger,
        gate,
        drone,
        velocity,
        B,
        F,
        atk_frames,
        hold_frames,
        dec_frames,
        sus_frames,
        rel_frames,
        sustain_level,
        send_reset_flag,
        state->u.envelope.stage,
        state->u.envelope.value,
        state->u.envelope.timer,
        state->u.envelope.velocity,
        state->u.envelope.activations,
        state->u.envelope.release_start,
        amp_plane,
        reset_plane
    );
    free(owned_trigger);
    free(owned_gate);
    free(owned_drone);
    free(owned_velocity);
    *out_buffer = buffer;
    *out_channels = 2;
    return 0;
}

static int run_pitch_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    (void)sample_rate;
    if (out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    char grid_csv[1024];
    double grid_values[256];
    double grid_sorted_vals[256];
    double grid_ext[257];
    int grid_count = 0;
    if (json_copy_string(descriptor->params_json, descriptor->params_len, "grid_cents", grid_csv, sizeof(grid_csv))) {
        grid_count = parse_csv_doubles(grid_csv, grid_values, 256);
    }
    if (grid_count <= 0) {
        for (int i = 0; i < 12; ++i) {
            grid_values[i] = (double)i * 100.0;
        }
        grid_count = 12;
    }
    int grid_size = build_sorted_grid(grid_values, grid_count, grid_sorted_vals, grid_ext);
    int is_free_mode = json_get_bool(descriptor->params_json, descriptor->params_len, "is_free_mode", 0);
    char variant_buf[32];
    if (!json_copy_string(descriptor->params_json, descriptor->params_len, "free_variant", variant_buf, sizeof(variant_buf))) {
        strcpy(variant_buf, "continuous");
    }
    double span_default = json_get_double(descriptor->params_json, descriptor->params_len, "span_default", 2.0);
    int slew_enabled = json_get_bool(descriptor->params_json, descriptor->params_len, "slew", 1);
    const EdgeRunnerParamView *input_view = find_param(inputs, "input");
    const EdgeRunnerParamView *root_view = find_param(inputs, "root_midi");
    const EdgeRunnerParamView *span_view = find_param(inputs, "span_oct");
    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    if (input_view != NULL && input_view->batches > 0) {
        B = (int)input_view->batches;
    }
    if (input_view != NULL && input_view->frames > 0) {
        F = (int)input_view->frames;
    }
    if (B <= 0) B = 1;
    if (F <= 0) F = 1;
    double *owned_input = NULL;
    double *owned_root = NULL;
    double *owned_span = NULL;
    const double *ctrl = ensure_param_plane(input_view, B, F, 0.0, &owned_input);
    const double *root = ensure_param_plane(root_view, B, F, 60.0, &owned_root);
    const double *span = ensure_param_plane(span_view, B, F, span_default, &owned_span);
    if (ctrl == NULL || root == NULL || span == NULL) {
        free(owned_input);
        free(owned_root);
        free(owned_span);
        return -1;
    }
    size_t total = (size_t)B * (size_t)F;
    double *freq_target = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (freq_target == NULL) {
        free(owned_input);
        free(owned_root);
        free(owned_span);
        return -1;
    }
    using RowArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowArrayXXd> ctrl_map(ctrl, B, F);
    Eigen::Map<const RowArrayXXd> span_map(span, B, F);
    Eigen::Map<const RowArrayXXd> root_map(root, B, F);
    RowArrayXXd ctrl_scaled = ctrl_map * span_map;
    RowArrayXXd root_freq_array = ((root_map - 69.0) / 12.0 * M_LN2).exp() * 440.0;
    Eigen::Map<RowArrayXXd> freq_target_map(freq_target, B, F);
    int variant_weighted = strcmp(variant_buf, "weighted") == 0;
    int variant_stepped = strcmp(variant_buf, "stepped") == 0;
    if (is_free_mode && !variant_weighted && !variant_stepped) {
        RowArrayXXd cents = ctrl_scaled * 1200.0;
        freq_target_map = root_freq_array * ((cents / 1200.0 * M_LN2).exp());
    } else {
        for (int b = 0; b < B; ++b) {
            for (int f = 0; f < F; ++f) {
                double ctrl_scaled_val = ctrl_scaled(b, f);
                double cents = 0.0;
                if (is_free_mode) {
                    if (variant_weighted) {
                        double u = ctrl_scaled_val * (double)grid_size;
                        cents = grid_warp_inverse_value(u, grid_sorted_vals, grid_ext, grid_size);
                    } else if (variant_stepped) {
                        double u = round(ctrl_scaled_val * (double)grid_size);
                        cents = grid_warp_inverse_value(u, grid_sorted_vals, grid_ext, grid_size);
                    } else {
                        cents = ctrl_scaled_val * 1200.0;
                    }
                } else {
                    double cents_unq = ctrl_scaled_val * 1200.0;
                    double u = grid_warp_forward_value(cents_unq, grid_sorted_vals, grid_ext, grid_size);
                    double u_round = round(u);
                    cents = grid_warp_inverse_value(u_round, grid_sorted_vals, grid_ext, grid_size);
                }
                freq_target_map(b, f) = root_freq_array(b, f) * exp((cents / 1200.0) * M_LN2);
            }
        }
    }
    free(owned_input);
    free(owned_root);
    free(owned_span);
    double *output = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (output == NULL) {
        free(freq_target);
        return -1;
    }
    if (slew_enabled) {
        if (state->u.pitch.last_freq == NULL || state->u.pitch.batches != B) {
            free(state->u.pitch.last_freq);
            state->u.pitch.last_freq = (double *)calloc((size_t)B, sizeof(double));
            state->u.pitch.batches = B;
        }
        if (state->u.pitch.last_freq == NULL) {
            free(freq_target);
            free(output);
            return -1;
        }
        Eigen::ArrayXd t_values;
        if (F > 1) {
            double t_last = (double)(F - 1) / (double)F;
            t_values = Eigen::ArrayXd::LinSpaced(F, 0.0, t_last);
        } else {
            t_values = Eigen::ArrayXd::Zero(1);
        }
        Eigen::ArrayXd hermite = 3.0 * t_values.square() - 2.0 * t_values.cube();
        for (int b = 0; b < B; ++b) {
            double y0 = state->u.pitch.last_freq[b];
            double y1 = freq_target_map(b, F - 1);
            Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1>> out_row(output + (size_t)b * (size_t)F, F);
            out_row = hermite * (y1 - y0) + y0;
            state->u.pitch.last_freq[b] = y1;
        }
    } else {
        memcpy(output, freq_target, total * sizeof(double));
    }
    free(freq_target);
    *out_buffer = output;
    *out_channels = 1;
    return 0;
}

static int run_oscillator_pitch_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }

    double min_freq = json_get_double(descriptor->params_json, descriptor->params_len, "min_freq", 0.0);
    double default_slew = json_get_double(descriptor->params_json, descriptor->params_len, "default_slew", 0.0);
    if (default_slew < 0.0) {
        default_slew = 0.0;
    }

    const EdgeRunnerParamView *pitch_view = find_param(inputs, "pitch_hz");
    const EdgeRunnerParamView *root_view = find_param(inputs, "root_hz");
    const EdgeRunnerParamView *offset_view = find_param(inputs, "offset_cents");
    const EdgeRunnerParamView *add_view = find_param(inputs, "add_hz");
    const EdgeRunnerParamView *slew_view = find_param(inputs, "slew_hz_per_s");

    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    const EdgeRunnerParamView *shape_source = pitch_view != NULL ? pitch_view : root_view;
    if (shape_source != NULL) {
        if (shape_source->batches > 0) {
            B = (int)shape_source->batches;
        }
        if (shape_source->frames > 0) {
            F = (int)shape_source->frames;
        }
    }
    if (B <= 0) {
        B = 1;
    }
    if (F <= 0) {
        F = 1;
    }

    double *owned_pitch = NULL;
    double *owned_root = NULL;
    double *owned_offset = NULL;
    double *owned_add = NULL;
    double *owned_slew = NULL;

    int has_direct = (pitch_view != NULL && pitch_view->data != NULL);
    const double *pitch_curve = ensure_param_plane(pitch_view, B, F, 0.0, &owned_pitch);
    const double *root_curve = NULL;
    const double *offset_curve = NULL;
    if (!has_direct) {
        root_curve = ensure_param_plane(root_view, B, F, min_freq > 0.0 ? min_freq : 0.0, &owned_root);
        offset_curve = ensure_param_plane(offset_view, B, F, 0.0, &owned_offset);
        if (root_curve == NULL || offset_curve == NULL) {
            free(owned_pitch);
            free(owned_root);
            free(owned_offset);
            return -1;
        }
    }
    const double *add_curve = ensure_param_plane(add_view, B, F, 0.0, &owned_add);
    const double *slew_curve = ensure_param_plane(slew_view, B, F, default_slew, &owned_slew);

    if (pitch_curve == NULL && !has_direct) {
        free(owned_pitch);
        free(owned_root);
        free(owned_offset);
        free(owned_add);
        free(owned_slew);
        return -1;
    }

    size_t total = (size_t)B * (size_t)F;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        free(owned_pitch);
        free(owned_root);
        free(owned_offset);
        free(owned_add);
        free(owned_slew);
        return -1;
    }

    if (state->u.osc_pitch.last_value == NULL || state->u.osc_pitch.batches != B) {
        free(state->u.osc_pitch.last_value);
        state->u.osc_pitch.last_value = (double *)calloc((size_t)B, sizeof(double));
        state->u.osc_pitch.batches = B;
    }
    if (state->u.osc_pitch.last_value == NULL) {
        free(buffer);
        free(owned_pitch);
        free(owned_root);
        free(owned_offset);
        free(owned_add);
        free(owned_slew);
        return -1;
    }

    using RowArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowArrayXXd target_array(B, F);
    if (has_direct && pitch_curve != NULL) {
        Eigen::Map<const RowArrayXXd> pitch_map(pitch_curve, B, F);
        target_array = pitch_map;
    } else if (!has_direct && root_curve != NULL && offset_curve != NULL) {
        Eigen::Map<const RowArrayXXd> root_map(root_curve, B, F);
        Eigen::Map<const RowArrayXXd> offset_map(offset_curve, B, F);
        target_array = root_map * ((offset_map / 1200.0 * M_LN2).exp());
    } else {
        target_array.setZero();
    }
    if (add_curve != NULL) {
        Eigen::Map<const RowArrayXXd> add_map(add_curve, B, F);
        target_array += add_map;
    }
    target_array = target_array.cwiseMax(min_freq);
    double default_limit = default_slew > 0.0 ? default_slew / sample_rate : 0.0;
    RowArrayXXd limit_array = RowArrayXXd::Constant(B, F, default_limit);
    if (slew_curve != NULL) {
        Eigen::Map<const RowArrayXXd> slew_map(slew_curve, B, F);
        limit_array = slew_map.unaryExpr([sample_rate](double per_sec) {
            if (per_sec <= 0.0) {
                return 0.0;
            }
            return per_sec / sample_rate;
        });
    }
    for (int b = 0; b < B; ++b) {
        double current = state->u.osc_pitch.last_value[b];
        size_t base = (size_t)b * (size_t)F;
        auto target_row = target_array.row(b);
        auto limit_row = limit_array.row(b);
        for (int f = 0; f < F; ++f) {
            size_t idx = base + (size_t)f;
            double target = target_row(f);
            double limit = limit_row(f);
            if (limit > 0.0) {
                double delta = target - current;
                if (delta > limit) {
                    delta = limit;
                } else if (delta < -limit) {
                    delta = -limit;
                }
                current += delta;
            } else {
                current = target;
            }
            if (current < min_freq) {
                current = min_freq;
            }
            buffer[idx] = current;
        }
        state->u.osc_pitch.last_value[b] = current;
    }

    state->u.osc_pitch.batches = B;

    free(owned_pitch);
    free(owned_root);
    free(owned_offset);
    free(owned_add);
    free(owned_slew);

    *out_buffer = buffer;
    *out_channels = 1;
    return 0;
}

static double alpha_lp(double fc, double sr) {
    if (fc < 1.0) {
        fc = 1.0;
    }
    return 1.0 - exp(-2.0 * M_PI * fc / sr);
}

static double alpha_hp(double fc, double sr) {
    if (fc < 1.0) {
        fc = 1.0;
    }
    double rc = 1.0 / (2.0 * M_PI * fc);
    return rc / (rc + 1.0 / sr);
}

static int run_subharm_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (inputs == NULL || !inputs->audio.has_audio || inputs->audio.data == NULL) {
        return -1;
    }
    int B = inputs->audio.batches > 0 ? (int)inputs->audio.batches : (batches > 0 ? batches : 1);
    int C = inputs->audio.channels > 0 ? (int)inputs->audio.channels : 1;
    int F = inputs->audio.frames > 0 ? (int)inputs->audio.frames : (frames > 0 ? frames : 1);
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    const double *audio = inputs->audio.data;
    double band_lo = json_get_double(descriptor->params_json, descriptor->params_len, "band_lo", 70.0);
    double band_hi = json_get_double(descriptor->params_json, descriptor->params_len, "band_hi", 160.0);
    double mix = json_get_double(descriptor->params_json, descriptor->params_len, "mix", 0.5);
    double drive = json_get_double(descriptor->params_json, descriptor->params_len, "drive", 1.0);
    double out_hp = json_get_double(descriptor->params_json, descriptor->params_len, "out_hp", 25.0);
    int use_div4 = json_get_bool(descriptor->params_json, descriptor->params_len, "use_div4", 0);
    double a_hp_in = alpha_hp(band_lo, sample_rate);
    double a_lp_in = alpha_lp(band_hi, sample_rate);
    double a_sub2 = alpha_lp(fmax(band_hi / 3.0, 30.0), sample_rate);
    double a_sub4 = use_div4 ? alpha_lp(fmax(band_hi / 5.0, 20.0), sample_rate) : 0.0;
    double a_env_attack = alpha_lp(100.0, sample_rate);
    double a_env_release = alpha_lp(5.0, sample_rate);
    double a_hp_out = alpha_hp(out_hp, sample_rate);
    size_t total = (size_t)B * (size_t)C * (size_t)F;
    double *buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        return -1;
    }
    int need_resize = state->u.subharm.batches != B || state->u.subharm.channels != C || state->u.subharm.use_div4 != use_div4;
    if (need_resize) {
        free(state->u.subharm.hp_y);
        free(state->u.subharm.lp_y);
        free(state->u.subharm.prev);
        free(state->u.subharm.sign);
        free(state->u.subharm.ff2);
        free(state->u.subharm.ff4);
        free(state->u.subharm.ff4_count);
        free(state->u.subharm.sub2_lp);
        free(state->u.subharm.sub4_lp);
        free(state->u.subharm.env);
        free(state->u.subharm.hp_out_y);
        free(state->u.subharm.hp_out_x);
        state->u.subharm.hp_y = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.lp_y = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.prev = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.sign = (int8_t *)calloc((size_t)B * (size_t)C, sizeof(int8_t));
        state->u.subharm.ff2 = (int8_t *)calloc((size_t)B * (size_t)C, sizeof(int8_t));
        state->u.subharm.sub2_lp = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.env = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.hp_out_y = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.hp_out_x = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        if (use_div4) {
            state->u.subharm.ff4 = (int8_t *)calloc((size_t)B * (size_t)C, sizeof(int8_t));
            state->u.subharm.ff4_count = (int32_t *)calloc((size_t)B * (size_t)C, sizeof(int32_t));
            state->u.subharm.sub4_lp = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        } else {
            free(state->u.subharm.ff4);
            free(state->u.subharm.ff4_count);
            free(state->u.subharm.sub4_lp);
            state->u.subharm.ff4 = NULL;
            state->u.subharm.ff4_count = NULL;
            state->u.subharm.sub4_lp = NULL;
        }
        state->u.subharm.batches = B;
        state->u.subharm.channels = C;
        state->u.subharm.use_div4 = use_div4;
    }
    if (state->u.subharm.hp_y == NULL || state->u.subharm.lp_y == NULL || state->u.subharm.prev == NULL || state->u.subharm.sign == NULL || state->u.subharm.ff2 == NULL || state->u.subharm.sub2_lp == NULL || state->u.subharm.env == NULL || state->u.subharm.hp_out_y == NULL || state->u.subharm.hp_out_x == NULL) {
        free(buffer);
        return -1;
    }
    subharmonic_process(
        audio,
        buffer,
        B,
        C,
        F,
        a_hp_in,
        a_lp_in,
        a_sub2,
        use_div4,
        a_sub4,
        a_env_attack,
        a_env_release,
        a_hp_out,
        drive,
        mix,
        state->u.subharm.hp_y,
        state->u.subharm.lp_y,
        state->u.subharm.prev,
        state->u.subharm.sign,
        state->u.subharm.ff2,
        state->u.subharm.ff4,
        state->u.subharm.ff4_count,
        state->u.subharm.sub2_lp,
        state->u.subharm.sub4_lp,
        state->u.subharm.env,
        state->u.subharm.hp_out_y,
        state->u.subharm.hp_out_x
    );
    *out_buffer = buffer;
    *out_channels = C;
    return 0;
}

static int run_osc_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (descriptor == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }

    char wave_buf[32];
    if (!json_copy_string(descriptor->params_json, descriptor->params_len, "wave", wave_buf, sizeof(wave_buf))) {
        strcpy(wave_buf, "sine");
    }
    int accept_reset = json_get_bool(descriptor->params_json, descriptor->params_len, "accept_reset", 1);
    int mode = parse_osc_mode(descriptor->params_json, descriptor->params_len, state->u.osc.mode);
    if (mode < OSC_MODE_POLYBLEP || mode > OSC_MODE_OP_AMP) {
        mode = OSC_MODE_POLYBLEP;
    }

    double integration_leak = json_get_double(descriptor->params_json, descriptor->params_len, "integration_leak", 0.9995);
    if (integration_leak < 0.0) integration_leak = 0.0;
    if (integration_leak > 0.999999) integration_leak = 0.999999;
    double integration_gain = json_get_double(descriptor->params_json, descriptor->params_len, "integration_gain", 1.0);
    double integration_clamp = json_get_double(descriptor->params_json, descriptor->params_len, "integration_clamp", 1.2);
    if (integration_clamp <= 0.0) integration_clamp = 1.2;

    double base_slew_rate = json_get_double(descriptor->params_json, descriptor->params_len, "slew_rate", 12000.0);
    if (base_slew_rate < 0.0) base_slew_rate = 0.0;
    double slew_clamp = json_get_double(descriptor->params_json, descriptor->params_len, "slew_clamp", 1.2);
    if (slew_clamp <= 0.0) slew_clamp = 1.2;

    const EdgeRunnerParamView *freq_view = find_param(inputs, "freq");
    if (freq_view == NULL) {
        freq_view = find_param(inputs, "frequency");
    }
    const EdgeRunnerParamView *amp_view = find_param(inputs, "amp");
    if (amp_view == NULL) {
        amp_view = find_param(inputs, "amplitude");
    }
    const EdgeRunnerParamView *pan_view = find_param(inputs, "pan");
    const EdgeRunnerParamView *reset_view = accept_reset ? find_param(inputs, "reset") : NULL;
    const EdgeRunnerParamView *phase_offset_view = find_param(inputs, "phase_offset");
    const EdgeRunnerParamView *frame_delay_view = find_param(inputs, "frame_delay");
    const EdgeRunnerParamView *slew_view = find_param(inputs, "slew");

    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    const EdgeRunnerParamView *shape_source = freq_view != NULL ? freq_view : amp_view;
    if (shape_source != NULL) {
        if (shape_source->batches > 0) {
            B = (int)shape_source->batches;
        }
        if (shape_source->frames > 0) {
            F = (int)shape_source->frames;
        }
    }
    if (B <= 0) B = 1;
    if (F <= 0) F = 1;

    double *owned_freq = NULL;
    double *owned_amp = NULL;
    double *owned_pan = NULL;
    double *owned_reset = NULL;
    double *owned_phase_offset = NULL;
    double *owned_frame_delay = NULL;
    double *owned_slew = NULL;

    const double *freq = ensure_param_plane(freq_view, B, F, 0.0, &owned_freq);
    const double *amp = ensure_param_plane(amp_view, B, F, 1.0, &owned_amp);
    const double *pan = ensure_param_plane(pan_view, B, F, 0.0, &owned_pan);
    const double *reset = ensure_param_plane(reset_view, B, F, 0.0, &owned_reset);
    const double *phase_offset = ensure_param_plane(phase_offset_view, B, F, 0.0, &owned_phase_offset);
    const double *frame_delay = ensure_param_plane(frame_delay_view, B, F, 0.0, &owned_frame_delay);
    const double *slew_curve = ensure_param_plane(slew_view, B, F, -1.0, &owned_slew);

    if ((mode != OSC_MODE_OP_AMP && freq == NULL) || amp == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_pan);
        free(owned_reset);
        free(owned_phase_offset);
        free(owned_frame_delay);
        free(owned_slew);
        return -1;
    }

    size_t total = (size_t)B * (size_t)F;
    if (state->u.osc.phase == NULL || state->u.osc.batches != B) {
        free(state->u.osc.phase);
        state->u.osc.phase = (double *)calloc((size_t)B, sizeof(double));
    }
    if (state->u.osc.phase_buffer == NULL || state->u.osc.batches != B) {
        free(state->u.osc.phase_buffer);
        state->u.osc.phase_buffer = (double *)malloc(total * sizeof(double));
    }
    if (state->u.osc.wave_buffer == NULL || state->u.osc.batches != B) {
        free(state->u.osc.wave_buffer);
        state->u.osc.wave_buffer = (double *)malloc(total * sizeof(double));
    }
    if (state->u.osc.dphi_buffer == NULL || state->u.osc.batches != B) {
        free(state->u.osc.dphi_buffer);
        state->u.osc.dphi_buffer = (double *)malloc(total * sizeof(double));
    }
    if (mode != OSC_MODE_OP_AMP && strcmp(wave_buf, "triangle") == 0) {
        if (state->u.osc.tri_state == NULL || state->u.osc.batches != B) {
            free(state->u.osc.tri_state);
            state->u.osc.tri_state = (double *)calloc((size_t)B, sizeof(double));
        }
    }
    if (mode == OSC_MODE_INTEGRATOR) {
        if (state->u.osc.integrator_state == NULL || state->u.osc.batches != B) {
            free(state->u.osc.integrator_state);
            state->u.osc.integrator_state = (double *)calloc((size_t)B, sizeof(double));
        }
    }

    const double *driver_data = NULL;
    int driver_channels = 1;
    if (mode == OSC_MODE_OP_AMP && inputs != NULL && inputs->audio.has_audio && inputs->audio.data != NULL) {
        driver_data = inputs->audio.data;
        driver_channels = inputs->audio.channels > 0 ? (int)inputs->audio.channels : 1;
    }
    if (mode == OSC_MODE_OP_AMP) {
        if (state->u.osc.op_amp_state == NULL || state->u.osc.batches != B || state->u.osc.driver_channels != driver_channels) {
            free(state->u.osc.op_amp_state);
            state->u.osc.op_amp_state = (double *)calloc((size_t)B, sizeof(double));
            state->u.osc.driver_channels = driver_channels;
        }
    }

    state->u.osc.batches = B;
    state->u.osc.mode = mode;

    if (state->u.osc.phase == NULL || state->u.osc.phase_buffer == NULL || state->u.osc.wave_buffer == NULL || state->u.osc.dphi_buffer == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_pan);
        free(owned_reset);
        free(owned_phase_offset);
        free(owned_frame_delay);
        free(owned_slew);
        return -1;
    }

    for (int b = 0; b < B; ++b) {
        for (int f = 0; f < F; ++f) {
            size_t idx = (size_t)b * (size_t)F + (size_t)f;
            double hz = (freq != NULL) ? freq[idx] : 0.0;
            state->u.osc.dphi_buffer[idx] = hz / sample_rate;
        }
    }

    const double *reset_ptr = accept_reset ? reset : NULL;
    phase_advance(state->u.osc.dphi_buffer, state->u.osc.phase_buffer, B, F, state->u.osc.phase, reset_ptr);

    if ((phase_offset != NULL && owned_phase_offset != NULL) || (frame_delay != NULL && owned_frame_delay != NULL)) {
        for (int b = 0; b < B; ++b) {
            size_t base = (size_t)b * (size_t)F;
            for (int f = 0; f < F; ++f) {
                size_t idx = base + (size_t)f;
                double ph = state->u.osc.phase_buffer[idx];
                if (phase_offset != NULL) {
                    ph += phase_offset[idx];
                }
                if (frame_delay != NULL) {
                    ph += state->u.osc.dphi_buffer[idx] * frame_delay[idx];
                }
                ph = ph - floor(ph);
                state->u.osc.phase_buffer[idx] = ph;
            }
        }
    }

    if (mode != OSC_MODE_OP_AMP) {
        if (strcmp(wave_buf, "saw") == 0) {
            osc_saw_blep_c(state->u.osc.phase_buffer, state->u.osc.dphi_buffer, state->u.osc.wave_buffer, B, F);
        } else if (strcmp(wave_buf, "square") == 0) {
            osc_square_blep_c(state->u.osc.phase_buffer, state->u.osc.dphi_buffer, 0.5, state->u.osc.wave_buffer, B, F);
        } else if (strcmp(wave_buf, "triangle") == 0) {
            if (state->u.osc.tri_state == NULL) {
                state->u.osc.tri_state = (double *)calloc((size_t)B, sizeof(double));
            }
            osc_triangle_blep_c(state->u.osc.phase_buffer, state->u.osc.dphi_buffer, state->u.osc.wave_buffer, B, F, state->u.osc.tri_state);
        } else {
            for (int b = 0; b < B; ++b) {
                size_t base = (size_t)b * (size_t)F;
                for (int f = 0; f < F; ++f) {
                    size_t idx = base + (size_t)f;
                    state->u.osc.wave_buffer[idx] = sin(state->u.osc.phase_buffer[idx] * 2.0 * M_PI);
                }
            }
        }
    } else {
        double per_sample_default = base_slew_rate > 0.0 ? base_slew_rate / sample_rate : 0.0;
        for (int b = 0; b < B; ++b) {
            double op_state = state->u.osc.op_amp_state != NULL ? state->u.osc.op_amp_state[b] : 0.0;
            size_t base_wave = (size_t)b * (size_t)F;
            size_t base_driver = ((size_t)b * (size_t)driver_channels) * (size_t)F;
            Eigen::ArrayXd target_row = Eigen::ArrayXd::Zero(F);
            if (driver_data != NULL) {
                if (driver_channels == 1) {
                    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> driver_vec(
                        driver_data + base_driver,
                        F
                    );
                    target_row = driver_vec;
                } else {
                    using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                    Eigen::Map<const RowMatrixXd> driver_block(driver_data + base_driver, driver_channels, F);
                    target_row = driver_block.colwise().mean().transpose().array();
                }
            }
            Eigen::ArrayXd slew_row = Eigen::ArrayXd::Constant(F, per_sample_default);
            if (slew_curve != NULL) {
                Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> slew_vec(slew_curve + base_wave, F);
                slew_row = slew_vec.unaryExpr([per_sample_default, sample_rate](double candidate) {
                    return candidate >= 0.0 ? candidate / sample_rate : per_sample_default;
                });
            }
            for (int f = 0; f < F; ++f) {
                double target = target_row[f];
                double per_sample_slew = slew_row[f];
                double delta = target - op_state;
                if (per_sample_slew > 0.0) {
                    if (delta > per_sample_slew) delta = per_sample_slew;
                    else if (delta < -per_sample_slew) delta = -per_sample_slew;
                }
                op_state += delta;
                if (op_state > slew_clamp) op_state = slew_clamp;
                else if (op_state < -slew_clamp) op_state = -slew_clamp;
                state->u.osc.wave_buffer[base_wave + (size_t)f] = op_state;
            }
            if (state->u.osc.op_amp_state != NULL) {
                state->u.osc.op_amp_state[b] = op_state;
            }
        }
    }

    if (mode == OSC_MODE_INTEGRATOR && state->u.osc.integrator_state != NULL) {
        for (int b = 0; b < B; ++b) {
            double accum = state->u.osc.integrator_state[b];
            size_t base = (size_t)b * (size_t)F;
            for (int f = 0; f < F; ++f) {
                size_t idx = base + (size_t)f;
                accum = accum * integration_leak + integration_gain * state->u.osc.wave_buffer[idx];
                if (accum > integration_clamp) accum = integration_clamp;
                else if (accum < -integration_clamp) accum = -integration_clamp;
                state->u.osc.wave_buffer[idx] = accum;
            }
            state->u.osc.integrator_state[b] = accum;
        }
    }

    int stereo = (pan_view != NULL && pan_view->data != NULL) ? 1 : 0;
    int channels = stereo ? 2 : 1;
    size_t total_out = (size_t)B * (size_t)channels * (size_t)F;
    double *buffer = (double *)malloc(total_out * sizeof(double));
    if (buffer == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_pan);
        free(owned_reset);
        free(owned_phase_offset);
        free(owned_frame_delay);
        free(owned_slew);
        return -1;
    }

    for (int b = 0; b < B; ++b) {
        for (int f = 0; f < F; ++f) {
            size_t idx = (size_t)b * (size_t)F + (size_t)f;
            double sample = state->u.osc.wave_buffer[idx] * amp[idx];
            if (stereo) {
                double pan_val = pan[idx];
                if (pan_val < -1.0) pan_val = -1.0;
                if (pan_val > 1.0) pan_val = 1.0;
                double angle = (pan_val + 1.0) * (M_PI / 4.0);
                double left = sample * cos(angle);
                double right = sample * sin(angle);
                buffer[((size_t)b * 2) * (size_t)F + (size_t)f] = left;
                buffer[((size_t)b * 2 + 1) * (size_t)F + (size_t)f] = right;
            } else {
                buffer[(size_t)b * (size_t)F + (size_t)f] = sample;
            }
        }
    }

    free(owned_freq);
    free(owned_amp);
    free(owned_pan);
    free(owned_reset);
    free(owned_phase_offset);
    free(owned_frame_delay);
    free(owned_slew);

    state->u.osc.channels = 1;
    state->u.osc.stereo = stereo;

    *out_buffer = buffer;
    *out_channels = stereo ? 2 : 1;
    return 0;
}

static int run_parametric_driver_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (descriptor == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }

    int mode = parse_driver_mode(descriptor->params_json, descriptor->params_len, state->u.driver.mode);
    double harmonics_buffer[32];
    int harmonic_count = 0;

    if (mode == DRIVER_MODE_PIEZO) {
        harmonics_buffer[0] = 1.0;
        harmonics_buffer[1] = 0.35;
        harmonics_buffer[2] = 0.12;
        harmonic_count = 3;
    } else if (mode == DRIVER_MODE_CUSTOM) {
        char harmonic_csv[256];
        if (json_copy_string(descriptor->params_json, descriptor->params_len, "harmonics", harmonic_csv, sizeof(harmonic_csv))) {
            harmonic_count = parse_csv_doubles(harmonic_csv, harmonics_buffer, 32);
        }
        if (harmonic_count <= 0) {
            harmonics_buffer[0] = 1.0;
            harmonic_count = 1;
        }
    } else {
        harmonics_buffer[0] = 1.0;
        harmonic_count = 1;
    }

    if (state->u.driver.harmonics == NULL || state->u.driver.harmonic_count != harmonic_count || state->u.driver.mode != mode) {
        free(state->u.driver.harmonics);
        state->u.driver.harmonics = (double *)malloc((size_t)harmonic_count * sizeof(double));
        if (state->u.driver.harmonics == NULL) {
            state->u.driver.harmonic_count = 0;
            return -1;
        }
        for (int i = 0; i < harmonic_count; ++i) {
            state->u.driver.harmonics[i] = harmonics_buffer[i];
        }
        state->u.driver.harmonic_count = harmonic_count;
    }
    state->u.driver.mode = mode;

    const EdgeRunnerParamView *freq_view = find_param(inputs, "frequency");
    if (freq_view == NULL) {
        freq_view = find_param(inputs, "freq");
    }
    const EdgeRunnerParamView *amp_view = find_param(inputs, "amplitude");
    if (amp_view == NULL) {
        amp_view = find_param(inputs, "amp");
    }
    const EdgeRunnerParamView *phase_view = find_param(inputs, "phase_offset");
    if (phase_view == NULL) {
        phase_view = find_param(inputs, "phase");
    }
    const EdgeRunnerParamView *render_view = find_param(inputs, "render_mode");

    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    const EdgeRunnerParamView *shape_source = freq_view != NULL ? freq_view : amp_view;
    if (shape_source != NULL) {
        if (shape_source->batches > 0) {
            B = (int)shape_source->batches;
        }
        if (shape_source->frames > 0) {
            F = (int)shape_source->frames;
        }
    }
    if (B <= 0) B = 1;
    if (F <= 0) F = 1;

    double *owned_freq = NULL;
    double *owned_amp = NULL;
    double *owned_phase = NULL;
    double *owned_render = NULL;

    const double *freq = ensure_param_plane(freq_view, B, F, 440.0, &owned_freq);
    const double *amp = ensure_param_plane(amp_view, B, F, 1.0, &owned_amp);
    const double *phase_offset = ensure_param_plane(phase_view, B, F, 0.0, &owned_phase);
    const double *render_mode = ensure_param_plane(render_view, B, F, 0.0, &owned_render);
    if (freq == NULL || amp == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_phase);
        free(owned_render);
        return -1;
    }

    if (state->u.driver.phase == NULL || state->u.driver.batches != B) {
        free(state->u.driver.phase);
        state->u.driver.phase = (double *)calloc((size_t)B, sizeof(double));
        state->u.driver.batches = B;
    }
    if (state->u.driver.phase == NULL || state->u.driver.harmonics == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_phase);
        return -1;
    }

    size_t total = (size_t)B * (size_t)F;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_phase);
        free(owned_render);
        return -1;
    }

    const double *driver_input = NULL;
    int driver_in_channels = 1;
    int driver_in_batches = B;
    int driver_in_frames = F;
    if (inputs != NULL && inputs->audio.has_audio && inputs->audio.data != NULL) {
        driver_input = inputs->audio.data;
        driver_in_channels = inputs->audio.channels > 0 ? (int)inputs->audio.channels : 1;
        if (inputs->audio.batches > 0) {
            driver_in_batches = (int)inputs->audio.batches;
        }
        if (inputs->audio.frames > 0) {
            driver_in_frames = (int)inputs->audio.frames;
        }
    }
    if (driver_in_batches <= 0) {
        driver_in_batches = B > 0 ? B : 1;
    }
    if (driver_in_frames <= 0) {
        driver_in_frames = F > 0 ? F : 1;
    }
    if (driver_in_channels <= 0) {
        driver_in_channels = 1;
    }

    for (int b = 0; b < B; ++b) {
        double phase = state->u.driver.phase[b];
        size_t base = (size_t)b * (size_t)F;
        for (int f = 0; f < F; ++f) {
            size_t idx = base + (size_t)f;
            double hz = freq[idx];
            if (hz < 0.0) hz = 0.0;
            double advance = hz / sample_rate;
            phase += advance;
            phase = phase - floor(phase);
            double ph = phase;
            if (phase_offset != NULL) {
                ph += phase_offset[idx];
                ph = ph - floor(ph);
            }
            double sample = 0.0;
            if (mode == DRIVER_MODE_QUARTZ && state->u.driver.harmonic_count == 1) {
                sample = sin(2.0 * M_PI * ph);
            } else {
                for (int h = 0; h < state->u.driver.harmonic_count; ++h) {
                    double coeff = state->u.driver.harmonics[h];
                    if (coeff == 0.0) {
                        continue;
                    }
                    double harmonic_phase = 2.0 * M_PI * ph * (double)(h + 1);
                    sample += coeff * sin(harmonic_phase);
                }
            }
            double blend = render_mode != NULL ? render_mode[idx] : 0.0;
            if (blend < 0.0) {
                blend = 0.0;
            } else if (blend > 1.0) {
                blend = 1.0;
            }
            double stream_val = 0.0;
            if (driver_input != NULL) {
                int bb = b;
                if (bb >= driver_in_batches) {
                    bb = driver_in_batches - 1;
                }
                if (bb < 0) {
                    bb = 0;
                }
                int ff = f;
                if (ff >= driver_in_frames) {
                    ff = driver_in_frames - 1;
                }
                if (ff < 0) {
                    ff = 0;
                }
                size_t base_audio = ((size_t)bb * (size_t)driver_in_channels) * (size_t)driver_in_frames;
                for (int c = 0; c < driver_in_channels; ++c) {
                    stream_val += driver_input[base_audio + (size_t)c * (size_t)driver_in_frames + (size_t)ff];
                }
                stream_val /= (double)driver_in_channels;
            }
            double combined = (1.0 - blend) * sample + blend * stream_val;
            buffer[idx] = combined * amp[idx];
        }
        state->u.driver.phase[b] = phase - floor(phase);
    }

    free(owned_freq);
    free(owned_amp);
    free(owned_phase);
    free(owned_render);

    *out_buffer = buffer;
    *out_channels = 1;
    return 0;
}

static int ensure_pitch_shift_state(node_state_t *state, int window_size, int hop_size, int resynthesis_hop) {
    if (state == NULL) {
        return -1;
    }
    if (window_size <= 0) {
        window_size = PITCH_SHIFT_DEFAULT_WINDOW;
    }
    if (!is_power_of_two_int(window_size)) {
        int next_power = 1;
        while (next_power < window_size && next_power < 131072) {
            next_power <<= 1;
        }
        window_size = next_power;
    }
    if (window_size <= 0) {
        window_size = PITCH_SHIFT_DEFAULT_WINDOW;
    }
    if (hop_size <= 0) {
        hop_size = PITCH_SHIFT_DEFAULT_HOP;
    }
    if (resynthesis_hop <= 0) {
        resynthesis_hop = PITCH_SHIFT_DEFAULT_RESYNTH_HOP;
    }

    int needs_resize = (state->u.pitch_shift.analysis_window == NULL)
        || (state->u.pitch_shift.window_size != window_size);
    if (needs_resize) {
        double *analysis = (double *)malloc((size_t)window_size * sizeof(double));
        double *synthesis = (double *)malloc((size_t)window_size * sizeof(double));
        double *prev_phase = (double *)malloc((size_t)window_size * sizeof(double));
        double *phase_accum = (double *)malloc((size_t)window_size * sizeof(double));
        if (analysis == NULL || synthesis == NULL || prev_phase == NULL || phase_accum == NULL) {
            free(analysis);
            free(synthesis);
            free(prev_phase);
            free(phase_accum);
            return -1;
        }
        free(state->u.pitch_shift.analysis_window);
        free(state->u.pitch_shift.synthesis_window);
        free(state->u.pitch_shift.prev_phase);
        free(state->u.pitch_shift.phase_accum);
        state->u.pitch_shift.analysis_window = analysis;
        state->u.pitch_shift.synthesis_window = synthesis;
        state->u.pitch_shift.prev_phase = prev_phase;
        state->u.pitch_shift.phase_accum = phase_accum;
        state->u.pitch_shift.window_size = window_size;
    }

    if (state->u.pitch_shift.analysis_window == NULL || state->u.pitch_shift.synthesis_window == NULL
        || state->u.pitch_shift.prev_phase == NULL || state->u.pitch_shift.phase_accum == NULL) {
        return -1;
    }

    state->u.pitch_shift.window_size = window_size;
    state->u.pitch_shift.hop_size = hop_size;
    state->u.pitch_shift.resynthesis_hop = resynthesis_hop;

    fill_window_weights(state->u.pitch_shift.analysis_window, window_size, FFT_WINDOW_HANN);
    fill_window_weights(state->u.pitch_shift.synthesis_window, window_size, FFT_WINDOW_HANN);
    memset(state->u.pitch_shift.prev_phase, 0, (size_t)window_size * sizeof(double));
    memset(state->u.pitch_shift.phase_accum, 0, (size_t)window_size * sizeof(double));

    return 0;
}

static double wrap_phase(double value) {
    while (value > M_PI) {
        value -= 2.0 * M_PI;
    }
    while (value < -M_PI) {
        value += 2.0 * M_PI;
    }
    return value;
}

static int pitch_shift_process(
    const double *input,
    int frames,
    double ratio,
    node_state_t *state,
    double *output
) {
    if (input == NULL || output == NULL || state == NULL) {
        return -1;
    }
    int window_size = state->u.pitch_shift.window_size;
    int hop_size = state->u.pitch_shift.hop_size;
    int resynthesis_hop = state->u.pitch_shift.resynthesis_hop;
    if (window_size <= 0 || hop_size <= 0) {
        return -1;
    }
    if (frames <= 0) {
        return 0;
    }

    if (ratio < 0.125) {
        ratio = 0.125;
    } else if (ratio > 8.0) {
        ratio = 8.0;
    }
    double time_scale = 1.0 / ratio;
    if (time_scale <= 0.0) {
        time_scale = 1.0;
    }
    double synthesis_hop = (double)resynthesis_hop * time_scale;
    if (synthesis_hop < 1e-6) {
        synthesis_hop = 1.0;
    }

    int padded_len = frames + window_size;
    int frame_count = 1;
    if (hop_size > 0) {
        frame_count = (frames + hop_size - 1) / hop_size + 1;
    }
    if (frame_count < 1) {
        frame_count = 1;
    }

    double *analysis_real = (double *)malloc((size_t)window_size * sizeof(double));
    double *analysis_imag = (double *)malloc((size_t)window_size * sizeof(double));
    double *synth_real = (double *)malloc((size_t)window_size * sizeof(double));
    double *synth_imag = (double *)malloc((size_t)window_size * sizeof(double));
    double *ifft_real = (double *)malloc((size_t)window_size * sizeof(double));
    double *ifft_imag = (double *)malloc((size_t)window_size * sizeof(double));
    double *frame_buffer = (double *)malloc((size_t)window_size * sizeof(double));
    double *input_padded = (double *)calloc((size_t)padded_len, sizeof(double));
    if (analysis_real == NULL || analysis_imag == NULL || synth_real == NULL || synth_imag == NULL
        || ifft_real == NULL || ifft_imag == NULL || frame_buffer == NULL || input_padded == NULL) {
        free(analysis_real);
        free(analysis_imag);
        free(synth_real);
        free(synth_imag);
        free(ifft_real);
        free(ifft_imag);
        free(frame_buffer);
        free(input_padded);
        return -1;
    }

    for (int i = 0; i < frames; ++i) {
        input_padded[i] = input[i];
    }

    double last_position = synthesis_hop * (double)(frame_count - 1);
    int stretched_len = (int)ceil(last_position + (double)window_size + 2.0);
    if (stretched_len < window_size) {
        stretched_len = window_size;
    }
    double *stretched = (double *)calloc((size_t)stretched_len, sizeof(double));
    if (stretched == NULL) {
        free(analysis_real);
        free(analysis_imag);
        free(synth_real);
        free(synth_imag);
        free(ifft_real);
        free(ifft_imag);
        free(frame_buffer);
        free(input_padded);
        return -1;
    }

    memset(state->u.pitch_shift.prev_phase, 0, (size_t)window_size * sizeof(double));
    memset(state->u.pitch_shift.phase_accum, 0, (size_t)window_size * sizeof(double));

    double out_pos = 0.0;
    for (int frame = 0; frame < frame_count; ++frame) {
        int start = frame * hop_size;
        for (int i = 0; i < window_size; ++i) {
            int idx = start + i;
            double sample = 0.0;
            if (idx >= 0 && idx < padded_len) {
                sample = input_padded[idx];
            }
            frame_buffer[i] = sample * state->u.pitch_shift.analysis_window[i];
        }

        fft_backend_transform(frame_buffer, NULL, analysis_real, analysis_imag, window_size, 0);

        for (int bin = 0; bin < window_size; ++bin) {
            double real = analysis_real[bin];
            double imag = analysis_imag[bin];
            double magnitude = sqrt(real * real + imag * imag);
            double phase = atan2(imag, real);
            double omega = 2.0 * M_PI * (double)bin * (double)hop_size / (double)window_size;
            double delta = wrap_phase(phase - state->u.pitch_shift.prev_phase[bin] - omega);
            double true_phase = omega + delta;
            state->u.pitch_shift.phase_accum[bin] += true_phase * time_scale;
            state->u.pitch_shift.prev_phase[bin] = phase;
            synth_real[bin] = magnitude * cos(state->u.pitch_shift.phase_accum[bin]);
            synth_imag[bin] = magnitude * sin(state->u.pitch_shift.phase_accum[bin]);
        }

        fft_backend_transform(synth_real, synth_imag, ifft_real, ifft_imag, window_size, 1);

        int pos_floor = (int)floor(out_pos);
        double pos_frac = out_pos - (double)pos_floor;
        for (int i = 0; i < window_size; ++i) {
            double sample = ifft_real[i] * state->u.pitch_shift.synthesis_window[i];
            int idx0 = pos_floor + i;
            int idx1 = idx0 + 1;
            if (idx0 >= 0 && idx0 < stretched_len) {
                stretched[idx0] += sample * (1.0 - pos_frac);
            }
            if (idx1 >= 0 && idx1 < stretched_len) {
                stretched[idx1] += sample * pos_frac;
            }
        }
        out_pos += synthesis_hop;
    }

    double input_energy = 0.0;
    for (int i = 0; i < frames; ++i) {
        input_energy += input[i] * input[i];
    }

    if (frames == 1) {
        output[0] = stretched_len > 0 ? stretched[0] : 0.0;
    } else {
        double scale = (double)(stretched_len - 1) / (double)(frames - 1);
        for (int i = 0; i < frames; ++i) {
            double pos = (double)i * scale;
            int base = (int)floor(pos);
            double frac = pos - (double)base;
            if (base < 0) {
                base = 0;
                frac = 0.0;
            }
            if (base >= stretched_len - 1) {
                base = stretched_len - 1;
                frac = 0.0;
            }
            double sample_a = stretched[base];
            double sample_b = stretched_len > base + 1 ? stretched[base + 1] : stretched[base];
            output[i] = sample_a + (sample_b - sample_a) * frac;
        }
    }

    double output_energy = 0.0;
    for (int i = 0; i < frames; ++i) {
        output_energy += output[i] * output[i];
    }
    if (output_energy > 1e-12 && input_energy > 1e-12) {
        double gain = sqrt(input_energy / output_energy);
        for (int i = 0; i < frames; ++i) {
            output[i] *= gain;
        }
    }

    free(analysis_real);
    free(analysis_imag);
    free(synth_real);
    free(synth_imag);
    free(ifft_real);
    free(ifft_imag);
    free(frame_buffer);
    free(input_padded);
    free(stretched);
    return 0;
}

static int run_pitch_shift_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    (void)sample_rate;
    if (descriptor == NULL || inputs == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }

    int window_size = json_get_int(descriptor->params_json, descriptor->params_len, "window_size", PITCH_SHIFT_DEFAULT_WINDOW);
    int hop_size = json_get_int(descriptor->params_json, descriptor->params_len, "hop_size", PITCH_SHIFT_DEFAULT_HOP);
    int resynthesis_hop = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "resynthesis_hop",
        PITCH_SHIFT_DEFAULT_RESYNTH_HOP
    );

    if (ensure_pitch_shift_state(state, window_size, hop_size, resynthesis_hop) != 0) {
        return -1;
    }

    int B = batches > 0 ? batches : 1;
    int channels = (int)inputs->audio.channels;
    if (channels <= 0) {
        channels = 1;
    }
    int input_frames = frames > 0 ? frames : 1;
    if (inputs->audio.frames > 0) {
        input_frames = (int)inputs->audio.frames;
    }
    if (input_frames <= 0) {
        input_frames = 1;
    }

    size_t total = (size_t)B * (size_t)channels * (size_t)input_frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }

    const EdgeRunnerParamView *ratio_view = find_param(inputs, "ratio");
    const EdgeRunnerParamView *semitone_view = find_param(inputs, "semitones");
    double ratio_default = json_get_double(descriptor->params_json, descriptor->params_len, "ratio", 1.0);
    double semitone_default = json_get_double(descriptor->params_json, descriptor->params_len, "semitones", 0.0);
    double base_ratio = ratio_default;
    if (semitone_default != 0.0) {
        base_ratio *= pow(2.0, semitone_default / 12.0);
    }
    if (base_ratio <= 0.0) {
        base_ratio = 1.0;
    }

    double *owned_ratio = NULL;
    double *owned_semitone = NULL;
    const double *ratio_plane = NULL;
    const double *semitone_plane = NULL;
    if (ratio_view != NULL) {
        ratio_plane = ensure_param_plane(ratio_view, B, input_frames, base_ratio, &owned_ratio);
    }
    if (semitone_view != NULL) {
        semitone_plane = ensure_param_plane(semitone_view, B, input_frames, 0.0, &owned_semitone);
    }

    if (!inputs->audio.has_audio || inputs->audio.data == NULL) {
        memset(buffer, 0, total * sizeof(double));
        *out_buffer = buffer;
        *out_channels = channels;
        free(owned_ratio);
        free(owned_semitone);
        return 0;
    }

    const double *audio = inputs->audio.data;
    for (int b = 0; b < B; ++b) {
        double ratio_value = base_ratio;
        size_t idx_base = (size_t)b * (size_t)input_frames;
        if (semitone_plane != NULL) {
            size_t idx = idx_base + (size_t)(input_frames > 0 ? input_frames - 1 : 0);
            ratio_value = pow(2.0, semitone_plane[idx] / 12.0);
        } else if (ratio_plane != NULL) {
            size_t idx = idx_base + (size_t)(input_frames > 0 ? input_frames - 1 : 0);
            ratio_value = ratio_plane[idx];
        }
        if (ratio_value <= 0.0) {
            ratio_value = base_ratio;
        }
        for (int c = 0; c < channels; ++c) {
            size_t base = ((size_t)b * (size_t)channels + (size_t)c) * (size_t)input_frames;
            const double *in_ptr = audio + base;
            double *out_ptr = buffer + base;
            if (pitch_shift_process(in_ptr, input_frames, ratio_value, state, out_ptr) != 0) {
                free(buffer);
                free(owned_ratio);
                free(owned_semitone);
                return -1;
            }
        }
    }

    free(owned_ratio);
    free(owned_semitone);
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

static int run_gain_node(
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels
) {
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    int channels = (int)inputs->audio.channels;
    if (!inputs->audio.has_audio || inputs->audio.data == NULL || channels <= 0) {
        channels = channels > 0 ? channels : 1;
        size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
        double *buffer = (double *)calloc(total, sizeof(double));
        if (buffer == NULL) {
            return -1;
        }
        *out_buffer = buffer;
        *out_channels = channels;
        return 0;
    }
    if (channels <= 0) {
        channels = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        return -1;
    }
    const double *audio = inputs->audio.data;
    const EdgeRunnerParamView *gain_view = find_param(inputs, "gain");
    const double *gain = (gain_view != NULL) ? gain_view->data : NULL;
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            size_t base = ((size_t)b * (size_t)channels + (size_t)c) * (size_t)frames;
            for (int f = 0; f < frames; ++f) {
                double sample = audio[base + (size_t)f];
                double g = gain != NULL ? gain[base + (size_t)f] : 1.0;
                buffer[base + (size_t)f] = sample * g;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

static int run_fft_division_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state,
    AmpNodeMetrics *metrics
) {
    if (descriptor == NULL || inputs == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    if (batches <= 0) {
        batches = 1;
    }
    int input_channels = channels;
    if (input_channels <= 0) {
        if (inputs->audio.channels > 0U) {
            input_channels = (int)inputs->audio.channels;
        } else {
            input_channels = 1;
        }
    }
    int slot_count = batches * input_channels;
    if (slot_count <= 0) {
        slot_count = 1;
    }

    int window_size = json_get_int(descriptor->params_json, descriptor->params_len, "window_size", 8);
    if (window_size <= 0) {
        window_size = 1;
    }
    double epsilon_default = json_get_double(descriptor->params_json, descriptor->params_len, "stabilizer", 1e-9);
    double epsilon_json = json_get_double(descriptor->params_json, descriptor->params_len, "epsilon", epsilon_default);
    if (epsilon_json <= 0.0) {
        epsilon_json = 1e-9;
    }
    int enable_remainder = json_get_bool(
        descriptor->params_json,
        descriptor->params_len,
        "enable_remainder",
        1
    );

    int default_algorithm = parse_algorithm_string(descriptor->params_json, descriptor->params_len, FFT_ALGORITHM_EIGEN);
    default_algorithm = clamp_algorithm_kind(default_algorithm);
    int default_window_kind = parse_window_string(descriptor->params_json, descriptor->params_len, FFT_WINDOW_HANN);
    default_window_kind = clamp_window_kind(default_window_kind);

    if (ensure_fft_state_buffers(state, slot_count, window_size) != 0) {
        return -1;
    }
    state->u.fftdiv.batches = batches;
    state->u.fftdiv.channels = input_channels;
    state->u.fftdiv.algorithm = default_algorithm;
    state->u.fftdiv.epsilon = epsilon_json;
    state->u.fftdiv.enable_remainder = enable_remainder ? 1 : 0;
    if (state->u.fftdiv.window_kind != default_window_kind) {
        fill_window_weights(state->u.fftdiv.window, window_size, default_window_kind);
        state->u.fftdiv.window_kind = default_window_kind;
    }

    const EdgeRunnerParamView *divisor_view = find_param(inputs, "divisor");
    const EdgeRunnerParamView *divisor_imag_view = find_param(inputs, "divisor_imag");
    const EdgeRunnerParamView *algorithm_view = find_param(inputs, "algorithm_selector");
    const EdgeRunnerParamView *window_view = find_param(inputs, "window_selector");
    const EdgeRunnerParamView *stabilizer_view = find_param(inputs, "stabilizer");
    const EdgeRunnerParamView *phase_view = find_param(inputs, "phase_offset");
    const EdgeRunnerParamView *lower_view = find_param(inputs, "lower_band");
    const EdgeRunnerParamView *upper_view = find_param(inputs, "upper_band");
    const EdgeRunnerParamView *filter_view = find_param(inputs, "filter_intensity");
    const EdgeRunnerParamView *carrier_views[FFT_DYNAMIC_CARRIER_LIMIT];
    uint32_t carrier_view_count = collect_dynamic_carrier_views(inputs, carrier_views, FFT_DYNAMIC_CARRIER_LIMIT);

    size_t total_samples = (size_t)slot_count * (size_t)frames;
    double *buffer = (double *)malloc(total_samples * sizeof(double));
    amp_last_alloc_count = total_samples;
    if (buffer == NULL) {
        return -1;
    }

    const double *audio_base = (inputs->audio.has_audio && inputs->audio.data != NULL) ? inputs->audio.data : NULL;
    size_t divisor_total = param_total_count(divisor_view);
    size_t divisor_imag_total = param_total_count(divisor_imag_view);
    size_t algorithm_total = param_total_count(algorithm_view);
    size_t window_total = param_total_count(window_view);
    size_t stabilizer_total = param_total_count(stabilizer_view);
    size_t phase_total = param_total_count(phase_view);
    size_t lower_total = param_total_count(lower_view);
    size_t upper_total = param_total_count(upper_view);
    size_t filter_total = param_total_count(filter_view);

    fft_dynamic_carrier_summary_t carrier_summary = summarize_dynamic_carriers(inputs);

    for (int frame_index = 0; frame_index < frames; ++frame_index) {
        size_t base_index = (size_t)frame_index * (size_t)slot_count;
        double epsilon_frame = epsilon_json;
        if (stabilizer_total > 0U) {
            double candidate = read_param_value(stabilizer_view, base_index, epsilon_frame);
            if (candidate < 0.0) {
                candidate = -candidate;
            }
            if (candidate > 0.0) {
                epsilon_frame = candidate;
            }
        }
        if (epsilon_frame < 1e-12) {
            epsilon_frame = 1e-12;
        }
        int algorithm_kind = default_algorithm;
        if (algorithm_total > 0U) {
            double raw = read_param_value(algorithm_view, base_index, (double)algorithm_kind);
            algorithm_kind = clamp_algorithm_kind(round_to_int(raw));
        }
        const fft_algorithm_class_t *algorithm_class = select_fft_algorithm(algorithm_kind);
        if (algorithm_class == NULL) {
            free(buffer);
            return AMP_E_UNSUPPORTED;
        }
        if (algorithm_class->requires_hook && !amp_fft_backend_has_hook()) {
            free(buffer);
            return AMP_E_UNSUPPORTED;
        }
        if (algorithm_class->requires_power_of_two && !is_power_of_two_int(window_size)) {
            algorithm_class = select_fft_algorithm(FFT_ALGORITHM_DFT);
            if (algorithm_class == NULL) {
                free(buffer);
                return AMP_E_UNSUPPORTED;
            }
        }
        algorithm_kind = algorithm_class->kind;
        int window_kind = default_window_kind;
        if (window_total > 0U) {
            double raw_w = read_param_value(window_view, base_index, (double)window_kind);
            window_kind = clamp_window_kind(round_to_int(raw_w));
        }
        if (state->u.fftdiv.window_kind != window_kind) {
            fill_window_weights(state->u.fftdiv.window, window_size, window_kind);
            state->u.fftdiv.window_kind = window_kind;
        }
        state->u.fftdiv.algorithm = algorithm_kind;
        state->u.fftdiv.epsilon = epsilon_frame;
        if (algorithm_kind == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            state->u.fftdiv.dynamic_carrier_band_count = 0;
            state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
            state->u.fftdiv.dynamic_k_active = 0;
        } else if (algorithm_class->supports_dynamic_carriers) {
            state->u.fftdiv.dynamic_carrier_band_count = (int)carrier_summary.band_count;
            state->u.fftdiv.dynamic_carrier_last_sum = carrier_summary.last_sum;
            state->u.fftdiv.dynamic_k_active = (int)carrier_summary.band_count;
        } else {
            state->u.fftdiv.dynamic_carrier_band_count = 0;
            state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
            state->u.fftdiv.dynamic_k_active = 0;
        }

        int filled = state->u.fftdiv.filled;
        double phase_last = state->u.fftdiv.last_phase;
        double lower_last = state->u.fftdiv.last_lower;
        double upper_last = state->u.fftdiv.last_upper;
        double filter_last = state->u.fftdiv.last_filter;
        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            double sample = 0.0;
            if (audio_base != NULL) {
                sample = audio_base[data_idx];
            }
            double divisor_sample = divisor_total > 0U ? read_param_value(divisor_view, base_index + (size_t)slot, 1.0) : 1.0;
            double divisor_imag_sample = divisor_imag_total > 0U ? read_param_value(divisor_imag_view, base_index + (size_t)slot, 0.0) : 0.0;
            double phase_sample = phase_total > 0U ? read_param_value(phase_view, base_index + (size_t)slot, phase_last) : phase_last;
            double lower_sample = lower_total > 0U ? read_param_value(lower_view, base_index + (size_t)slot, lower_last) : lower_last;
            double upper_sample = upper_total > 0U ? read_param_value(upper_view, base_index + (size_t)slot, upper_last) : upper_last;
            double filter_sample = filter_total > 0U ? read_param_value(filter_view, base_index + (size_t)slot, filter_last) : filter_last;
            size_t offset = (size_t)slot * (size_t)window_size;
            double *input_ring = state->u.fftdiv.input_buffer + offset;
            double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
            double *divisor_imag_ring = state->u.fftdiv.divisor_imag_buffer + offset;
            double *phase_ring = state->u.fftdiv.phase_buffer + offset;
            double *lower_ring = state->u.fftdiv.lower_buffer + offset;
            double *upper_ring = state->u.fftdiv.upper_buffer + offset;
            double *filter_ring = state->u.fftdiv.filter_buffer + offset;
            if (filled < window_size) {
                input_ring[filled] = sample;
                divisor_ring[filled] = divisor_sample;
                divisor_imag_ring[filled] = divisor_imag_sample;
                phase_ring[filled] = phase_sample;
                lower_ring[filled] = lower_sample;
                upper_ring[filled] = upper_sample;
                filter_ring[filled] = filter_sample;
            } else {
                if (window_size > 1) {
                    memmove(input_ring, input_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(divisor_ring, divisor_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(divisor_imag_ring, divisor_imag_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(phase_ring, phase_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(lower_ring, lower_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(upper_ring, upper_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(filter_ring, filter_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                }
                input_ring[window_size - 1] = sample;
                divisor_ring[window_size - 1] = divisor_sample;
                divisor_imag_ring[window_size - 1] = divisor_imag_sample;
                phase_ring[window_size - 1] = phase_sample;
                lower_ring[window_size - 1] = lower_sample;
                upper_ring[window_size - 1] = upper_sample;
                filter_ring[window_size - 1] = filter_sample;
            }
            phase_last = phase_sample;
            lower_last = lower_sample;
            upper_last = upper_sample;
            filter_last = filter_sample;
        }
        state->u.fftdiv.last_phase = phase_last;
        state->u.fftdiv.last_lower = lower_last;
        state->u.fftdiv.last_upper = upper_last;
        state->u.fftdiv.last_filter = filter_last;
        if (filled < window_size) {
            filled += 1;
            state->u.fftdiv.filled = filled;
        }
        if (state->u.fftdiv.filled < window_size) {
            for (int slot = 0; slot < slot_count; ++slot) {
                double divisor_sample = divisor_total > 0U ? read_param_value(divisor_view, base_index + (size_t)slot, 1.0) : 1.0;
                double safe_div = fabs(divisor_sample) < epsilon_frame ? (divisor_sample >= 0.0 ? epsilon_frame : -epsilon_frame) : divisor_sample;
                double sample = audio_base != NULL ? audio_base[base_index + (size_t)slot] : 0.0;
                buffer[base_index + (size_t)slot] = sample / safe_div;
            }
            continue;
        }
        double *work_real = state->u.fftdiv.work_real;
        double *work_imag = state->u.fftdiv.work_imag;
        double *div_real = state->u.fftdiv.div_real;
        double *div_imag = state->u.fftdiv.div_imag;
        double *ifft_real = state->u.fftdiv.ifft_real;
        double *ifft_imag = state->u.fftdiv.ifft_imag;
        double *window_weights = state->u.fftdiv.window;
        int dynamic_active_frame = 0;
        double dynamic_sum_frame = 0.0;
        state->u.fftdiv.dynamic_k_active = 0;
        state->u.fftdiv.remainder_energy = 0.0;
        for (int slot = 0; slot < slot_count; ++slot) {
            size_t offset = (size_t)slot * (size_t)window_size;
            double *input_ring = state->u.fftdiv.input_buffer + offset;
            double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
            double *divisor_imag_ring = state->u.fftdiv.divisor_imag_buffer + offset;
            double *phase_ring = state->u.fftdiv.phase_buffer + offset;
            double *lower_ring = state->u.fftdiv.lower_buffer + offset;
            double *upper_ring = state->u.fftdiv.upper_buffer + offset;
            double *filter_ring = state->u.fftdiv.filter_buffer + offset;
            double phase_mod = phase_ring[window_size > 0 ? window_size - 1 : 0];
            double lower_mod = lower_ring[window_size > 0 ? window_size - 1 : 0];
            double upper_mod = upper_ring[window_size > 0 ? window_size - 1 : 0];
            double filter_mod = filter_ring[window_size > 0 ? window_size - 1 : 0];
            double lower_clamped = clamp_unit_double(lower_mod);
            double upper_clamped = clamp_unit_double(upper_mod);
            if (upper_clamped < lower_clamped) {
                double tmp_bounds = lower_clamped;
                lower_clamped = upper_clamped;
                upper_clamped = tmp_bounds;
            }
            double intensity_clamped = clamp_unit_double(filter_mod);
            double cos_phase = cos(phase_mod);
            double sin_phase = sin(phase_mod);

            if (algorithm_kind == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
                double carrier_fnorm[FFT_DYNAMIC_CARRIER_LIMIT];
                int carrier_indices_local[FFT_DYNAMIC_CARRIER_LIMIT];
                double theta0[FFT_DYNAMIC_CARRIER_LIMIT];
                double step_re_local[FFT_DYNAMIC_CARRIER_LIMIT];
                double step_im_local[FFT_DYNAMIC_CARRIER_LIMIT];
                double b_re[FFT_DYNAMIC_CARRIER_LIMIT];
                double b_im[FFT_DYNAMIC_CARRIER_LIMIT];
                double div_re_acc[FFT_DYNAMIC_CARRIER_LIMIT];
                double div_im_acc[FFT_DYNAMIC_CARRIER_LIMIT];
                double last_re[FFT_DYNAMIC_CARRIER_LIMIT];
                double last_im[FFT_DYNAMIC_CARRIER_LIMIT];
                double gram_real[FFT_DYNAMIC_CARRIER_LIMIT * FFT_DYNAMIC_CARRIER_LIMIT];
                double gram_imag[FFT_DYNAMIC_CARRIER_LIMIT * FFT_DYNAMIC_CARRIER_LIMIT];
                double amp_re[FFT_DYNAMIC_CARRIER_LIMIT];
                double amp_im[FFT_DYNAMIC_CARRIER_LIMIT];
                double ph_re_buffer[FFT_DYNAMIC_CARRIER_LIMIT];
                double ph_im_buffer[FFT_DYNAMIC_CARRIER_LIMIT];
                int active_count = 0;
                double carrier_sum_slot = 0.0;
                for (uint32_t idx = 0; idx < carrier_view_count && idx < FFT_DYNAMIC_CARRIER_LIMIT; ++idx) {
                    const EdgeRunnerParamView *carrier_view = carrier_views[idx];
                    if (carrier_view == NULL) {
                        continue;
                    }
                    size_t total = param_total_count(carrier_view);
                    if (total == 0U) {
                        continue;
                    }
                    double raw_value = read_param_value(
                        carrier_view,
                        base_index + (size_t)slot,
                        0.0
                    );
                    carrier_sum_slot += raw_value;
                    double normalized = raw_value;
                    if (sample_rate > 0.0 && fabs(normalized) > 1.0) {
                        normalized = raw_value / sample_rate;
                    }
                    normalized = clamp_unit_double(normalized);
                    carrier_fnorm[active_count] = normalized;
                    carrier_indices_local[active_count] = (int)idx;
                    active_count += 1;
                }
                double *remainder_ring = state->u.fftdiv.result_buffer + offset;
                for (int i = 0; i < window_size; ++i) {
                    state->u.fftdiv.div_fft_real[offset + (size_t)i] = 0.0;
                    state->u.fftdiv.div_fft_imag[offset + (size_t)i] = 0.0;
                    remainder_ring[i] = 0.0;
                }
                if (active_count > 0) {
                    size_t phase_offset = (size_t)slot * (size_t)FFT_DYNAMIC_CARRIER_LIMIT;
                    double inv_window = window_size > 0 ? 1.0 / (double)window_size : 1.0;
                    int has_divisor = (divisor_total > 0U) || (divisor_imag_total > 0U);
                    for (int k = 0; k < active_count; ++k) {
                        size_t carrier_index = (size_t)carrier_indices_local[k];
                        size_t phase_index = phase_offset + carrier_index;
                        double theta = state->u.fftdiv.dynamic_phase[phase_index];
                        theta0[k] = theta;
                        double fnorm = carrier_fnorm[k];
                        double angle = 2.0 * M_PI * fnorm;
                        double step_re = cos(angle);
                        double step_im = sin(angle);
                        step_re_local[k] = step_re;
                        step_im_local[k] = step_im;
                        state->u.fftdiv.dynamic_step_re[phase_index] = step_re;
                        state->u.fftdiv.dynamic_step_im[phase_index] = step_im;
                        ph_re_buffer[k] = cos(theta);
                        ph_im_buffer[k] = sin(theta);
                        b_re[k] = 0.0;
                        b_im[k] = 0.0;
                        div_re_acc[k] = 0.0;
                        div_im_acc[k] = 0.0;
                        last_re[k] = ph_re_buffer[k];
                        last_im[k] = ph_im_buffer[k];
                        for (int j = 0; j < active_count; ++j) {
                            gram_real[k * active_count + j] = 0.0;
                            gram_imag[k * active_count + j] = 0.0;
                        }
                    }
                    double phi_re_local[FFT_DYNAMIC_CARRIER_LIMIT];
                    double phi_im_local[FFT_DYNAMIC_CARRIER_LIMIT];
                    for (int n = 0; n < window_size; ++n) {
                        double w = window_weights != NULL ? window_weights[n] : 1.0;
                        double xw = input_ring[n] * w;
                        for (int k = 0; k < active_count; ++k) {
                            double ph_re = ph_re_buffer[k];
                            double ph_im = ph_im_buffer[k];
                            double wr = ph_re * w;
                            double wi = ph_im * w;
                            phi_re_local[k] = wr;
                            phi_im_local[k] = wi;
                            b_re[k] += xw * wr;
                            b_im[k] -= xw * wi;
                            if (has_divisor) {
                                double dw_re = divisor_ring[n] * w;
                                double dw_im = divisor_imag_ring[n] * w;
                                div_re_acc[k] += dw_re * ph_re + dw_im * ph_im;
                                div_im_acc[k] += -dw_re * ph_im + dw_im * ph_re;
                            }
                            if (n == window_size - 1) {
                                last_re[k] = ph_re;
                                last_im[k] = ph_im;
                            }
                        }
                        for (int i_local = 0; i_local < active_count; ++i_local) {
                            for (int j_local = i_local; j_local < active_count; ++j_local) {
                                double a_re = phi_re_local[i_local];
                                double a_im = phi_im_local[i_local];
                                double b_re_local = phi_re_local[j_local];
                                double b_im_local = phi_im_local[j_local];
                                double re = a_re * b_re_local + a_im * b_im_local;
                                double im = a_re * b_im_local - a_im * b_re_local;
                                gram_real[i_local * active_count + j_local] += re;
                                gram_imag[i_local * active_count + j_local] += im;
                                if (i_local != j_local) {
                                    gram_real[j_local * active_count + i_local] += re;
                                    gram_imag[j_local * active_count + i_local] -= im;
                                }
                            }
                        }
                        for (int k = 0; k < active_count; ++k) {
                            double next_re = ph_re_buffer[k] * step_re_local[k] - ph_im_buffer[k] * step_im_local[k];
                            double next_im = ph_re_buffer[k] * step_im_local[k] + ph_im_buffer[k] * step_re_local[k];
                            ph_re_buffer[k] = next_re;
                            ph_im_buffer[k] = next_im;
                        }
                    }
                    for (int k = 0; k < active_count; ++k) {
                        size_t carrier_index = (size_t)carrier_indices_local[k];
                        size_t phase_index = phase_offset + carrier_index;
                        double theta_next = theta0[k] + 2.0 * M_PI * carrier_fnorm[k] * (double)window_size;
                        state->u.fftdiv.dynamic_phase[phase_index] = wrap_phase_two_pi(theta_next);
                        if (!has_divisor) {
                            div_re_acc[k] = 1.0;
                            div_im_acc[k] = 0.0;
                        }
                    }
                    double coeff_re_local[FFT_DYNAMIC_CARRIER_LIMIT];
                    double coeff_im_local[FFT_DYNAMIC_CARRIER_LIMIT];
                    if (state->u.fftdiv.enable_remainder) {
                        int dim = active_count * 2;
                        double normal_matrix[FFT_DYNAMIC_CARRIER_LIMIT * 2 * FFT_DYNAMIC_CARRIER_LIMIT * 2];
                        double rhs_vec[FFT_DYNAMIC_CARRIER_LIMIT * 2];
                        for (int i_local = 0; i_local < dim * dim; ++i_local) {
                            normal_matrix[i_local] = 0.0;
                        }
                        for (int i_local = 0; i_local < dim; ++i_local) {
                            rhs_vec[i_local] = 0.0;
                        }
                        double trace = 0.0;
                        for (int k = 0; k < active_count; ++k) {
                            trace += gram_real[k * active_count + k];
                        }
                        double lambda = 0.0;
                        if (active_count > 0 && trace > 0.0) {
                            lambda = 1e-8 * (trace / (double)active_count);
                        }
                        for (int i_local = 0; i_local < active_count; ++i_local) {
                            for (int j_local = 0; j_local < active_count; ++j_local) {
                                double real_part = gram_real[i_local * active_count + j_local];
                                double imag_part = gram_imag[i_local * active_count + j_local];
                                if (i_local == j_local) {
                                    real_part += lambda;
                                }
                                normal_matrix[i_local * dim + j_local] = real_part;
                                normal_matrix[i_local * dim + (j_local + active_count)] = -imag_part;
                                normal_matrix[(i_local + active_count) * dim + j_local] = imag_part;
                                normal_matrix[(i_local + active_count) * dim + (j_local + active_count)] = real_part;
                            }
                            rhs_vec[i_local] = b_re[i_local];
                            rhs_vec[i_local + active_count] = b_im[i_local];
                        }
                        if (solve_linear_system(normal_matrix, rhs_vec, dim) == 0) {
                            for (int k = 0; k < active_count; ++k) {
                                coeff_re_local[k] = rhs_vec[k];
                                coeff_im_local[k] = rhs_vec[k + active_count];
                            }
                        } else {
                            for (int k = 0; k < active_count; ++k) {
                                coeff_re_local[k] = b_re[k];
                                coeff_im_local[k] = b_im[k];
                            }
                        }
                    } else {
                        for (int k = 0; k < active_count; ++k) {
                            coeff_re_local[k] = b_re[k];
                            coeff_im_local[k] = b_im[k];
                        }
                    }
                    double sample_dynamic = 0.0;
                    for (int k = 0; k < active_count; ++k) {
                        double coeff_re = coeff_re_local[k];
                        double coeff_im = coeff_im_local[k];
                        double div_re = div_re_acc[k];
                        double div_im = div_im_acc[k];
                        if (has_divisor) {
                            double denom = div_re * div_re + div_im * div_im;
                            if (denom < epsilon_frame) {
                                denom = epsilon_frame;
                            }
                            double real_tmp = (coeff_re * div_re + coeff_im * div_im) / denom;
                            double imag_tmp = (coeff_im * div_re - coeff_re * div_im) / denom;
                            coeff_re = real_tmp;
                            coeff_im = imag_tmp;
                        }
                        double ratio = carrier_fnorm[k];
                        if (ratio < 0.0) {
                            ratio = 0.0;
                        } else if (ratio > 1.0) {
                            ratio = 1.0;
                        }
                        double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
                        double gated_re = coeff_re * gain;
                        double gated_im = coeff_im * gain;
                        double rotated_re = gated_re * cos_phase - gated_im * sin_phase;
                        double rotated_im = gated_re * sin_phase + gated_im * cos_phase;
                        amp_re[k] = rotated_re;
                        amp_im[k] = rotated_im;
                        sample_dynamic += (rotated_re * last_re[k] - rotated_im * last_im[k]) * inv_window;
                    }
                    if (state->u.fftdiv.enable_remainder) {
                        double *modeled_ring = state->u.fftdiv.work_real;
                        for (int i = 0; i < window_size; ++i) {
                            modeled_ring[i] = 0.0;
                        }
                        for (int k = 0; k < active_count; ++k) {
                            ph_re_buffer[k] = cos(theta0[k]);
                            ph_im_buffer[k] = sin(theta0[k]);
                        }
                        double remainder_energy_local = 0.0;
                        for (int n = 0; n < window_size; ++n) {
                            double sum_val = 0.0;
                            for (int k = 0; k < active_count; ++k) {
                                sum_val += amp_re[k] * ph_re_buffer[k] - amp_im[k] * ph_im_buffer[k];
                            }
                            modeled_ring[n] = sum_val * inv_window;
                            double residual_val = input_ring[n] - modeled_ring[n];
                            remainder_ring[n] = residual_val;
                            remainder_energy_local += residual_val * residual_val;
                            for (int k = 0; k < active_count; ++k) {
                                double next_re = ph_re_buffer[k] * step_re_local[k] - ph_im_buffer[k] * step_im_local[k];
                                double next_im = ph_re_buffer[k] * step_im_local[k] + ph_im_buffer[k] * step_re_local[k];
                                ph_re_buffer[k] = next_re;
                                ph_im_buffer[k] = next_im;
                            }
                        }
                        state->u.fftdiv.remainder_energy += remainder_energy_local;
                    } else {
                        size_t tail_index = (size_t)(window_size > 0 ? window_size - 1 : 0);
                        remainder_ring[tail_index] = sample_dynamic;
                    }
                    buffer[base_index + (size_t)slot] = sample_dynamic;
                    if (active_count > dynamic_active_frame) {
                        dynamic_active_frame = active_count;
                    }
                    dynamic_sum_frame += carrier_sum_slot;
                    continue;
                }
                buffer[base_index + (size_t)slot] = 0.0;
                continue;
            }

            for (int i = 0; i < window_size; ++i) {
                double w = window_weights != NULL ? window_weights[i] : 1.0;
                work_real[i] = input_ring[i] * w;
                work_imag[i] = 0.0;
                div_real[i] = divisor_ring[i] * w;
                div_imag[i] = divisor_imag_ring[i] * w;
            }
            algorithm_class->forward(work_real, work_imag, work_real, work_imag, window_size);
            algorithm_class->forward(div_real, div_imag, div_real, div_imag, window_size);
            for (int i = 0; i < window_size; ++i) {
                double a = work_real[i];
                double b = work_imag[i];
                double c = div_real[i];
                double d = div_imag[i];
                state->u.fftdiv.div_fft_real[offset + (size_t)i] = c;
                state->u.fftdiv.div_fft_imag[offset + (size_t)i] = d;
                double denom = c * c + d * d;
                if (denom < epsilon_frame) {
                    denom = epsilon_frame;
                }
                double real = (a * c + b * d) / denom;
                double imag = (b * c - a * d) / denom;
                double ratio = window_size > 1 ? (double)i / (double)(window_size - 1) : 0.0;
                double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
                double gated_real = real * gain;
                double gated_imag = imag * gain;
                double rotated_real = gated_real * cos_phase - gated_imag * sin_phase;
                double rotated_imag = gated_real * sin_phase + gated_imag * cos_phase;
                work_real[i] = rotated_real;
                work_imag[i] = rotated_imag;
            }
            algorithm_class->inverse(work_real, work_imag, ifft_real, ifft_imag, window_size);
            for (int i = 0; i < window_size; ++i) {
                state->u.fftdiv.result_buffer[offset + (size_t)i] = ifft_real[i];
            }
            buffer[base_index + (size_t)slot] = ifft_real[window_size - 1];
        }
        if (algorithm_kind == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            state->u.fftdiv.dynamic_carrier_band_count = dynamic_active_frame;
            state->u.fftdiv.dynamic_carrier_last_sum = dynamic_sum_frame;
            state->u.fftdiv.dynamic_k_active = dynamic_active_frame;
        }
        state->u.fftdiv.position = window_size - 1;
    }

    *out_buffer = buffer;
    *out_channels = input_channels;
    if (metrics != NULL) {
        metrics->measured_delay_frames = (uint32_t)(window_size > 0 ? window_size - 1 : 0);
        double complexity = 0.0;
        if (state->u.fftdiv.algorithm == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            complexity = (double)slot_count * (double)state->u.fftdiv.dynamic_k_active * (double)window_size * 8.0;
        } else if (
            state->u.fftdiv.algorithm == FFT_ALGORITHM_EIGEN
            || state->u.fftdiv.algorithm == FFT_ALGORITHM_HOOK
        ) {
            double stages = log((double)window_size) / log(2.0);
            if (stages < 1.0) {
                stages = 1.0;
            }
            complexity = (double)slot_count * (double)window_size * (stages * 6.0 + 4.0);
        } else {
            complexity = (double)slot_count * (double)window_size * (double)window_size * 2.0;
        }
        float heat = (float)(complexity / 1000.0);
        if (heat < 0.001f) {
            heat = 0.001f;
        }
        metrics->accumulated_heat = heat;
        state->u.fftdiv.total_heat += (double)heat;
        metrics->reserved[0] = state->u.fftdiv.last_phase;
        metrics->reserved[1] = state->u.fftdiv.last_lower;
        metrics->reserved[2] = state->u.fftdiv.last_upper;
        metrics->reserved[3] = state->u.fftdiv.last_filter;
        metrics->reserved[4] = (double)window_size;
        metrics->reserved[5] = (double)state->u.fftdiv.algorithm;
    }
    return 0;
}

static int run_fft_division_node_backward(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state,
    AmpNodeMetrics *metrics
) {
    if (descriptor == NULL || inputs == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    if (batches <= 0) {
        batches = 1;
    }
    int input_channels = channels;
    if (input_channels <= 0) {
        if (inputs->audio.channels > 0U) {
            input_channels = (int)inputs->audio.channels;
        } else {
            input_channels = 1;
        }
    }
    int slot_count = batches * input_channels;
    if (slot_count <= 0) {
        slot_count = 1;
    }

    int window_size = json_get_int(descriptor->params_json, descriptor->params_len, "window_size", 8);
    if (window_size <= 0) {
        window_size = 1;
    }
    double epsilon_default = json_get_double(descriptor->params_json, descriptor->params_len, "stabilizer", 1e-9);
    double epsilon_json = json_get_double(descriptor->params_json, descriptor->params_len, "epsilon", epsilon_default);
    if (epsilon_json <= 0.0) {
        epsilon_json = 1e-9;
    }

    int default_algorithm = parse_algorithm_string(descriptor->params_json, descriptor->params_len, FFT_ALGORITHM_EIGEN);
    default_algorithm = clamp_algorithm_kind(default_algorithm);
    int default_window_kind = parse_window_string(descriptor->params_json, descriptor->params_len, FFT_WINDOW_HANN);
    default_window_kind = clamp_window_kind(default_window_kind);

    if (ensure_fft_state_buffers(state, slot_count, window_size) != 0) {
        return -1;
    }
    if (state->u.fftdiv.window_kind != default_window_kind) {
        fill_window_weights(state->u.fftdiv.window, window_size, default_window_kind);
        state->u.fftdiv.window_kind = default_window_kind;
    }

    const EdgeRunnerParamView *divisor_view = find_param(inputs, "divisor");
    const EdgeRunnerParamView *divisor_imag_view = find_param(inputs, "divisor_imag");
    const EdgeRunnerParamView *stabilizer_view = find_param(inputs, "stabilizer");
    const EdgeRunnerParamView *algorithm_view = find_param(inputs, "algorithm_selector");
    const EdgeRunnerParamView *window_view = find_param(inputs, "window_selector");
    const EdgeRunnerParamView *phase_view = find_param(inputs, "phase_offset");
    const EdgeRunnerParamView *lower_view = find_param(inputs, "lower_band");
    const EdgeRunnerParamView *upper_view = find_param(inputs, "upper_band");
    const EdgeRunnerParamView *filter_view = find_param(inputs, "filter_intensity");

    size_t divisor_total = param_total_count(divisor_view);
    size_t divisor_imag_total = param_total_count(divisor_imag_view);
    size_t stabilizer_total = param_total_count(stabilizer_view);
    size_t algorithm_total = param_total_count(algorithm_view);
    size_t window_total = param_total_count(window_view);
    size_t phase_total = param_total_count(phase_view);
    size_t lower_total = param_total_count(lower_view);
    size_t upper_total = param_total_count(upper_view);
    size_t filter_total = param_total_count(filter_view);

    const EdgeRunnerParamView *carrier_views[FFT_DYNAMIC_CARRIER_LIMIT];
    uint32_t carrier_view_count = collect_dynamic_carrier_views(inputs, carrier_views, FFT_DYNAMIC_CARRIER_LIMIT);

    size_t total_samples = (size_t)slot_count * (size_t)frames;
    double *buffer = (double *)malloc(total_samples * sizeof(double));
    amp_last_alloc_count = total_samples;
    if (buffer == NULL) {
        return -1;
    }

    const double *audio_base = (inputs->audio.has_audio && inputs->audio.data != NULL) ? inputs->audio.data : NULL;
    int recomb_filled = state->u.fftdiv.recomb_filled;
    double phase_last_state = state->u.fftdiv.last_phase;
    double lower_last_state = state->u.fftdiv.last_lower;
    double upper_last_state = state->u.fftdiv.last_upper;
    double filter_last_state = state->u.fftdiv.last_filter;

    fft_dynamic_carrier_summary_t carrier_summary = summarize_dynamic_carriers(inputs);

    for (int frame_index = 0; frame_index < frames; ++frame_index) {
        size_t base_index = (size_t)frame_index * (size_t)slot_count;
        double epsilon_frame = epsilon_json;
        if (stabilizer_total > 0U) {
            double candidate = read_param_value(stabilizer_view, base_index, epsilon_frame);
            if (candidate < 0.0) {
                candidate = -candidate;
            }
            if (candidate > 0.0) {
                epsilon_frame = candidate;
            }
        }
        if (epsilon_frame < 1e-12) {
            epsilon_frame = 1e-12;
        }
        int algorithm_kind = default_algorithm;
        if (algorithm_total > 0U) {
            double raw = read_param_value(algorithm_view, base_index, (double)algorithm_kind);
            algorithm_kind = clamp_algorithm_kind(round_to_int(raw));
        }
        const fft_algorithm_class_t *algorithm_class = select_fft_algorithm(algorithm_kind);
        if (algorithm_class == NULL) {
            free(buffer);
            return AMP_E_UNSUPPORTED;
        }
        if (algorithm_class->requires_hook && !amp_fft_backend_has_hook()) {
            free(buffer);
            return AMP_E_UNSUPPORTED;
        }
        if (algorithm_class->requires_power_of_two && !is_power_of_two_int(window_size)) {
            algorithm_class = select_fft_algorithm(FFT_ALGORITHM_DFT);
            if (algorithm_class == NULL) {
                free(buffer);
                return AMP_E_UNSUPPORTED;
            }
        }
        algorithm_kind = algorithm_class->kind;
        if (algorithm_kind == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            double *window_weights = state->u.fftdiv.window;
            double inv_window = window_size > 0 ? 1.0 / (double)window_size : 1.0;
            int dynamic_active_frame = 0;
            double dynamic_sum_frame = 0.0;
            state->u.fftdiv.dynamic_k_active = 0;
            state->u.fftdiv.algorithm = algorithm_kind;
            state->u.fftdiv.epsilon = epsilon_frame;
            for (int slot = 0; slot < slot_count; ++slot) {
                size_t offset = (size_t)slot * (size_t)window_size;
                double *recomb_ring = state->u.fftdiv.recomb_buffer + offset;
                double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
                double *divisor_imag_ring = state->u.fftdiv.divisor_imag_buffer + offset;
                double *phase_ring = state->u.fftdiv.phase_buffer + offset;
                double *lower_ring = state->u.fftdiv.lower_buffer + offset;
                double *upper_ring = state->u.fftdiv.upper_buffer + offset;
                double *filter_ring = state->u.fftdiv.filter_buffer + offset;
                double phase_mod = phase_ring[window_size > 0 ? window_size - 1 : 0];
                double lower_mod = lower_ring[window_size > 0 ? window_size - 1 : 0];
                double upper_mod = upper_ring[window_size > 0 ? window_size - 1 : 0];
                double filter_mod = filter_ring[window_size > 0 ? window_size - 1 : 0];
                double lower_clamped = clamp_unit_double(lower_mod);
                double upper_clamped = clamp_unit_double(upper_mod);
                if (upper_clamped < lower_clamped) {
                    double tmp_bounds = lower_clamped;
                    lower_clamped = upper_clamped;
                    upper_clamped = tmp_bounds;
                }
                double intensity_clamped = clamp_unit_double(filter_mod);
                double cos_phase = cos(phase_mod);
                double sin_phase = sin(phase_mod);
                int has_divisor = (divisor_total > 0U) || (divisor_imag_total > 0U);
                double carrier_fnorm[FFT_DYNAMIC_CARRIER_LIMIT];
                int carrier_indices_local[FFT_DYNAMIC_CARRIER_LIMIT];
                int active_count = 0;
                double carrier_sum_slot = 0.0;
                for (uint32_t idx = 0; idx < carrier_view_count && idx < FFT_DYNAMIC_CARRIER_LIMIT; ++idx) {
                    const EdgeRunnerParamView *carrier_view = carrier_views[idx];
                    if (carrier_view == NULL) {
                        continue;
                    }
                    size_t total = param_total_count(carrier_view);
                    if (total == 0U) {
                        continue;
                    }
                    double raw_value = read_param_value(
                        carrier_view,
                        base_index + (size_t)slot,
                        0.0
                    );
                    carrier_sum_slot += raw_value;
                    double normalized = raw_value;
                    if (sample_rate > 0.0 && fabs(normalized) > 1.0) {
                        normalized = raw_value / sample_rate;
                    }
                    normalized = clamp_unit_double(normalized);
                    carrier_fnorm[active_count] = normalized;
                    carrier_indices_local[active_count] = (int)idx;
                    active_count += 1;
                }
                if (active_count > 0) {
                    size_t phase_offset = (size_t)slot * (size_t)FFT_DYNAMIC_CARRIER_LIMIT;
                    double *phasor_re = state->u.fftdiv.work_real;
                    double *phasor_im = state->u.fftdiv.work_imag;
                    double sample_dynamic = 0.0;
                    for (int k = 0; k < active_count; ++k) {
                        size_t carrier_index = (size_t)carrier_indices_local[k];
                        size_t phase_index = phase_offset + carrier_index;
                        double theta = state->u.fftdiv.dynamic_phase[phase_index];
                        double fnorm = carrier_fnorm[k];
                        double angle = 2.0 * M_PI * fnorm;
                        double step_re = cos(angle);
                        double step_im = sin(angle);
                        state->u.fftdiv.dynamic_step_re[phase_index] = step_re;
                        state->u.fftdiv.dynamic_step_im[phase_index] = step_im;
                        double ph_re = cos(theta);
                        double ph_im = sin(theta);
                        double div_re_acc = 0.0;
                        double div_im_acc = 0.0;
                        double last_re = ph_re;
                        double last_im = ph_im;
                        for (int n = 0; n < window_size; ++n) {
                            double w = window_weights != NULL ? window_weights[n] : 1.0;
                            phasor_re[n] = ph_re;
                            phasor_im[n] = ph_im;
                            if (has_divisor) {
                                double dw_re = divisor_ring[n] * w;
                                double dw_im = divisor_imag_ring[n] * w;
                                div_re_acc += dw_re * ph_re + dw_im * ph_im;
                                div_im_acc += -dw_re * ph_im + dw_im * ph_re;
                            }
                            if (n == window_size - 1) {
                                last_re = ph_re;
                                last_im = ph_im;
                            }
                            double next_re = ph_re * step_re - ph_im * step_im;
                            double next_im = ph_re * step_im + ph_im * step_re;
                            ph_re = next_re;
                            ph_im = next_im;
                        }
                        double theta_next = state->u.fftdiv.dynamic_phase[phase_index] + angle * (double)window_size;
                        state->u.fftdiv.dynamic_phase[phase_index] = wrap_phase_two_pi(theta_next);
                        if (!has_divisor) {
                            div_re_acc = 1.0;
                            div_im_acc = 0.0;
                        }
                        double ratio = clamp_unit_double(fnorm);
                        double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
                        if (gain < 1e-6) {
                            gain = 1e-6;
                        }
                        double rotated_re = last_re * cos_phase - last_im * sin_phase;
                        double rotated_im = last_re * sin_phase + last_im * cos_phase;
                        double scale_re = rotated_re * gain;
                        double scale_im = rotated_im * gain;
                        if (has_divisor) {
                            double denom = div_re_acc * div_re_acc + div_im_acc * div_im_acc;
                            if (denom < epsilon_frame) {
                                denom = epsilon_frame;
                            }
                            double tmp_re = (scale_re * div_re_acc + scale_im * div_im_acc) / denom;
                            double tmp_im = (scale_im * div_re_acc - scale_re * div_im_acc) / denom;
                            scale_re = tmp_re;
                            scale_im = tmp_im;
                        }
                        for (int n = 0; n < window_size; ++n) {
                            double w = window_weights != NULL ? window_weights[n] : 1.0;
                            double coeff = inv_window * w * (scale_re * phasor_re[n] + scale_im * phasor_im[n]);
                            sample_dynamic += coeff * recomb_ring[n];
                        }
                    }
                    buffer[base_index + (size_t)slot] = sample_dynamic;
                    if (active_count > dynamic_active_frame) {
                        dynamic_active_frame = active_count;
                    }
                    dynamic_sum_frame += carrier_sum_slot;
                    continue;
                }
                buffer[base_index + (size_t)slot] = 0.0;
            }
            state->u.fftdiv.dynamic_carrier_band_count = dynamic_active_frame;
            state->u.fftdiv.dynamic_carrier_last_sum = dynamic_sum_frame;
            state->u.fftdiv.dynamic_k_active = dynamic_active_frame;
            state->u.fftdiv.position = window_size > 0 ? window_size - 1 : 0;
            continue;
        }
        int window_kind = default_window_kind;
        if (window_total > 0U) {
            double raw_w = read_param_value(window_view, base_index, (double)window_kind);
            window_kind = clamp_window_kind(round_to_int(raw_w));
        }
        if (state->u.fftdiv.window_kind != window_kind) {
            fill_window_weights(state->u.fftdiv.window, window_size, window_kind);
            state->u.fftdiv.window_kind = window_kind;
        }
        state->u.fftdiv.algorithm = algorithm_kind;
        state->u.fftdiv.epsilon = epsilon_frame;
        if (algorithm_class->supports_dynamic_carriers) {
            state->u.fftdiv.dynamic_carrier_band_count = (int)carrier_summary.band_count;
            state->u.fftdiv.dynamic_carrier_last_sum = carrier_summary.last_sum;
        } else {
            state->u.fftdiv.dynamic_carrier_band_count = 0;
            state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
        }

        double phase_last = phase_last_state;
        double lower_last = lower_last_state;
        double upper_last = upper_last_state;
        double filter_last = filter_last_state;

        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            double sample = audio_base != NULL ? audio_base[data_idx] : 0.0;
            size_t offset = (size_t)slot * (size_t)window_size;
            double *recomb_ring = state->u.fftdiv.recomb_buffer + offset;
            double *phase_ring = state->u.fftdiv.phase_buffer + offset;
            double *lower_ring = state->u.fftdiv.lower_buffer + offset;
            double *upper_ring = state->u.fftdiv.upper_buffer + offset;
            double *filter_ring = state->u.fftdiv.filter_buffer + offset;
            double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
            double *divisor_imag_ring = state->u.fftdiv.divisor_imag_buffer + offset;

            size_t tail_index = (size_t)(window_size > 0 ? window_size - 1 : 0);
            double phase_default = (recomb_filled < window_size) ? phase_ring[recomb_filled] : phase_ring[tail_index];
            double lower_default = (recomb_filled < window_size) ? lower_ring[recomb_filled] : lower_ring[tail_index];
            double upper_default = (recomb_filled < window_size) ? upper_ring[recomb_filled] : upper_ring[tail_index];
            double filter_default = (recomb_filled < window_size) ? filter_ring[recomb_filled] : filter_ring[tail_index];
            double divisor_default = (recomb_filled < window_size) ? divisor_ring[recomb_filled] : divisor_ring[tail_index];
            double divisor_imag_default = (recomb_filled < window_size) ? divisor_imag_ring[recomb_filled] : divisor_imag_ring[tail_index];

            double phase_sample = phase_total > 0U
                ? read_param_value(phase_view, base_index + (size_t)slot, phase_default)
                : phase_default;
            double lower_sample = lower_total > 0U
                ? read_param_value(lower_view, base_index + (size_t)slot, lower_default)
                : lower_default;
            double upper_sample = upper_total > 0U
                ? read_param_value(upper_view, base_index + (size_t)slot, upper_default)
                : upper_default;
            double filter_sample = filter_total > 0U
                ? read_param_value(filter_view, base_index + (size_t)slot, filter_default)
                : filter_default;
            double divisor_sample = divisor_total > 0U
                ? read_param_value(divisor_view, base_index + (size_t)slot, divisor_default)
                : divisor_default;
            double divisor_imag_sample = divisor_imag_total > 0U
                ? read_param_value(divisor_imag_view, base_index + (size_t)slot, divisor_imag_default)
                : divisor_imag_default;

            if (recomb_filled < window_size) {
                recomb_ring[recomb_filled] = sample;
                phase_ring[recomb_filled] = phase_sample;
                lower_ring[recomb_filled] = lower_sample;
                upper_ring[recomb_filled] = upper_sample;
                filter_ring[recomb_filled] = filter_sample;
                divisor_ring[recomb_filled] = divisor_sample;
                divisor_imag_ring[recomb_filled] = divisor_imag_sample;
            } else {
                if (window_size > 1) {
                    memmove(recomb_ring, recomb_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(phase_ring, phase_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(lower_ring, lower_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(upper_ring, upper_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(filter_ring, filter_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(divisor_ring, divisor_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(divisor_imag_ring, divisor_imag_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                }
                recomb_ring[window_size - 1] = sample;
                phase_ring[window_size - 1] = phase_sample;
                lower_ring[window_size - 1] = lower_sample;
                upper_ring[window_size - 1] = upper_sample;
                filter_ring[window_size - 1] = filter_sample;
                divisor_ring[window_size - 1] = divisor_sample;
                divisor_imag_ring[window_size - 1] = divisor_imag_sample;
            }

            phase_last = phase_sample;
            lower_last = lower_sample;
            upper_last = upper_sample;
            filter_last = filter_sample;
        }

        phase_last_state = phase_last;
        lower_last_state = lower_last;
        upper_last_state = upper_last;
        filter_last_state = filter_last;
        state->u.fftdiv.last_phase = phase_last;
        state->u.fftdiv.last_lower = lower_last;
        state->u.fftdiv.last_upper = upper_last;
        state->u.fftdiv.last_filter = filter_last;
        if (recomb_filled < window_size) {
            recomb_filled += 1;
            if (recomb_filled > window_size) {
                recomb_filled = window_size;
            }
            for (int slot = 0; slot < slot_count; ++slot) {
                size_t data_idx = base_index + (size_t)slot;
                size_t offset = (size_t)slot * (size_t)window_size;
                double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
                double divisor_sample = divisor_ring[recomb_filled > 0 ? recomb_filled - 1 : 0];
                if (divisor_total > 0U) {
                    divisor_sample = read_param_value(divisor_view, base_index + (size_t)slot, divisor_sample);
                }
                double safe_div = fabs(divisor_sample) < epsilon_frame
                    ? (divisor_sample >= 0.0 ? epsilon_frame : -epsilon_frame)
                    : divisor_sample;
                double quotient_sample = audio_base != NULL ? audio_base[data_idx] : 0.0;
                buffer[data_idx] = quotient_sample * safe_div;
            }
            continue;
        }

        double *work_real = state->u.fftdiv.work_real;
        double *work_imag = state->u.fftdiv.work_imag;
        double *div_real = state->u.fftdiv.div_real;
        double *div_imag = state->u.fftdiv.div_imag;
        double *ifft_real = state->u.fftdiv.ifft_real;
        double *ifft_imag = state->u.fftdiv.ifft_imag;
        double *window_weights = state->u.fftdiv.window;

        for (int slot = 0; slot < slot_count; ++slot) {
            size_t offset = (size_t)slot * (size_t)window_size;
            double *recomb_ring = state->u.fftdiv.recomb_buffer + offset;
            double *phase_ring = state->u.fftdiv.phase_buffer + offset;
            double *lower_ring = state->u.fftdiv.lower_buffer + offset;
            double *upper_ring = state->u.fftdiv.upper_buffer + offset;
            double *filter_ring = state->u.fftdiv.filter_buffer + offset;
            double phase_mod = phase_ring[window_size > 0 ? window_size - 1 : 0];
            double lower_mod = lower_ring[window_size > 0 ? window_size - 1 : 0];
            double upper_mod = upper_ring[window_size > 0 ? window_size - 1 : 0];
            double filter_mod = filter_ring[window_size > 0 ? window_size - 1 : 0];
            double lower_clamped = clamp_unit_double(lower_mod);
            double upper_clamped = clamp_unit_double(upper_mod);
            if (upper_clamped < lower_clamped) {
                double tmp_bounds = lower_clamped;
                lower_clamped = upper_clamped;
                upper_clamped = tmp_bounds;
            }
            double intensity_clamped = clamp_unit_double(filter_mod);
            double cos_phase = cos(phase_mod);
            double sin_phase = sin(phase_mod);

            for (int i = 0; i < window_size; ++i) {
                double w = window_weights != NULL ? window_weights[i] : 1.0;
                div_real[i] = state->u.fftdiv.divisor_buffer[offset + (size_t)i] * w;
                div_imag[i] = state->u.fftdiv.divisor_imag_buffer[offset + (size_t)i] * w;
            }
            algorithm_class->forward(div_real, div_imag, div_real, div_imag, window_size);
            for (int i = 0; i < window_size; ++i) {
                state->u.fftdiv.div_fft_real[offset + (size_t)i] = div_real[i];
                state->u.fftdiv.div_fft_imag[offset + (size_t)i] = div_imag[i];
            }

            for (int i = 0; i < window_size; ++i) {
                double w = window_weights != NULL ? window_weights[i] : 1.0;
                work_real[i] = recomb_ring[i] * w;
                work_imag[i] = 0.0;
            }
            algorithm_class->forward(work_real, work_imag, work_real, work_imag, window_size);
            for (int i = 0; i < window_size; ++i) {
                double ratio = window_size > 1 ? (double)i / (double)(window_size - 1) : 0.0;
                double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
                if (gain < 1e-6) {
                    gain = 1e-6;
                }
                double rotated_real = work_real[i];
                double rotated_imag = work_imag[i];
                double inv_real = rotated_real * cos_phase + rotated_imag * sin_phase;
                double inv_imag = -rotated_real * sin_phase + rotated_imag * cos_phase;
                inv_real /= gain;
                inv_imag /= gain;
                double c = state->u.fftdiv.div_fft_real[offset + (size_t)i];
                double d = state->u.fftdiv.div_fft_imag[offset + (size_t)i];
                double out_real_freq = inv_real * c - inv_imag * d;
                double out_imag_freq = inv_real * d + inv_imag * c;
                work_real[i] = out_real_freq;
                work_imag[i] = out_imag_freq;
            }
            algorithm_class->inverse(work_real, work_imag, ifft_real, ifft_imag, window_size);
            buffer[base_index + (size_t)slot] = ifft_real[window_size - 1];
        }
    }

    state->u.fftdiv.recomb_filled = recomb_filled;
    *out_buffer = buffer;
    *out_channels = input_channels;
    if (metrics != NULL) {
        metrics->measured_delay_frames = (uint32_t)(window_size > 0 ? window_size - 1 : 0);
        double complexity = 0.0;
        if (state->u.fftdiv.algorithm == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            complexity = (double)slot_count * (double)state->u.fftdiv.dynamic_k_active * (double)window_size * 8.0;
        } else if (
            state->u.fftdiv.algorithm == FFT_ALGORITHM_EIGEN
            || state->u.fftdiv.algorithm == FFT_ALGORITHM_HOOK
        ) {
            double stages = log((double)window_size) / log(2.0);
            if (stages < 1.0) {
                stages = 1.0;
            }
            complexity = (double)slot_count * (double)window_size * (stages * 6.0 + 4.0);
        } else {
            complexity = (double)slot_count * (double)window_size * (double)window_size * 2.0;
        }
        float heat = (float)(complexity / 1000.0);
        if (heat < 0.001f) {
            heat = 0.001f;
        }
        metrics->accumulated_heat = heat;
        state->u.fftdiv.total_heat += (double)heat;
        metrics->reserved[0] = state->u.fftdiv.last_phase;
        metrics->reserved[1] = state->u.fftdiv.last_lower;
        metrics->reserved[2] = state->u.fftdiv.last_upper;
        metrics->reserved[3] = state->u.fftdiv.last_filter;
        metrics->reserved[4] = (double)window_size;
        metrics->reserved[5] = (double)state->u.fftdiv.algorithm;
    }
    return 0;
}

static int run_mix_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels
) {
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    int target_channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", 1);
    if (target_channels <= 0) {
        target_channels = 1;
    }
    size_t total = (size_t)batches * (size_t)target_channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    if (!inputs->audio.has_audio || inputs->audio.data == NULL || inputs->audio.channels == 0) {
        memset(buffer, 0, total * sizeof(double));
        *out_buffer = buffer;
        *out_channels = target_channels;
        return 0;
    }
    int in_channels = (int)inputs->audio.channels;
    const double *audio = inputs->audio.data;
    for (int b = 0; b < batches; ++b) {
        for (int f = 0; f < frames; ++f) {
            double sum = 0.0;
            for (int c = 0; c < in_channels; ++c) {
                size_t idx = ((size_t)b * (size_t)in_channels + (size_t)c) * (size_t)frames + (size_t)f;
                sum += audio[idx];
            }
            for (int oc = 0; oc < target_channels; ++oc) {
                size_t out_idx = ((size_t)b * (size_t)target_channels + (size_t)oc) * (size_t)frames + (size_t)f;
                buffer[out_idx] = sum;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = target_channels;
    return 0;
}

static int run_sine_osc_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    int channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", 1);
    if (channels <= 0) {
        channels = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    double initial_phase = json_get_double(descriptor->params_json, descriptor->params_len, "phase", 0.0);
    double normalized_phase = initial_phase - floor(initial_phase);
    if (state != NULL) {
        if (state->u.sine.phase == NULL || state->u.sine.batches != batches || state->u.sine.channels != channels) {
            free(state->u.sine.phase);
            state->u.sine.phase = (double *)calloc((size_t)batches * (size_t)channels, sizeof(double));
            if (state->u.sine.phase == NULL) {
                free(buffer);
                state->u.sine.batches = 0;
                state->u.sine.channels = 0;
                return -1;
            }
            state->u.sine.batches = batches;
            state->u.sine.channels = channels;
            state->u.sine.base_phase = normalized_phase;
            for (size_t idx = 0; idx < (size_t)batches * (size_t)channels; ++idx) {
                state->u.sine.phase[idx] = normalized_phase;
            }
        }
    }
    double *phase = state != NULL ? state->u.sine.phase : NULL;
    double base_freq = json_get_double(descriptor->params_json, descriptor->params_len, "frequency", 440.0);
    double base_amp = json_get_double(descriptor->params_json, descriptor->params_len, "amplitude", 0.5);
    const EdgeRunnerParamView *freq_view = find_param(inputs, "frequency");
    const EdgeRunnerParamView *amp_view = find_param(inputs, "amplitude");
    const double *freq_data = freq_view != NULL ? freq_view->data : NULL;
    const double *amp_data = amp_view != NULL ? amp_view->data : NULL;
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            size_t bc = (size_t)b * (size_t)channels + (size_t)c;
            double phase_acc = phase != NULL ? phase[bc] : normalized_phase;
            for (int f = 0; f < frames; ++f) {
                size_t idx = bc * (size_t)frames + (size_t)f;
                double freq = freq_data != NULL ? freq_data[idx] : base_freq;
                double amp = amp_data != NULL ? amp_data[idx] : base_amp;
                double step = freq / sample_rate;
                phase_acc += step;
                phase_acc -= floor(phase_acc);
                buffer[idx] = sin(phase_acc * 2.0 * M_PI) * amp;
            }
            if (phase != NULL) {
                phase[bc] = phase_acc;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

static int run_safety_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    (void)sample_rate;
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    int channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", (int)inputs->audio.channels);
    if (channels <= 0) {
        channels = (int)inputs->audio.channels;
    }
    if (channels <= 0) {
        channels = 1;
    }
    double alpha = json_get_double(descriptor->params_json, descriptor->params_len, "dc_alpha", 0.995);
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    if (state != NULL) {
        if (state->u.safety.state == NULL || state->u.safety.batches != batches || state->u.safety.channels != channels) {
            free(state->u.safety.state);
            state->u.safety.state = (double *)calloc((size_t)batches * (size_t)channels, sizeof(double));
            if (state->u.safety.state == NULL) {
                free(buffer);
                state->u.safety.batches = 0;
                state->u.safety.channels = 0;
                return -1;
            }
            state->u.safety.batches = batches;
            state->u.safety.channels = channels;
        }
        state->u.safety.alpha = alpha;
    }
    if (!inputs->audio.has_audio || inputs->audio.data == NULL) {
        memset(buffer, 0, total * sizeof(double));
    } else {
        double *dc_state = NULL;
        if (state != NULL) {
            dc_state = state->u.safety.state;
        }
        if (dc_state == NULL) {
            dc_state = (double *)calloc((size_t)batches * (size_t)channels, sizeof(double));
            if (dc_state == NULL) {
                free(buffer);
                return -1;
            }
            if (state != NULL) {
                state->u.safety.state = dc_state;
                state->u.safety.batches = batches;
                state->u.safety.channels = channels;
            }
        }
        dc_block(inputs->audio.data, buffer, batches, channels, frames, alpha, dc_state);
        for (size_t i = 0; i < total; ++i) {
            double v = buffer[i];
            if (v > 1.0) {
                buffer[i] = 1.0;
            } else if (v < -1.0) {
                buffer[i] = -1.0;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

static void amp_reset_metrics(AmpNodeMetrics *metrics) {
    if (metrics == NULL) {
        return;
    }
    metrics->measured_delay_frames = 0U;
    metrics->accumulated_heat = 0.0f;
    metrics->processing_time_seconds = 0.0;
    metrics->logging_time_seconds = 0.0;
    metrics->total_time_seconds = 0.0;
    metrics->thread_cpu_time_seconds = 0.0;
    for (size_t i = 0; i < sizeof(metrics->reserved) / sizeof(metrics->reserved[0]); ++i) {
        metrics->reserved[i] = 0.0;
    }
}

/* Thread-local accumulators used to separate time spent in logging helpers
   from the node processing time. We use clock() to measure CPU time which is
   sufficient for profiling relative contributions of logging vs processing. */
#if defined(_MSC_VER)
__declspec(thread) static double _tl_logging_accum = 0.0;
__declspec(thread) static const char *_tl_current_node = NULL;
__declspec(thread) static double _tl_thread_cpu_start = 0.0;
#else
static __thread double _tl_logging_accum = 0.0;
static __thread const char *_tl_current_node = NULL;
static __thread double _tl_thread_cpu_start = 0.0;
#endif

/* High-resolution monotonic time (seconds). Use platform-specific APIs for
   better resolution than clock() which may be coarse on some platforms. */
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static inline double _now_clock_seconds(void) {
    static LARGE_INTEGER _freq = {0};
    LARGE_INTEGER v;
    if (_freq.QuadPart == 0) {
        QueryPerformanceFrequency(&_freq);
    }
    QueryPerformanceCounter(&v);
    return (double)v.QuadPart / (double)_freq.QuadPart;
}
#else
#include <time.h>
static inline double _now_clock_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return (double)time(NULL);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static inline double _now_thread_cpu_seconds(void) {
#if defined(_WIN32) || defined(_WIN64)
    FILETIME creation_time, exit_time, kernel_time, user_time;
    HANDLE thread = GetCurrentThread();
    if (GetThreadTimes(thread, &creation_time, &exit_time, &kernel_time, &user_time)) {
        ULARGE_INTEGER kernel_ticks;
        ULARGE_INTEGER user_ticks;
        kernel_ticks.LowPart = kernel_time.dwLowDateTime;
        kernel_ticks.HighPart = kernel_time.dwHighDateTime;
        user_ticks.LowPart = user_time.dwLowDateTime;
        user_ticks.HighPart = user_time.dwHighDateTime;
        unsigned long long total_ticks = kernel_ticks.QuadPart + user_ticks.QuadPart;
        return (double)total_ticks * 1.0e-7; /* FILETIME is 100-ns units */
    }
    return 0.0;
#else
#ifdef CLOCK_THREAD_CPUTIME_ID
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    clock_t ticks = clock();
    if (ticks == (clock_t)-1) {
        return 0.0;
    }
    return (double)ticks / (double)CLOCKS_PER_SEC;
#endif
}

typedef struct {
    double total_seconds;
    double processing_seconds;
    double logging_seconds;
    double thread_cpu_seconds;
} node_timing_info;

static double _node_timing_begin(const char *node_name) {
    _tl_logging_accum = 0.0;
    _tl_thread_cpu_start = _now_thread_cpu_seconds();
    _tl_current_node = node_name;
    return _now_clock_seconds();
}

static node_timing_info _node_timing_end(double start_clock) {
    node_timing_info info;
    double end_clock = _now_clock_seconds();
    info.total_seconds = end_clock - start_clock;
    if (info.total_seconds < 0.0) {
        info.total_seconds = 0.0;
    }
    double logging = _tl_logging_accum;
    if (logging < 0.0) logging = 0.0;
    info.logging_seconds = logging;
    info.processing_seconds = info.total_seconds - logging;
    if (info.processing_seconds < 0.0) {
        info.processing_seconds = 0.0;
    }
    double cpu_elapsed = _now_thread_cpu_seconds() - _tl_thread_cpu_start;
    if (cpu_elapsed < 0.0) {
        cpu_elapsed = 0.0;
    }
    info.thread_cpu_seconds = cpu_elapsed;
    _tl_current_node = NULL;
    _tl_logging_accum = 0.0;
    _tl_thread_cpu_start = 0.0;
    return info;
}

static const char *_node_dump_root_dir(void) {
    static int initialised = 0;
    static char root_dir[1024];
    if (!initialised) {
        const char *env = getenv("AMP_NODE_DUMP_DIR");
        if (env != NULL && env[0] != '\0') {
            size_t len = strlen(env);
            if (len >= sizeof(root_dir)) {
                len = sizeof(root_dir) - 1;
            }
            memcpy(root_dir, env, len);
            root_dir[len] = '\0';
        } else {
            root_dir[0] = '\0';
        }
        initialised = 1;
    }
    if (root_dir[0] == '\0') {
        return NULL;
    }
    return root_dir;
}

static void _sanitize_node_name(const char *name, char *out, size_t out_len) {
    if (out_len == 0) {
        return;
    }
    const char *src = (name != NULL && name[0] != '\0') ? name : "unnamed";
    size_t written = 0;
    for (; src[0] != '\0' && written + 1 < out_len; ++src) {
        unsigned char ch = (unsigned char)*src;
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
            out[written++] = (char)ch;
        } else if (ch == '-' || ch == '_') {
            out[written++] = (char)ch;
        } else {
            out[written++] = '_';
        }
    }
    if (written == 0) {
        out[written++] = 'n';
        if (written + 1 < out_len) {
            out[written++] = 'o';
        }
        if (written + 1 < out_len) {
            out[written++] = 'd';
        }
    }
    out[written] = '\0';
}

static void _write_json_string(FILE *stream, const char *text) {
    if (stream == NULL) {
        return;
    }
    if (text == NULL) {
        fputs("null", stream);
        return;
    }
    fputc('"', stream);
    for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
        unsigned char ch = *p;
        switch (ch) {
            case '\\':
                fputs("\\\\", stream);
                break;
            case '"':
                fputs("\\\"", stream);
                break;
            case '\n':
                fputs("\\n", stream);
                break;
            case '\r':
                fputs("\\r", stream);
                break;
            case '\t':
                fputs("\\t", stream);
                break;
            default:
                if (ch < 0x20U) {
                    fprintf(stream, "\\u%04x", ch);
                } else {
                    fputc((int)ch, stream);
                }
                break;
        }
    }
    fputc('"', stream);
}

#if defined(_WIN32) || defined(_WIN64)
static volatile LONG64 _node_dump_sequence_counter = 0;
#else
static volatile uint64_t _node_dump_sequence_counter = 0;
#endif

static uint64_t _next_dump_sequence(void) {
#if defined(_WIN32) || defined(_WIN64)
    LONG64 value = InterlockedIncrement64(&_node_dump_sequence_counter);
    if (value <= 0) {
        return 0ULL;
    }
    return (uint64_t)(value - 1);
#else
    return __sync_fetch_and_add(&_node_dump_sequence_counter, 1ULL);
#endif
}

static void maybe_dump_node_output(
    const char *node_name,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    const double *buffer,
    const AmpNodeMetrics *metrics,
    node_timing_info timing
) {
    (void)metrics;
    if (buffer == NULL) {
        return;
    }
    if (batches <= 0 || channels <= 0 || frames <= 0) {
        return;
    }
    const char *root = _node_dump_root_dir();
    if (root == NULL) {
        return;
    }
    char safe_name[128];
    _sanitize_node_name(node_name, safe_name, sizeof(safe_name));
    char node_dir[1024];
    if (snprintf(node_dir, sizeof(node_dir), "%s/%s", root, safe_name) >= (int)sizeof(node_dir)) {
        return;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (!CreateDirectoryA(node_dir, NULL)) {
        DWORD err = GetLastError();
        if (err != ERROR_ALREADY_EXISTS) {
            return;
        }
    }
#else
    if (mkdir(node_dir, 0775) != 0 && errno != EEXIST) {
        return;
    }
#endif
    uint64_t sequence = _next_dump_sequence();
    char base_path[1024];
    if (snprintf(base_path, sizeof(base_path), "%s/%s/%s_%06llu", root, safe_name, safe_name, (unsigned long long)sequence)
        >= (int)sizeof(base_path)) {
        return;
    }
    char raw_path[1024];
    if (snprintf(raw_path, sizeof(raw_path), "%s.raw", base_path) >= (int)sizeof(raw_path)) {
        return;
    }
    char meta_path[1024];
    if (snprintf(meta_path, sizeof(meta_path), "%s.meta.json", base_path) >= (int)sizeof(meta_path)) {
        return;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    float *temp = (float *)malloc(total * sizeof(float));
    if (temp == NULL) {
        return;
    }
    for (size_t i = 0; i < total; ++i) {
        temp[i] = (float)buffer[i];
    }
    FILE *raw_file = fopen(raw_path, "wb");
    if (raw_file == NULL) {
        free(temp);
        return;
    }
    size_t written = fwrite(temp, sizeof(float), total, raw_file);
    fclose(raw_file);
    free(temp);
    if (written != total) {
        remove(raw_path);
        return;
    }
    FILE *meta_file = fopen(meta_path, "w");
    if (meta_file == NULL) {
        return;
    }
    fprintf(meta_file, "{\n");
    fprintf(meta_file, "  \"node_name\": ");
    _write_json_string(meta_file, node_name);
    fprintf(meta_file, ",\n  \"safe_node_name\": ");
    _write_json_string(meta_file, safe_name);
    fprintf(meta_file, ",\n  \"sequence\": %llu,\n", (unsigned long long)sequence);
    fprintf(meta_file, "  \"batches\": %d,\n", batches);
    fprintf(meta_file, "  \"channels\": %d,\n", channels);
    fprintf(meta_file, "  \"frames\": %d,\n", frames);
    fprintf(meta_file, "  \"sample_rate\": %.9f,\n", sample_rate);
    fprintf(meta_file, "  \"total_time_seconds\": %.12g,\n", timing.total_seconds);
    fprintf(meta_file, "  \"processing_time_seconds\": %.12g,\n", timing.processing_seconds);
    fprintf(meta_file, "  \"logging_time_seconds\": %.12g,\n", timing.logging_seconds);
    fprintf(meta_file, "  \"thread_cpu_time_seconds\": %.12g,\n", timing.thread_cpu_seconds);
    fprintf(meta_file, "  \"dtype\": \"float32\",\n");
    fprintf(meta_file, "  \"layout\": \"BCF\"\n");
    fprintf(meta_file, "}\n");
    fclose(meta_file);
}

static int amp_run_node_impl(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history,
    AmpExecutionMode mode,
    AmpNodeMetrics *metrics
) {
    amp_reset_metrics(metrics);
    (void)channels;
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    node_kind_t kind = determine_node_kind(descriptor);
    if (kind == NODE_KIND_UNKNOWN) {
        return -3;
    }
    node_state_t *node_state = NULL;
    if (state != NULL && *state != NULL) {
        node_state = (node_state_t *)(*state);
    }
    if (node_state != NULL && node_state->kind != kind) {
        release_node_state(node_state);
        node_state = NULL;
        if (state != NULL) {
            *state = NULL;
        }
    }
    if (node_state == NULL) {
        node_state = (node_state_t *)calloc(1, sizeof(node_state_t));
        if (node_state == NULL) {
            return -1;
        }
        node_state->kind = kind;
        if (state != NULL) {
            *state = node_state;
        }
    }

    int rc = 0;
    /* Begin per-node timing window: set current node context and mark start. */
    double _node_start_clock = _node_timing_begin(descriptor != NULL ? descriptor->name : NULL);
    switch (kind) {
        case NODE_KIND_CONSTANT:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_constant_node(descriptor, batches, frames, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_GAIN:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_gain_node(inputs, batches, frames, out_buffer, out_channels)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_MIX:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_mix_node(descriptor, inputs, batches, frames, out_buffer, out_channels)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_SAFETY:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_safety_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_SINE_OSC:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_sine_osc_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_CONTROLLER:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_controller_node(descriptor, inputs, batches, frames, out_buffer, out_channels, history)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_LFO:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_lfo_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_ENVELOPE:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_envelope_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_PITCH:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_pitch_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_PITCH_SHIFT:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_pitch_shift_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_OSC_PITCH:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_oscillator_pitch_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_OSC:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_osc_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_DRIVER:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_parametric_driver_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_SUBHARM:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_subharm_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_FFT_DIV:
            if (mode == AMP_EXECUTION_MODE_BACKWARD) {
                rc = run_fft_division_node_backward(
                    descriptor,
                    inputs,
                    batches,
                    channels,
                    frames,
                    sample_rate,
                    out_buffer,
                    out_channels,
                    node_state,
                    metrics
                );
            } else {
                rc = run_fft_division_node(
                    descriptor,
                    inputs,
                    batches,
                    channels,
                    frames,
                    sample_rate,
                    out_buffer,
                    out_channels,
                    node_state,
                    metrics
                );
            }
            break;
        case NODE_KIND_SPECTRAL_DRIVE:
            rc = AMP_E_UNSUPPORTED;
            break;
        default:
            rc = -3;
            break;
    }
    /* End timing window and attribute timing to metrics if requested. */
    node_timing_info _timing = _node_timing_end(_node_start_clock);
    if (metrics != NULL) {
        if (rc == 0 && kind != NODE_KIND_FFT_DIV) {
            metrics->measured_delay_frames = 0U;
            metrics->accumulated_heat = 0.0f;
        }
        metrics->processing_time_seconds = _timing.processing_seconds;
        metrics->logging_time_seconds = _timing.logging_seconds;
        metrics->total_time_seconds = _timing.total_seconds;
        metrics->thread_cpu_time_seconds = _timing.thread_cpu_seconds;
    }
    maybe_dump_node_output(
        descriptor != NULL ? descriptor->name : NULL,
        batches,
        (out_channels != NULL) ? *out_channels : 0,
        frames,
        sample_rate,
        (out_buffer != NULL && *out_buffer != NULL) ? *out_buffer : NULL,
        metrics,
        _timing
    );
    return rc;
}

AMP_CAPI int amp_run_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history
) {
    AMP_LOG_NATIVE_CALL("amp_run_node", (size_t)batches, (size_t)frames);
    return amp_run_node_impl(
        descriptor,
        inputs,
        batches,
        channels,
        frames,
        sample_rate,
        out_buffer,
        out_channels,
        state,
        history,
        AMP_EXECUTION_MODE_FORWARD,
        NULL
    );
}

AMP_CAPI int amp_run_node_v2(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history,
    AmpExecutionMode mode,
    AmpNodeMetrics *metrics
) {
    AMP_LOG_NATIVE_CALL("amp_run_node_v2", (size_t)batches, (size_t)frames);
    return amp_run_node_impl(
        descriptor,
        inputs,
        batches,
        channels,
        frames,
        sample_rate,
        out_buffer,
        out_channels,
        state,
        history,
        mode,
        metrics
    );
}

AMP_CAPI void amp_free(double *buffer) {
    AMP_LOG_NATIVE_CALL("amp_free", (size_t)(buffer != NULL), 0);
    AMP_LOG_GENERATED("amp_free", (size_t)buffer, 0);
    if (buffer != NULL) {
        free(buffer);
    }
}

AMP_CAPI void amp_release_state(void *state_ptr) {
    AMP_LOG_NATIVE_CALL("amp_release_state", (size_t)(state_ptr != NULL), 0);
    AMP_LOG_GENERATED("amp_release_state", (size_t)state_ptr, 0);
    if (state_ptr == NULL) {
        return;
    }
    node_state_t *node_state = (node_state_t *)state_ptr;
    release_node_state(node_state);
}
