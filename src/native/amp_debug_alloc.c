#define AMP_DEBUG_ALLOC_IMPLEMENTATION
#include "amp_debug_alloc.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#if defined(__GNUC__) && !defined(_WIN32) && !defined(_WIN64)
#include <execinfo.h>
#endif

AMP_THREAD_LOCAL double amp_debug_logging_accum = 0.0;
AMP_THREAD_LOCAL const char *amp_debug_current_node = NULL;

double amp_debug_now_seconds(void) {
#if defined(_WIN32) || defined(_WIN64)
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    return (double)time(NULL);
#endif
}

#if defined(AMP_NATIVE_ENABLE_LOGGING)

/* Persistent log file handles. */
static FILE *log_f_alloc = NULL;
static FILE *log_f_memops = NULL;
static FILE *log_f_ccalls = NULL;
static FILE *log_f_cgenerated = NULL;

#if defined(_WIN32) || defined(_WIN64)
static CRITICAL_SECTION log_lock;
static int log_lock_initialized = 0;
#define LOG_LOCK_INIT() do { if (!log_lock_initialized) { InitializeCriticalSection(&log_lock); log_lock_initialized = 1; } } while (0)
#define LOG_LOCK() EnterCriticalSection(&log_lock)
#define LOG_UNLOCK() LeaveCriticalSection(&log_lock)
#else
static pthread_mutex_t log_lock;
static int log_lock_initialized = 0;
#define LOG_LOCK_INIT() do { if (!log_lock_initialized) { pthread_mutex_init(&log_lock, NULL); log_lock_initialized = 1; } } while (0)
#define LOG_LOCK() pthread_mutex_lock(&log_lock)
#define LOG_UNLOCK() pthread_mutex_unlock(&log_lock)
#endif

static void amp_debug_close_all_logs(void);

static int logging_mode_enabled = 0;

static int amp_native_logging_enabled_internal(void) {
    return logging_mode_enabled;
}

AMP_CAPI int amp_native_logging_enabled(void) {
    return logging_mode_enabled;
}

static void amp_debug_ensure_log_files_open(void) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    if (!log_lock_initialized) LOG_LOCK_INIT();
    LOG_LOCK();
#if defined(_WIN32) || defined(_WIN64)
    CreateDirectoryA("logs", NULL);
#else
    mkdir("logs", 0775);
#endif
    if (log_f_alloc == NULL) log_f_alloc = fopen("logs/native_alloc_trace.log", "a");
    if (log_f_memops == NULL) log_f_memops = fopen("logs/native_mem_ops.log", "a");
    if (log_f_ccalls == NULL) log_f_ccalls = fopen("logs/native_c_calls.log", "a");
    if (log_f_cgenerated == NULL) log_f_cgenerated = fopen("logs/native_c_generated.log", "a");
    LOG_UNLOCK();
    static int atexit_registered = 0;
    if (!atexit_registered) {
        atexit(amp_debug_close_all_logs);
        atexit_registered = 1;
    }
}

static void amp_debug_close_all_logs(void) {
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

AMP_CAPI void amp_native_logging_set(int enabled) {
    int normalised = enabled ? 1 : 0;
    if (!log_lock_initialized) LOG_LOCK_INIT();
    LOG_LOCK();
    logging_mode_enabled = normalised;
    LOG_UNLOCK();
    if (!normalised) {
        amp_debug_close_all_logs();
        return;
    }
    amp_debug_ensure_log_files_open();
}

static void amp_debug_capture_stack_frames(void **out_frames, unsigned short *out_count);
static void amp_debug_dump_alloc_backtrace(FILE *stream, struct alloc_rec *rec);
static void amp_debug_dump_backtrace(FILE *stream);

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

struct alloc_rec {
    void *ptr;
    size_t size;
    struct alloc_rec *next;
    void *bt[32];
    unsigned short bt_count;
};

static struct alloc_rec *alloc_list = NULL;

static void amp_debug_dump_alloc_snapshot(FILE *g) {
    if (!amp_native_logging_enabled_internal()) return;
    fprintf(g, "ALLOC_SNAPSHOT_BEGIN\n");
    struct alloc_rec *it = alloc_list;
    while (it) {
        fprintf(g, "ALLOC_ENTRY ptr=%p size=%zu bt_count=%u\n", it->ptr, it->size, (unsigned)it->bt_count);
        for (unsigned short i = 0; i < it->bt_count; ++i) {
            fprintf(g, "RBT %p\n", it->bt[i]);
        }
        it = it->next;
    }
    fprintf(g, "ALLOC_SNAPSHOT_END\n");
}

static void amp_debug_register_alloc_internal(void *ptr, size_t size) {
    if (!amp_native_logging_enabled_internal()) return;
    if (ptr == NULL) return;
    struct alloc_rec *it = alloc_list;
    while (it) {
        if (it->ptr == ptr) {
            if (it->size != size) {
                amp_debug_ensure_log_files_open();
                if (log_f_alloc) {
                    fprintf(log_f_alloc, "REGISTER_UPDATE %p old=%zu new=%zu\n", ptr, it->size, size);
                    amp_debug_dump_alloc_snapshot(log_f_alloc);
                    amp_debug_dump_backtrace(log_f_alloc);
                    fflush(log_f_alloc);
                }
                it->size = size;
            }
            return;
        }
        it = it->next;
    }
    struct alloc_rec *rec = (struct alloc_rec *)real_malloc_fn(sizeof(struct alloc_rec));
    if (rec == NULL) return;
    rec->ptr = ptr;
    rec->size = size;
    rec->next = alloc_list;
    rec->bt_count = 0;
    amp_debug_capture_stack_frames(rec->bt, &rec->bt_count);
    alloc_list = rec;
    amp_debug_ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "REGISTER %p size=%zu\n", ptr, size);
        amp_debug_dump_alloc_backtrace(log_f_alloc, rec);
        fflush(log_f_alloc);
    }
}

static void amp_debug_unregister_alloc_internal(void *ptr) {
    if (!amp_native_logging_enabled_internal()) return;
    if (ptr == NULL) return;
    struct alloc_rec **pp = &alloc_list;
    int removed = 0;
    while (*pp) {
        if ((*pp)->ptr == ptr) {
            struct alloc_rec *remove = *pp;
            *pp = remove->next;
            amp_debug_ensure_log_files_open();
            if (log_f_alloc) {
                fprintf(log_f_alloc, "UNREGISTER %p\n", ptr);
                amp_debug_dump_alloc_snapshot(log_f_alloc);
                amp_debug_dump_alloc_backtrace(log_f_alloc, remove);
                amp_debug_dump_backtrace(log_f_alloc);
                fflush(log_f_alloc);
            }
            real_free_fn(remove);
            removed = 1;
            continue;
        }
        pp = &(*pp)->next;
    }
    if (!removed) {
        amp_debug_ensure_log_files_open();
        if (log_f_alloc) {
            fprintf(log_f_alloc, "UNREGISTER_NOTFOUND %p\n", ptr);
            amp_debug_dump_alloc_snapshot(log_f_alloc);
            amp_debug_dump_backtrace(log_f_alloc);
            fflush(log_f_alloc);
        }
    }
}

static int amp_debug_range_within_alloc(void *addr, size_t len) {
    if (addr == NULL) return 0;
    unsigned char *a = (unsigned char *)addr;
    struct alloc_rec *rec = alloc_list;
    while (rec) {
        unsigned char *base = (unsigned char *)rec->ptr;
        if (a >= base && (a + len) <= (base + rec->size)) return 1;
        rec = rec->next;
    }
    return 0;
}

static void amp_debug_dump_backtrace(FILE *g) {
    if (!amp_native_logging_enabled_internal()) return;
    if (g == NULL) return;
#if defined(_WIN32) || defined(_WIN64)
    void *frames[64];
    USHORT count = CaptureStackBackTrace(0, (ULONG)(sizeof(frames) / sizeof(frames[0])), frames, NULL);
    fprintf(g, "BACKTRACE_FRAMES %u\n", (unsigned)count);
    for (USHORT i = 0; i < count; ++i) {
        fprintf(g, "BT %p\n", frames[i]);
    }
#else
    void *frames[64];
    int count = backtrace(frames, (int)(sizeof(frames) / sizeof(frames[0])));
    char **symbols = backtrace_symbols(frames, count);
    if (symbols != NULL) {
        fprintf(g, "BACKTRACE_FRAMES %d\n", count);
        for (int i = 0; i < count; ++i) {
            fprintf(g, "BT %s\n", symbols[i]);
        }
        free(symbols);
    } else {
        fprintf(g, "BACKTRACE_FRAMES %d\n", count);
        for (int i = 0; i < count; ++i) {
            fprintf(g, "BT %p\n", frames[i]);
        }
    }
#endif
}

static void amp_debug_capture_stack_frames(void **out_frames, unsigned short *out_count) {
    if (!amp_native_logging_enabled_internal()) return;
    if (out_frames == NULL || out_count == NULL) return;
#if defined(_WIN32) || defined(_WIN64)
    void *frames[64];
    USHORT count = CaptureStackBackTrace(0, (ULONG)(sizeof(frames) / sizeof(frames[0])), frames, NULL);
    unsigned short copy_count = (unsigned short)((count > 32) ? 32 : count);
    for (unsigned short i = 0; i < copy_count; ++i) out_frames[i] = frames[i];
    *out_count = copy_count;
#else
    void *frames[64];
    int count = backtrace(frames, (int)(sizeof(frames) / sizeof(frames[0])));
    int copy_count = (count > 32) ? 32 : count;
    for (int i = 0; i < copy_count; ++i) out_frames[i] = frames[i];
    *out_count = (unsigned short)copy_count;
#endif
}

static void amp_debug_dump_alloc_backtrace(FILE *g, struct alloc_rec *rec) {
    if (!amp_native_logging_enabled_internal()) return;
    if (g == NULL || rec == NULL) return;
    fprintf(g, "REGISTER_BACKTRACE %u\n", (unsigned)rec->bt_count);
    for (unsigned short i = 0; i < rec->bt_count; ++i) {
        fprintf(g, "RBT %p\n", rec->bt[i]);
    }
}

void amp_debug_register_alloc(void *ptr, size_t size) {
    amp_debug_register_alloc_internal(ptr, size);
}

void amp_debug_unregister_alloc(void *ptr) {
    amp_debug_unregister_alloc_internal(ptr);
}

static void amp_debug_log_native_call_impl(const char *fn, size_t a, size_t b) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    amp_debug_ensure_log_files_open();
    if (log_f_ccalls == NULL) {
        return;
    }
    double t = (double)time(NULL);
#ifdef PyThreadState_Get
    void *py_ts = (void *)PyThreadState_Get();
    fprintf(log_f_ccalls, "%.3f %p %s %zu %zu\n", t, py_ts, fn, a, b);
#else
#if defined(_WIN32) || defined(_WIN64)
    unsigned long tid = (unsigned long)GetCurrentThreadId();
    fprintf(log_f_ccalls, "%.3f %lu %s %zu %zu\n", t, tid, fn, a, b);
#else
    unsigned long tid = (unsigned long)pthread_self();
    fprintf(log_f_ccalls, "%.3f %lu %s %zu %zu\n", t, tid, fn, a, b);
#endif
#endif
    fflush(log_f_ccalls);
}

void amp_debug_log_native_call(const char *fn, size_t a, size_t b) {
    amp_debug_log_native_call_impl(fn, a, b);
}

static void amp_debug_log_generated_impl(const char *fn, size_t a, size_t b) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    amp_debug_ensure_log_files_open();
    if (log_f_cgenerated == NULL) return;
#ifdef PyThreadState_Get
    void *py_ts = (void *)PyThreadState_Get();
    fprintf(log_f_cgenerated, "%s %p %zu %zu\n", fn, py_ts, a, b);
#else
    fprintf(log_f_cgenerated, "%s %p %zu %zu\n", fn, (void *)0, a, b);
#endif
    fflush(log_f_cgenerated);
}

void amp_debug_log_generated(const char *fn, size_t a, size_t b) {
    amp_debug_log_generated_impl(fn, a, b);
}

void amp_debug_log_memops(const char *fmt, ...) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    amp_debug_ensure_log_files_open();
    if (log_f_memops == NULL) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(log_f_memops, fmt, args);
    va_end(args);
}

void amp_debug_flush_memops(void) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    amp_debug_ensure_log_files_open();
    if (log_f_memops != NULL) {
        fflush(log_f_memops);
    }
}

void *amp_debug_malloc(size_t s, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_malloc_fn(s);
    }
    void *p = real_malloc_fn(s);
    amp_debug_log_native_call_impl("malloc", (size_t)s, (size_t)(uintptr_t)p);
    amp_debug_ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "MALLOC %s:%d %s size=%zu ptr=%p\n", file, line, func, s, p);
        fflush(log_f_alloc);
    }
    if (p != NULL) amp_debug_register_alloc_internal(p, s);
    return p;
}

void *amp_debug_calloc(size_t n, size_t size, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_calloc_fn(n, size);
    }
    void *p = real_calloc_fn(n, size);
    amp_debug_log_native_call_impl("calloc", (size_t)(n * size), (size_t)(uintptr_t)p);
    amp_debug_ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "CALLOC %s:%d %s nmemb=%zu size=%zu ptr=%p\n", file, line, func, n, size, p);
        fflush(log_f_alloc);
    }
    if (p != NULL) amp_debug_register_alloc_internal(p, n * size);
    return p;
}

void *amp_debug_realloc(void *ptr, size_t s, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_realloc_fn(ptr, s);
    }
    void *p = real_realloc_fn(ptr, s);
    amp_debug_log_native_call_impl("realloc_old", (size_t)(uintptr_t)ptr, (size_t)(uintptr_t)p);
    amp_debug_log_native_call_impl("realloc_new", (size_t)s, (size_t)(uintptr_t)p);
    amp_debug_ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "REALLOC %s:%d %s old=%p new=%p size=%zu\n", file, line, func, ptr, p, s);
        amp_debug_dump_backtrace(log_f_alloc);
        fflush(log_f_alloc);
    }
    if (ptr != NULL) amp_debug_unregister_alloc_internal(ptr);
    if (p != NULL) amp_debug_register_alloc_internal(p, s);
    return p;
}

void amp_debug_free(void *ptr, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        if (ptr != NULL) real_free_fn(ptr);
        return;
    }
    amp_debug_log_native_call_impl("free", (size_t)(uintptr_t)ptr, 0);
    amp_debug_ensure_log_files_open();
    if (log_f_alloc) {
        fprintf(log_f_alloc, "FREE %s:%d %s ptr=%p\n", file, line, func, ptr);
        amp_debug_dump_backtrace(log_f_alloc);
        fflush(log_f_alloc);
    }
    if (ptr != NULL) {
        amp_debug_unregister_alloc_internal(ptr);
        real_free_fn(ptr);
    }
}

void *amp_debug_memcpy(void *dest, const void *src, size_t n, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_memcpy_fn(dest, src, n);
    }
    amp_debug_ensure_log_files_open();
    if (log_f_memops) {
        fprintf(log_f_memops, "MEMCPY %s:%d %s dest=%p src=%p n=%zu\n", file, line, func, dest, src, n);
    }
    if (dest == NULL || src == NULL || n == 0) {
        return dest;
    }
    if (!amp_debug_range_within_alloc(dest, n)) {
        uintptr_t stack_probe = (uintptr_t)&stack_probe;
        uintptr_t d = (uintptr_t)dest;
        const uintptr_t STACK_WINDOW = 0x100000;
        if (!(d >= stack_probe - STACK_WINDOW && d <= stack_probe + STACK_WINDOW)) {
            amp_debug_ensure_log_files_open();
            if (log_f_memops) {
                fprintf(log_f_memops, "BAD_MEMCPY %s:%d %s dest=%p n=%zu stack_probe=%p (no matching alloc or too small)\n", file, line, func, dest, n, (void *)stack_probe);
                struct alloc_rec *it = alloc_list;
                while (it) {
                    fprintf(log_f_memops, "ALLOC_SNAPSHOT ptr=%p size=%zu\n", it->ptr, it->size);
                    it = it->next;
                }
                amp_debug_dump_backtrace(log_f_memops);
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

void *amp_debug_memset(void *dest, int c, size_t n, const char *file, int line, const char *func) {
    if (!amp_native_logging_enabled_internal()) {
        return real_memset_fn(dest, c, n);
    }
    amp_debug_ensure_log_files_open();
    if (log_f_memops) {
        fprintf(log_f_memops, "MEMSET %s:%d %s ptr=%p val=%d n=%zu\n", file, line, func, dest, c, n);
    }
    if (dest == NULL || n == 0) {
        return dest;
    }
    if (!amp_debug_range_within_alloc(dest, n)) {
        uintptr_t stack_probe = (uintptr_t)&stack_probe;
        uintptr_t d = (uintptr_t)dest;
        const uintptr_t STACK_WINDOW = 0x100000;
        if (!(d >= stack_probe - STACK_WINDOW && d <= stack_probe + STACK_WINDOW)) {
            amp_debug_ensure_log_files_open();
            if (log_f_memops) {
                fprintf(log_f_memops, "BAD_MEMSET %s:%d %s ptr=%p n=%zu stack_probe=%p (no matching alloc or too small)\n", file, line, func, dest, n, (void *)stack_probe);
                amp_debug_dump_backtrace(log_f_memops);
                fflush(log_f_memops);
            }
        }
    }
    unsigned char *p = (unsigned char *)dest;
    unsigned char v = (unsigned char)c;
    for (size_t i = 0; i < n; ++i) p[i] = v;
    return dest;
}

AMP_CAPI void amp_log_generated(const char *fn, void *py_ts, size_t a, size_t b) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    amp_debug_ensure_log_files_open();
    if (!log_f_cgenerated) return;
    double start = amp_debug_now_seconds();
    fprintf(log_f_cgenerated, "%s %p %zu %zu\n", fn, py_ts, a, b);
    fflush(log_f_cgenerated);
    double end = amp_debug_now_seconds();
    if (amp_debug_current_node != NULL) {
        amp_debug_logging_accum += (end - start);
    }
}

AMP_CAPI void amp_log_native_call_external(const char *fn, size_t a, size_t b) {
    if (!amp_native_logging_enabled_internal()) {
        return;
    }
    amp_debug_ensure_log_files_open();
    if (!log_f_ccalls) return;
    double start = amp_debug_now_seconds();
    double t = (double)time(NULL);
#if defined(_WIN32) || defined(_WIN64)
    unsigned long tid = (unsigned long)GetCurrentThreadId();
    fprintf(log_f_ccalls, "%.3f %lu %s %zu %zu\n", t, tid, fn, a, b);
#else
    unsigned long tid = (unsigned long)pthread_self();
    fprintf(log_f_ccalls, "%.3f %lu %s %zu %zu\n", t, tid, fn, a, b);
#endif
    fflush(log_f_ccalls);
    double end = amp_debug_now_seconds();
    if (amp_debug_current_node != NULL) {
        amp_debug_logging_accum += (end - start);
    }
}

#else /* !AMP_NATIVE_ENABLE_LOGGING */

AMP_CAPI int amp_native_logging_enabled(void) {
    return 0;
}

AMP_CAPI void amp_native_logging_set(int enabled) {
    (void)enabled;
}

AMP_CAPI void amp_log_generated(const char *fn, void *py_ts, size_t a, size_t b) {
    (void)fn;
    (void)py_ts;
    (void)a;
    (void)b;
}

AMP_CAPI void amp_log_native_call_external(const char *fn, size_t a, size_t b) {
    (void)fn;
    (void)a;
    (void)b;
}

void amp_debug_register_alloc(void *ptr, size_t size) {
    (void)ptr;
    (void)size;
}

void amp_debug_unregister_alloc(void *ptr) {
    (void)ptr;
}

void amp_debug_log_native_call(const char *fn, size_t a, size_t b) {
    (void)fn;
    (void)a;
    (void)b;
}

void amp_debug_log_generated(const char *fn, size_t a, size_t b) {
    (void)fn;
    (void)a;
    (void)b;
}

#endif /* AMP_NATIVE_ENABLE_LOGGING */
