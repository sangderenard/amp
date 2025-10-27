#pragma once

#include <stddef.h>

#include "amp_native.h"

#ifndef AMP_THREAD_LOCAL
#  if defined(_MSC_VER)
#    define AMP_THREAD_LOCAL __declspec(thread)
#  else
#    define AMP_THREAD_LOCAL __thread
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern AMP_THREAD_LOCAL double amp_debug_logging_accum;
extern AMP_THREAD_LOCAL const char *amp_debug_current_node;

double amp_debug_now_seconds(void);

AMP_CAPI int amp_native_logging_enabled(void);
AMP_CAPI void amp_native_logging_set(int enabled);
AMP_CAPI void amp_log_generated(const char *fn, void *py_ts, size_t a, size_t b);
AMP_CAPI void amp_log_native_call_external(const char *fn, size_t a, size_t b);

#if defined(AMP_NATIVE_ENABLE_LOGGING)
void *amp_debug_malloc(size_t s, const char *file, int line, const char *func);
void *amp_debug_calloc(size_t n, size_t size, const char *file, int line, const char *func);
void *amp_debug_realloc(void *ptr, size_t s, const char *file, int line, const char *func);
void amp_debug_free(void *ptr, const char *file, int line, const char *func);
void *amp_debug_memcpy(void *dest, const void *src, size_t n, const char *file, int line, const char *func);
void *amp_debug_memset(void *dest, int c, size_t n, const char *file, int line, const char *func);

void amp_debug_register_alloc(void *ptr, size_t size);
void amp_debug_unregister_alloc(void *ptr);

void amp_debug_log_native_call(const char *fn, size_t a, size_t b);
void amp_debug_log_generated(const char *fn, size_t a, size_t b);
void amp_debug_log_memops(const char *fmt, ...);
void amp_debug_flush_memops(void);
#else
#if !defined(AMP_DEBUG_ALLOC_IMPLEMENTATION)
static inline void amp_debug_register_alloc(void *ptr, size_t size) {
    (void)ptr;
    (void)size;
}
static inline void amp_debug_unregister_alloc(void *ptr) {
    (void)ptr;
}
static inline void amp_debug_log_native_call(const char *fn, size_t a, size_t b) {
    (void)fn;
    (void)a;
    (void)b;
}
static inline void amp_debug_log_generated(const char *fn, size_t a, size_t b) {
    (void)fn;
    (void)a;
    (void)b;
}
static inline void amp_debug_log_memops(const char *fmt, ...) {
    (void)fmt;
}
static inline void amp_debug_flush_memops(void) {}
#endif
#endif

#ifdef __cplusplus
}
#endif

#if defined(AMP_NATIVE_ENABLE_LOGGING) && !defined(AMP_DEBUG_ALLOC_IMPLEMENTATION)
#undef malloc
#undef calloc
#undef realloc
#undef free
#undef memcpy
#undef memset

#define malloc(s) amp_debug_malloc((s), __FILE__, __LINE__, __func__)
#define calloc(n, s) amp_debug_calloc((n), (s), __FILE__, __LINE__, __func__)
#define realloc(p, s) amp_debug_realloc((p), (s), __FILE__, __LINE__, __func__)
#define free(p) amp_debug_free((p), __FILE__, __LINE__, __func__)
#define memcpy(d, s, n) amp_debug_memcpy((d), (s), (n), __FILE__, __LINE__, __func__)
#define memset(p, c, n) amp_debug_memset((p), (c), (n), __FILE__, __LINE__, __func__)

#undef AMP_LOG_NATIVE_CALL
#undef AMP_LOG_GENERATED
#define AMP_LOG_NATIVE_CALL(fn, a, b) amp_debug_log_native_call((fn), (a), (b))
#define AMP_LOG_GENERATED(fn, a, b) amp_debug_log_generated((fn), (a), (b))
#define AMP_DEBUG_LOG_MEMOPS(...) amp_debug_log_memops(__VA_ARGS__)
#define AMP_DEBUG_FLUSH_MEMOPS() amp_debug_flush_memops()
#else
#ifndef AMP_LOG_NATIVE_CALL
#define AMP_LOG_NATIVE_CALL(fn, a, b) ((void)0)
#endif
#ifndef AMP_LOG_GENERATED
#define AMP_LOG_GENERATED(fn, a, b) ((void)0)
#endif
#ifndef AMP_DEBUG_LOG_MEMOPS
#define AMP_DEBUG_LOG_MEMOPS(...) ((void)0)
#endif
#ifndef AMP_DEBUG_FLUSH_MEMOPS
#define AMP_DEBUG_FLUSH_MEMOPS() ((void)0)
#endif
#endif

#ifndef AMP_LIKELY
#  if defined(__GNUC__)
#    define AMP_LIKELY(x) __builtin_expect(!!(x), 1)
#    define AMP_UNLIKELY(x) __builtin_expect(!!(x), 0)
#  else
#    define AMP_LIKELY(x) (x)
#    define AMP_UNLIKELY(x) (x)
#  endif
#endif
