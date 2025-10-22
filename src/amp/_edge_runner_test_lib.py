"""Minimal CFFI test kernel used by the edge runner during development.

The production build is expected to ship a compiled ``_amp_ckernels_cffi``
module exposing the same entrypoints. When it is missing (typical in test
runs on CI), we JIT-compile a very small C module that validates the data
plumbing and allocates a dummy audio buffer in C.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

C_SOURCE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t has_audio;
    uint32_t batches;
    uint32_t channels;
    uint32_t frames;
    const double *data;
} EdgeRunnerAudioView;

typedef struct {
    const char *name;
    uint32_t batches;
    uint32_t channels;
    uint32_t frames;
    const double *data;
} EdgeRunnerParamView;

typedef struct {
    uint32_t count;
    EdgeRunnerParamView *items;
} EdgeRunnerParamSet;

typedef struct {
    EdgeRunnerAudioView audio;
    EdgeRunnerParamSet params;
} EdgeRunnerNodeInputs;

static void dump_bytes(const char *label, const uint8_t *data, size_t size) {
    if (data == NULL) {
        printf("[edge-runner-test] %s: <null>\n", label);
        return;
    }
    printf("[edge-runner-test] %s: %zu bytes\n", label, (size_t)size);
    size_t preview = size < 16 ? size : 16;
    printf("[edge-runner-test] %s preview:", label);
    for (size_t i = 0; i < preview; ++i) {
        printf(" %02x", (unsigned int)data[i]);
    }
    printf("\n");
}

static void dump_plan(const uint8_t *plan, size_t size) {
    if (plan == NULL || size < 12) {
        printf("[edge-runner-test] plan blob too small (%zu)\n", (size_t)size);
        return;
    }
    const uint8_t *ptr = plan;
    char magic[5];
    memcpy(magic, ptr, 4);
    magic[4] = '\0';
    ptr += 4;
    uint32_t version = ((const uint32_t *)ptr)[0];
    ptr += 4;
    uint32_t node_count = ((const uint32_t *)ptr)[0];
    ptr += 4;
    printf("[edge-runner-test] plan magic='%s' version=%u nodes=%u\n",
           magic, (unsigned int)version, (unsigned int)node_count);
    for (uint32_t idx = 0; idx < node_count; ++idx) {
        if ((size_t)(ptr - plan) + 20 > size) {
            printf("[edge-runner-test] node %u truncated\n", (unsigned int)idx);
            return;
        }
        uint32_t function_id = ((const uint32_t *)ptr)[0];
        ptr += 4;
        uint32_t name_len = ((const uint32_t *)ptr)[0];
        ptr += 4;
        uint32_t audio_offset = ((const uint32_t *)ptr)[0];
        ptr += 4;
        uint32_t audio_span = ((const uint32_t *)ptr)[0];
        ptr += 4;
        uint32_t param_count = ((const uint32_t *)ptr)[0];
        ptr += 4;
        size_t remaining = (size_t)(plan + size - ptr);
        if (remaining < name_len) {
            printf("[edge-runner-test] node %u name truncated\n", (unsigned int)idx);
            return;
        }
        printf("[edge-runner-test] node %u func=%u audio_off=%u span=%u params=%u\n",
               (unsigned int)idx,
               (unsigned int)function_id,
               (unsigned int)audio_offset,
               (unsigned int)audio_span,
               (unsigned int)param_count);
        printf("[edge-runner-test]   name='%.*s'\n", (int)name_len, (const char *)ptr);
        ptr += name_len;
        for (uint32_t p = 0; p < param_count; ++p) {
            if ((size_t)(ptr - plan) + 12 > size) {
                printf("[edge-runner-test]   param %u truncated\n", (unsigned int)p);
                return;
            }
            uint32_t param_name_len = ((const uint32_t *)ptr)[0];
            ptr += 4;
            uint32_t param_offset = ((const uint32_t *)ptr)[0];
            ptr += 4;
            uint32_t param_span = ((const uint32_t *)ptr)[0];
            ptr += 4;
            if ((size_t)(plan + size - ptr) < param_name_len) {
                printf("[edge-runner-test]   param %u name truncated\n", (unsigned int)p);
                return;
            }
            printf("[edge-runner-test]   param %u name='%.*s' offset=%u span=%u\n",
                   (unsigned int)p,
                   (int)param_name_len,
                   (const char *)ptr,
                   (unsigned int)param_offset,
                   (unsigned int)param_span);
            ptr += param_name_len;
        }
    }
}

int amp_test_run_graph(
    const uint8_t *control_blob,
    size_t control_size,
    const uint8_t *node_blob,
    size_t node_size,
    const uint8_t *plan_blob,
    size_t plan_size,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_batches,
    int *out_channels,
    int *out_frames
) {
    dump_bytes("control", control_blob, control_size);
    dump_bytes("nodes", node_blob, node_size);
    dump_plan(plan_blob, plan_size);
    printf("[edge-runner-test] batches=%d channels=%d frames=%d sample_rate=%.2f\n",
           batches, channels, frames, sample_rate);
    if (batches <= 0) {
        batches = 1;
    }
    if (channels <= 0) {
        channels = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    for (size_t i = 0; i < total; ++i) {
        buffer[i] = 0.5;
    }
    *out_buffer = buffer;
    *out_batches = batches;
    *out_channels = channels;
    *out_frames = frames;
    return 0;
}

void amp_test_free(double *buffer) {
    if (buffer != NULL) {
        free(buffer);
    }
}
"""


@lru_cache(maxsize=1)
def load(ffi: Any):
    """Compile (or load the cached) C helper library using ``ffi``."""

    return ffi.verify(C_SOURCE)
