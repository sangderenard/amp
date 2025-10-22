#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple test kernel: report sizes and fill output buffer with zeros

// control_blob: pointer to bytes, control_size: size
// node_blob: pointer to bytes, node_size: size
// out_buf: pointer to double buffer (batches * channels * frames)
// batches, channels, frames: dimensions

void test_run_graph(const char *control_blob, unsigned long control_size, const char *node_blob, unsigned long node_size, double *out_buf, unsigned int batches, unsigned int channels, unsigned int frames) {
    printf("[c_test] Received control_blob size=%lu node_blob size=%lu\n", control_size, node_size);
    printf("[c_test] Expected output shape: %u x %u x %u\n", batches, channels, frames);
    fflush(stdout);
    unsigned long count = (unsigned long)batches * channels * frames;
    for (unsigned long i = 0; i < count; ++i) {
        out_buf[i] = 0.0; // silence
    }
}
