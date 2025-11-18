#pragma once

#if !defined(__cplusplus)
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FftDivStageLockSnapshot {
    bool stage1_ready;
    bool stage2_ready;
    bool stage3_ready;
    bool stage4_ready;
    bool stage5_ready;
} FftDivStageLockSnapshot;

#ifdef __cplusplus
}
#endif
