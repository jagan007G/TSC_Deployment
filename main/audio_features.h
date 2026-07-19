#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void extract_features(
    int16_t *audio,
    int samples,
    float *features
);

#ifdef __cplusplus
}
#endif