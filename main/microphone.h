#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t microphone_init(void);

size_t microphone_read( int16_t *buffer, size_t samples);

#ifdef __cplusplus
}
#endif