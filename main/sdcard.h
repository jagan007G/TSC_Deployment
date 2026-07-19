#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t sd_card_mount(void);
void sd_card_unmount(void);

#ifdef __cplusplus
}
#endif