#pragma once

#include "driver/i2c.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t shared_i2c_init(void);

i2c_port_t shared_i2c_port(void);

#ifdef __cplusplus
}
#endif