#include "i2c_manager.h"

#include "esp_log.h"

static const char *TAG = "i2c_manager";

// ======================================================
// SHARED I2C CONFIG
// ======================================================

#define SHARED_I2C_PORT     I2C_NUM_1

#define SHARED_I2C_SDA      GPIO_NUM_7
#define SHARED_I2C_SCL      GPIO_NUM_8

#define SHARED_I2C_FREQ     100000

// ======================================================
// INTERNAL FLAG
// ======================================================

static bool i2c_initialized = false;

// ======================================================
// INIT SHARED I2C
// ======================================================

esp_err_t shared_i2c_init(void)
{
    // ==============================================
    // PREVENT DOUBLE INITIALIZATION
    // ==============================================

    if (i2c_initialized)
    {
        ESP_LOGI(TAG, "I2C already initialized");

        return ESP_OK;
    }

    ESP_LOGI(TAG, "Initializing shared I2C bus");

    // ==============================================
    // I2C CONFIG
    // ==============================================

    i2c_config_t conf = {};

    conf.mode = I2C_MODE_MASTER;

    conf.sda_io_num = SHARED_I2C_SDA;
    conf.scl_io_num = SHARED_I2C_SCL;

    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;

    conf.master.clk_speed = SHARED_I2C_FREQ;

    // ==============================================
    // APPLY CONFIG
    // ==============================================

    esp_err_t ret;

    ret = i2c_param_config(
        SHARED_I2C_PORT,
        &conf
    );

    if (ret != ESP_OK)
    {
        ESP_LOGE(
            TAG,
            "i2c_param_config failed"
        );

        return ret;
    }

    // ==============================================
    // INSTALL DRIVER
    // ==============================================

    ret = i2c_driver_install(
        SHARED_I2C_PORT,
        conf.mode,
        0,
        0,
        0
    );

    if (ret != ESP_OK)
    {
        ESP_LOGE(
            TAG,
            "i2c_driver_install failed"
        );

        return ret;
    }

    // ==============================================
    // SUCCESS
    // ==============================================

    i2c_initialized = true;

    ESP_LOGI(TAG, "Shared I2C initialized");

    return ESP_OK;
}

// ======================================================
// GET I2C PORT
// ======================================================

i2c_port_t shared_i2c_port(void)
{
    return SHARED_I2C_PORT;
}