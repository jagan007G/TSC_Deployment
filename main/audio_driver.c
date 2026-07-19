#include "audio_driver.h"
#include "audio_pin_config.h"
#include "i2c_manager.h"

#include "driver/i2s_std.h"

#include "esp_log.h"
#include "esp_check.h"

#include "es8311.h"

static const char *TAG = "audio_driver";

static i2s_chan_handle_t tx_handle = NULL;
static i2s_chan_handle_t rx_handle = NULL;

static es8311_handle_t codec_handle = NULL;

// ============================================================
// ES8311 INIT
// ============================================================

static esp_err_t es8311_codec_init(void)
{
    // =====================================================
    // USE SHARED I2C
    // =====================================================

    codec_handle =
        es8311_create(
            shared_i2c_port(),
            ES8311_ADDRRES_0);

    ESP_RETURN_ON_FALSE(
        codec_handle,
        ESP_FAIL,
        TAG,
        "es8311_create failed");

    es8311_clock_config_t clk_cfg = {
        .mclk_inverted = false,
        .sclk_inverted = false,
        .mclk_from_mclk_pin = true,
        .mclk_frequency = MCLK_FREQ_HZ,
        .sample_frequency = SAMPLE_RATE,
    };

    ESP_RETURN_ON_ERROR(
        es8311_init(
            codec_handle,
            &clk_cfg,
            ES8311_RESOLUTION_16,
            ES8311_RESOLUTION_16),
        TAG,
        "es8311_init failed");

    ESP_RETURN_ON_ERROR(
        es8311_sample_frequency_config(
            codec_handle,
            MCLK_FREQ_HZ,
            SAMPLE_RATE),
        TAG,
        "sample freq config failed");

    ESP_RETURN_ON_ERROR(
        es8311_voice_volume_set(
            codec_handle,
            VOICE_VOLUME,
            NULL),
        TAG,
        "volume set failed");

    ESP_RETURN_ON_ERROR(
        es8311_microphone_config(
            codec_handle,
            false),
        TAG,
        "mic config failed");

    ESP_RETURN_ON_ERROR(
        es8311_microphone_gain_set(
            codec_handle,
            MIC_GAIN),
        TAG,
        "mic gain failed");

    ESP_LOGI(TAG, "ES8311 initialized");

    return ESP_OK;
}

// ============================================================
// I2S INIT
// ============================================================

static esp_err_t i2s_bus_init(void)
{
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM,
            I2S_ROLE_MASTER);

    chan_cfg.auto_clear = true;

    ESP_RETURN_ON_ERROR(
        i2s_new_channel(
            &chan_cfg,
            &tx_handle,
            &rx_handle),
        TAG,
        "i2s_new_channel failed");

    i2s_std_config_t std_cfg = {
        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_16BIT,
                I2S_SLOT_MODE_STEREO),

        .gpio_cfg = {
            .mclk = I2S_MCK_IO,
            .bclk = I2S_BCK_IO,
            .ws   = I2S_WS_IO,
            .dout = I2S_DO_IO,
            .din  = I2S_DI_IO,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    std_cfg.clk_cfg.mclk_multiple =
        MCLK_MULTIPLE;

    ESP_RETURN_ON_ERROR(
        i2s_channel_init_std_mode(
            tx_handle,
            &std_cfg),
        TAG,
        "tx init failed");

    ESP_RETURN_ON_ERROR(
        i2s_channel_init_std_mode(
            rx_handle,
            &std_cfg),
        TAG,
        "rx init failed");

    ESP_RETURN_ON_ERROR(
        i2s_channel_enable(tx_handle),
        TAG,
        "tx enable failed");

    ESP_RETURN_ON_ERROR(
        i2s_channel_enable(rx_handle),
        TAG,
        "rx enable failed");

    ESP_LOGI(TAG, "I2S initialized");

    return ESP_OK;
}

// ============================================================
// PUBLIC INIT
// ============================================================

esp_err_t audio_driver_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << GPIO_PA_ENABLE),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };

    gpio_config(&io_conf);

    gpio_set_level(GPIO_PA_ENABLE, 1);

    ESP_RETURN_ON_ERROR(
        i2s_bus_init(),
        TAG,
        "i2s init failed");

    ESP_RETURN_ON_ERROR(
        es8311_codec_init(),
        TAG,
        "codec init failed");

    return ESP_OK;
}

// ============================================================
// GETTERS
// ============================================================

i2s_chan_handle_t audio_driver_get_rx_handle(void)
{
    return rx_handle;
}

i2s_chan_handle_t audio_driver_get_tx_handle(void)
{
    return tx_handle;
}