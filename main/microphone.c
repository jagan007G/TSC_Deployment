#include "microphone.h"
#include "audio_driver.h"
#include "freertos/FreeRTOS.h"
#include "esp_log.h"

static const char *TAG = "microphone";

// ============================================================
// READ AUDIO
// ============================================================

size_t microphone_read(int16_t *buffer,
                       size_t samples_to_read)
{
    if (buffer == NULL)
    {
        return 0;
    }

    i2s_chan_handle_t rx =
        audio_driver_get_rx_handle();

    if (rx == NULL)
    {
        ESP_LOGE(TAG, "RX handle NULL");
        return 0;
    }

    size_t bytes_read = 0;

    // stereo 16-bit
    size_t bytes_to_read =
        samples_to_read *
        sizeof(int16_t) *
        2;

    esp_err_t ret =
        i2s_channel_read(
            rx,
            buffer,
            bytes_to_read,
            &bytes_read,
            portMAX_DELAY);

    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG,
                 "i2s read failed");
        return 0;
    }

    // return number of int16 samples
    return bytes_read / sizeof(int16_t);
}