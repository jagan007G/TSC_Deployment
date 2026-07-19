#ifndef AUDIO_DRIVER_H
#define AUDIO_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"
#include "driver/i2s_std.h"


// ============================================================
// AUDIO DRIVER INIT
// ============================================================

/**
 * @brief Initialize ES8311 codec + I2S bus
 *
 * Initializes:
 * - I2C communication
 * - ES8311 codec
 * - I2S TX/RX channels
 * - speaker amplifier GPIO
 *
 * @return ESP_OK on success
 */
esp_err_t audio_driver_init(void);

// ============================================================
// HANDLE GETTERS
// ============================================================

/**
 * @brief Get I2S RX channel handle
 *
 * Used for microphone/audio input.
 */
i2s_chan_handle_t audio_driver_get_rx_handle(void);

/**
 * @brief Get I2S TX channel handle
 *
 * Used for speaker/audio output.
 */
i2s_chan_handle_t audio_driver_get_tx_handle(void);

#ifdef __cplusplus
}
#endif

#endif // AUDIO_DRIVER_H