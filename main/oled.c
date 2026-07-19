/*
 * SH1107 OLED Driver for ESP32-P4 Random Forest Project
 */

#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_timer.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"

#include "driver/i2c.h"

#include "esp_err.h"
#include "esp_log.h"

#include "lvgl.h"
#include "esp_lvgl_port.h"
#include "i2c_manager.h"
#include "oled.h"

#include "esp_lcd_sh1107.h"

static const char *TAG = "OLED";

static lv_obj_t *label = NULL;

// ======================================================
// I2C CONFIG
// ======================================================

#define I2C_HOST                   I2C_NUM_1

#define EXAMPLE_LCD_PIXEL_CLOCK_HZ (400 * 1000)

#define EXAMPLE_PIN_NUM_SDA        7
#define EXAMPLE_PIN_NUM_SCL        8

#define EXAMPLE_PIN_NUM_RST        -1

#define EXAMPLE_I2C_HW_ADDR        0x3C

// ======================================================
// SH1107 DISPLAY CONFIG
// ======================================================

#define EXAMPLE_LCD_H_RES          128
#define EXAMPLE_LCD_V_RES          64

#define EXAMPLE_LCD_CMD_BITS       8
#define EXAMPLE_LCD_PARAM_BITS     8

// ======================================================
// OLED TEXT DISPLAY
// ======================================================

void oled_show_text(const char *text)
{
    if (label == NULL || text == NULL)
    {
        return;
    }

    if (lvgl_port_lock(pdMS_TO_TICKS(500)))
    {

        lv_obj_set_width(label, 128);
        
        lv_label_set_text(label, text);

        lv_obj_align(label, LV_ALIGN_TOP_MID, 0, 0);

        lvgl_port_unlock();
    }
}

void oled_clear(void)
{
    oled_show_text("");
}

// ======================================================
// OLED INIT
// ======================================================

esp_err_t oled_init(void)
{
    ESP_LOGI(TAG, "Initializing I2C");

    i2c_config_t i2c_conf =
    {
        .mode = I2C_MODE_MASTER,

        .sda_io_num = EXAMPLE_PIN_NUM_SDA,
        .scl_io_num = EXAMPLE_PIN_NUM_SCL,

        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,

        .master.clk_speed = EXAMPLE_LCD_PIXEL_CLOCK_HZ,
    };

    // ==================================================
    // PANEL IO
    // ==================================================

    ESP_LOGI(TAG, "Installing panel IO");

    esp_lcd_panel_io_handle_t io_handle = NULL;

    esp_lcd_panel_io_i2c_config_t io_config =
    {
        .dev_addr = EXAMPLE_I2C_HW_ADDR,

        .control_phase_bytes = 1,

        .lcd_cmd_bits = EXAMPLE_LCD_CMD_BITS,
        .lcd_param_bits = EXAMPLE_LCD_PARAM_BITS,

        .dc_bit_offset = 0,

        .flags =
        {
            .disable_control_phase = 1,
        }
    };

    ESP_ERROR_CHECK(
        esp_lcd_new_panel_io_i2c(
            (esp_lcd_i2c_bus_handle_t)I2C_HOST,
            &io_config,
            &io_handle
        )
    );

    // ==================================================
    // SH1107 PANEL DRIVER
    // ==================================================

    ESP_LOGI(TAG, "Installing SH1107 driver");

    esp_lcd_panel_handle_t panel_handle = NULL;

    esp_lcd_panel_dev_config_t panel_config =
    {
        .bits_per_pixel = 1,

        .reset_gpio_num = EXAMPLE_PIN_NUM_RST,
    };

    ESP_ERROR_CHECK(
        esp_lcd_new_panel_sh1107(
            io_handle,
            &panel_config,
            &panel_handle
        )
    );

    ESP_ERROR_CHECK(
        esp_lcd_panel_reset(panel_handle)
    );

    ESP_ERROR_CHECK(
        esp_lcd_panel_init(panel_handle)
    );

    ESP_ERROR_CHECK(
        esp_lcd_panel_disp_on_off(
            panel_handle,
            true
        )
    );

    // ==================================================
    // SH1107 FIXES
    // ==================================================

    esp_lcd_panel_mirror(panel_handle, false, false);

    ESP_ERROR_CHECK(
        esp_lcd_panel_invert_color(
            panel_handle,
            false
        )
    );

    // ==================================================
    // LVGL INIT
    // ==================================================

    ESP_LOGI(TAG, "Initializing LVGL");

    const lvgl_port_cfg_t lvgl_cfg =
        ESP_LVGL_PORT_INIT_CONFIG();

    ESP_ERROR_CHECK(
        lvgl_port_init(&lvgl_cfg)
    );

    const lvgl_port_display_cfg_t disp_cfg =
    {
        .io_handle = io_handle,

        .panel_handle = panel_handle,

        .buffer_size =
            EXAMPLE_LCD_H_RES *
            EXAMPLE_LCD_V_RES,

        .double_buffer = true,

        .hres = EXAMPLE_LCD_H_RES,
        .vres = EXAMPLE_LCD_V_RES,

        .monochrome = true,

        .rotation =
        {
            .swap_xy = false,
            .mirror_x = false,
            .mirror_y = false,
        }
    };

    lv_disp_t *disp =
        lvgl_port_add_disp(&disp_cfg);

    lv_disp_set_rotation(disp, LV_DISP_ROT_180);

    // ==================================================
    // UI
    // ==================================================

    if (lvgl_port_lock(portMAX_DELAY))
    {
        lv_obj_t *scr =
            lv_disp_get_scr_act(disp);

        lv_obj_set_style_bg_color(
            scr,
            lv_color_black(),
            0
        );

        lv_obj_set_style_bg_opa(
            scr,
            LV_OPA_COVER,
            0
        );

        label = lv_label_create(scr);

        //lv_label_set_long_mode(label, LV_LABEL_LONG_SCROLL_CIRCULAR );

        lv_obj_set_width(label, 128);

        lv_obj_set_style_text_align(label, LV_TEXT_ALIGN_CENTER, 0);

        lv_obj_set_style_text_color(label, lv_color_white(), 0);

        lv_label_set_text(label, "RF Ready");

        lv_obj_align(label, LV_ALIGN_CENTER, 0,0);

        lvgl_port_unlock();
    }

    ESP_LOGI(TAG, "OLED initialized");

    return ESP_OK;
}