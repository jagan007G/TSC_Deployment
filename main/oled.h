#ifndef OLED_DISPLAY_H
#define OLED_DISPLAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lvgl.h"

/* Initialize OLED display */
esp_err_t oled_init(void);

/* Show scrolling text on OLED */
void oled_show_text(const char *text);

/* Clear OLED screen */
void oled_clear(void);

#ifdef __cplusplus
}
#endif

#endif /* OLED_DISPLAY_H */