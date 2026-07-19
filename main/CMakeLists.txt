idf_component_register(
    SRCS
        "main.cpp"
        "audio_features.c"
        "i2c_manager.c"
        "audio_driver.c"
        "microphone.c"
        "sdcard.c"
        "oled.c"
        "DPNet_ori_a_hat_55x3.cc"
       
        "wav_header.h"

    INCLUDE_DIRS
        "."
    
    REQUIRES
        espressif__esp-tflite-micro
        fatfs
        vfs
        sdmmc
        lvgl
        esp-dl
        espressif__esp-nn
        

)