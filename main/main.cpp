/*
 * DPNet Real-Time Audio Classification — ESP32-P4 Nano
 *
 * ─────────────────────────────────────────────────────────────────
 * NEW IN THIS VERSION — GATE LOG (Phase 0 instrumentation)
 * ─────────────────────────────────────────────────────────────────
 * GATE LOG  A second CSV (/sdcard/gate_log.csv) now records ONE ROW
 *           FOR EVERY FRESH FRAME that reaches process_task — not
 *           just inferred frames.  This is the dataset used to derive
 *           thresholds / weights / frequencies statistically instead
 *           of hand-picking them.
 *
 *           Columns:
 *             time_ms, frame_id, wav_file,
 *             raw_energy,                (pre-norm, from record_task)
 *             energy, peak, zcr,         (extract_features)
 *             b0..b5,                    (Goertzel band fractions)
 *             novelty,                   (-1 on first frame)
 *             spec_dist,                 (-1 until first inference;
 *                                         now computed on EVERY frame
 *                                         for logging, even when Gate 2
 *                                         is not reached)
 *             gate1_pass,                (1 = novelty gate wants infer)
 *             gate2_pass,                (1 = passed, 0 = vetoed,
 *                                         -1 = not evaluated)
 *             inferred,                  (1 = DPNet actually ran)
 *             pred_class, pred_score,    ("-" / -1 when not inferred)
 *             infer_ms                   (Invoke() duration, -1 if
 *                                         not inferred — gives the
 *                                         latency distribution for
 *                                         P99-based buffer sizing)
 *
 *           Supporting changes:
 *           - RingSlotMeta_t carries raw_energy through the ring so
 *             the value computed in record_task (CPU0) reaches the
 *             gate log written in process_task (CPU1).
 *             ring_push()/ring_pop() signatures extended.
 *           - run_inference() now returns the predicted class index
 *             and reports score + Invoke() duration via out-params.
 *           - frame_id is captured once per fresh frame and used for
 *             BOTH the wav filename and all log rows, and wav_counter
 *             now increments once per processed frame unconditionally.
 *             (Previously, if sd_buf_sem timed out, the counter did
 *             not advance and the next frame reused the same number —
 *             so a classify_log row could point at a WAV that was
 *             actually written by a LATER frame.  IDs are now unique.)
 *           - compute_novelty(): band_flux term restored
 *             (+ band_flux * GOERTZEL_NOVELTY_WEIGHT).  It was being
 *             computed and silently discarded.  With the weight at
 *             0.00f behaviour is IDENTICAL, but the term is wired up
 *             again for when a derived weight replaces 0.00f.
 *
 * ─────────────────────────────────────────────────────────────────
 * BUG FIXES APPLIED (previous versions — unchanged)
 * ─────────────────────────────────────────────────────────────────
 * BUG 1  Energy threshold was evaluated on POST-normalised audio.
 *        FIX: capture_audio() computes mean-abs on current_frame[]
 *        BEFORE DC removal / normalisation / Hamming.
 *
 * BUG 2  wav_pcm_buffer race between record_task and sd_task.
 *        FIX: wav_pcm_buffer written exclusively by save_wav_float().
 *
 * BUG 3  BUFFER_MAX_AGE_MS reduced 5000 → 2000 ms.
 *
 * BUG 4  Bulk stale flush after STALE_FLUSH_LIMIT consecutive drops.
 *
 * BUG 5  audio_initialized flag/log moved after audio_driver_init().
 *
 * IMP 1  NOVELTY_THRESHOLD 0.05 → 0.10.
 * IMP 2  Soft update of prev_features during skipped frames.
 * IMP 3  RING_BUFFER_SLOTS 8 → 4.
 * IMP 4  sdQueue depth 2 → 1.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <dirent.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_task_wdt.h"
#include "esp_nn.h"
#include "esp_timer.h"

#include "audio_pin_config.h"
#include "i2c_manager.h"
#include "audio_driver.h"
#include "microphone.h"
#include "oled.h"
#include "sdcard.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/api/error_reporter.h"

#include "DPNet_ori_a_hat_55x3.h"

static const char *TAG = "DPNet";

// ============================================================
// CONFIG — audio / model (unchanged)
// ============================================================

#define AUDIO_SAMPLES       30225
#define TENSOR_ARENA_SIZE   (4 * 1024 * 1024)

#define RECORD_TASK_STACK   (8  * 1024)
#define PROCESS_TASK_STACK  (32 * 1024)

#define RECORD_TASK_PRIO    6
#define INFERENCE_TASK_PRIO 5

#define GAIN_SHIFT          4
#define OVERLAP_SAMPLES     (AUDIO_SAMPLES / 2)
#define NEW_SAMPLES         (AUDIO_SAMPLES - OVERLAP_SAMPLES)

// ============================================================
// CONFIG — Goertzel spectral bank (unchanged)
// ============================================================
#define GOERTZEL_N            256   // sub-frame length (12.8 ms @ 20 kHz)
#define GOERTZEL_BANDS        6
#define GOERTZEL_SUBFRAMES    16    // sub-frames spread across the window

// Weight of the spectral-flux term inside compute_novelty().
// 0.0f = VERIFICATION MODE: gate behaves EXACTLY as before; bands only logged.
// After confirming logs look sane, set to ~0.5f to activate the gate term.
#define GOERTZEL_NOVELTY_WEIGHT   0.00f
#define SPECTRAL_VETO_THRESHOLD   0.35f

// ============================================================
// CONFIG — Ring Buffer (unchanged)
// ============================================================
#define RING_BUFFER_SLOTS   4   /* WAS 8 — IMP 3 */

// ============================================================
// CONFIG — Novelty Detection (unchanged)
// ============================================================
#define NOVELTY_THRESHOLD   0.10f   /* WAS 0.05 — IMP 1 */
#define NOVELTY_SOFT_ALPHA  0.10f   /* NEW — IMP 2 */

// ============================================================
// CONFIG — Energy / Age (unchanged)
// ============================================================
#define ENERGY_THRESHOLD    0.020f  /* pre-norm — BUG 1 */
#define BUFFER_MAX_AGE_MS   2000    /* WAS 5000 — BUG 3 */
#define STALE_FLUSH_LIMIT   4       /* NEW — BUG 4 */

// ── Classification log (CSV) — inferred frames only ────────
#define CLASSIFY_CSV_PATH  "/sdcard/classify_log.csv"

// ── GATE LOG (CSV) — EVERY fresh frame ─────────────────────
#define GATE_CSV_PATH      "/sdcard/gate_log.csv"

// ============================================================
// STRUCT DEFINITIONS
// ============================================================

// Novelty feature vector (unchanged)
typedef struct
{
    float energy;   // RMS = sqrt( mean(x[i]²) )
    float peak;     // max( |x[i]| )
    float zcr;      // zero-crossing rate in [0, 1]
    float band[GOERTZEL_BANDS]; // normalized energy fraction per band
} AudioFeatures_t;

// Per-slot ring metadata (lives in internal SRAM)
// GATE LOG: raw_energy added so the pre-normalisation energy computed
// in record_task travels with its frame to process_task for logging.
typedef struct
{
    int64_t capture_time_us;
    float   raw_energy;     /* NEW — GATE LOG */
    bool    valid;
} RingSlotMeta_t;

// SD queue item (unchanged)
typedef struct
{
    char   filename[64];
    float *buffer;          // always points to sd_buf
} SdItem;

// OLED queue item (unchanged)
typedef struct { char text[64]; } DisplayItem;

// ============================================================
// GLOBALS — Ring Buffer
// ============================================================

static float           *ring_pool       = nullptr;
static RingSlotMeta_t   ring_meta[RING_BUFFER_SLOTS];
static volatile int     ring_write_idx  = 0;
static volatile int     ring_read_idx   = 0;
static volatile int     ring_count      = 0;
static SemaphoreHandle_t ring_mutex     = NULL;

// ============================================================
// GLOBALS — Working Buffers (PSRAM)
// ============================================================

static float *capture_tmp_buf = nullptr;  // record_task private
static float *process_buf     = nullptr;  // process_task private
static float *sd_buf          = nullptr;  // sd_task write buffer

// Semaphore: 1 = "sd_buf is free for process_task to write"
//            0 = "sd_task is currently reading sd_buf"
static SemaphoreHandle_t sd_buf_sem = NULL;

// ============================================================
// GLOBALS — Novelty Detection
// ============================================================

static AudioFeatures_t  prev_features   = {0};
static bool             prev_valid      = false;
static uint32_t         skipped_frames  = 0;
static uint32_t         inferred_frames = 0;

// ============================================================
// GLOBALS — Misc
// ============================================================

static SemaphoreHandle_t oledMutex;
static SemaphoreHandle_t wav_mutex;

static QueueHandle_t sdQueue;
static QueueHandle_t displayQueue;

static uint32_t wav_counter = 0;

static uint8_t *tensor_arena   = nullptr;

// DMA-capable stereo I2S scratch (internal SRAM, MALLOC_CAP_DMA)
static int16_t *temp_buffer    = nullptr;

// PCM16 output for WAV saving only (internal SRAM).
static int16_t *wav_pcm_buffer = nullptr;

static int  last_class         = -1;
static bool audio_initialized  = false;

// Overlap and current-frame staging (internal BSS, accessed
// by record_task only — no mutex needed)
static float overlap_buffer[OVERLAP_SAMPLES];
static float current_frame[AUDIO_SAMPLES];

// Band fingerprint of the most recently INFERRED frame — Gate 2's reference.
static float last_inferred_band[GOERTZEL_BANDS] = {0.0f};
static bool  last_inferred_valid                = false;

// Set in app_main before record_task starts; used by ring_push()
static TaskHandle_t processTaskHandle = NULL;

//-------- Classification log (CSV) — inferred frames --------
static SemaphoreHandle_t csv_mutex = NULL;

//-------- GATE LOG (CSV) — every fresh frame -----------------
static SemaphoreHandle_t gate_csv_mutex = NULL;   /* NEW — GATE LOG */

const tflite::Model      *model          = nullptr;
tflite::MicroInterpreter *interpreter    = nullptr;
TfLiteTensor             *input          = nullptr;
TfLiteTensor             *output         = nullptr;
tflite::ErrorReporter    *error_reporter = nullptr;

// ============================================================
// LABEL MAP (unchanged)
// ============================================================

const char *label_map[] = {
    "Flush", "Door", "Basin_Tap", "Walker_Crutch",
    "Bathroom_Tap", "No_Class", "Shower"
};

// ============================================================
// RING BUFFER API
//
// GATE LOG CHANGE: push/pop now carry raw_energy per slot.
// ============================================================

void ring_init(void)
{
    ring_mutex     = xSemaphoreCreateMutex();
    ring_write_idx = 0;
    ring_read_idx  = 0;
    ring_count     = 0;
    memset(ring_meta, 0, sizeof(ring_meta));

    ESP_LOGI(TAG,
             "Ring buffer: %d slots × %d samples × 4 B = %u KB (PSRAM)",
             RING_BUFFER_SLOTS, AUDIO_SAMPLES,
             (unsigned)((size_t)RING_BUFFER_SLOTS
                        * AUDIO_SAMPLES * sizeof(float) / 1024));
}

void ring_push(const float *src, int64_t capture_time_us, float raw_energy)
{
    xSemaphoreTake(ring_mutex, portMAX_DELAY);

    float *dst = ring_pool + (size_t)ring_write_idx * AUDIO_SAMPLES;
    memcpy(dst, src, AUDIO_SAMPLES * sizeof(float));

    ring_meta[ring_write_idx].capture_time_us = capture_time_us;
    ring_meta[ring_write_idx].raw_energy      = raw_energy;   /* GATE LOG */
    ring_meta[ring_write_idx].valid           = true;

    if (ring_count == RING_BUFFER_SLOTS)
    {
        // Overwrite oldest — advance read pointer silently
        ring_read_idx = (ring_read_idx + 1) % RING_BUFFER_SLOTS;
    }
    else
    {
        ring_count++;
    }

    ring_write_idx = (ring_write_idx + 1) % RING_BUFFER_SLOTS;

    xSemaphoreGive(ring_mutex);

    if (processTaskHandle)
        xTaskNotifyGive(processTaskHandle);
}

bool ring_pop(float *dst, int64_t *out_capture_time, float *out_raw_energy)
{
    xSemaphoreTake(ring_mutex, portMAX_DELAY);

    if (ring_count == 0)
    {
        xSemaphoreGive(ring_mutex);
        return false;
    }

    const float *src = ring_pool + (size_t)ring_read_idx * AUDIO_SAMPLES;
    memcpy(dst, src, AUDIO_SAMPLES * sizeof(float));

    *out_capture_time = ring_meta[ring_read_idx].capture_time_us;
    if (out_raw_energy)                                        /* GATE LOG */
        *out_raw_energy = ring_meta[ring_read_idx].raw_energy;
    ring_meta[ring_read_idx].valid = false;

    ring_read_idx = (ring_read_idx + 1) % RING_BUFFER_SLOTS;
    ring_count--;

    xSemaphoreGive(ring_mutex);
    return true;
}

void ring_flush(void)
{
    xSemaphoreTake(ring_mutex, portMAX_DELAY);
    ring_read_idx = ring_write_idx;
    ring_count    = 0;
    for (int i = 0; i < RING_BUFFER_SLOTS; i++)
        ring_meta[i].valid = false;
    xSemaphoreGive(ring_mutex);
}

int ring_get_count(void)
{
    if (xSemaphoreTake(ring_mutex, pdMS_TO_TICKS(10)) != pdTRUE)
        return -1;
    int c = ring_count;
    xSemaphoreGive(ring_mutex);
    return c;
}

// ============================================================
// NOVELTY DETECTION
// ============================================================

// Summed power in each band across the window, returned as FRACTIONS
// (band / sum-of-bands). Fractions are invariant to peak-normalization.
static void goertzel_band_fractions(const float *buf, size_t n,
                                    float out_frac[GOERTZEL_BANDS])
{
    // coeff = 2*cos(2*pi*k/N), N=256, SR=20kHz  (from dataset analysis)
    static const float coeff[GOERTZEL_BANDS] = {
        1.997591f,  //  156.3 Hz (k=2)    Door / Walker low-frequency impact
        1.913881f,  //  937.5 Hz (k=12)   Door / Walker onset
        1.481902f,  // 2343.8 Hz (k=30)   Flush transient
        1.151616f,  // 3046.9 Hz (k=39)   Best transient vs No_Class
        0.000000f,  // 5000.0 Hz (k=64)   Water energy
        -0.627363f, // 6015.6 Hz (k=77)   Water + Walker high-frequency
    };

    float acc[GOERTZEL_BANDS] = {0.0f};

    if (n < GOERTZEL_N) {
        for (int b = 0; b < GOERTZEL_BANDS; b++) out_frac[b] = 0.0f;
        return;
    }

    float sub[GOERTZEL_N];                 // SRAM stack scratch (1 KB)
    const size_t span = n - GOERTZEL_N;

    for (int s = 0; s < GOERTZEL_SUBFRAMES; s++)
    {
        size_t start = (GOERTZEL_SUBFRAMES > 1)
                     ? (size_t)((uint64_t)span * s / (GOERTZEL_SUBFRAMES - 1))
                     : 0;
        memcpy(sub, buf + start, GOERTZEL_N * sizeof(float));  // 1 PSRAM pass

        for (int b = 0; b < GOERTZEL_BANDS; b++)
        {
            const float c = coeff[b];
            float s1 = 0.0f, s2 = 0.0f;
            for (int i = 0; i < GOERTZEL_N; i++) {
                float s0 = sub[i] + c * s1 - s2;
                s2 = s1; s1 = s0;
            }
            acc[b] += s1*s1 + s2*s2 - c*s1*s2;   // |X[k]|^2, no sqrt
        }
    }

    float total = 0.0f;
    for (int b = 0; b < GOERTZEL_BANDS; b++) total += acc[b];
    if (total > 1e-12f) {
        float inv = 1.0f / total;
        for (int b = 0; b < GOERTZEL_BANDS; b++) out_frac[b] = acc[b] * inv;
    } else {
        for (int b = 0; b < GOERTZEL_BANDS; b++) out_frac[b] = 0.0f;
    }
}

AudioFeatures_t extract_features(const float *buf, size_t n)
{
    AudioFeatures_t f = {0};
    float sum_sq    = 0.0f;
    int   crossings = 0;

    for (size_t i = 0; i < n; i++)
    {
        const float v  = buf[i];
        const float av = fabsf(v);

        sum_sq += v * v;
        if (av > f.peak) f.peak = av;
        if (i > 0 && ((buf[i - 1] >= 0.0f) != (v >= 0.0f)))
            crossings++;
    }

    f.energy = sqrtf(sum_sq / (float)n);
    f.zcr    = (n > 1) ? (float)crossings / (float)(n - 1) : 0.0f;
    goertzel_band_fractions(buf, n, f.band);
    return f;
}

// GATE LOG NOTE: the band_flux term was being computed and then
// discarded (dropped from the return in a previous edit).  Restored
// here with GOERTZEL_NOVELTY_WEIGHT — still 0.00f, so behaviour is
// IDENTICAL, but the wiring is back for when a derived weight lands.
float compute_novelty(const AudioFeatures_t *curr, const AudioFeatures_t *prev)
{
    float band_flux = 0.0f;
    for (int b = 0; b < GOERTZEL_BANDS; b++)
        band_flux += fabsf(curr->band[b] - prev->band[b]);

    return (fabsf(curr->energy - prev->energy) * 0.5f)
         + (fabsf(curr->peak   - prev->peak  ) * 0.3f)
         + (fabsf(curr->zcr    - prev->zcr   ) * 0.2f)
         + (band_flux * GOERTZEL_NOVELTY_WEIGHT);
}

// ============================================================
// AUDIO CAPTURE (unchanged)
// ============================================================

bool capture_audio(float *target_buffer, float *raw_energy_out)
{
    size_t samples_read = microphone_read(temp_buffer, AUDIO_SAMPLES);

    ESP_LOGI(TAG, "RAW PCM: %d %d %d %d",
             temp_buffer[0], temp_buffer[1],
             temp_buffer[2], temp_buffer[3]);

    if (samples_read == 0)
    {
        ESP_LOGE(TAG, "No audio captured");
        return false;
    }

    size_t mono_samples = samples_read / 2;
    if (mono_samples > AUDIO_SAMPLES)
        mono_samples = AUDIO_SAMPLES;

    // ─── Stereo → Mono ───────────────────────────────────────
    for (size_t i = 0; i < mono_samples; i++)
    {
        int16_t left_sample = temp_buffer[i * 2];
        current_frame[i] = (float)left_sample / 32768.0f;
    }

    // ─── Raw energy on pre-normalised float values (BUG 1) ───
    if (raw_energy_out != nullptr)
    {
        float acc = 0.0f;
        size_t cnt = 0;
        for (size_t i = 0; i < mono_samples; i += 100)
        {
            acc += fabsf(current_frame[i]);
            cnt++;
        }
        *raw_energy_out = (cnt > 0) ? acc / (float)cnt : 0.0f;
    }

    // ─── Sliding Window — 50 % overlap (unchanged) ───────────
    memcpy(target_buffer,
           overlap_buffer,
           OVERLAP_SAMPLES * sizeof(float));
    memcpy(target_buffer + OVERLAP_SAMPLES,
           current_frame,
           NEW_SAMPLES * sizeof(float));
    memcpy(overlap_buffer,
           target_buffer + NEW_SAMPLES,
           OVERLAP_SAMPLES * sizeof(float));

    // ─── DC Offset Removal (unchanged) ───────────────────────
    float mean = 0.0f;
    for (size_t i = 0; i < mono_samples; i++)
        mean += target_buffer[i];
    mean /= (float)mono_samples;
    for (size_t i = 0; i < mono_samples; i++)
        target_buffer[i] -= mean;

    // ─── Peak Normalisation (unchanged) ──────────────────────
    float max_val = 0.0f;
    for (size_t i = 0; i < mono_samples; i++)
    {
        float v = fabsf(target_buffer[i]);
        if (v > max_val) max_val = v;
    }
    if (max_val > 1e-6f)
    {
        float inv = 1.0f / max_val;
        for (size_t i = 0; i < mono_samples; i++)
            target_buffer[i] *= inv;
    }

    // ─── Hamming Window (unchanged) ──────────────────────────
    const float alpha = 0.54f;
    const float beta  = 0.46f;
    for (size_t i = 0; i < mono_samples; i++)
    {
        float w = alpha
                  - beta * cosf((2.0f * M_PI * i) / (float)(mono_samples - 1));
        target_buffer[i] *= w;
    }

    /* StandardScaler disabled — see previous version */

    // ─── Zero-pad remaining (unchanged) ──────────────────────
    for (size_t i = mono_samples; i < AUDIO_SAMPLES; i++)
        target_buffer[i] = 0.0f;

    ESP_LOGI(TAG, "Audio[0]=%f Audio[100]=%f Audio[500]=%f",
             target_buffer[0], target_buffer[100], target_buffer[500]);

    return true;
}

// ============================================================
// RECORD TASK
//
// GATE LOG CHANGE: raw_energy is passed into ring_push() so it
// rides along with its frame to process_task.
// ============================================================

void record_task(void *arg)
{
    esp_task_wdt_add(NULL);

    while (1)
    {
        esp_task_wdt_reset();

        float raw_energy = 0.0f;

        const int64_t capture_time = esp_timer_get_time();

        if (!capture_audio(capture_tmp_buf, &raw_energy))
            continue;

        // ─────────────────────────────────────────────────────
        // ENERGY-BASED RING FLUSH (unchanged)
        // ─────────────────────────────────────────────────────
        if (raw_energy > ENERGY_THRESHOLD)
        {
            int before = ring_get_count();
            if (before > 0)
            {
                ring_flush();
                ESP_LOGW(TAG,
                         "⚡ High energy (raw=%.4f) — flushed %d ring slot(s)",
                         raw_energy, before);
            }
        }

        // Push into ring; overwrites oldest if full.
        ring_push(capture_tmp_buf, capture_time, raw_energy);  /* GATE LOG */

        ESP_LOGD(TAG, "Pushed frame - ring=%d raw_energy=%.4f",
                 ring_get_count(), raw_energy);
    }
}

// ============================================================
// WAV HEADER WRITER (unchanged)
// ============================================================

void write_wav_header(FILE *file, uint32_t data_size)
{
    uint32_t sampleRate    = SAMPLE_RATE;
    uint16_t bitsPerSample = 16;
    uint16_t numChannels   = 1;
    uint32_t byteRate      = sampleRate * numChannels * bitsPerSample / 8;
    uint16_t blockAlign    = numChannels * bitsPerSample / 8;
    uint32_t chunkSize     = 36 + data_size;

    fwrite("RIFF",         1, 4, file);
    fwrite(&chunkSize,     4, 1, file);
    fwrite("WAVE",         1, 4, file);
    fwrite("fmt ",         1, 4, file);
    uint32_t sub1 = 16; uint16_t fmt = 1;
    fwrite(&sub1,          4, 1, file);
    fwrite(&fmt,           2, 1, file);
    fwrite(&numChannels,   2, 1, file);
    fwrite(&sampleRate,    4, 1, file);
    fwrite(&byteRate,      4, 1, file);
    fwrite(&blockAlign,    2, 1, file);
    fwrite(&bitsPerSample, 2, 1, file);
    fwrite("data",         1, 4, file);
    fwrite(&data_size,     4, 1, file);
}

// ============================================================
// WAV SAVER (unchanged)
// ============================================================

bool save_wav_float(const char *filename, float *audio_buffer, size_t samples)
{
    struct stat st;
    if (stat(filename, &st) == 0)
    {
        ESP_LOGW(TAG, "EXISTS: %s  size=%ld  mode=0%03o",
            filename, (long)st.st_size, (unsigned)(st.st_mode & 0777));
    }
    else
    {
        ESP_LOGI(TAG, "NEW:    %s", filename);
    }

    xSemaphoreTake(wav_mutex, portMAX_DELAY);

    ESP_LOGI(TAG, "Saving WAV: %s", filename);

    FILE *f = fopen(filename, "wb");

    if (!f)
    {
        int e = errno;
        ESP_LOGE(TAG, "fopen failed: errno=%d (%s)", e, strerror(e));

        if (e == EACCES)
            ESP_LOGE(TAG,
                     "SD write-protected or FAT corrupted "
                     "(FR_DENIED). Check the physical write-protect tab.");

        xSemaphoreGive(wav_mutex);
        return false;
    }

    for (size_t i = 0; i < samples; i++)
    {
        float s = audio_buffer[i];
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        wav_pcm_buffer[i] = (int16_t)(s * 32767.0f);
    }

    uint32_t data_size = samples * sizeof(int16_t);
    write_wav_header(f, data_size);

    size_t written = fwrite(wav_pcm_buffer, sizeof(int16_t), samples, f);
    fflush(f);
    fclose(f);

    if (written != samples)
    {
        ESP_LOGE(TAG, "WAV write incomplete (%u/%u samples)",
                 (unsigned)written, (unsigned)samples);
        xSemaphoreGive(wav_mutex);
        return false;
    }

    ESP_LOGI(TAG, "WAV saved: %s", filename);
    xSemaphoreGive(wav_mutex);
    return true;
}

// ============================================================
// CLEAR RECORDINGS — wipe previous session's WAVs at boot so
// numbering restarts from 0 with no leftover files.
// ============================================================
static void recordings_clear(void)
{
    mkdir("/sdcard/recordings", 0777);   // ensure it exists

    DIR *dir = opendir("/sdcard/recordings");
    if (!dir) {
        ESP_LOGW(TAG, "recordings dir unavailable — nothing to clear");
        return;
    }

    struct dirent *e;
    char path[96];
    int  removed = 0;

    while ((e = readdir(dir)) != NULL) {
        // only delete our WAV files, leave anything else alone
        if (strstr(e->d_name, ".wav") == NULL) continue;
        snprintf(path, sizeof(path), "/sdcard/recordings/%s", e->d_name);
        if (unlink(path) == 0) removed++;
        else ESP_LOGW(TAG, "could not delete %s (errno=%d)", path, errno);
    }
    closedir(dir);

    wav_counter = 0;   // restart numbering from the beginning
    ESP_LOGI(TAG, "recordings cleared: %d file(s) removed, wav_counter=0", removed);
}

// ============================================================
// CLASSIFY CSV (inferred frames only — unchanged)
// ============================================================

static void classify_csv_init(void)
{
    csv_mutex = xSemaphoreCreateMutex();

    struct stat st;
    bool need_header = (stat(CLASSIFY_CSV_PATH, &st) != 0) || (st.st_size == 0);
    if (!need_header) return;

    FILE *f = fopen(CLASSIFY_CSV_PATH, "w");
    if (!f) {
        ESP_LOGE(TAG, "classify_csv: cannot open %s (errno=%d)", CLASSIFY_CSV_PATH, errno);
        return;
    }
    fprintf(f,
        "time_ms,frame_idx,top1,top1_score,top2,top2_score,margin,"
        "b0,b1,b2,b3,b4,b5,env_decay,env_var\n");

    fclose(f);
    ESP_LOGI(TAG, "classify_csv: header written to %s", CLASSIFY_CSV_PATH);
}

// ============================================================
// GATE LOG CSV (every fresh frame) — NEW
// ============================================================

static void gate_csv_init(void)
{
    gate_csv_mutex = xSemaphoreCreateMutex();

    struct stat st;
    bool need_header = (stat(GATE_CSV_PATH, &st) != 0) || (st.st_size == 0);
    if (!need_header) return;

    FILE *f = fopen(GATE_CSV_PATH, "w");
    if (!f) {
        ESP_LOGE(TAG, "gate_csv: cannot open %s (errno=%d)", GATE_CSV_PATH, errno);
        return;
    }
    fprintf(f,
        "time_ms,frame_id,wav_file,raw_energy,energy,peak,zcr,"
        "b0,b1,b2,b3,b4,b5,novelty,spec_dist,"
        "gate1_pass,gate2_pass,inferred,pred_class,pred_score,infer_ms\n");
    fclose(f);
    ESP_LOGI(TAG, "gate_csv: header written to %s", GATE_CSV_PATH);
}

// One row per fresh frame.  Sentinels: novelty / spec_dist / pred_score /
// infer_ms are -1 when not applicable; gate2_pass is -1 when not evaluated;
// pred_class is "-" when no inference ran.
static void gate_log_row(uint32_t frame_id, const char *wav_file,
                         float raw_energy, const AudioFeatures_t *f,
                         float novelty, float spec_dist,
                         int gate1_pass, int gate2_pass, int inferred,
                         const char *pred_class, float pred_score,
                         int64_t infer_ms)
{
    if (!gate_csv_mutex) return;
    if (xSemaphoreTake(gate_csv_mutex, pdMS_TO_TICKS(200)) != pdTRUE) {
        ESP_LOGW(TAG, "gate_csv: mutex timeout — row dropped");
        return;
    }

    FILE *gf = fopen(GATE_CSV_PATH, "a");
    if (gf) {
        fprintf(gf,
            "%lld,%lu,%s,%.5f,%.5f,%.5f,%.5f,"
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
            "%.4f,%.4f,%d,%d,%d,%s,%.3f,%lld\n",
            esp_timer_get_time() / 1000LL,
            (unsigned long)frame_id, wav_file,
            raw_energy, f->energy, f->peak, f->zcr,
            f->band[0], f->band[1], f->band[2],
            f->band[3], f->band[4], f->band[5],
            novelty, spec_dist,
            gate1_pass, gate2_pass, inferred,
            pred_class, pred_score, infer_ms);
        fclose(gf);
    } else {
        ESP_LOGE(TAG, "gate_csv: cannot open %s (errno=%d)", GATE_CSV_PATH, errno);
    }

    xSemaphoreGive(gate_csv_mutex);
}

// ============================================================
// WATER-CLASS HELPER + ENVELOPE (unchanged)
// ============================================================

static inline bool is_water_class(int idx)
{
    return (idx == 0 || idx == 2 || idx == 4 || idx == 6);
}

static void window_envelope(const float *buf, size_t n,
                            float *out_decay, float *out_var)
{
    const int SUB = GOERTZEL_SUBFRAMES;
    float e[GOERTZEL_SUBFRAMES];
    const size_t span = (n > GOERTZEL_N) ? (n - GOERTZEL_N) : 0;

    for (int s = 0; s < SUB; s++) {
        size_t start = (SUB > 1) ? (size_t)((uint64_t)span * s / (SUB - 1)) : 0;
        const float *p = buf + start;
        float acc = 0.0f;
        for (int i = 0; i < GOERTZEL_N; i++) acc += p[i] * p[i];
        e[s] = acc;
    }

    float mean = 0.0f;
    for (int s = 0; s < SUB; s++) mean += e[s];
    mean /= SUB;

    float var = 0.0f, first = 0.0f, second = 0.0f;
    for (int s = 0; s < SUB; s++) {
        float d = e[s] - mean; var += d * d;
        if (s < SUB / 2) first += e[s]; else second += e[s];
    }
    var /= SUB;

    float denom = (mean > 1e-12f) ? mean : 1e-12f;
    *out_var   = sqrtf(var) / denom;
    *out_decay = (first - second) / (first + second + 1e-12f);
}

// ============================================================
// SD TASK (unchanged)
// ============================================================

void sd_task(void *arg)
{
    SdItem item;
    while (1)
    {
        if (xQueueReceive(sdQueue, &item, portMAX_DELAY) == pdTRUE)
        {
            save_wav_float(item.filename, item.buffer, AUDIO_SAMPLES);
            // Return sd_buf ownership to process_task
            xSemaphoreGive(sd_buf_sem);
        }
    }
}

// ============================================================
// OLED TASK (unchanged)
// ============================================================

void oled_task(void *arg)
{
    DisplayItem item;
    while (1)
    {
        if (xQueueReceive(displayQueue, &item, portMAX_DELAY) == pdTRUE)
        {
            if (xSemaphoreTake(oledMutex, pdMS_TO_TICKS(100)) == pdTRUE)
            {
                oled_clear();
                oled_show_text(item.text);
                xSemaphoreGive(oledMutex);
            }
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }
}

// ============================================================
// RUN INFERENCE
//
// GATE LOG CHANGES:
//   - Returns the predicted class index (or -1 on failure)
//   - Reports top-1 score via *out_score
//   - Times Invoke() and reports duration via *out_infer_ms.
//     This is the raw data for the P99 latency distribution
//     used to derive ring-buffer sizing.
// Everything else (top-2 scan, classify CSV row, OLED) unchanged.
// ============================================================

int run_inference(float *audio_buffer, const AudioFeatures_t *curr_feat,
                  uint32_t frame_idx, float *out_score, int64_t *out_infer_ms)
{
    ESP_LOGI(TAG, "Running inference...");

    memcpy(interpreter->typed_input_tensor<float>(0),
           audio_buffer,
           AUDIO_SAMPLES * sizeof(float));

    ESP_LOGI(TAG, "Invoke start");
    const int64_t t0 = esp_timer_get_time();               /* GATE LOG */
    TfLiteStatus status = interpreter->Invoke();
    const int64_t infer_ms =
        (esp_timer_get_time() - t0) / 1000LL;              /* GATE LOG */
    ESP_LOGI(TAG, "Invoke returned (%lld ms)", infer_ms);

    if (out_infer_ms) *out_infer_ms = infer_ms;            /* GATE LOG */

    if (status != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Invoke failed");
        return -1;
    }
    ESP_LOGI(TAG, "Inference complete");

    float *out = interpreter->typed_output_tensor<float>(0);
    int n = output->bytes / sizeof(float);
    int loop_n = (n < 7) ? n : 7;

    // top-2 scan
    float s1 = -9999.0f, s2 = -9999.0f;
    int   i1 = -1,       i2 = -1;
    for (int i = 0; i < loop_n; i++) {
        if (out[i] > s1)      { s2 = s1; i2 = i1; s1 = out[i]; i1 = i; }
        else if (out[i] > s2) { s2 = out[i]; i2 = i; }
    }
    float best_score = s1;
    int   best_index = i1;

    if (out_score) *out_score = best_score;                /* GATE LOG */

    // envelope + CSV row (classify log — unchanged)
    float env_decay = 0.0f, env_var = 0.0f;
    window_envelope(audio_buffer, AUDIO_SAMPLES, &env_decay, &env_var);

    if (csv_mutex && xSemaphoreTake(csv_mutex, pdMS_TO_TICKS(200)) == pdTRUE) {
        FILE *cf = fopen(CLASSIFY_CSV_PATH, "a");
        if (cf) {
            int both_water = (is_water_class(i1) && is_water_class(i2)) ? 1 : 0;
            fprintf(cf,
                "%lld,%lu,%s,%.3f,%s,%.3f,%.3f,%d,"
                "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f\n",
                esp_timer_get_time() / 1000LL,
                (unsigned long)frame_idx,
                (i1 >= 0 ? label_map[i1] : "?"), s1,
                (i2 >= 0 ? label_map[i2] : "?"), s2,
                s1 - s2, both_water,
                curr_feat->band[0], curr_feat->band[1], curr_feat->band[2],
                curr_feat->band[3], curr_feat->band[4], curr_feat->band[5],
                env_decay, env_var);
            fclose(cf);
        }
        xSemaphoreGive(csv_mutex);
    }

    if (best_index >= 0 && best_index < 7)
    {
        if (best_index != last_class)
        {
            last_class = best_index;
            char text[64];
            snprintf(text, sizeof(text), "%s %.1f%%",
                     label_map[best_index], best_score * 100.0f);
            ESP_LOGI(TAG, "Prediction: %s", text);

            DisplayItem disp;
            snprintf(disp.text, sizeof(disp.text), "%s %.1f%%",
                     label_map[best_index], best_score * 100.0f);
            xQueueSend(displayQueue, &disp, 0);
        }
    }
    else
    {
        ESP_LOGE(TAG, "Invalid class index");
        return -1;
    }

    return best_index;                                     /* GATE LOG */
}

// ============================================================
// TFLITE INIT (unchanged)
// ============================================================

bool init_tflite()
{
    ESP_LOGI(TAG, "Initializing TensorFlow Lite...");

    tensor_arena = (uint8_t *)heap_caps_malloc(
        TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena)
    {
        ESP_LOGE(TAG, "Tensor arena alloc failed");
        return false;
    }

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "Schema mismatch");
        return false;
    }

    static tflite::MicroMutableOpResolver<25> resolver;
    resolver.AddConv2D();          resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();  resolver.AddSoftmax();
    resolver.AddReshape();         resolver.AddQuantize();
    resolver.AddDequantize();      resolver.AddAdd();
    resolver.AddMul();             resolver.AddRelu();
    resolver.AddPad();             resolver.AddMean();
    resolver.AddTranspose();       resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();       resolver.AddLogistic();
    resolver.AddConcatenation();   resolver.AddPack();
    resolver.AddSplit();           resolver.AddSplitV();
    resolver.AddStridedSlice();    resolver.AddShape();
    resolver.AddGather();          resolver.AddSlice();
    resolver.AddSub();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return false;
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "Arena used: %d B  Input: %d B  type: %d",
             interpreter->arena_used_bytes(), input->bytes, input->type);
    return true;
}

// ============================================================
// PROCESS TASK
//
// GATE LOG CHANGES:
//   1. ring_pop() now yields the frame's raw_energy.
//   2. frame_id captured ONCE per fresh frame; used for the gate
//      row, the classify row (via run_inference), and the WAV
//      filename — so every artefact of a frame shares one ID.
//   3. spec_dist computed on EVERY frame (when a Gate-2 reference
//      exists), even if Gate 1 already skipped — costs 6 subtracts,
//      gives the full distribution needed for threshold derivation.
//   4. gate_log_row() written for every fresh frame after all gate
//      decisions and (optional) inference are complete.
//   5. wav_counter increments once per processed frame regardless
//      of SD-save outcome, guaranteeing unique frame IDs.
// ============================================================

void process_task(void *arg)
{
    while (1)
    {
        // Block until ring_push() wakes us, or 2 s safety timeout.
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(2000));

        int64_t capture_time_us   = 0;
        float   frame_raw_energy  = 0.0f;   /* GATE LOG */
        int     consecutive_stale = 0;      // BUG 4: stale-run counter

        int ring_before = ring_get_count();
        while (ring_pop(process_buf, &capture_time_us, &frame_raw_energy))
        {
            // ─────────────────────────────────────────────────
            // BUFFER AGE GATE  (BUG 3 + BUG 4 — unchanged)
            // ─────────────────────────────────────────────────
            int64_t age_ms = (esp_timer_get_time() - capture_time_us) / 1000LL;

            if (age_ms > BUFFER_MAX_AGE_MS)
            {
                consecutive_stale++;
                ESP_LOGW(TAG,
                         "⏱ Stale frame (age=%lld ms) dropped [run=%d]",
                         age_ms, consecutive_stale);

                if (consecutive_stale >= STALE_FLUSH_LIMIT)
                {
                    ring_flush();
                    ESP_LOGW(TAG,
                             "Bulk-flushed ring after %d consecutive "
                             "stale frames",
                             consecutive_stale);
                    consecutive_stale = 0;
                    break;
                }
                continue;
            }

            // Valid (fresh) frame — reset stale counter
            consecutive_stale = 0;

            // GATE LOG: one ID per fresh frame, shared by the gate row,
            // the classify row, and the WAV filename.
            const uint32_t frame_id = wav_counter;
            char wav_name[32];
            snprintf(wav_name, sizeof(wav_name),
                     "rec_%06lu.wav", (unsigned long)frame_id);

            ESP_LOGI(TAG, "Frame: age=%lld ms  ring=%d  skip=%lu  infer=%lu",
                     age_ms, ring_before, skipped_frames, inferred_frames);

            ring_before = ring_get_count();

            // ─────────────────────────────────────────────────
            // NOVELTY DETECTION (Gate 1)
            // ─────────────────────────────────────────────────
            const AudioFeatures_t curr =
                extract_features(process_buf, AUDIO_SAMPLES);

            ESP_LOGI(TAG, "BANDS %.3f %.3f %.3f %.3f %.3f %.3f",
                     curr.band[0], curr.band[1], curr.band[2],
                     curr.band[3], curr.band[4], curr.band[5]);

            bool  run_infer   = true;
            float novelty_val = -1.0f;   /* GATE LOG: -1 = first frame */
            int   gate1_pass  = 1;       /* GATE LOG */

            if (prev_valid)
            {
                float nov   = compute_novelty(&curr, &prev_features);
                novelty_val = nov;                          /* GATE LOG */

                if (nov < NOVELTY_THRESHOLD)
                {
                    // Frame is similar — skip DPNet
                    run_infer  = false;
                    gate1_pass = 0;                         /* GATE LOG */
                    skipped_frames++;

                    // IMP 2: SOFT UPDATE of novelty reference (unchanged)
                    prev_features.energy +=
                        NOVELTY_SOFT_ALPHA
                        * (curr.energy - prev_features.energy);
                    prev_features.peak +=
                        NOVELTY_SOFT_ALPHA
                        * (curr.peak   - prev_features.peak);
                    prev_features.zcr +=
                        NOVELTY_SOFT_ALPHA
                        * (curr.zcr    - prev_features.zcr);

                    for (int b = 0; b < GOERTZEL_BANDS; b++)
                        prev_features.band[b] += NOVELTY_SOFT_ALPHA
                            * (curr.band[b] - prev_features.band[b]);

                    ESP_LOGI(TAG,
                             "🔇 Similar (novelty=%.4f < %.4f) — skip"
                             "  [s=%lu i=%lu]",
                             nov, NOVELTY_THRESHOLD,
                             skipped_frames, inferred_frames);
                }
                else
                {
                    ESP_LOGI(TAG,
                             "🔔 Novel  (novelty=%.4f threshold=%.4f) — infer"
                             "  [s=%lu i=%lu]",
                             nov, NOVELTY_THRESHOLD,
                             skipped_frames, inferred_frames);
                }
            }
            else
            {
                ESP_LOGI(TAG, "First frame — always infer");
            }

            // ─────────────────────────────────────────────────
            // GATE 2 — FREQUENCY REDUNDANCY VETO
            //
            // GATE LOG CHANGE: spec_dist is computed whenever a
            // reference exists — even if Gate 1 already skipped —
            // so the gate log captures its value on EVERY frame.
            // The veto DECISION still only applies on the original
            // path (run_infer && last_inferred_valid); behaviour
            // of the gate itself is unchanged.
            // ─────────────────────────────────────────────────
            float spec_dist  = -1.0f;   /* GATE LOG: -1 = no reference yet */
            int   gate2_pass = -1;      /* GATE LOG: -1 = not evaluated   */

            if (last_inferred_valid)
            {
                spec_dist = 0.0f;
                for (int b = 0; b < GOERTZEL_BANDS; b++)
                    spec_dist += fabsf(curr.band[b] - last_inferred_band[b]);
            }

            if (run_infer && last_inferred_valid)
            {
                if (spec_dist < SPECTRAL_VETO_THRESHOLD)
                {
                    run_infer  = false;
                    gate2_pass = 0;                         /* GATE LOG */
                    skipped_frames++;
                    ESP_LOGI(TAG,
                             "🎵 Freq-veto (dist=%.3f < %.3f) skip redundant"
                             "  [s=%lu i=%lu]",
                             spec_dist, SPECTRAL_VETO_THRESHOLD,
                             skipped_frames, inferred_frames);
                }
                else
                {
                    gate2_pass = 1;                         /* GATE LOG */
                    ESP_LOGI(TAG, "🎵 Freq-pass (dist=%.3f) — infer", spec_dist);
                }
            }

            // ─────────────────────────────────────────────────
            // INFERENCE (unchanged decision; result captured for log)
            // ─────────────────────────────────────────────────
            int     pred_idx   = -1;    /* GATE LOG */
            float   pred_score = -1.0f; /* GATE LOG */
            int64_t infer_ms   = -1;    /* GATE LOG */

            if (run_infer)
            {
                // Hard update: replace reference with current frame
                prev_features   = curr;
                prev_valid      = true;
                inferred_frames++;

                // Gate 2 reference: write-on-infer ONLY.
                for (int b = 0; b < GOERTZEL_BANDS; b++)
                    last_inferred_band[b] = curr.band[b];
                last_inferred_valid = true;

                pred_idx = run_inference(process_buf, &curr, frame_id,
                                         &pred_score, &infer_ms);
            }

            // ─────────────────────────────────────────────────
            // GATE LOG ROW — every fresh frame, decision complete
            // ─────────────────────────────────────────────────
            gate_log_row(frame_id, wav_name,
                         frame_raw_energy, &curr,
                         novelty_val, spec_dist,
                         gate1_pass, gate2_pass,
                         run_infer ? 1 : 0,
                         (pred_idx >= 0 && pred_idx < 7)
                             ? label_map[pred_idx] : "-",
                         pred_score, infer_ms);

            // ─────────────────────────────────────────────────
            // SD SAVE (all frames saved regardless of novelty)
            // ─────────────────────────────────────────────────
            if (xSemaphoreTake(sd_buf_sem, pdMS_TO_TICKS(5000)) == pdTRUE)
            {
                memcpy(sd_buf, process_buf, AUDIO_SAMPLES * sizeof(float));

                SdItem sd;
                snprintf(sd.filename, sizeof(sd.filename),
                         "/sdcard/recordings/%s", wav_name);  /* GATE LOG */
                sd.buffer = sd_buf;

                if (xQueueSend(sdQueue, &sd, pdMS_TO_TICKS(1000)) != pdTRUE)
                {
                    xSemaphoreGive(sd_buf_sem);
                    ESP_LOGW(TAG, "SD queue full — skipping save");
                }
                // sd_task gives sd_buf_sem back after the file write
            }
            else
            {
                ESP_LOGW(TAG, "sd_buf_sem timeout — skipping save");
            }

            // GATE LOG: unique ID per processed frame, regardless of
            // whether the SD save succeeded (see header note).
            wav_counter++;

        } // while (ring_pop)

    } // while (1)
}

// ============================================================
// APP MAIN
//
// GATE LOG CHANGE: gate_csv_init() called after SD mount.
// ============================================================

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting DPNet System");
    vTaskDelay(pdMS_TO_TICKS(1000));

    esp_task_wdt_config_t wdt_cfg = {
        .timeout_ms     = 30000,
        .idle_core_mask = (1 << 0) | (1 << 1),
        .trigger_panic  = false,
    };
    esp_task_wdt_reconfigure(&wdt_cfg);

    // ── SD CARD ─────────────────────────────────────────────
    if (sd_card_mount() != ESP_OK)
        ESP_LOGW(TAG, "SD card mount failed");
    else
    {
        ESP_LOGI(TAG, "SD card mounted");
        recordings_clear();     /* NEW — wipe old WAVs, wav_counter=0 */
        classify_csv_init();
        gate_csv_init();                                   /* GATE LOG */
    }

    // ── OLED ────────────────────────────────────────────────
    ESP_ERROR_CHECK(shared_i2c_init());
    oledMutex = xSemaphoreCreateMutex();
    wav_mutex = xSemaphoreCreateMutex();

    if (oled_init() == ESP_OK)
    {
        oled_show_text("Booting...");
        vTaskDelay(pdMS_TO_TICKS(2000));
        ESP_LOGI(TAG, "OLED WORKING");
    }
    else
    {
        ESP_LOGE(TAG, "OLED init failed");
    }

    // ── AUDIO DRIVER ────────────────────────────────────────
    ESP_LOGI(TAG, "Initializing audio driver...");

    temp_buffer = (int16_t *)heap_caps_malloc(
        AUDIO_SAMPLES * 2 * sizeof(int16_t),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);

    wav_pcm_buffer = (int16_t *)heap_caps_malloc(
        AUDIO_SAMPLES * sizeof(int16_t),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    if (!temp_buffer || !wav_pcm_buffer)
    {
        ESP_LOGE(TAG, "Audio scratch buffer alloc failed");
        return;
    }

    ESP_ERROR_CHECK(audio_driver_init());

    // BUG 5 FIX: flag and log AFTER successful init.
    audio_initialized = true;
    ESP_LOGI(TAG, "Audio initialized");

    // ── TFLITE ──────────────────────────────────────────────
    if (!init_tflite())
    {
        ESP_LOGE(TAG, "TFLite init failed");
        oled_show_text("Model Fail");
        return;
    }
    oled_show_text("DPNet Ready");

    // ── RING BUFFER POOL (PSRAM) ─────────────────────────────
    const size_t ring_bytes =
        (size_t)RING_BUFFER_SLOTS * AUDIO_SAMPLES * sizeof(float);

    ring_pool = (float *)heap_caps_malloc(
        ring_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!ring_pool)
    {
        ESP_LOGE(TAG, "ring_pool alloc failed (%u B)", (unsigned)ring_bytes);
        return;
    }
    memset(ring_pool, 0, ring_bytes);

    // ── WORKING BUFFERS (PSRAM) ──────────────────────────────
    capture_tmp_buf = (float *)heap_caps_malloc(
        AUDIO_SAMPLES * sizeof(float),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    process_buf = (float *)heap_caps_malloc(
        AUDIO_SAMPLES * sizeof(float),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    sd_buf = (float *)heap_caps_malloc(
        AUDIO_SAMPLES * sizeof(float),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!capture_tmp_buf || !process_buf || !sd_buf)
    {
        ESP_LOGE(TAG, "Working buffer alloc failed");
        return;
    }

    // ── SYNCHRONISATION ─────────────────────────────────────
    sd_buf_sem = xSemaphoreCreateBinary();
    xSemaphoreGive(sd_buf_sem);

    // ── RING BUFFER INIT ────────────────────────────────────
    ring_init();

    // ── QUEUES ──────────────────────────────────────────────
    displayQueue = xQueueCreate(5, sizeof(DisplayItem));
    sdQueue      = xQueueCreate(1, sizeof(SdItem));  /* WAS 2 — IMP 4 */

    // ── TASKS ───────────────────────────────────────────────
    xTaskCreatePinnedToCore(
        process_task, "process_task", PROCESS_TASK_STACK, NULL,
        INFERENCE_TASK_PRIO, &processTaskHandle, 1);  // CPU1

    xTaskCreatePinnedToCore(
        record_task, "record_task", RECORD_TASK_STACK, NULL,
        RECORD_TASK_PRIO, NULL, 0);                   // CPU0

    xTaskCreatePinnedToCore(
        oled_task, "oled_task", 4096, NULL, 1, NULL, 0);

    xTaskCreatePinnedToCore(
        sd_task, "sd_task", 8192, NULL, 3, NULL, 0);

    // ── STARTUP SUMMARY ─────────────────────────────────────
    ESP_LOGI(TAG, "All tasks started");
    ESP_LOGI(TAG,
             "Ring: %d slots × %u KB = %u KB PSRAM",
             RING_BUFFER_SLOTS,
             (unsigned)(AUDIO_SAMPLES * sizeof(float) / 1024),
             (unsigned)(ring_bytes / 1024));
    ESP_LOGI(TAG,
             "ENERGY_THRESHOLD=%.3f (pre-norm)  "
             "NOVELTY_THRESHOLD=%.2f  "
             "BUFFER_MAX_AGE=%d ms  "
             "STALE_FLUSH_LIMIT=%d",
             ENERGY_THRESHOLD,
             NOVELTY_THRESHOLD,
             BUFFER_MAX_AGE_MS,
             STALE_FLUSH_LIMIT);
    ESP_LOGI(TAG, "Gate log: %s (one row per fresh frame)", GATE_CSV_PATH);
}