/*
 * AccessiSound — ESP32-S3 Accessibility Sound Recognition Assistant
 * Hackathon Prototype v1.0
 *
 * Target Board : ESP32-S3-DevKitC-1 (or any ESP32-S3 with MEMS mic)
 * Framework    : Arduino (ESP-IDF compatible)
 *
 * Audio pipeline:
 *   I2S mic  →  ring buffer  →  FFT feature extraction
 *   →  threshold classifier  →  vibration / speaker alert
 *
 * Recognised sound classes (index → label):
 *   0  doorbell      3  smoke_alarm
 *   1  microwave     4  phone_ring
 *   2  fire_alarm    5  background (negative class)
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include "sound_model.h"        // generated TFLite flatbuffer header
#include "feature_extractor.h"  // MFCC / FFT helpers
#include "alert_manager.h"      // vibration + speaker output

// ── Pin definitions ──────────────────────────────────────────────────────────
#define I2S_WS_PIN      42
#define I2S_SCK_PIN     41
#define I2S_SD_PIN      2

#define VIBRATION_PIN   10
#define SPEAKER_PIN     11
#define LED_RED_PIN     38
#define LED_GREEN_PIN   39
#define BOOT_BTN_PIN    0       // onboard BOOT button (mute toggle)

// ── Audio constants ───────────────────────────────────────────────────────────
#define SAMPLE_RATE     16000   // Hz — matches training pipeline
#define FRAME_SIZE      512     // samples per FFT frame
#define HOP_SIZE        256     // 50 % overlap
#define N_MFCC          13      // MFCC coefficients per frame
#define WINDOW_FRAMES   32      // frames stacked into one inference window
#define CONFIDENCE_THR  0.72f   // minimum confidence to fire an alert

// ── Sound label table ─────────────────────────────────────────────────────────
static const char* SOUND_LABELS[] = {
    "Doorbell",
    "Microwave beep",
    "Fire alarm",
    "Smoke alarm",
    "Phone ringing",
    "Background noise"
};
static const uint8_t N_CLASSES = sizeof(SOUND_LABELS) / sizeof(SOUND_LABELS[0]);
static const uint8_t BACKGROUND_CLASS = N_CLASSES - 1;

// ── Globals ───────────────────────────────────────────────────────────────────
static int16_t  audioBuffer[FRAME_SIZE * 2];   // double-buffer
static float    mfccWindow[WINDOW_FRAMES][N_MFCC];
static uint8_t  frameIdx = 0;
static bool     muteMode = false;

AlertManager    alertMgr(VIBRATION_PIN, SPEAKER_PIN);
FeatureExtractor feat(SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, N_MFCC);

// TFLite micro runtime (initialised in setup)
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const int kTensorArenaSize = 60 * 1024;
static uint8_t   tensorArena[kTensorArenaSize];
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* inputTensor  = nullptr;
static TfLiteTensor* outputTensor = nullptr;

// ── I2S initialisation ────────────────────────────────────────────────────────
void initI2S() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 4,
        .dma_buf_len          = FRAME_SIZE,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0
    };
    i2s_pin_config_t pins = {
        .bck_io_num   = I2S_SCK_PIN,
        .ws_io_num    = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_SD_PIN
    };
    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pins));
    Serial.println("[I2S] Driver installed at 16 kHz, 16-bit mono.");
}

// ── TFLite micro initialisation ───────────────────────────────────────────────
void initModel() {
    static tflite::AllOpsResolver resolver;
    const tflite::Model* model = tflite::GetModel(sound_model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[TFLite] ERROR: Model schema version mismatch!");
        while (true) delay(1000);
    }

    static tflite::MicroInterpreter staticInterp(
        model, resolver, tensorArena, kTensorArenaSize);
    interpreter = &staticInterp;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[TFLite] ERROR: AllocateTensors() failed!");
        while (true) delay(1000);
    }

    inputTensor  = interpreter->input(0);
    outputTensor = interpreter->output(0);
    Serial.printf("[TFLite] Model loaded. Input shape: [1, %d, %d, 1]\n",
                  WINDOW_FRAMES, N_MFCC);
}

// ── Read one audio frame from I2S ─────────────────────────────────────────────
bool readAudioFrame() {
    size_t bytesRead = 0;
    esp_err_t err = i2s_read(I2S_NUM_0,
                             audioBuffer,
                             FRAME_SIZE * sizeof(int16_t),
                             &bytesRead,
                             pdMS_TO_TICKS(100));
    return (err == ESP_OK && bytesRead > 0);
}

// ── Run inference and return best class index ─────────────────────────────────
int8_t runInference(float confidence[]) {
    // Copy MFCC window into model input tensor
    for (int f = 0; f < WINDOW_FRAMES; ++f)
        for (int c = 0; c < N_MFCC; ++c)
            inputTensor->data.f[f * N_MFCC + c] = mfccWindow[f][c];

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[TFLite] Invoke() failed.");
        return -1;
    }

    // Softmax output — find argmax
    int8_t bestClass = 0;
    float  bestScore = 0.0f;
    for (uint8_t i = 0; i < N_CLASSES; ++i) {
        float score = outputTensor->data.f[i];
        confidence[i] = score;
        if (score > bestScore) {
            bestScore = score;
            bestClass = i;
        }
    }
    return bestClass;
}

// ── setup() ───────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== AccessiSound v1.0 — ESP32-S3 ===");

    // GPIO setup
    pinMode(LED_RED_PIN,   OUTPUT);
    pinMode(LED_GREEN_PIN, OUTPUT);
    pinMode(BOOT_BTN_PIN,  INPUT_PULLUP);
    digitalWrite(LED_GREEN_PIN, HIGH);  // ready indicator

    alertMgr.begin();
    feat.begin();
    initI2S();
    initModel();

    Serial.println("[SYSTEM] Listening for sounds...");
}

// ── loop() ────────────────────────────────────────────────────────────────────
void loop() {
    // Mute toggle via BOOT button
    if (digitalRead(BOOT_BTN_PIN) == LOW) {
        muteMode = !muteMode;
        Serial.printf("[BUTTON] Mute %s\n", muteMode ? "ON" : "OFF");
        delay(300);  // debounce
    }

    if (!readAudioFrame()) return;

    // Compute MFCCs for this frame and slide into window
    float frameMFCC[N_MFCC];
    feat.computeMFCC(audioBuffer, FRAME_SIZE, frameMFCC);

    // Shift window left, append new frame
    memmove(mfccWindow, mfccWindow + N_MFCC,
            (WINDOW_FRAMES - 1) * N_MFCC * sizeof(float));
    memcpy(mfccWindow[WINDOW_FRAMES - 1], frameMFCC,
           N_MFCC * sizeof(float));

    frameIdx++;
    // Run inference every HOP_SIZE frames (50 % overlap)
    if (frameIdx < (WINDOW_FRAMES / 2)) return;
    frameIdx = 0;

    float confidence[N_CLASSES];
    int8_t predicted = runInference(confidence);

    if (predicted < 0 || predicted == BACKGROUND_CLASS) return;
    if (confidence[predicted] < CONFIDENCE_THR) return;
    if (muteMode) return;

    // ── Alert! ──────────────────────────────────────────────────────────────
    Serial.printf("[DETECTED] %s (%.1f%%)\n",
                  SOUND_LABELS[predicted],
                  confidence[predicted] * 100.0f);

    digitalWrite(LED_RED_PIN, HIGH);
    alertMgr.triggerAlert(predicted, confidence[predicted]);
    digitalWrite(LED_RED_PIN, LOW);
}
