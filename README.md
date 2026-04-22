# AccessiSound — ESP32-S3 Accessibility Sound Recognition Assistant

<div align="center">

![Platform](https://img.shields.io/badge/Platform-ESP32--S3-red?logo=espressif)
![Framework](https://img.shields.io/badge/Framework-Arduino%20%7C%20PlatformIO-blue)
![ML](https://img.shields.io/badge/ML-TFLite%20Micro-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-ESC--50%20(CC--BY)-green)
![License](https://img.shields.io/badge/License-MIT-purple)
![Status](https://img.shields.io/badge/Status-Hackathon%20Prototype-yellow)

**A compact, offline assistive device that recognises environmental sounds and alerts visually impaired users via vibration and audio feedback — no cloud, no phone required.**

[Features](#-features) · [Hardware](#-hardware) · [Quick Start](#-quick-start) · [Training](#-training-your-own-model) · [Architecture](#-architecture) · [Roadmap](#-roadmap)

</div>

---

## What It Does

AccessiSound listens continuously for specific sounds in the user's environment and provides immediate tactile (vibration) and audio feedback so that a visually impaired user knows what's happening around them — even without looking at a screen or phone.

| Sound Detected | Vibration Pattern | Audio Alert |
|---|---|---|
|  Doorbell | 2 short pulses | Ding-dong tone |
|  Microwave | 3 rapid pulses | High beep |
|  Fire alarm | Rapid continuous | Alternating tones |
|  Smoke alarm | Rapid continuous | Alternating tones |
|  Phone ring | Triple-pulse × 2 | Repeating tone |

All processing happens **on-device** — the ESP32-S3 runs the TFLite Micro model with no internet connection required, making it suitable for use anywhere.

---

##  Features

- **Real-time on-device inference** — ~50 ms latency on ESP32-S3
- **MFCC feature extraction** implemented in C++ (no external DSP library needed)
- **Compact CNN model** — ~40–60 kB quantised, fits in ESP32-S3 SRAM
- **Unique alert patterns** per sound class so users can distinguish events by feel
- **Mute toggle** via onboard BOOT button
- **Offline training pipeline** using the open-source [ESC-50 dataset](https://github.com/karolpiczak/ESC-50)
- **C header export** — model auto-converted and ready to include in firmware

---

##  Hardware

### Required Components

| Component | Notes | Approximate Cost |
|---|---|---|
| ESP32-S3-DevKitC-1 | Main MCU + USB-C | ~$10 USD |
| INMP441 I2S MEMS Microphone | 16-bit, low-noise | ~$2 |
| ERM Vibration Motor (3 V) | With transistor driver | ~$1 |
| Passive Buzzer / Small Speaker | 8Ω, ≤1W | ~$0.50 |
| NPN Transistor (e.g. 2N2222) | For vibration motor | ~$0.10 |
| 100Ω resistor (×2), 1kΩ resistor | Pull-ups / base resistor | ~$0.05 |
| Breadboard + jumper wires | Prototyping | ~$3 |

**Total: ~$17 USD**

### Wiring Diagram

```
ESP32-S3-DevKitC-1
┌──────────────────────────────────────┐
│  GPIO 42 ──────────────── WS   │INMP441
│  GPIO 41 ──────────────── SCK  │(I2S Mic)
│  GPIO  2 ──────────────── SD   │
│  3.3 V  ──────────────── VDD  │
│  GND    ──────────────── GND  │
│                                      │
│  GPIO 10 ── 100Ω ── Base(2N2222)     │ Vibration Motor
│             Emitter ── GND           │ (Collector → Motor → 3.3V)
│                                      │
│  GPIO 11 ── 100Ω ── Buzzer+ ── GND  │ Passive Buzzer
└──────────────────────────────────────┘
```

**No hardware?** You can still train the model and explore the codebase — the ML pipeline runs entirely on your laptop using the ESC-50 dataset.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/accessisound-esp32s3.git
cd accessisound-esp32s3
```

### 2. Train the model (on your laptop)

```bash
cd ml
pip install -r requirements.txt
python train_model.py
```

This will:
- Download ESC-50 automatically (~600 MB, one time)
- Extract MFCC features
- Train a compact CNN (~50 epochs)
- Export `models/sound_classifier.tflite`
- Copy `models/sound_model.h` → `firmware/src/sound_model.h` automatically

Expected test accuracy: **~78–85%** on the 5-class subset.

### 3. Flash the firmware

Make sure [PlatformIO](https://platformio.org/) is installed (VS Code extension recommended).

```bash
cd firmware
pio run --target upload --upload-port /dev/ttyUSB0   # Linux
# or /dev/cu.usbmodem*  on macOS
# or  COM3              on Windows
```

Open the serial monitor to see live detections:
```bash
pio device monitor
```

### 4. Test it

Play any of these sounds near the microphone:
- Doorbell sound on your phone
- Set a microwave timer beep
- Play a fire alarm clip from YouTube

The device should vibrate and beep within ~1 second.

---

## Training Your Own Model

### Dataset: ESC-50

[ESC-50](https://github.com/karolpiczak/ESC-50) is a free, openly licensed dataset of 2,000 environmental audio recordings across 50 classes. The training script automatically downloads and filters the 5 classes we need.

```
data/
└── ESC-50-master/
    ├── audio/          ← 2000 WAV clips (5 s each, 44.1 kHz)
    └── meta/
        └── esc50.csv   ← labels and fold assignments
```

### Add Your Own Sounds

To add a custom sound class (e.g. a specific appliance in your home):

1. Record 20–40 five-second WAV clips of your sound
2. Place them in `data/custom/<your_class_name>/`
3. Add the class to `TARGET_CLASSES` in `train_model.py`
4. Re-run training

### Model Architecture

```
Input: (32 frames × 13 MFCC coefficients × 1 channel)
  ↓
Conv2D(16, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
  ↓
Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
  ↓
GlobalAveragePooling2D
  ↓
Dense(64, ReLU) → Dropout(0.4)
  ↓
Dense(5, Softmax)

Parameters: ~18,000  |  Quantised size: ~45 kB
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ESP32-S3 Firmware                     │
│                                                          │
│  I2S DMA Read (16 kHz) → Ring Buffer                     │
│       ↓                                                  │
│  FeatureExtractor: Pre-emphasis → Hamming window         │
│       → Power spectrum → Mel filterbank → DCT → MFCC    │
│       ↓                                                  │
│  Sliding Window (32 frames, 50% overlap)                 │
│       ↓                                                  │
│  TFLite Micro Interpreter (quantised CNN)                │
│       ↓                                                  │
│  Confidence threshold (0.72)                             │
│       ↓                                                  │
│  AlertManager → Vibration + Buzzer patterns              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  ML Training Pipeline                    │
│  (runs on your laptop, produces firmware-ready header)   │
│                                                          │
│  ESC-50 WAV files → librosa MFCC extraction             │
│       → CNN training (TensorFlow/Keras)                  │
│       → int8 quantisation (TFLite converter)             │
│       → sound_model.h (C byte array for firmware)       │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
accessisound-esp32s3/
├── firmware/
│   ├── platformio.ini          ← PlatformIO build config
│   └── src/
│       ├── main.cpp            ← Main firmware (I2S + inference loop)
│       ├── feature_extractor.h ← MFCC computation in C++
│       ├── alert_manager.h     ← Vibration + buzzer patterns
│       └── sound_model.h       ← Auto-generated (after training)
│
├── ml/
│   ├── train_model.py          ← Full training pipeline
│   └── requirements.txt        ← Python dependencies
│
├── docs/
│   └── wiring_diagram.png      ← Hardware connection diagram
│
└── README.md
```

---

## Performance

| Metric | Value |
|---|---|
| Inference latency | ~45–60 ms |
| Model size (quantised) | ~45 kB |
| Tensor arena | 60 kB |
| Test accuracy (5 classes) | ~80 % |
| False positive rate | <5 % at threshold 0.72 |
| Power consumption | ~120 mA active, ~15 mA deep sleep |

---

## Roadmap

This was built during a hackathon. Future improvements:

- [ ] **v1.1** — Wake-word detection to reduce false positives
- [ ] **v1.2** — BLE companion app (iOS / Android) for configuration
- [ ] **v2.0** — Upgrade to ESP32-CAM for vision-based alerts ([Ultimate Version](docs/ultimate-version.md))
- [ ] **v2.1** — Larger dataset with real microwave / doorbell recordings
- [ ] **v2.2** — OTA model updates over Wi-Fi

---

## Contributing

Pull requests are welcome! Please open an issue first to discuss what you'd like to change. If you have access to real microwave or doorbell recording equipment, audio clip contributions are especially valuable.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) used for training is licensed under **CC BY (Creative Commons Attribution)**.

---

## Acknowledgements

- [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) by Karol Piczak
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [librosa](https://librosa.org/) — audio feature extraction
- Hackathon teammates and mentors

