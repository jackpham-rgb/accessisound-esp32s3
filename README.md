# AccessiSound: ESP32-S3 Sound Recognition Assistant

Use this if you need a small, offline device that listens for a few specific
household sounds (doorbell, microwave beep, fire/smoke alarm, phone ring) and
turns them into a vibration + buzzer pattern, for someone who can't rely on
hearing them. Everything runs on the ESP32-S3 itself: no phone, no cloud, no
internet connection.

Built as a hackathon prototype. It has not been assembled on physical
hardware yet; there's no photo, video, or bench-test proof to show here. The
wiring below is exactly what `firmware/src/main.cpp` expects, and the ML
pipeline is real and runs end to end on your laptop (see "Try it without any
hardware" below), but the ESP32-S3 side is untested until someone actually
builds it. Treat the pin list as a build plan, not a demonstrated result.

## What it does

The device listens continuously through a microphone. When it recognizes one
of 5 trained sounds with enough confidence, it vibrates and beeps in a
pattern unique to that sound, so you can tell events apart by feel alone.

| Sound | Vibration | Buzzer |
|---|---|---|
| Doorbell | 2 pulses, twice | rising two-tone "ding-dong" |
| Microwave beep | 3 quick pulses | 3 high beeps |
| Fire alarm | continuous rapid | alternating high/low tones |
| Smoke alarm | same as fire alarm | same as fire alarm |
| Phone ringing | 3 pulses, twice | 3 mid tones, twice |

All of the above is decided on-device by a small CNN (about 45 KB, quantized
to int8) running on TensorFlow Lite Micro. No data leaves the board.

## Parts list

| Part | Notes | Rough cost |
|---|---|---|
| ESP32-S3-DevKitC-1 | the main board | ~$10 |
| INMP441 I2S microphone | digital MEMS mic | ~$2 |
| Vibration motor (3V ERM) | needs a transistor to drive it, GPIO can't power it directly | ~$1 |
| Passive buzzer | 8 ohm, small | ~$0.50 |
| NPN transistor (2N2222 or similar) | switches the vibration motor | ~$0.10 |
| Red LED + green LED | status indicators | ~$0.10 |
| Resistors: 100 ohm x2, 1 kOhm x1, 220 ohm x2 | 100R for buzzer/transistor base, 1k transistor base, 220R for the two LEDs | ~$0.10 |
| Breadboard + jumper wires | prototyping | ~$3 |

Total: roughly $17.

## Wiring

Every pin number below comes straight from `firmware/src/main.cpp`'s `#define`
block. If you change a pin there, update it here too.

**Microphone (INMP441, I2S):**
- ESP32-S3 GPIO 42 to mic WS
- ESP32-S3 GPIO 41 to mic SCK
- ESP32-S3 GPIO 2 to mic SD
- ESP32-S3 3.3V to mic VDD
- ESP32-S3 GND to mic GND

**Vibration motor (through a transistor, since a GPIO pin can't supply enough current for a motor):**
- ESP32-S3 GPIO 10 to a 100 ohm resistor to the transistor's base
- Transistor emitter to GND
- Transistor collector to the motor, and the motor's other lead to 3.3V

**Buzzer:**
- ESP32-S3 GPIO 11 to a 100 ohm resistor to the buzzer's positive lead
- Buzzer negative lead to GND

**Status LEDs:**
- ESP32-S3 GPIO 38 to a 220 ohm resistor to the red LED's anode, LED cathode to GND
- ESP32-S3 GPIO 39 to a 220 ohm resistor to the green LED's anode, LED cathode to GND

**Mute button:**
- GPIO 0 is the DevKitC-1's onboard BOOT button. It's already wired on the
  board, nothing extra to connect. Press it to toggle mute.

## No hardware? Try the ML pipeline anyway

The training and demo scripts are plain Python and don't need an ESP32-S3 at
all.

### 1. Clone the repo

```bash
git clone https://github.com/jackpham-rgb/accessisound-esp32s3.git
cd accessisound-esp32s3
```

### 2. Set up and train

```bash
cd ml
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python train_model.py
```

This downloads ESC-50 (a free, CC-BY-licensed sound dataset, about 600 MB,
one-time), extracts MFCC features, trains the CNN, and writes:
- `models/sound_classifier.tflite`: flash this onto the ESP32-S3
- `models/sound_model.h`: also auto-copied to `firmware/src/sound_model.h`
- `models/training_report.png`: accuracy/loss curves and a confusion matrix

Measured test accuracy on this dataset: about 47-53% across 5 classes (a
random guess would get 20%). That's real but not great, and there's a
concrete reason: ESC-50 doesn't actually contain a doorbell, smoke alarm, or
phone ring sound. Every class is trained on the closest available stand-in
(see the comment above `TARGET_CLASSES` in `train_model.py` for exactly
which). Swap in real recordings of your own doorbell, alarm, and phone (see
"Use your own sounds" below) and accuracy should improve a lot, since the
model would then be learning the actual sounds instead of proxies for them.

Want a faster run to check the pipeline works, without waiting for full
training: `python train_model.py --skip-download --epochs 5`.

### 3. Try it without any hardware

```bash
python demo.py
```

With no argument, this grabs a random labeled clip from ESC-50 and runs it
through the trained model, exactly like the firmware would: same feature
extraction, same model, same 72% confidence threshold. It prints the
confidence for every class and tells you whether the real device would have
fired an alert. Pass your own WAV file instead: `python demo.py path/to/clip.wav`.

### 4. Flash the firmware (needs the hardware built)

Install [PlatformIO](https://platformio.org/) (there's a VS Code extension),
then:

```bash
cd firmware
pio run --target upload --upload-port /dev/ttyUSB0   # Linux
# or /dev/cu.usbmodem* on macOS, or COM3 on Windows
pio device monitor
```

## Use your own sounds

To train on real recordings instead of the ESC-50 stand-ins:

1. Record 20-40 five-second WAV clips of the real sound
2. Put them in `data/custom/<class_name>/`
3. Add that folder to `TARGET_CLASSES` in `train_model.py`
4. Re-run `python train_model.py`

## How the firmware pipeline works

1. I2S reads 16 kHz audio into a ring buffer.
2. `feature_extractor.h` turns each frame into 13 MFCC coefficients (the
   same pre-emphasis, Hamming window, Mel filterbank, and DCT steps as
   `train_model.py`, just written in C++ instead of relying on librosa).
3. 32 of those frames are stacked into one sliding window (50% overlap).
4. The window goes into the TFLite Micro interpreter, which outputs a
   confidence score per class.
5. If the top class scores above 0.72 confidence, `alert_manager.h` fires
   that class's vibration + buzzer pattern.

The training pipeline (`ml/train_model.py`) is separate and runs on your
laptop: it downloads ESC-50, extracts the same kind of MFCC features with
librosa, trains the CNN in TensorFlow/Keras, quantizes it to int8, and
writes out both the `.tflite` file and a `.h` file the firmware can compile
directly.

Model itself: 2 convolution blocks (16 then 32 filters), global average
pooling, one 64-unit dense layer, then 5-way softmax. About 7,400
parameters, roughly 45 KB after quantization.

## Repository layout

```
firmware/
  platformio.ini            build config
  src/main.cpp               I2S read + inference loop
  src/feature_extractor.h    MFCC in C++
  src/alert_manager.h        vibration/buzzer patterns
  src/sound_model.h          generated by train_model.py, not in git

ml/
  train_model.py             full training pipeline
  demo.py                    hardware-free demo (see above)
  requirements.txt
```

## Status and known limits

- Firmware has not been run on real hardware. Wiring above is a build plan
  from the pin definitions, not a tested result.
- Training data is proxy sounds, not the real target sounds (see above).
  Expect the shipped model to work best as a proof of concept, not as a
  finished product.
- No wake-word or noise-gating, and no trained "background/silence" class:
  the model is a plain 5-way softmax over the 5 sound classes, so it always
  picks one of them as "most likely" even during silence. The 72%
  confidence threshold is the only thing stopping constant false alerts,
  not an actual rejection class.

## License

Code: MIT, see [LICENSE](LICENSE). The [ESC-50 dataset](https://github.com/karolpiczak/ESC-50)
used for training is CC-BY licensed, separately from this repo's code.

## Credits

- [ESC-50](https://github.com/karolpiczak/ESC-50) dataset by Karol Piczak
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [librosa](https://librosa.org/) for audio feature extraction
