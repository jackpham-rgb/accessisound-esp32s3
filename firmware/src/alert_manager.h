/*
 * alert_manager.h — vibration motor + passive buzzer alert patterns
 *
 * Each sound class gets a unique haptic/audio pattern so a visually
 * impaired user can distinguish events without looking at any display.
 *
 *  Class 0 — Doorbell    : two short vibration pulses + two rising tones
 *  Class 1 — Microwave   : three rapid pulses + high-pitch beep
 *  Class 2 — Fire alarm  : continuous rapid buzz + alternating tones
 *  Class 3 — Smoke alarm : same as fire alarm (safety critical)
 *  Class 4 — Phone ring  : repeating triple pulse
 */

#pragma once
#include <Arduino.h>

class AlertManager {
public:
    AlertManager(uint8_t vibPin, uint8_t spkPin)
        : _vibPin(vibPin), _spkPin(spkPin) {}

    void begin() {
        pinMode(_vibPin, OUTPUT);
        digitalWrite(_vibPin, LOW);
        // Speaker pin handled by tone()
        Serial.printf("[ALERT] AlertManager ready. Vib=%d Spk=%d\n",
                      _vibPin, _spkPin);
    }

    /*
     * triggerAlert()
     * soundClass : index matching SOUND_LABELS in main.cpp
     * confidence : 0.0–1.0 (used to scale vibration intensity in future HW)
     */
    void triggerAlert(uint8_t soundClass, float /*confidence*/) {
        switch (soundClass) {
            case 0: _doorbellPattern();    break;
            case 1: _microwavePattern();   break;
            case 2: /* fall through */
            case 3: _fireAlarmPattern();   break;
            case 4: _phonePattern();       break;
            default: _genericBuzz();       break;
        }
    }

private:
    uint8_t _vibPin, _spkPin;

    void _vib(uint16_t onMs, uint16_t offMs = 100) {
        digitalWrite(_vibPin, HIGH);
        delay(onMs);
        digitalWrite(_vibPin, LOW);
        delay(offMs);
    }

    // Doorbell: ding-dong (two tones + two pulses)
    void _doorbellPattern() {
        tone(_spkPin, 880, 150);  _vib(150, 80);
        tone(_spkPin, 659, 250);  _vib(250, 0);
        delay(100);
        tone(_spkPin, 880, 150);  _vib(150, 80);
        tone(_spkPin, 659, 250);  _vib(250);
    }

    // Microwave: three rapid high beeps
    void _microwavePattern() {
        for (uint8_t i = 0; i < 3; i++) {
            tone(_spkPin, 1760, 80);
            _vib(80, 60);
        }
    }

    // Fire / smoke alarm: urgent rapid pattern
    void _fireAlarmPattern() {
        for (uint8_t i = 0; i < 6; i++) {
            tone(_spkPin, (i % 2 == 0) ? 2000 : 2400, 120);
            _vib(120, 40);
        }
    }

    // Phone ring: triple-pulse, repeated twice
    void _phonePattern() {
        for (uint8_t rep = 0; rep < 2; rep++) {
            for (uint8_t i = 0; i < 3; i++) {
                tone(_spkPin, 1047, 100);
                _vib(100, 50);
            }
            delay(300);
        }
    }

    // Generic fallback
    void _genericBuzz() {
        tone(_spkPin, 1200, 200);
        _vib(200);
    }
};
