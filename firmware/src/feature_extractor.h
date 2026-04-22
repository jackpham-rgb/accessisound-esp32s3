/*
 * feature_extractor.h — MFCC / FFT feature extraction helper
 * Runs entirely on ESP32-S3 without external DSP library dependencies.
 */

#pragma once
#include <Arduino.h>
#include <math.h>

#define MEL_FILTERBANKS  26
#define PRE_EMPHASIS     0.97f

class FeatureExtractor {
public:
    FeatureExtractor(uint32_t sampleRate, uint16_t frameSize,
                     uint16_t hopSize, uint8_t nMFCC)
        : _sr(sampleRate), _frameSize(frameSize),
          _hopSize(hopSize), _nMFCC(nMFCC) {}

    void begin() {
        _buildMelFilterbank();
        _buildDCTMatrix();
        Serial.printf("[DSP] FeatureExtractor ready. Frame=%d Hop=%d MFCC=%d\n",
                      _frameSize, _hopSize, _nMFCC);
    }

    /*
     * computeMFCC()
     * Input : raw int16 PCM samples (frameSize samples)
     * Output: mfcc[nMFCC] — normalised float coefficients
     */
    void computeMFCC(const int16_t* samples, uint16_t len, float* mfcc) {
        static float frame[512];
        static float spectrum[512];

        // 1. Convert int16 → float, apply pre-emphasis
        float prevSample = 0.0f;
        for (uint16_t i = 0; i < len; i++) {
            float s = samples[i] / 32768.0f;
            frame[i] = s - PRE_EMPHASIS * prevSample;
            prevSample = s;
        }

        // 2. Hamming window
        for (uint16_t i = 0; i < len; i++) {
            frame[i] *= 0.54f - 0.46f * cosf(2.0f * M_PI * i / (len - 1));
        }

        // 3. Power spectrum via simple DFT (FFT would be faster;
        //    replace with esp_dsp_fft if available)
        _powerSpectrum(frame, len, spectrum);

        // 4. Mel filterbank energies
        float melEnergies[MEL_FILTERBANKS];
        for (uint8_t m = 0; m < MEL_FILTERBANKS; m++) {
            float energy = 0.0f;
            for (uint16_t k = _fbStart[m]; k <= _fbStop[m]; k++) {
                energy += spectrum[k] * _fbWeights[m][k - _fbStart[m]];
            }
            melEnergies[m] = logf(energy + 1e-9f);
        }

        // 5. DCT → MFCC coefficients
        for (uint8_t i = 0; i < _nMFCC; i++) {
            float sum = 0.0f;
            for (uint8_t m = 0; m < MEL_FILTERBANKS; m++) {
                sum += melEnergies[m] * _dctMatrix[i][m];
            }
            mfcc[i] = sum;
        }

        // 6. Mean normalisation (cepstral mean subtraction approximation)
        float mean = 0.0f;
        for (uint8_t i = 0; i < _nMFCC; i++) mean += mfcc[i];
        mean /= _nMFCC;
        for (uint8_t i = 0; i < _nMFCC; i++) mfcc[i] -= mean;
    }

private:
    uint32_t _sr;
    uint16_t _frameSize, _hopSize;
    uint8_t  _nMFCC;

    // Mel filterbank
    uint16_t _fbStart[MEL_FILTERBANKS];
    uint16_t _fbStop[MEL_FILTERBANKS];
    float    _fbWeights[MEL_FILTERBANKS][64];  // max 64 bins per filter

    // DCT matrix [nMFCC × MEL_FILTERBANKS]
    float    _dctMatrix[16][MEL_FILTERBANKS];

    static float _hzToMel(float hz) {
        return 2595.0f * log10f(1.0f + hz / 700.0f);
    }
    static float _melToHz(float mel) {
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }

    void _buildMelFilterbank() {
        uint16_t nFFT = _frameSize / 2 + 1;
        float melLow  = _hzToMel(80.0f);
        float melHigh = _hzToMel(_sr / 2.0f);

        float melPoints[MEL_FILTERBANKS + 2];
        for (uint8_t i = 0; i < MEL_FILTERBANKS + 2; i++) {
            melPoints[i] = _melToHz(
                melLow + i * (melHigh - melLow) / (MEL_FILTERBANKS + 1));
        }

        for (uint8_t m = 0; m < MEL_FILTERBANKS; m++) {
            _fbStart[m] = (uint16_t)(melPoints[m]   * _frameSize / _sr);
            uint16_t center = (uint16_t)(melPoints[m+1] * _frameSize / _sr);
            _fbStop[m]  = (uint16_t)(melPoints[m+2] * _frameSize / _sr);

            uint8_t wIdx = 0;
            for (uint16_t k = _fbStart[m]; k <= _fbStop[m] && wIdx < 64; k++, wIdx++) {
                if (k <= center) {
                    _fbWeights[m][wIdx] =
                        (float)(k - _fbStart[m]) / (center - _fbStart[m] + 1);
                } else {
                    _fbWeights[m][wIdx] =
                        (float)(_fbStop[m] - k) / (_fbStop[m] - center + 1);
                }
            }
        }
    }

    void _buildDCTMatrix() {
        for (uint8_t i = 0; i < _nMFCC; i++) {
            for (uint8_t m = 0; m < MEL_FILTERBANKS; m++) {
                _dctMatrix[i][m] = cosf(M_PI * i * (m + 0.5f) / MEL_FILTERBANKS)
                                   * sqrtf(2.0f / MEL_FILTERBANKS);
            }
        }
    }

    void _powerSpectrum(const float* frame, uint16_t len, float* spectrum) {
        uint16_t half = len / 2 + 1;
        for (uint16_t k = 0; k < half; k++) {
            float re = 0.0f, im = 0.0f;
            for (uint16_t n = 0; n < len; n++) {
                float angle = 2.0f * M_PI * k * n / len;
                re += frame[n] * cosf(angle);
                im -= frame[n] * sinf(angle);
            }
            spectrum[k] = (re * re + im * im) / len;
        }
    }
};
