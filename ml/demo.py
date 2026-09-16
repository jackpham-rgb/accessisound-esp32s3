#!/usr/bin/env python3
"""
demo.py: AccessiSound hardware-free demo
=========================================
Runs the SAME feature extraction and model as the firmware, on your laptop,
against a real audio clip. No ESP32-S3, microphone, or wiring needed. This
is here because a from-scratch electronics build is easy to claim and hard
to prove without the physical board in front of you; this script proves the
detection logic actually works on real audio, right now, on any machine.

Usage:
    python demo.py                       # picks a random ESC-50 test clip
    python demo.py path/to/some_clip.wav # runs on a specific WAV file

Requires a trained model. Run train_model.py first if you see a "no model
found" message.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from train_model import (
    DATA_DIR, MODELS_DIR, LABEL_NAMES, TARGET_CLASSES,
    extract_mfcc_window,
)

MODEL_PATH = os.path.join(MODELS_DIR, "sound_classifier.keras")
CONFIDENCE_THR = 0.72  # matches CONFIDENCE_THR in firmware/src/main.cpp

# Plain-English version of firmware/src/alert_manager.h's patterns. The
# firmware fires these with a vibration motor and a piezo buzzer; here we
# just describe them, since there's no motor or buzzer attached to a laptop.
ALERT_DESCRIPTIONS = {
    0: "2 vibration pulses + a rising two-tone 'ding-dong', twice",
    1: "3 quick vibration pulses + 3 high beeps",
    2: "continuous rapid buzz + alternating high/low tones (6x)",
    3: "same pattern as fire alarm (the firmware treats them as equally urgent)",
    4: "3 pulses + 3 mid tones, repeated twice",
}


def pick_random_test_clip() -> str:
    """Grab a random clip from one of our 5 target ESC-50 categories, so the
    demo works with zero arguments if ESC-50 is already downloaded."""
    meta_path = os.path.join(DATA_DIR, "meta", "esc50.csv")
    if not os.path.exists(meta_path):
        print(f"[ERROR] No WAV file given, and {meta_path} doesn't exist.")
        print("        Either pass a WAV file path, or run train_model.py")
        print("        once first (it downloads ESC-50 for you).")
        sys.exit(1)
    meta = pd.read_csv(meta_path)
    meta = meta[meta["category"].isin(TARGET_CLASSES.keys())]
    row = meta.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
    true_label = LABEL_NAMES[TARGET_CLASSES[row["category"]]]
    print(f"[DEMO] No file given: picked a random ESC-50 clip "
          f"({row['category']}, our '{true_label}' proxy) to run through the model.")
    return os.path.join(DATA_DIR, "audio", row["filename"])


def main():
    parser = argparse.ArgumentParser(description="AccessiSound hardware-free demo")
    parser.add_argument("wav_file", nargs="?", default=None,
                         help="Path to a WAV file. If omitted, picks a random ESC-50 test clip.")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No trained model at {MODEL_PATH}.")
        print("        Run: python train_model.py")
        sys.exit(1)

    wav_path = args.wav_file or pick_random_test_clip()
    if not os.path.exists(wav_path):
        print(f"[ERROR] File not found: {wav_path}")
        sys.exit(1)

    print(f"[DEMO] Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"[DEMO] Extracting MFCC features from {wav_path}")
    features = extract_mfcc_window(wav_path)
    features = features[np.newaxis, ..., np.newaxis]  # (1, frames, mfcc, 1)

    confidence = model.predict(features, verbose=0)[0]
    predicted = int(np.argmax(confidence))

    print()
    print("Confidence per class:")
    for i, label in enumerate(LABEL_NAMES):
        bar = "#" * int(confidence[i] * 40)
        print(f"  {label:<12} {confidence[i]*100:5.1f}%  {bar}")
    print()

    if confidence[predicted] < CONFIDENCE_THR:
        print(f"[RESULT] Below the {CONFIDENCE_THR:.0%} confidence threshold. "
              f"The real firmware would NOT fire an alert here (it would stay quiet).")
    else:
        label = LABEL_NAMES[predicted]
        print(f"[RESULT] Detected: {label} ({confidence[predicted]*100:.1f}% confidence)")
        print(f"         Alert the firmware would trigger: {ALERT_DESCRIPTIONS[predicted]}")


if __name__ == "__main__":
    main()
