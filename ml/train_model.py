#!/usr/bin/env python3
"""
train_model.py — AccessiSound Training Pipeline
================================================
Downloads ESC-50 (free, CC-BY licence), filters the 5 target sound classes,
extracts MFCC features, trains a compact CNN, and exports a TFLite flatbuffer
ready to flash onto the ESP32-S3.

Requirements:
    pip install -r requirements.txt

Usage:
    python train_model.py               # full pipeline
    python train_model.py --skip-download  # if ESC-50 already in ./data/
    python train_model.py --evaluate    # evaluate existing model

Output files:
    models/sound_classifier.tflite     ← flash this onto ESP32-S3
    models/sound_model.h               ← include in firmware/src/
    models/training_report.png         ← accuracy / loss curves
"""

import argparse
import os
import sys
import zipfile
import urllib.request
import shutil

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Silence TF verbose output ─────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ── Configuration ─────────────────────────────────────────────────────────────
ESC50_URL    = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
DATA_DIR     = "data/ESC-50-master"
MODELS_DIR   = "models"
SAMPLE_RATE  = 16000
DURATION_S   = 5          # ESC-50 clips are 5 s
N_MFCC       = 13
N_FRAMES     = 32         # frames per window (matches firmware WINDOW_FRAMES)
BATCH_SIZE   = 32
EPOCHS       = 50
SEED         = 42

# Map ESC-50 category names → our class indices
# ESC-50 categories: https://github.com/karolpiczak/ESC-50#license
TARGET_CLASSES = {
    "door_wood_knock": 0,   # proxy for doorbell (ESC-50 has no doorbell)
    "clock_alarm":     2,   # fire_alarm proxy
    "smoke_detector":  3,   # exact match
    "phone":           4,   # phone_ring
    # class 1 (microwave) — use hand_saw as a synthetic stand-in;
    # replace with real microwave clips in production
    "hand_saw":        1,
}
LABEL_NAMES = [
    "Doorbell", "Microwave", "Fire alarm",
    "Smoke alarm", "Phone ring"
]
N_CLASSES = len(LABEL_NAMES)

# ── Download ESC-50 ───────────────────────────────────────────────────────────
def download_dataset():
    zip_path = "data/ESC-50-master.zip"
    os.makedirs("data", exist_ok=True)

    if os.path.isdir(DATA_DIR):
        print("[DATA] ESC-50 already present, skipping download.")
        return

    print(f"[DATA] Downloading ESC-50 from GitHub (~600 MB)…")
    urllib.request.urlretrieve(ESC50_URL, zip_path,
        reporthook=lambda b, bs, tot: print(
            f"\r  {min(b*bs, tot)/tot*100:.1f}%", end="", flush=True))
    print()

    print("[DATA] Extracting…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("data/")
    os.remove(zip_path)
    print("[DATA] ESC-50 ready.")


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_mfcc_window(filepath: str) -> np.ndarray:
    """
    Load a WAV clip, resample to 16 kHz, compute overlapping MFCC frames,
    and return a fixed-size window of shape (N_FRAMES, N_MFCC).
    """
    y, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION_S)
    # Pad / trim to exactly DURATION_S * SAMPLE_RATE samples
    target_len = DURATION_S * SAMPLE_RATE
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfccs = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=512, hop_length=256, n_mels=26
    )  # shape: (N_MFCC, time_frames)

    # Normalise (cepstral mean / variance normalisation)
    mfccs = (mfccs - mfccs.mean(axis=1, keepdims=True)) / \
            (mfccs.std(axis=1, keepdims=True) + 1e-9)

    # Fixed-length window via linear interpolation
    from scipy.ndimage import zoom
    current_frames = mfccs.shape[1]
    mfccs_resized = zoom(mfccs, (1, N_FRAMES / current_frames))
    # Transpose → (N_FRAMES, N_MFCC) to match firmware layout
    return mfccs_resized.T[:N_FRAMES, :N_MFCC].astype(np.float32)


def build_dataset():
    meta_path = os.path.join(DATA_DIR, "meta", "esc50.csv")
    meta = pd.read_csv(meta_path)
    meta = meta[meta["category"].isin(TARGET_CLASSES.keys())]

    X, y = [], []
    audio_dir = os.path.join(DATA_DIR, "audio")
    total = len(meta)

    print(f"[FEAT] Extracting features from {total} clips…")
    for i, (_, row) in enumerate(meta.iterrows()):
        fpath = os.path.join(audio_dir, row["filename"])
        try:
            features = extract_mfcc_window(fpath)
            X.append(features)
            y.append(TARGET_CLASSES[row["category"]])
        except Exception as e:
            print(f"  [WARN] Skipped {row['filename']}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{total} clips processed…")

    X = np.array(X)            # (N, N_FRAMES, N_MFCC)
    y = np.array(y, dtype=np.int32)
    print(f"[FEAT] Dataset: X={X.shape}, y={y.shape}")
    print(f"       Class distribution: { {LABEL_NAMES[c]: int((y==c).sum()) for c in range(N_CLASSES)} }")
    return X, y


# ── Model definition ──────────────────────────────────────────────────────────
def build_model() -> tf.keras.Model:
    """
    Compact CNN designed to fit within the ESP32-S3's 60 kB tensor arena.
    Input shape : (N_FRAMES, N_MFCC, 1)  →  e.g. (32, 13, 1)
    Output      : softmax over N_CLASSES
    """
    inp = tf.keras.Input(shape=(N_FRAMES, N_MFCC, 1), name="mfcc_input")

    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inp, out, name="AccessiSound_v1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ── Conversion to TFLite ──────────────────────────────────────────────────────
def convert_to_tflite(model: tf.keras.Model, X_train: np.ndarray):
    os.makedirs(MODELS_DIR, exist_ok=True)
    tflite_path = os.path.join(MODELS_DIR, "sound_classifier.tflite")

    def representative_dataset():
        for sample in X_train[:200]:
            yield [sample.reshape(1, N_FRAMES, N_MFCC, 1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[EXPORT] TFLite model saved → {tflite_path} ({len(tflite_model)/1024:.1f} kB)")

    # Generate C header for firmware inclusion
    _generate_c_header(tflite_model)
    return tflite_path


def _generate_c_header(model_data: bytes):
    header_path = os.path.join(MODELS_DIR, "sound_model.h")
    hex_array = ", ".join(f"0x{b:02x}" for b in model_data)
    header = f"""/*
 * sound_model.h — Auto-generated TFLite model data
 * Generated by train_model.py — do not edit manually.
 * Copy this file into firmware/src/ before building.
 */
#pragma once
#include <stdint.h>

const uint8_t sound_model_data[] = {{
  {hex_array}
}};
const uint32_t sound_model_data_len = {len(model_data)};
"""
    with open(header_path, "w") as f:
        f.write(header)
    print(f"[EXPORT] C header saved → {header_path}")
    shutil.copy(header_path, "firmware/src/sound_model.h")
    print(f"[EXPORT] Copied → firmware/src/sound_model.h")


# ── Training report ───────────────────────────────────────────────────────────
def save_report(history, y_test, y_pred):
    os.makedirs(MODELS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("AccessiSound — Training Report", fontsize=14, weight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[2],
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                cmap="Blues")
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "training_report.png")
    plt.savefig(out, dpi=150)
    print(f"[REPORT] Saved → {out}")
    plt.close()

    # Text report
    print("\n[REPORT] Classification report on test split:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AccessiSound training pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (ESC-50 already in ./data/)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Only evaluate the existing .tflite model")
    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # ── Step 1: Dataset ──────────────────────────────────────────────────────
    if not args.skip_download and not args.evaluate:
        download_dataset()

    X, y = build_dataset()

    # Add channel dim for CNN: (N, frames, mfcc) → (N, frames, mfcc, 1)
    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

    print(f"[SPLIT] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    if args.evaluate:
        print("[EVAL] Load existing model and evaluate.")
        model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "sound_classifier_keras"))
        y_pred = model.predict(X_test).argmax(axis=1)
        save_report(None, y_test, y_pred)
        return

    # ── Step 2: Train ────────────────────────────────────────────────────────
    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ── Step 3: Evaluate ─────────────────────────────────────────────────────
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[EVAL] Test accuracy: {test_acc*100:.1f}%")
    y_pred = model.predict(X_test).argmax(axis=1)
    save_report(history, y_test, y_pred)

    # ── Step 4: Export ───────────────────────────────────────────────────────
    model.save(os.path.join(MODELS_DIR, "sound_classifier_keras"))
    convert_to_tflite(model, X_train)

    print("\n✅ Pipeline complete!")
    print("   Next steps:")
    print("   1. Copy models/sound_model.h → firmware/src/")
    print("   2. Open firmware/ in VS Code + PlatformIO")
    print("   3. Connect ESP32-S3 and run: pio run --target upload")


if __name__ == "__main__":
    main()
