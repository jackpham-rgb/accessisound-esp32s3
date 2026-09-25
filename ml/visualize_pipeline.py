#!/usr/bin/env python3
"""
visualize_pipeline.py: make the AccessiSound algorithm visible
================================================================
Draws every step from raw audio to an alert decision, using a from-scratch
NumPy version of the firmware's own math (firmware/src/feature_extractor.h
and the inference rule in firmware/src/main.cpp). It does not need TensorFlow,
a dataset download or any dedicated hardware.

    python visualize_pipeline.py                 # built-in synthetic chime
    python visualize_pipeline.py clip.wav        # any 16-bit WAV
    python visualize_pipeline.py --out ../docs/imgs

Writes four PNGs:
    pipeline_steps.png              one frame, step by step (the DSP)
    pipeline_features.png           whole clip: spectrogram -> mel -> MFCC
    pipeline_classifier.png         CNN input, architecture, alert rule
    pipeline_train_vs_firmware.png  firmware MFCC vs the librosa MFCC that
                                    train_model.py uses (needs librosa)

Requires only: numpy, scipy, matplotlib. librosa is optional (last figure).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

# Constants copied from firmware/src/main.cpp and feature_extractor.h.
SR = 16000
FRAME = 512          # FRAME_SIZE: 32 ms
HOP = 256            # HOP_SIZE: 16 ms (50% overlap)
N_MFCC = 13
N_MELS = 26          # MEL_FILTERBANKS
PRE_EMPH = 0.97      # PRE_EMPHASIS
F_LOW = 80.0         # melLow in _buildMelFilterbank
WINDOW_FRAMES = 32   # frames stacked per inference
INFER_EVERY = WINDOW_FRAMES // 2
CONF_THR = 0.72      # CONFIDENCE_THR
LABELS = ["Doorbell", "Microwave beep", "Fire alarm", "Smoke alarm", "Phone ringing"]


# ---------------------------------------------------------------------------
# The firmware's DSP, in NumPy
# ---------------------------------------------------------------------------
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=float) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def mel_filterbank():
    """Same integer bin arithmetic as _buildMelFilterbank(): returns a
    (N_MELS, FRAME//2+1) weight matrix plus each filter's (start, center, stop)."""
    n_bins = FRAME // 2 + 1
    lo, hi = hz_to_mel(F_LOW), hz_to_mel(SR / 2.0)
    pts = mel_to_hz(lo + np.arange(N_MELS + 2) * (hi - lo) / (N_MELS + 1))
    fb = np.zeros((N_MELS, n_bins))
    edges = []
    for m in range(N_MELS):
        start = int(pts[m] * FRAME / SR)
        center = int(pts[m + 1] * FRAME / SR)
        stop = int(pts[m + 2] * FRAME / SR)
        for k in range(start, min(stop, n_bins - 1) + 1):
            if k <= center:
                fb[m, k] = (k - start) / (center - start + 1)
            else:
                fb[m, k] = (stop - k) / (stop - center + 1)
        edges.append((start, center, stop))
    return fb, edges


def dct_matrix():
    """_buildDCTMatrix(): DCT-II rows scaled by sqrt(2/M)."""
    i = np.arange(N_MFCC)[:, None]
    m = np.arange(N_MELS)[None, :]
    return np.cos(np.pi * i * (m + 0.5) / N_MELS) * np.sqrt(2.0 / N_MELS)


FB, FB_EDGES = mel_filterbank()
DCT = dct_matrix()


def frame_steps(x):
    """Run computeMFCC() on one frame and keep every intermediate."""
    pre = np.empty_like(x)
    prev = 0.0  # the firmware resets prev to 0 at the start of every frame
    for n, s in enumerate(x):
        pre[n] = s - PRE_EMPH * prev
        prev = s
    ham = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(len(x)) / (len(x) - 1))
    windowed = pre * ham
    power = np.abs(np.fft.rfft(windowed)) ** 2 / len(x)   # == the C++ DFT loop
    mel_energy = FB @ power
    log_mel = np.log(mel_energy + 1e-9)
    mfcc_raw = DCT @ log_mel
    mfcc = mfcc_raw - mfcc_raw.mean()                     # firmware step 6
    return dict(pre=pre, ham=ham, windowed=windowed, power=power,
                mel_energy=mel_energy, log_mel=log_mel, mfcc_raw=mfcc_raw, mfcc=mfcc)


def clip_features(sig):
    n = 1 + (len(sig) - FRAME) // HOP
    spec = np.zeros((FRAME // 2 + 1, n))
    logmel = np.zeros((N_MELS, n))
    mfcc = np.zeros((N_MFCC, n))
    for t in range(n):
        st = frame_steps(sig[t * HOP: t * HOP + FRAME])
        spec[:, t] = st["power"]
        logmel[:, t] = st["log_mel"]
        mfcc[:, t] = st["mfcc"]
    return spec, logmel, mfcc


# ---------------------------------------------------------------------------
# Test signal
# ---------------------------------------------------------------------------
def synthetic_chime(seconds=1.6, seed=0):
    """A two-note 'ding-dong' (E5 then C5, with harmonics and decay) plus a
    little room noise. A stand-in so the script runs with no data files."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    sig = np.zeros_like(t)

    def note(f0, t0, dur):
        m = (t >= t0) & (t < t0 + dur)
        tt = t[m] - t0
        env = np.exp(-4.5 * tt) * (1 - np.exp(-200 * tt))
        tone = np.sin(2 * np.pi * f0 * tt) + 0.5 * np.sin(2 * np.pi * 2 * f0 * tt) \
            + 0.25 * np.sin(2 * np.pi * 3 * f0 * tt)
        sig[m] += env * tone

    note(659.25, 0.20, 0.60)
    note(523.25, 0.80, 0.75)
    sig = sig / np.max(np.abs(sig)) * 0.6
    sig += rng.normal(0, 0.01, size=sig.shape)
    return sig.astype(np.float64)


def load_wav(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sr, data = wavfile.read(path)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind in "iu" or np.max(np.abs(data)) > 1.5:
        data = data / 32768.0
    if sr != SR:
        g = np.gcd(sr, SR)
        data = resample_poly(data, SR // g, sr // g)
    return data


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_steps(sig, f_idx, out):
    x = sig[f_idx * HOP: f_idx * HOP + FRAME]
    st = frame_steps(x)
    t_ms = np.arange(FRAME) / SR * 1000
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)

    fig, ax = plt.subplots(4, 2, figsize=(13, 15))
    ax = ax.ravel()

    tt = np.arange(len(sig)) / SR
    ax[0].plot(tt, sig, lw=0.6, color="C0")
    ax[0].axvspan(f_idx * HOP / SR, (f_idx * HOP + FRAME) / SR, color="C3", alpha=0.35)
    ax[0].set_title("1. Raw audio at 16 kHz (red = the 512-sample frame used below)")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("amplitude")

    ax[1].plot(t_ms, x, label="raw x[n]", color="C0")
    ax[1].plot(t_ms, st["pre"], label="pre-emphasized y[n]", color="C1")
    ax[1].set_title("2. Pre-emphasis: y[n] = x[n] - 0.97 x[n-1]\n(boosts high frequencies, which carry the alarm and chime detail)")
    ax[1].set_xlabel("time in frame (ms)"); ax[1].legend(fontsize=8)

    ax[2].plot(t_ms, st["ham"], color="gray", label="Hamming w[n]")
    ax[2].plot(t_ms, st["windowed"], color="C2", label="y[n] * w[n]")
    ax[2].set_title("3. Hamming window: w[n] = 0.54 - 0.46 cos(2 pi n / (N-1))\n(tapers the frame edges so the DFT does not smear)")
    ax[2].set_xlabel("time in frame (ms)"); ax[2].legend(fontsize=8)

    ax[3].plot(freqs, 10 * np.log10(st["power"] + 1e-12), color="C4")
    ax[3].set_title("4. Power spectrum: P[k] = |X[k]|^2 / N, N = 512\n(each bin is 31.25 Hz wide)")
    ax[3].set_xlabel("frequency (Hz)"); ax[3].set_ylabel("dB")

    cols = plt.cm.viridis(np.linspace(0, 1, N_MELS))
    for m in range(N_MELS):
        ax[4].plot(freqs, FB[m], color=cols[m], lw=1.0)
    ax[4].set_title("5. Mel filterbank: 26 triangles, 80 Hz to 8 kHz\n(narrow at low pitch, wide at high pitch, like human hearing)")
    ax[4].set_xlabel("frequency (Hz)"); ax[4].set_ylabel("filter weight")

    ax[5].bar(np.arange(N_MELS), st["log_mel"], color="C5")
    ax[5].set_title("6. Log mel energies: E_m = ln( sum_k P[k] W_m[k] + 1e-9 )")
    ax[5].set_xlabel("mel filter index m"); ax[5].set_ylabel("ln energy")

    ax[6].bar(np.arange(N_MFCC), st["mfcc_raw"], color="C6", alpha=0.45, label="DCT output")
    ax[6].bar(np.arange(N_MFCC), st["mfcc"], color="C6", width=0.5, label="after mean subtraction")
    ax[6].set_title("7. DCT to 13 MFCCs: c_i = sqrt(2/26) sum_m E_m cos( pi i (m + 0.5) / 26 )\nthen subtract the mean of the 13 values")
    ax[6].set_xlabel("coefficient i"); ax[6].legend(fontsize=8)

    ax[7].axis("off")
    ax[7].text(0.0, 1.0,
        "Constants used (from the firmware)\n\n"
        f"sample rate        {SR} Hz\n"
        f"frame              {FRAME} samples = 32 ms\n"
        f"hop                {HOP} samples = 16 ms (50% overlap)\n"
        f"pre-emphasis       {PRE_EMPH}\n"
        f"mel filters        {N_MELS}, from {F_LOW:.0f} Hz to {SR // 2} Hz\n"
        f"MFCCs kept         {N_MFCC}\n"
        f"window for the CNN {WINDOW_FRAMES} frames (0.53 s)\n"
        f"inference every    {INFER_EVERY} frames (0.26 s)\n"
        f"alert threshold    {CONF_THR:.2f} softmax confidence",
        va="top", family="monospace", fontsize=10)
    fig.suptitle("AccessiSound: from one 32 ms frame of audio to 13 numbers", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "pipeline_steps.png", dpi=140)
    plt.close(fig)


def pick_window(mfcc, spec):
    """The 32-frame window with the most energy, so the demo shows the sound."""
    e = spec.sum(axis=0)
    n = spec.shape[1]
    best, best_s = 0, -1.0
    for s in range(0, n - WINDOW_FRAMES + 1):
        v = e[s: s + WINDOW_FRAMES].sum()
        if v > best_s:
            best, best_s = s, v
    return best


def fig_features(sig, spec, logmel, mfcc, w0, out):
    n = spec.shape[1]
    t_axis = (np.arange(n) * HOP + FRAME / 2) / SR
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(np.arange(len(sig)) / SR, sig, lw=0.6)
    ax[0].set_title("Raw waveform")
    ax[0].set_ylabel("amplitude")
    im = ax[1].imshow(10 * np.log10(spec + 1e-12), origin="lower", aspect="auto",
                      extent=(t_axis[0], t_axis[-1], 0, SR / 2), cmap="magma")
    ax[1].set_title("Spectrogram: power spectrum of every frame (dB)")
    ax[1].set_ylabel("Hz")
    ax[2].imshow(logmel, origin="lower", aspect="auto",
                 extent=(t_axis[0], t_axis[-1], 0, N_MELS), cmap="magma")
    ax[2].set_title("Log mel spectrogram: 513 bins folded into 26 mel bands")
    ax[2].set_ylabel("mel band")
    ax[3].imshow(mfcc, origin="lower", aspect="auto",
                 extent=(t_axis[0], t_axis[-1], 0, N_MFCC), cmap="coolwarm")
    ax[3].set_title("MFCCs: DCT of the log mel bands, 13 coefficients per frame")
    ax[3].set_ylabel("coefficient"); ax[3].set_xlabel("time (s)")
    x0, x1 = t_axis[w0], t_axis[w0 + WINDOW_FRAMES - 1]
    for a, top in ((ax[1], SR / 2), (ax[2], N_MELS), (ax[3], N_MFCC)):
        a.add_patch(Rectangle((x0, 0), x1 - x0, top, fill=False, ec="cyan", lw=2))
    ax[3].text(x0 + 0.01, N_MFCC - 0.3, "32-frame window sent to the CNN", color="black", fontsize=9, va="top",
               bbox=dict(fc="white", ec="cyan", alpha=0.85, pad=2))
    fig.tight_layout()
    fig.savefig(out / "pipeline_features.png", dpi=140)
    plt.close(fig)


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def arch_rows():
    """Layer shapes and parameter counts for AccessiSound_v1 (train_model.py)."""
    rows = []
    h, w, c = 32, 13, 1
    rows.append(("Input", f"{h} x {w} x {c}", 0))
    p = 16 * (3 * 3 * c + 1); c = 16
    rows.append(("Conv 3x3, 16, ReLU", f"{h} x {w} x {c}", p))
    rows.append(("BatchNorm", f"{h} x {w} x {c}", 4 * c))
    h, w = h // 2, w // 2
    rows.append(("MaxPool 2x2", f"{h} x {w} x {c}", 0))
    p = 32 * (3 * 3 * c + 1); c = 32
    rows.append(("Conv 3x3, 32, ReLU", f"{h} x {w} x {c}", p))
    rows.append(("BatchNorm", f"{h} x {w} x {c}", 4 * c))
    h, w = h // 2, w // 2
    rows.append(("MaxPool 2x2", f"{h} x {w} x {c}", 0))
    rows.append(("Global avg pool", f"{c}", 0))
    rows.append(("Dense 64, ReLU", "64", c * 64 + 64))
    rows.append(("Dense 5, softmax", "5", 64 * 5 + 5))
    return rows


def fig_classifier(mfcc, w0, out, logits_confident, logits_unsure):
    win = mfcc[:, w0: w0 + WINDOW_FRAMES].T  # (32, 13)
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 0.7], hspace=0.55, wspace=0.28)

    a = fig.add_subplot(gs[0, 0])
    a.imshow(win, aspect="auto", cmap="coolwarm", origin="lower")
    a.set_title("CNN input: 32 frames x 13 MFCCs (the highlighted window)")
    a.set_xlabel("MFCC coefficient"); a.set_ylabel("frame (16 ms each)")

    b = fig.add_subplot(gs[0, 1]); b.axis("off")
    rows = arch_rows()
    total = sum(r[2] for r in rows)
    b.set_title(f"CNN (AccessiSound_v1 in train_model.py): {total:,} parameters", loc="left", pad=22)
    n = len(rows)
    for i, (name, shape, params) in enumerate(rows):
        y = 1 - (i + 1) / (n + 0.4)
        b.add_patch(FancyBboxPatch((0.02, y), 0.55, 0.075, boxstyle="round,pad=0.005",
                                   fc="#dbe7f5", ec="#5b7fae", transform=b.transAxes))
        b.text(0.04, y + 0.037, name, transform=b.transAxes, va="center", fontsize=9)
        b.text(0.60, y + 0.037, shape, transform=b.transAxes, va="center", fontsize=9, family="monospace")
        if params:
            b.text(0.85, y + 0.037, f"{params:,}", transform=b.transAxes, va="center", fontsize=9, ha="left")
    b.text(0.60, 1.0, "output shape", transform=b.transAxes, fontsize=8, color="gray")
    b.text(0.85, 1.0, "params", transform=b.transAxes, fontsize=8, color="gray")

    for col, (title, logits) in enumerate((("Case A: a confident sound", logits_confident),
                                           ("Case B: an unsure sound", logits_unsure))):
        p = softmax(logits)
        c = fig.add_subplot(gs[1, col])
        colors = ["C0"] * len(p)
        top = int(np.argmax(p))
        colors[top] = "C3" if p[top] >= CONF_THR else "C7"
        c.bar(range(len(p)), p, color=colors)
        c.axhline(CONF_THR, color="k", ls="--", lw=1)
        c.text(len(p) - 0.5, CONF_THR + 0.02, "threshold 0.72", ha="right", fontsize=8)
        c.set_ylim(0, 1.05); c.set_xticks(range(len(p)))
        c.set_xticklabels(LABELS, rotation=25, ha="right", fontsize=8)
        c.set_ylabel("softmax confidence")
        verdict = (f"ALERT: {LABELS[top]} ({p[top]:.0%}); vibrate + buzz"
                   if p[top] >= CONF_THR else f"no alert (top guess {LABELS[top]} is only {p[top]:.0%})")
        c.set_title(f"{title}\n{verdict}", fontsize=10)

    d = fig.add_subplot(gs[2, :]); d.axis("off")
    d.text(0.0, 1.0,
        "The decision rule (firmware/src/main.cpp)\n"
        "  1. softmax:  p_i = exp(z_i) / sum_j exp(z_j)   (the CNN's 5 raw scores z become probabilities that sum to 1)\n"
        "  2. best = argmax p;   fire the alert only if  p[best] >= 0.72  and mute is off\n"
        "  3. inference runs every 16 frames (0.26 s) on the newest 32 frames, so windows overlap by 50%\n\n"
        "Cases A and B use ILLUSTRATIVE scores chosen to show the rule. They are not outputs of a trained model.\n"
        "Run demo.py after train_model.py to see the real model's scores for a real clip.",
        va="top", family="monospace", fontsize=9)
    fig.suptitle("AccessiSound: from MFCC window to alert decision", fontsize=14, weight="bold")
    fig.savefig(out / "pipeline_classifier.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_train_vs_firmware(sig, mfcc_fw, out):
    try:
        import librosa
        from scipy.ndimage import zoom
    except Exception as exc:  # librosa is optional
        print(f"[skip] pipeline_train_vs_firmware.png needs librosa ({exc})")
        return None
    m = librosa.feature.mfcc(y=sig.astype(np.float32), sr=SR, n_mfcc=N_MFCC,
                             n_fft=512, hop_length=256, n_mels=26)
    m_cmvn = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)
    n = min(m.shape[1], mfcc_fw.shape[1])
    fw = mfcc_fw[:, :n]
    fw_z = (fw - fw.mean(axis=1, keepdims=True)) / (fw.std(axis=1, keepdims=True) + 1e-9)
    lb_z = m_cmvn[:, :n]
    corr = [np.corrcoef(fw_z[i], lb_z[i])[0, 1] for i in range(1, N_MFCC)]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].imshow(fw_z, origin="lower", aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax[0].set_title("Firmware MFCC (feature_extractor.h)\nper-coefficient z-scored here, for display only")
    ax[1].imshow(lb_z, origin="lower", aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax[1].set_title("Training MFCC (librosa, as in train_model.py)\nafter its per-coefficient normalization")
    for a in ax[:2]:
        a.set_xlabel("frame"); a.set_ylabel("coefficient")
    ax[2].bar(range(1, N_MFCC), corr, color="C1")
    ax[2].set_ylim(-1, 1); ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_title("Correlation per coefficient (c1 to c12)\nof firmware vs librosa on this clip")
    ax[2].set_xlabel("coefficient")
    fig.suptitle("The firmware and the training script compute similar, but not identical, features", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "pipeline_train_vs_firmware.png", dpi=140)
    plt.close(fig)
    return corr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?", help="16-bit WAV file (default: built-in synthetic chime)")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "docs" / "imgs"),
                    help="output folder for the PNGs")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.wav:
        sig = load_wav(args.wav)
        print(f"[input] {args.wav}: {len(sig) / SR:.2f} s")
    else:
        sig = synthetic_chime()
        print("[input] built-in synthetic two-note chime (no data files needed)")
    if len(sig) < FRAME + (WINDOW_FRAMES - 1) * HOP:
        print("[error] clip is shorter than one 32-frame window (0.53 s)")
        sys.exit(1)

    spec, logmel, mfcc = clip_features(sig)
    w0 = pick_window(mfcc, spec)
    f_idx = w0 + int(np.argmax(spec[:, w0: w0 + WINDOW_FRAMES].sum(axis=0)))

    fig_steps(sig, f_idx, out)
    fig_features(sig, spec, logmel, mfcc, w0, out)
    fig_classifier(mfcc, w0, out,
                   logits_confident=[0.3, -0.4, 3.4, 0.5, 0.1],
                   logits_unsure=[0.9, 0.5, 1.5, 1.1, 0.7])
    corr = fig_train_vs_firmware(sig, mfcc, out)
    if corr is not None:
        print(f"[compare] firmware vs librosa MFCC correlation, c1-c12: min {min(corr):.2f}, mean {np.mean(corr):.2f}")
    print(f"[done] wrote figures to {out}")


if __name__ == "__main__":
    main()
