#!/usr/bin/env python3
"""
check_against_firmware.py: prove visualize_pipeline.py matches the firmware
==========================================================================
Compiles the real firmware/src/feature_extractor.h on your computer (with a
tiny stand-in for Arduino.h), feeds it a few frames of audio, and compares
its MFCC output to the NumPy version in visualize_pipeline.py.

    python check_against_firmware.py

Needs a C++ compiler (g++) on your PATH. Expect a max difference around 1e-5,
which is float32 rounding.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import visualize_pipeline as vp

FIRMWARE_SRC = pathlib.Path(__file__).resolve().parent.parent / "firmware" / "src"

ARDUINO_STUB = """#pragma once
#include <stdint.h>
#include <stdio.h>
#include <math.h>
struct SerialStub { template<typename... A> void printf(const char* f, A... a) { ::printf(f, a...); } };
static SerialStub Serial;
"""

HARNESS = """#include "feature_extractor.h"
#include <vector>
int main(int, char** argv) {
    FeatureExtractor fx(16000, 512, 256, 13);
    fx.begin();
    std::vector<int16_t> s(512);
    FILE* f = fopen(argv[1], "rb"); fread(s.data(), 2, 512, f); fclose(f);
    float mfcc[13];
    fx.computeMFCC(s.data(), 512, mfcc);
    FILE* o = fopen(argv[2], "wb"); fwrite(mfcc, 4, 13, o); fclose(o);
    return 0;
}
"""


def main() -> int:
    gxx = shutil.which("g++")
    if not gxx:
        print("g++ not found on PATH; install a C++ compiler to run this check.")
        return 1
    sig = vp.synthetic_chime()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "Arduino.h").write_text(ARDUINO_STUB)
        (tmp / "harness.cpp").write_text(HARNESS)
        exe = tmp / ("harness.exe" if sys.platform == "win32" else "harness")
        cmd = [gxx, "-std=c++17", "-O2", "-D_USE_MATH_DEFINES", f"-I{tmp}", f"-I{FIRMWARE_SRC}",
               str(tmp / "harness.cpp"), "-o", str(exe)]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stderr)
            return 1

        worst = 0.0
        for start in (3000, 6400, 9000, 12000):
            frame = np.clip(np.round(sig[start:start + 512] * 32768), -32768, 32767).astype("<i2")
            (tmp / "frame.bin").write_bytes(frame.tobytes())
            subprocess.run([str(exe), str(tmp / "frame.bin"), str(tmp / "out.bin")], check=True, capture_output=True)
            c_out = np.frombuffer((tmp / "out.bin").read_bytes(), dtype="<f4")
            py_out = vp.frame_steps(frame.astype(np.float64) / 32768.0)["mfcc"]
            err = float(np.max(np.abs(c_out - py_out)))
            worst = max(worst, err)
            print(f"frame at sample {start}: max |firmware C++ - NumPy| = {err:.2e}")
        print(f"worst difference over all frames: {worst:.2e}")
        return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
