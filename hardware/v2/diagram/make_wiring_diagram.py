#!/usr/bin/env python3
"""Draw the AccessiSound v2 wiring diagram from part images.

Part images live in the same folder as this script (see parts.json). Each
face-on part image is cropped to the part's silhouette, so a pin's position in
millimetres maps linearly onto the image. To use a real product photo instead
of a rendered part, add the photo, then give it a `parts.json` entry (size in
mm) and update the pin positions used below.

    python make_wiring_diagram.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
PARTS = HERE
META = json.loads((PARTS / "parts.json").read_text())

W, H = 2800, 2000
COL = dict(red="#e02020", orange="#f28a12", black="#1a1a1a", yellow="#f2c200", green="#3aa843",
           blue="#2f7fe0", purple="#9a3fd0", white="#9a9a9a", cyan="#22b8cf", brown="#8a5a2b")

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def load(name):
    return Image.open(PARTS / name).convert("RGBA")


def put(img, x, y, scale_h):
    h = scale_h; w = img.width * scale_h / img.height
    ax.imshow(np.asarray(img), extent=(x, x + w, y + h, y), zorder=2, interpolation="lanczos")
    return w, h


class Face:
    """Face-on part image with a model (mm) to canvas mapping."""

    def __init__(self, key, x, y, s):
        m = META[key]
        self.m, self.x, self.y, self.s = m, x, y, s
        self.w = (m["xmax"] - m["xmin"]) * s
        self.h = (m["ymax"] - m["ymin"]) * s
        ax.imshow(np.asarray(load(m["file"])), extent=(x, x + self.w, y + self.h, y), zorder=2, interpolation="lanczos")

    def pt(self, X, Y):
        m = self.m
        px = self.x + ((X - m["xmin"]) if m["side"] == "front" else (m["xmax"] - X)) * self.s
        py = self.y + (m["ymax"] - Y) * self.s
        return (px, py)


def wire(pts, color, lw=6, z=5):
    pts = np.array(pts, dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], color=COL[color], lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot(*pts[0], "o", color=COL[color], ms=lw * 0.9, zorder=z + 1)
    ax.plot(*pts[-1], "o", color=COL[color], ms=lw * 0.9, zorder=z + 1)


def label(x, y, text, size=15, color="#222", ha="center", va="center", weight="normal", z=8, rot=0, box=False):
    kw = dict(bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5)) if box else {}
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=z, rotation=rot, **kw)


NET = dict(GND=("black", "GND"), V3V3=("red", "3V3"), V5=("orange", "5V"))


def flag(x, y, net, dy=-40):
    """Power-net symbol on a vertical stub: ground bars for GND, a tag for 3V3 / 5V."""
    c, text = NET[net]
    ex, ey = x, y + dy
    ax.plot([x, ex], [y, ey], color=COL[c], lw=4, zorder=6)
    sgn = -1 if dy < 0 else 1
    if net == "GND":
        for i, wdt in enumerate((30, 20, 10)):
            ax.plot([ex - wdt / 2, ex + wdt / 2], [ey + sgn * i * 7] * 2, color=COL[c], lw=3.5, zorder=6)
    else:
        label(ex, ey + sgn * 18, text, size=13, color=COL[c], weight="bold")
    ax.plot(x, y, "o", color=COL[c], ms=7, zorder=7)


# ================================================================ parts
dk = Face("devkit", 1380, 440, 9.6)
cam = Face("cam", 520, 800, 9.4)
mic = Face("mic", 1900, 200, 14.0)
drv = Face("driver", 1250, 1380, 14.5)
chg = Face("charger", 470, 1380, 11.5)
lip = Face("lipo", 120, 1300, 8.4)

# display: real product photo, upright, header along the top edge
disp = load("display_photo_crop.png")
DSC = 0.40
DX0, DY0 = 440, 250
dw, dh = put(disp, DX0, DY0, disp.height * DSC)


def disp_pin(i):  # 0..7 = GND VCC SCL SDA RST DC CS BL, measured on the product photo
    return (DX0 + (174 + 62.28 * i) * DSC, DY0 + 34 * DSC)


def icon(name, x, y, h):
    im = load(name)
    w, hh = put(im, x, y, h)
    return x, y, w, hh


label(W / 2, 46, "AccessiSound v2 wiring", size=34, weight="bold")

# ================================================================ DevKit
label(dk.x + dk.w / 2, dk.y - 36, "ESP32-S3-DevKitC-1", size=21, weight="bold")
J1 = ['3V3', '3V3', 'RST', 'IO4', 'IO5', 'IO6', 'IO7', 'IO15', 'IO16', 'IO17', 'IO18', 'IO8', 'IO3', 'IO46', 'IO9', 'IO10', 'IO11', 'IO12', 'IO13', 'IO14', '5V', 'GND']
J3 = ['GND', 'IO43', 'IO44', 'IO1', 'IO2', 'IO42', 'IO41', 'IO40', 'IO39', 'IO38', 'IO37', 'IO36', 'IO35', 'IO0', 'IO45', 'IO48', 'IO47', 'IO21', 'IO20', 'IO19', 'GND', 'GND']


def dkp(name, nth=0):
    hits = [(X, 58.84 - 2.54 * k) for hdr, X in ((J1, -4.43), (J3, 18.43)) for k, l in enumerate(hdr) if l == name]
    return dk.pt(*hits[nth])


used = {"IO12", "IO13", "IO14", "IO15", "IO16", "IO17", "IO18", "IO8", "IO10", "IO11", "IO38", "IO39",
        "IO2", "IO42", "IO41", "IO0", "3V3", "5V", "GND"}
for hdr, X, side in ((J1, -4.43, -1), (J3, 18.43, 1)):
    for k, l in enumerate(hdr):
        px, py = dk.pt(X, 58.84 - 2.54 * k)
        ax.plot(px, py, "o", color="#d8b23a", ms=4.2, zorder=4)
        if l in used:
            label(px + side * 44, py, l, size=11.5, weight="bold", ha="center", z=6)
# DevKit GND / 3V3 pins get symbols too
flag(*dkp("3V3", 0), "V3V3", dy=0) if False else None

# ================================================================ display (signals)
label(DX0 + dw / 2, DY0 + dh + 30, "1.54 in ST7789 display, 240 x 240, SPI", size=18, weight="bold")
sig = [(2, "IO12", "yellow", 95, 1270), (3, "IO13", "green", 118, 1250), (4, "IO16", "white", 141, 1230),
       (5, "IO15", "purple", 164, 1210), (6, "IO14", "blue", 187, 1190), (7, "IO17", "orange", 210, 1170)]
for i, pin, c, ly, lx in sig:
    p0 = disp_pin(i); p1 = dkp(pin)
    wire([p0, (p0[0], ly), (lx, ly), (lx, p1[1]), (p1[0] - 8, p1[1])], c)
flag(*disp_pin(0), "GND", dy=-52)
flag(*disp_pin(1), "V3V3", dy=-52)

# ================================================================ camera
label(cam.x + cam.w / 2, cam.y - 34, "ESP32-CAM (AI-Thinker)", size=18, weight="bold")
for i in range(8):
    for X in (-36.23, -11.77):
        px, py = cam.pt(X, 78.5 + 2.54 * i)
        ax.plot(px, py, "o", color="#d8b23a", ms=4, zorder=4)
p_u0r = cam.pt(-11.77, 78.5 + 5 * 2.54)
p_u0t = cam.pt(-11.77, 78.5 + 6 * 2.54)
label(p_u0r[0] - 52, p_u0r[1], "U0R", size=10.5, box=True)
label(p_u0t[0] - 52, p_u0t[1], "U0T", size=10.5, box=True)
t18, t8 = dkp("IO18"), dkp("IO8")
wire([p_u0r, (1060, p_u0r[1]), (1060, t18[1]), (t18[0] - 8, t18[1])], "cyan")
wire([p_u0t, (1085, p_u0t[1]), (1085, t8[1]), (t8[0] - 8, t8[1])], "brown")
c5 = cam.pt(-36.23, 78.5); cg = cam.pt(-36.23, 78.5 + 2.54)
flag(c5[0], c5[1], "V5", dy=46)
flag(cg[0] + 22, cg[1], "GND", dy=46)

# ================================================================ microphone
label(mic.x + mic.w / 2, mic.y - 30, "INMP441 I2S microphone", size=18, weight="bold")
mp = [mic.pt(30 + (i - 2.5) * 2.54, 94.3) for i in range(6)]  # order on this module: VDD GND SD WS SCK L/R
names = ("VDD", "GND", "SD", "WS", "SCK", "L/R")
for nm, p in zip(names, mp):
    label(p[0], p[1] + 30, nm, size=10.5, box=True)
flag(mp[0][0], mp[0][1] + 44, "V3V3", dy=44)
flag(mp[1][0], mp[1][1] + 44, "GND", dy=44)
flag(mp[5][0], mp[5][1] + 44, "GND", dy=44)
for idx, pin, c, off, lx in ((2, "IO2", "green", 78, 1700), (3, "IO42", "yellow", 100, 1730), (4, "IO41", "blue", 122, 1760)):
    p = mp[idx]; t = dkp(pin)
    wire([(p[0], p[1] + 44), (p[0], p[1] + off), (lx, p[1] + off), (lx, t[1]), (t[0] + 8, t[1])], c)

# ================================================================ mute button
ix, iy, iw, ih = icon("icon_button_crop.png", 1800, 760, 120)
label(ix + iw / 2 + 60, iy + ih + 100, "Mute button\n(parallel with BOOT)", size=15, weight="bold")
t0 = dkp("IO0")
wire([(ix + 12, iy + ih * 0.55), (1650, iy + ih * 0.55), (1650, t0[1]), (t0[0] + 8, t0[1])], "yellow")
flag(ix + iw * 0.5, iy + ih, "GND", dy=40)

# ================================================================ driver board
label(drv.x + drv.w + 20, drv.y + drv.h / 2, "Driver board\n2N2222, 1N4148,\nresistors", size=16, weight="bold", ha="left")
in_pins = [drv.pt(-33.5 + 2.54 * i, 25.5) for i in range(6)]
out_pins = [drv.pt(-33.5 + 2.54 * i, 10.5) for i in range(6)]
for nm, p in zip(("IO10", "IO11", "IO38", "IO39", "3V3", "GND"), in_pins):
    label(p[0], p[1], nm, size=8.5, color="white", rot=90, z=9)
for nm, p in zip(("BUZ+", "LED R", "LED G", "MOTOR-", "MOTOR+", "GND"), out_pins):
    label(p[0], p[1], nm, size=8.5, color="white", rot=90, z=9)
d10, d11, d38, d39 = dkp("IO10"), dkp("IO11"), dkp("IO38"), dkp("IO39")
top = in_pins[0][1] - 20
wire([(d10[0] - 8, d10[1]), (1130, d10[1]), (1130, top - 50), (in_pins[0][0], top - 50), (in_pins[0][0], in_pins[0][1] - 8)], "purple")
wire([(d11[0] - 8, d11[1]), (1110, d11[1]), (1110, top - 78), (in_pins[1][0], top - 78), (in_pins[1][0], in_pins[1][1] - 8)], "blue")
wire([(d38[0] + 8, d38[1]), (1730 + 60, d38[1]), (1790, top - 30), (in_pins[2][0], top - 30), (in_pins[2][0], in_pins[2][1] - 8)], "cyan")
wire([(d39[0] + 8, d39[1]), (1810, d39[1]), (1810, top - 58), (in_pins[3][0], top - 58), (in_pins[3][0], in_pins[3][1] - 8)], "brown")
flag(in_pins[4][0], in_pins[4][1] - 8, "V3V3", dy=-40)
flag(in_pins[5][0], in_pins[5][1] - 8, "GND", dy=-40)

# outputs
by = 1780
bx, byy, bw, bh = icon("icon_buzzer_crop.png", 1010, by, 110)
label(bx + bw / 2, byy + bh + 28, "Buzzer", size=15, weight="bold")
mx, my, mw, mh = icon("icon_motor_crop.png", 1260, by, 110)
label(mx + mw / 2, my + mh + 28, "Vibration motor", size=15, weight="bold")
lx1, ly1, lw1, lh1 = icon("icon_led_red_crop.png", 1560, by, 90)
lx2, ly2, lw2, lh2 = icon("icon_led_green_crop.png", 1790, by, 90)
label(lx1 + lw1 / 2, ly1 + lh1 + 28, "Red LED", size=15, weight="bold")
label(lx2 + lw2 / 2, ly2 + lh2 + 28, "Green LED", size=15, weight="bold")
oy = out_pins[0][1] + 8
wire([(out_pins[0][0], oy), (out_pins[0][0], 1720), (bx + bw * 0.45, 1720), (bx + bw * 0.45, by + 20)], "blue")
wire([(out_pins[3][0], oy), (out_pins[3][0], 1745), (mx + mw * 0.5, 1745), (mx + mw * 0.5, my + 20)], "purple")
wire([(out_pins[1][0], oy), (out_pins[1][0], 1700), (lx1 + lw1 * 0.32, 1700), (lx1 + lw1 * 0.32, ly1 + 22)], "cyan")
wire([(out_pins[2][0], oy), (out_pins[2][0], 1680), (lx2 + lw2 * 0.32, 1680), (lx2 + lw2 * 0.32, ly2 + 22)], "brown")
flag(out_pins[4][0], oy, "V3V3", dy=30)
flag(out_pins[5][0], oy, "GND", dy=44)
ax.text(1500, 1960, "LED cathodes and buzzer minus return to GND", fontsize=11.5, color="#444", ha="left", zorder=8)

# ================================================================ power chain
label(lip.x + lip.w / 2, lip.y + lip.h + 32, "LiPo 3.7 V", size=18, weight="bold")
label(chg.x + chg.w / 2, chg.y + chg.h + 62, "TP4056 USB-C charger", size=18, weight="bold")
ax.text(chg.x + chg.w / 2, chg.y + chg.h + 30, "USB-C charging port", fontsize=12, color="#333", ha="center", zorder=8)
ax.add_patch(plt.Rectangle((770, 1470), 210, 120, fc="#eef3fb", ec="#5b7fae", lw=3, zorder=2))
label(875, 1520, "5 V boost\nconverter", size=17, weight="bold")
ax.text(875, 1614, "(generic module)", fontsize=12, color="#555", ha="center", zorder=8)
lp = lip.pt(-11, 52); ln = lip.pt(-9, 52)
pads = {n: chg.pt(x, 27.9) for n, x in (("OUT+", 22.0), ("OUT-", 25.5), ("B+", 29.0), ("B-", 32.5))}
for n, p in pads.items():
    label(p[0], p[1] - 24, n, size=9, box=True)
wire([lp, (lp[0], lp[1] - 46), (pads["B+"][0], lp[1] - 46), (pads["B+"][0], pads["B+"][1])], "red")
wire([ln, (ln[0], ln[1] - 76), (pads["B-"][0], ln[1] - 76), (pads["B-"][0], pads["B-"][1])], "black")
wire([pads["OUT+"], (pads["OUT+"][0], 1430), (730, 1430), (730, 1500), (770, 1500)], "red")
wire([pads["OUT-"], (pads["OUT-"][0], 1408), (700, 1408), (700, 1560), (770, 1560)], "black")
d5, dg = dkp("5V"), dkp("GND", 0)
wire([(980, 1500), (1010, 1500), (1010, d5[1]), (d5[0] - 8, d5[1])], "orange")
wire([(980, 1560), (1040, 1560), (1040, dg[1] + 26), (dg[0] - 8, dg[1] + 26), (dg[0] - 8, dg[1])], "black")

# ================================================================ legend
leg = [("red", "3V3 and battery +"), ("orange", "5 V"), ("black", "GND and battery -")]
label(2330, 1500, "Wire colors", size=17, weight="bold", ha="left")
for i, (c, t) in enumerate(leg):
    ax.plot([2330, 2385], [1545 + i * 36] * 2, color=COL[c], lw=6, zorder=6)
    label(2400, 1545 + i * 36, t, size=14, ha="left")
label(2330, 1675, "Other colors mark the signal lines.", size=13, ha="left")
label(2330, 1705, "Ground symbols connect to a common GND.", size=13, ha="left")
label(2330, 1735, "Pin labels on the DevKit are GPIO numbers.", size=13, ha="left")

out = HERE / "wiring_diagram.png"
fig.savefig(out, dpi=100, facecolor="white")
print("wrote", out)
