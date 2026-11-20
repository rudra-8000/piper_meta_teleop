#!/usr/bin/env python3
"""Sweep an Orbbec camera's colour settings in YOUR lighting and score each one, so the recorder config comes from data.

    .venv/bin/python tools/camera_tune.py --serial CC1365303LL --out logs/tune_high [--wb 4000,5000,5600] [--exposures 50,100,200,300]
    .venv/bin/python tools/camera_tune.py --serial CC1365303LL --out logs/card --white-card     # colour: calibrate on a white sheet

For every preset it starts the camera with those settings, lets it settle, grabs ~3 s and prints:
  fps     distinct frames per second actually delivered (auto-exposure priority can quietly lower it)
  clipHi  % of pixels at or near white (blown out)        clipLo  % at or near black (crushed)
  mean    average brightness 0..255                       p5/p95  the 5th / 95th percentile (the usable range)
  R/G B/G gray-world colour balance (1.0 = neutral; >1 warm/red or blue cast.  Rough: assumes a mostly grey scene)
  whiteR/G, whiteB/G   colour balance of the near-white pixels (walls, paper): 1.00 = neutral white, >1 = reddish/magenta... (the
          better test - a big coloured object skews the gray-world R/G B/G)
  table   the same measured on the lower part of the frame only (the work surface)
  sharp   focus / blur score (variance of the Laplacian; bigger = crisper; only comparable between presets of ONE scene)
and saves one frame per preset plus a labelled contact sheet.  Put the scene (and the lighting) you will record in front of it.
Exposure units are 100 us (100 = 10 ms): multiples of 10 ms avoid banding under 50 Hz mains lights.  At 30 fps the longest
usable exposure is about 330.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop.recording.cameras import make_camera  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--serial", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--exposures", default="50,100,200,300")
ap.add_argument("--wb", default="4000,5000,5600", help="white balance values (K) tried at the middle exposure")
ap.add_argument("--gains", default="20,40,60", help="gain values tried at --gain-exposure (auto white balance)")
ap.add_argument("--gain-exposure", type=int, default=200)
ap.add_argument("--wb-gain", type=int, default=0, help="gain used for the white-balance presets")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--white-card", action="store_true", help="instead of the sweep: hold WHITE paper over the middle of the view; prints rgb_gain")
ap.add_argument("--exposure", type=int, default=200, help="--white-card: the exposure the recorder will use")
ap.add_argument("--gain", type=int, default=40, help="--white-card: the gain the recorder will use")
ap.add_argument("--card-wb", type=int, default=None, help="--white-card: a fixed white balance in K (default: the camera's auto)")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

if a.white_card:
    from piper_teleop.recording.cameras import suggest_rgb_gain
    st = {"exposure": a.exposure, "gain": a.gain}
    if a.card_wb:
        st["white_balance"] = a.card_wb
    cam = make_camera("t", {"type": "orbbec", "serial": a.serial, "width": 640, "height": 480, "fps": a.fps, "settings": st})
    cam.connect()
    try:
        time.sleep(1.5)
        fs = [cam.read() for _ in range(30) if not time.sleep(0.03)]
    finally:
        cam.disconnect()
    f = np.mean(fs, axis=0)
    h, w = f.shape[:2]
    roi = f[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    m = roi.reshape(-1, 3).mean(0)
    luma = 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]
    print(f"middle of the frame averages R {m[0]:.0f}  G {m[1]:.0f}  B {m[2]:.0f}   (brightness {luma:.0f})")
    if luma < 90 or luma > 240 or (roi.reshape(-1, 3).std(0).max() > 40):
        sys.exit("that does not look like an evenly lit WHITE card filling the middle of the view (too dark, blown out or "
                 "patchy). Hold a plain white sheet flat, about 20 cm from the lens, evenly lit, and try again.")
    g = suggest_rgb_gain(m)
    after = np.clip(roi * np.array(g), 0, 255).reshape(-1, 3).mean(0)
    print(f"rgb_gain {g}  ->  the card then reads R {after[0]:.0f} G {after[1]:.0f} B {after[2]:.0f}")
    print("\nput this in config/teleop.yaml under the camera:")
    print(f"    settings: {{exposure: {a.exposure}, gain: {a.gain}" + (f", white_balance: {a.card_wb}" if a.card_wb else "") + f", rgb_gain: {g}}}")
    Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).save(out / "white_card_before.png")
    Image.fromarray(np.clip(f * np.array(g), 0, 255).astype(np.uint8)).save(out / "white_card_after.png")
    sys.exit(0)

AUTO = {"auto_exposure": True, "auto_white_balance": True}
presets = [("auto (SDK default)", {**AUTO, "ae_priority": 1}), ("auto, fixed fps", {**AUTO, "ae_priority": 0})]
exps = [int(x) for x in a.exposures.split(",")]
presets += [(f"manual exp {e}", {"exposure": e, "gain": 0, "auto_white_balance": True}) for e in exps]
mid = exps[len(exps) // 2]
presets += [(f"exp {mid} g{a.wb_gain} wb {k}K", {"exposure": mid, "gain": a.wb_gain, "white_balance": int(k)}) for k in a.wb.split(",") if k]


if a.gains:
    presets += [(f"exp {a.gain_exposure} gain {g}", {"exposure": a.gain_exposure, "gain": int(g), "auto_white_balance": True}) for g in a.gains.split(",")]


def score(frames, fps):
    f = frames[-1]
    luma = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])
    m = f.reshape(-1, 3).astype(float).mean(0)
    ff = f.astype(float)
    near = (luma > 90) & (luma < 240) & ((ff.max(-1) - ff.min(-1)) < 70)      # bright, weakly coloured pixels: walls, paper, ceiling
    wm = ff[near].mean(0) if near.sum() > 500 else np.array([np.nan] * 3)
    tab = luma[int(0.45 * luma.shape[0]):]                       # lower part of the frame: the work surface
    return {"wR/G": float(wm[0] / wm[1]), "wB/G": float(wm[2] / wm[1]), "fps": fps, "tabHi": 100 * float((tab >= 245).mean()), "tabLo": 100 * float((tab <= 10).mean()), "tabMean": float(tab.mean()), "clipHi": 100 * float((luma >= 245).mean()), "clipLo": 100 * float((luma <= 10).mean()), "mean": float(luma.mean()),
            "p5": float(np.percentile(luma, 5)), "p95": float(np.percentile(luma, 95)), "R/G": float(m[0] / m[1]), "B/G": float(m[2] / m[1]),
            "sharp": float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())}


rows, thumbs = [], []
print(f"{'preset':24s} {'fps':>5s} {'clipHi%':>8s} {'clipLo%':>8s} {'mean':>6s} {'p5':>5s} {'p95':>5s} {'R/G':>5s} {'B/G':>5s} {'sharp':>7s} | {'whiteR/G':>8s} {'whiteB/G':>8s} | table: {'mean':>5s} {'hi%':>5s} {'lo%':>5s}")
for name, settings in presets:
    cam = make_camera("t", {"type": "orbbec", "serial": a.serial, "width": 640, "height": 480, "fps": a.fps, "settings": settings})
    try:
        cam.connect()
        time.sleep(1.5)                                          # let auto-exposure / white balance settle
        t0, seen, frames = time.time(), set(), []
        while time.time() - t0 < 3.0:
            f = cam.read()
            key = f[::53, ::53].tobytes()
            if key not in seen:
                seen.add(key)
                frames.append(f)
            time.sleep(0.004)
        s = score(frames, len(seen) / 3.0)
    except Exception as e:                                       # noqa: BLE001
        print(f"{name:24s} FAILED: {e}")
        continue
    finally:
        cam.disconnect()
    rows.append((name, settings, s))
    Image.fromarray(frames[-1]).save(out / (name.replace(" ", "_").replace(",", "") + ".png"))
    thumbs.append((name, frames[-1]))
    print(f"{name:24s} {s['fps']:5.1f} {s['clipHi']:8.1f} {s['clipLo']:8.1f} {s['mean']:6.0f} {s['p5']:5.0f} {s['p95']:5.0f} {s['R/G']:5.2f} {s['B/G']:5.2f} {s['sharp']:7.0f} | {s['wR/G']:8.2f} {s['wB/G']:8.2f} |        {s['tabMean']:5.0f} {s['tabHi']:5.1f} {s['tabLo']:5.1f}", flush=True)

if thumbs:                                                       # contact sheet: 3 per row, labelled
    w, h, cols = 320, 240, 3
    sheet = Image.new("RGB", (w * cols, (h + 18) * ((len(thumbs) + cols - 1) // cols)))
    d = ImageDraw.Draw(sheet)
    for i, (name, f) in enumerate(thumbs):
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        sheet.paste(Image.fromarray(f).resize((w, h)), (x, y + 18))
        d.text((x + 4, y + 3), name, fill=(255, 255, 255))
    sheet.save(out / "contact_sheet.png")
    print(f"\ncontact sheet: {out / 'contact_sheet.png'}")
