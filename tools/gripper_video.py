#!/usr/bin/env python3
"""Camera frames + command / measured plots side by side, from a tools/gripper_showcase.py run.  PIL + ffmpeg only.

    python tools/gripper_video.py FRAMES_DIR logs/gripper_showcase.csv OUT.mp4 OUT_overview.png

FRAMES_DIR is a tools/xtion_snap.py recording (color_NNN.png + times.txt, wall-clock times aligned with the CSV).
Video: camera on the left; on the right the last 12 s of target vs measured, the error, and the whole run with a
cursor.  The PNG has one small panel per pattern with its stats.
"""
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gripper_showcase import segments, stats  # noqa: E402

SURF, INK, INK2, GRID, BAND = (252, 252, 251), (11, 11, 11), (82, 81, 78), (228, 228, 225), (243, 243, 240)
TGT, MEAS = (235, 104, 52), (42, 120, 214)          # validated categorical slots 2 (orange) and 1 (blue)
DJ = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
F = {k: ImageFont.truetype(f"{DJ}{'-Bold' if b else ''}.ttf", s) for k, (b, s) in
     {"title": (1, 20), "big": (1, 26), "h": (1, 16), "t": (0, 14), "s": (0, 12)}.items()}
WIN = 12.0                                           # rolling window, s


def load(csv_path):
    R = list(csv.DictReader(open(csv_path)))
    t = np.array([float(r["t"]) for r in R])
    return (t, np.array([float(r["target_mm"]) for r in R]), np.array([float(r["measured_mm"]) for r in R]),
            [r["pattern"] for r in R])


def seg_spans(t, pat):
    spans, start = [], 0
    for i in range(1, len(pat) + 1):
        if i == len(pat) or pat[i] != pat[start]:
            spans.append((pat[start], start, i))
            start = i
    return spans


def legend(d, x, y):
    for col, txt in ((TGT, "target (command)"), (MEAS, "measured opening")):
        d.line([(x, y + 8), (x + 22, y + 8)], fill=col, width=3)
        d.text((x + 28, y), txt, fill=INK2, font=F["t"])
        x += 28 + d.textlength(txt, font=F["t"]) + 22


def stats_for(spans, t, tg, ms):
    kind = {n: k for n, _, _, k in segments(random.Random(7))}
    return {name: stats(kind.get(name, "smooth"), list(zip(t[a:b] - t[a], tg[a:b], ms[a:b]))) for name, a, b in spans}


def overview_png(path, spans, t, tg, ms, st):
    cols, pw, ph = 4, 390, 230
    rows = (len(spans) + cols - 1) // cols
    img = Image.new("RGB", (cols * pw + 20, rows * ph + 70), SURF)
    d = ImageDraw.Draw(img)
    d.text((16, 12), "Piper gripper at the shipped settings: target vs measured, per pattern", fill=INK, font=F["title"])
    legend(d, 16, 42)
    for n, (name, a, b) in enumerate(spans):
        x0, y0 = 10 + (n % cols) * pw, 70 + (n // cols) * ph
        X0, Y0, Wp, Hp = x0 + 34, y0 + 52, pw - 50, ph - 82
        d.text((x0 + 6, y0 + 2), name, fill=INK, font=F["h"])
        parts = st[name].split(" | ")
        for i, part in enumerate((parts[0], " | ".join(parts[1:]))):
            d.text((x0 + 6, y0 + 21 + 14 * i), part, fill=INK2, font=F["s"])
        for mm in (0, 35, 70):
            yy = Y0 + Hp - mm / 70 * Hp
            d.line([(X0, yy), (X0 + Wp, yy)], fill=GRID)
            d.text((X0 - 26, yy - 7), f"{mm:>2d}", fill=INK2, font=F["s"])
        tt = t[a:b] - t[a]
        d.text((X0 + Wp - 40, Y0 + Hp + 4), f"{tt[-1]:.1f} s", fill=INK2, font=F["s"])
        xs = X0 + tt / max(tt[-1], 1e-6) * Wp
        for arr, col in ((tg[a:b], TGT), (ms[a:b], MEAS)):
            d.line(list(zip(xs[::2], Y0 + Hp - np.clip(arr[::2], 0, 70) / 70 * Hp)), fill=col, width=2, joint="curve")
    img.save(path)


def main():
    frames_dir, csv_path, out_mp4, out_png = sys.argv[1:5]
    t, tg, ms, pat = load(csv_path)
    spans = seg_spans(t, pat)
    st = stats_for(spans, t, tg, ms)
    overview_png(out_png, spans, t, tg, ms, st)
    print(out_png)

    W, H = 1280, 720
    base = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(base)
    d.text((16, 10), "Piper gripper - shipped gripper layer (no ramp, +-0.5 mm sine @ 6 Hz), real backend at 200 Hz",
           fill=INK, font=F["title"])
    # plot A: rolling target vs measured
    AX, AY, AW, AH = 624, 74, 636, 290
    legend(d, AX, 44)
    for mm in (0, 15, 30, 45, 60, 70):
        yy = AY + AH - mm / 70 * AH
        d.line([(AX, yy), (AX + AW, yy)], fill=GRID)
        d.text((AX - 30, yy - 8), f"{mm:>2d}", fill=INK2, font=F["t"])
    d.text((AX - 30, 46), "mm", fill=INK2, font=F["s"])
    for s in range(0, int(WIN) + 1, 2):
        xx = AX + AW - s / WIN * AW
        d.text((xx - 14, AY + AH + 6), "now" if s == 0 else f"-{s} s", fill=INK2, font=F["s"])
    # plot B: error
    BX, BY, BW, BH = AX, 420, AW, 110
    d.text((BX, BY - 24), "error = measured - target (mm)", fill=INK2, font=F["t"])
    for e in (-10, -5, 0, 5, 10):
        yy = BY + BH / 2 - e / 10 * BH / 2
        d.line([(BX, yy), (BX + BW, yy)], fill=INK2 if e == 0 else GRID)
        d.text((BX - 34, yy - 8), f"{e:>3d}", fill=INK2, font=F["s"])
    # plot C: whole run with pattern bands
    CX, CY, CW, CH = 40, 604, 1220, 96
    T0, T1 = t[0], t[-1]
    cx = lambda tt: CX + (tt - T0) / (T1 - T0) * CW
    for n, (name, a, b) in enumerate(spans):
        if n % 2:
            d.rectangle([cx(t[a]), CY, cx(t[b - 1]), CY + CH], fill=BAND)
    for arr, col in ((tg, TGT), (ms, MEAS)):
        d.line(list(zip(cx(t[::20]), CY + CH - np.clip(arr[::20], 0, 70) / 70 * CH)), fill=col, width=1)
    d.text((CX, CY - 22), f"whole run ({T1 - T0:.0f} s), shaded = pattern changes", fill=INK2, font=F["t"])

    times = [tuple(map(float, l.split())) for l in open(f"{frames_dir}/times.txt")]
    keep = [(int(i), ft) for i, ft in times if T0 - 0.3 <= ft <= T1 + 0.3 and os.path.exists(f"{frames_dir}/color_{int(i):03d}.png")]
    fps = (len(keep) - 1) / (keep[-1][1] - keep[0][1])
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-framerate", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                           "-movflags", "+faststart", out_mp4], stdin=subprocess.PIPE)
    seg_of = {name: n for n, (name, _, _) in enumerate(spans)}
    for i, ft in keep:
        c = base.copy()
        c.paste(Image.open(f"{frames_dir}/color_{i:03d}.png").convert("RGB").resize((560, 420), Image.BICUBIC), (16, 44))
        d = ImageDraw.Draw(c)
        k = int(np.clip(np.searchsorted(t, ft), 0, len(t) - 1))
        d.text((16, 472), f"{seg_of[pat[k]] + 1}/{len(spans)}  {pat[k]}", fill=INK, font=F["h"])
        d.text((16, 494), f"t = {ft - T0:5.1f} s", fill=INK2, font=F["t"])
        for x, col, lab, v in ((16, TGT, "target", tg[k]), (206, MEAS, "measured", ms[k]), (396, None, "error", ms[k] - tg[k])):
            if col:
                d.rectangle([x, 523, x + 10, 533], fill=col)
            d.text((x + (16 if col else 0), 518), lab, fill=INK2, font=F["t"])
            d.text((x, 536), f"{v:+5.1f} mm" if lab == "error" else f"{v:5.1f} mm", fill=INK, font=F["big"])
        sel = (t >= ft - WIN) & (t <= ft)
        if sel.sum() > 2:
            xs = AX + AW - (ft - t[sel]) / WIN * AW
            for arr, col in ((tg[sel], TGT), (ms[sel], MEAS)):
                d.line(list(zip(xs[::2], AY + AH - np.clip(arr[::2], 0, 70) / 70 * AH)), fill=col, width=2, joint="curve")
            er = np.clip(ms[sel] - tg[sel], -10, 10)
            d.line(list(zip(xs[::2], BY + BH / 2 - er[::2] / 10 * BH / 2)), fill=INK2, width=2, joint="curve")
        d.line([(cx(ft), CY - 2), (cx(ft), CY + CH + 2)], fill=INK, width=2)
        ff.stdin.write(c.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out_mp4, f"{len(keep)} frames @ {fps:.1f} fps")


if __name__ == "__main__":
    main()
