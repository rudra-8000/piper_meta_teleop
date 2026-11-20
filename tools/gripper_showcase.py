#!/usr/bin/env python3
"""Show the gripper at the shipped settings: smooth targets, steps and random targets, through the REAL backend + the
GripperLayer configured in config/teleop.yaml, at 200 Hz.

    python tools/gripper_showcase.py left logs/gripper_showcase.csv [--wait-camera DIR]

Smooth: sines (0.2 / 0.5 / 1 Hz), a 0.1 -> 1.5 Hz chirp, linear ramps (30 and 100 mm/s), exponential approach and
exponential growth / decay, cosine ease in/out, a 5 mm staircase.  Steps: 0/15/30/45/60/70 mm up and down.  Random:
picks from those steps, uniform 0-70 mm, and a random smooth curve.  The joints are held where they are (on a
joint-locked arm nothing but gripper frames is sent).  --wait-camera waits until tools/xtion_snap.py is recording into
DIR.  Logs wall time, pattern, target, raw command (with the sine) and measured opening per tick.
"""
import argparse
import csv
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop import config as cfgmod  # noqa: E402
from piper_teleop.gripper_layer import GripperLayer  # noqa: E402
from piper_teleop.joint_lock import lock_until  # noqa: E402

DT = 0.005
STEPS = [0, 15, 30, 45, 60, 70]


def lin(pts):
    """Piecewise-linear target through (t, mm) points."""
    ts, vs = zip(*pts)
    return lambda t: float(np.interp(t, ts, vs))


def held(vals, dwell):
    return lambda t: float(vals[min(int(t / dwell), len(vals) - 1)])


def ease(pts):
    """Cosine ease in/out between (t, mm) points."""
    def f(t):
        for (t0, a), (t1, b) in zip(pts, pts[1:]):
            if t <= t1:
                u = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
                return a + (b - a) * (1 - math.cos(math.pi * u)) / 2
        return pts[-1][1]
    return f


def exp_approach(t, tau=0.4):                    # first-order moves: 35 -> 70, then 70 -> 2
    return 70 - 35 * math.exp(-t / tau) if t < 2.5 else 2 + 68 * math.exp(-(t - 2.5) / tau)


def exp_growth(t, k=math.log(35) / 3):           # accelerating 2 -> 70 in 3 s, then decaying back to 2
    return 2 * math.exp(k * t) if t < 3 else 70 * math.exp(-k * (t - 3))


def segments(rng):
    pick, last = [], None
    for _ in range(12):
        last = rng.choice([s for s in STEPS if s != last])
        pick.append(last)
    uni = [rng.uniform(0, 70) for _ in range(12)]
    fr = [rng.uniform(0.1, 0.6) for _ in range(3)]
    ph = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
    smooth = lambda t: 35 + 30 * sum(math.sin(2 * math.pi * f * t + p) for f, p in zip(fr, ph)) / 3 * math.sin(math.pi * min(t, 10 - t, 1) / 2)
    stair = list(range(0, 71, 5)) + list(range(65, -1, -5))
    return [  # (name, seconds, target(t) mm, kind: smooth / steps / hold)
        ("start: hold 35 mm", 2.0, lambda t: 35.0, "hold"),
        ("sine 0.2 Hz, 5-65 mm", 10.0, lambda t: 35 + 30 * math.sin(2 * math.pi * 0.2 * t), "smooth"),
        ("sine 0.5 Hz, 5-65 mm", 6.0, lambda t: 35 + 30 * math.sin(2 * math.pi * 0.5 * t), "smooth"),
        ("sine 1 Hz, 20-50 mm", 4.0, lambda t: 35 + 15 * math.sin(2 * math.pi * t), "smooth"),
        ("chirp 0.1 -> 1.5 Hz, 10-60 mm", 10.0, lambda t: 35 + 25 * math.sin(2 * math.pi * (0.1 * t + 1.4 * t * t / 20)), "smooth"),
        ("linear ramps 30 mm/s", 6.0, lin([(0, 35), (1.17, 70), (1.67, 70), (4.0, 0), (4.5, 0), (5.67, 35)]), "smooth"),
        ("linear ramps 100 mm/s", 5.3, lin([(0, 35), (0.35, 70), (0.85, 70), (1.55, 0), (2.05, 0), (2.75, 70), (3.25, 70),
                                            (3.95, 0), (4.45, 0), (4.8, 35)]), "smooth"),
        ("exponential approach (tau 0.4 s)", 6.0, exp_approach, "smooth"),
        ("exponential growth / decay", 6.0, exp_growth, "smooth"),
        ("cosine ease in/out, 4 s each way", 8.0, ease([(0, 2), (4, 70), (8, 0)]), "smooth"),
        ("staircase 5 mm every 0.4 s", 0.4 * len(stair), held(stair, 0.4), "steps"),
        ("steps 0/15/30/45/60/70 up + down", 16.5, held([0, 15, 30, 45, 60, 70, 60, 45, 30, 15, 0], 1.5), "steps"),
        ("random steps from 0/15/.../70", 14.4, held(pick, 1.2), "steps"),
        ("random targets 0-70 mm", 12.0, held(uni, 1.0), "steps"),
        ("random smooth curve", 10.0, smooth, "smooth"),
        ("finish: open (default)", 1.5, lambda t: 70.0, "hold"),
    ]


def smooth_stats(tr):
    tg = np.array([x[1] for x in tr])
    ms = np.array([x[2] for x in tr])
    err = ms - tg
    best = min(range(0, 80), key=lambda k: float(np.mean((tg[:len(tg) - k] - ms[k:]) ** 2)))
    sh = ms[best:] - tg[:len(tg) - best]
    return (f"RMS {math.sqrt(np.mean(err ** 2)):4.1f} mm | lag {best * DT:4.2f} s | RMS after lag "
            f"{math.sqrt(np.mean(sh ** 2)):4.1f} mm | max {np.max(np.abs(sh)):4.1f} mm")


def step_stats(tr, first=False):
    """Holds are split where the target changes (not by time, so logged timestamps can't shift a tick across)."""
    holds = []
    for p in tr:
        if not holds or p[1] != holds[-1][-1][1]:
            holds.append([])
        holds[-1].append(p)
    if not first:
        holds = holds[1:]                        # the first hold starts from the previous segment
    ends = [abs(h[-1][2] - h[-1][1]) for h in holds]
    t3 = [next((t - h[0][0] for t, tg, m in h if abs(m - tg) < 3.0), None) for h in holds]
    ok = [x for x in t3 if x is not None]
    return (f"{sum(e < 3 for e in ends)}/{len(ends)} within 3 mm at hold end | end error median "
            f"{statistics.median(ends):4.2f} mm | to 3 mm median {statistics.median(ok) if ok else float('nan'):4.2f} s")


def stats(kind, tr):
    return smooth_stats(tr) if kind == "smooth" else step_stats(tr, first=kind == "hold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("out")
    ap.add_argument("--wait-camera")
    ap.add_argument("--hold-joints", action="store_true",
                    help="allow an arm WITHOUT a joint lock: it is enabled and its joints held where they are")
    a = ap.parse_args()
    raw = cfgmod.load()
    section = raw["arms"][a.arm]
    cfg = cfgmod.arm_config(a.arm, raw, section)
    backend = cfgmod.make_backend(section, raw, False)
    backend.connect()
    if not lock_until(backend.can_name) and not a.hold_joints:          # backends only set joint_locked in enable()
        backend.close()
        sys.exit(f"{a.arm} has no joint lock; this would enable the arm and hold its joints - pass --hold-joints for that")
    if not backend.enable():
        sys.exit("enable failed: " + getattr(backend, "error", ""))
    for n in getattr(backend, "notes", []):
        print(f"[{a.arm}] {n}")
    layer = GripperLayer(cfg.gripper_slew_mm_s, cfg.gripper_dither_mm, cfg.gripper_dither_hz, cfg.gripper_max)
    print(f"gripper layer: ramp {cfg.gripper_slew_mm_s or 'none'} mm/s, sine +-{cfg.gripper_dither_mm} mm @ "
          f"{cfg.gripper_dither_hz} Hz, loop {1 / DT:.0f} Hz", flush=True)
    if a.wait_camera:
        t_end = time.time() + 20
        while time.time() < t_end:
            p = Path(a.wait_camera) / "times.txt"
            if p.exists() and len(p.read_text().splitlines()) > 15:
                break
            time.sleep(0.1)
    st = backend.read()
    q_hold = st.q.copy()
    layer.reset(st.gripper)
    rows = []
    for name, secs, fn, kind in segments(random.Random(7)):
        t0 = time.perf_counter()
        nxt = t0
        tr = []
        while (t := time.perf_counter() - t0) < secs:
            tgt = min(70.0, max(0.0, fn(t)))
            cmd = layer.update(tgt / 1000, DT)
            backend.command(q_hold, cmd)
            meas = backend.read().gripper * 1000
            tr.append((t, tgt, meas))
            rows.append([f"{time.time():.4f}", name, kind, f"{tgt:.2f}", f"{cmd * 1000:.2f}", f"{meas:.2f}"])
            nxt += DT
            while time.perf_counter() < nxt:
                time.sleep(0.0003)
        print(f"  {name:34s} {stats(kind, tr)}", flush=True)
    backend.close()
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "pattern", "kind", "target_mm", "cmd_mm", "measured_mm"])
        w.writerows(rows)
    print(f"{len(rows)} ticks -> {a.out}")


if __name__ == "__main__":
    main()
