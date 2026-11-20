#!/usr/bin/env python3
"""Replay the gripper targets of a leader_follower log on a real gripper, per input-layer setting.  (gripper frames only)

    python tools/gripper_log_replay.py can_right logs/leader_follower_X.csv logs/gripper_lf_replay.csv \
        [--windows 32.5-38,70.3-75] [--reps 2] [--settings JSON]

Why: with a leader pendant the follower gripper stalled half way on fast closes (commanded 30 mm, stuck at 54 mm for good).
Each trial: open to 70 mm and hold 1.5 s, replay the logged `target_mm` of one window at its original timing through
GripperLayer, then hold the last target 2 s.  Reported per setting: final error and whether it stalled (still > 4 mm off
after the hold).  Settings are interleaved in a random order every rep.  Only 0x159 frames are sent (no joint frames);
every frame is checked against config/joint_lock.yaml.
"""
import argparse
import csv
import json
import random
import statistics
import struct
import sys
import time
from pathlib import Path

import can

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop.gripper_layer import GripperLayer  # noqa: E402
from piper_teleop.joint_lock import check  # noqa: E402

DT = 0.005
SETTINGS = {                                        # name: (slew mm/s, sine mm, sine Hz)
    "shipped: step + 0.5mm@6Hz": (0, 0.5, 6),
    "slew 200 mm/s": (200, 0.5, 6),
    "slew 120 mm/s": (120, 0.5, 6),
    "step + 1mm@6Hz": (0, 1.0, 6),
}
ap = argparse.ArgumentParser()
ap.add_argument("iface")
ap.add_argument("log")
ap.add_argument("out")
ap.add_argument("--windows", default="32.5-38,70.3-75")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--settings", help='JSON {"name": [slew, amp, hz]}')
a = ap.parse_args()
if a.settings:
    SETTINGS = {k: tuple(v) for k, v in json.loads(a.settings).items()}

rows = list(csv.DictReader(open(a.log)))
t0 = float(rows[0]["t"])
T = [float(r["t"]) - t0 for r in rows]
TG = [float(r["target_mm"]) for r in rows]
WIN = []
for w in a.windows.split(","):
    lo, hi = (float(x) for x in w.split("-"))
    WIN.append((w, [g for t, g in zip(T, TG) if lo <= t <= hi]))

bus = can.Bus(channel=a.iface, interface="socketcan")
fb, out = {}, []


def poll(dt):
    end = time.time() + dt
    while (left := end - time.time()) > 0:
        m = bus.recv(left)
        if m is not None and m.arbitration_id == 0x2A8:
            fb["w"] = struct.unpack(">ih", bytes(m.data[:6]))[0] / 1000


def send(layer, target_mm, label):
    check(a.iface, 0x159)
    bus.send(can.Message(arbitration_id=0x159, is_extended_id=False, data=struct.pack(
        ">ihBB", int(round(layer.update(target_mm / 1000, DT) * 1e6)), 1000, 0x01, 0)))
    out.append([f"{time.time():.4f}", label, f"{target_mm:.1f}", f"{fb.get('w', 0):.2f}"])


def hold(layer, mm, secs, label):
    nxt = time.time()
    end = nxt + secs
    while time.time() < end:
        send(layer, mm, label)
        nxt += DT
        poll(max(0.0, nxt - time.time()))


poll(0.4)
if "w" not in fb:
    sys.exit(f"no gripper feedback on {a.iface}")
res = {(n, w): [] for n in SETTINGS for w, _ in WIN}
rng = random.Random(3)
for rep in range(a.reps):
    combos = list(res)
    rng.shuffle(combos)
    for name, wname in combos:
        slew, amp, hz = SETTINGS[name]
        seq = dict(WIN)[wname]
        layer = GripperLayer(slew_mm_s=slew, dither_mm=amp, dither_hz=hz)
        layer.reset(fb["w"] / 1000)
        lab = f"{name} [{wname}] rep{rep}"
        hold(layer, 70.0, 1.5, lab + " open")
        nxt = time.time()
        for g in seq:                                   # the logged targets, at their original 200 Hz timing
            send(layer, g, lab + " replay")
            nxt += DT
            poll(max(0.0, nxt - time.time()))
        hold(layer, seq[-1], 2.0, lab + " hold")
        err = abs(fb["w"] - seq[-1])
        res[(name, wname)].append(err)
        print(f"rep {rep + 1}/{a.reps} {name:26s} window {wname:9s} final target {seq[-1]:5.1f}  measured {fb['w']:5.1f}  "
              f"error {err:5.1f} mm  {'STALLED' if err > 4 else 'ok'}", flush=True)
bus.shutdown()
print("\nfinal error after the 2 s hold (median / worst mm; 'x/n' = trials that stalled, > 4 mm)")
for name in SETTINGS:
    cells = []
    for w, _ in WIN:
        e = res[(name, w)]
        cells.append(f"{statistics.median(e):4.1f}/{max(e):4.1f}  {sum(x > 4 for x in e)}/{len(e)}")
    print(f"{name:28s} " + "   ".join(f"[{w}] {c}" for (w, _), c in zip(WIN, cells)))
with open(a.out, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["t", "pattern", "target_mm", "measured_mm"])
    wr.writerows(out)
