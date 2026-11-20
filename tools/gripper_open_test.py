#!/usr/bin/env python3
"""Opening / closing delay of a Piper gripper for different input-layer settings (gripper frames only).

    python tools/gripper_open_test.py can_right logs/gripper_open_right.csv [--reps 5] [--settings JSON]

Each trial: target 70 mm for 1.5 s (open), then 0 mm for 1.5 s (close), through GripperLayer at 200 Hz.  Settings are
interleaved in a random order every rep, so drift (warm-up, the gripper's mood) does not favour one of them.  Reports,
per setting and direction: time until the measured opening has moved 2 mm toward the target (start), time until it is
within 3 mm of it (arrive), and how often it never arrived.  Every 0x159 frame is checked against config/joint_lock.yaml.
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
HOLD = 1.5
SETTINGS = {                                        # name: (slew mm/s, sine mm, sine Hz)
    "shipped: step + 0.5mm@6Hz": (0, 0.5, 6),
    "ramp 300 mm/s": (300, 0.5, 6),
    "ramp 600 mm/s": (600, 0.5, 6),
    "ramp 1000 mm/s": (1000, 0.5, 6),
    "step + 1mm@6Hz": (0, 1.0, 6),
}

ap = argparse.ArgumentParser()
ap.add_argument("iface")
ap.add_argument("out")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--settings", help='JSON {"name": [slew, amp, hz], ...} instead of the defaults')
a = ap.parse_args()
if a.settings:
    SETTINGS = {k: tuple(v) for k, v in json.loads(a.settings).items()}
bus = can.Bus(channel=a.iface, interface="socketcan")
fb = {}
rows = []


def poll(dt):
    end = time.time() + dt
    while (left := end - time.time()) > 0:
        m = bus.recv(left)
        if m is not None and m.arbitration_id == 0x2A8:
            fb["w"] = struct.unpack(">ih", bytes(m.data[:6]))[0] / 1000


def send_mm(mm):
    check(a.iface, 0x159)
    bus.send(can.Message(arbitration_id=0x159, data=struct.pack(">ihBB", int(round(mm * 1000)), 1000, 0x01, 0),
                         is_extended_id=False))


def hold(layer, target_mm, label):
    t0 = time.time()
    nxt = t0
    tr = []
    while (t := time.time() - t0) < HOLD:
        send_mm(layer.update(target_mm / 1000, DT) * 1000)
        nxt += DT
        poll(max(0.0, nxt - time.time()))
        tr.append((t, fb["w"]))
        rows.append([f"{time.time():.4f}", label, f"{target_mm:.1f}", f"{fb['w']:.2f}"])
    return tr


poll(0.4)
if "w" not in fb:
    sys.exit(f"no gripper feedback (0x2A8) on {a.iface}")
res = {(n, d): [] for n in SETTINGS for d in ("open", "close")}
order_rng = random.Random(5)
for rep in range(a.reps):
    names = list(SETTINGS)
    order_rng.shuffle(names)
    for name in names:
        slew, amp, hz = SETTINGS[name]
        layer = GripperLayer(slew_mm_s=slew, dither_mm=amp, dither_hz=hz)
        layer.reset(fb["w"] / 1000)
        hold(layer, 0.0, f"{name} rep{rep} settle")                      # start every trial from closed
        for direction, tgt in (("open", 70.0), ("close", 0.0)):
            start_mm = fb["w"]
            tr = hold(layer, tgt, f"{name} rep{rep} {direction}")
            sgn = 1 if tgt > start_mm else -1
            st = next((t for t, m in tr if (m - start_mm) * sgn > 2.0), None)
            ar = next((t for t, m in tr if abs(m - tgt) < 3.0), None)
            res[(name, direction)].append((st, ar))
        print(f"rep {rep + 1}/{a.reps}  {name:28s} open start/arrive {res[(name, 'open')][-1]}  "
              f"close {res[(name, 'close')][-1]}", flush=True)
bus.shutdown()


def fmt(xs):
    v = [x for x in xs if x is not None]
    return f"{statistics.median(v):4.2f} (worst {max(v):4.2f})" if v else "   never"


print(f"\n{'setting':28s} {'dir':5s} {'start moving, median s':>26s} {'arrive within 3 mm, median s':>30s}  never arrived")
for name in SETTINGS:
    for d in ("open", "close"):
        r = res[(name, d)]
        print(f"{name:28s} {d:5s} {fmt([x[0] for x in r]):>26s} {fmt([x[1] for x in r]):>30s}  "
              f"{sum(x[1] is None for x in r)}/{len(r)}")
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t", "pattern", "target_mm", "measured_mm"])
    w.writerows(rows)
