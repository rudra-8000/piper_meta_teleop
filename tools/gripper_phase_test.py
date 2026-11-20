#!/usr/bin/env python3
"""Does the gripper's response to a step depend on WHERE in the dither sine the step lands?  (gripper frames only)

    python tools/gripper_phase_test.py can_right logs/gripper_phase_right.csv [--reps 3] [--amp 0.5] [--hz 6]

Opens to 70 mm / closes to 0 mm in steps; just before each step the layer's sine phase is forced to PHASE (12 values, 0-330
deg).  Reports the delay until the measured opening has moved 2 mm toward the target, per phase and direction.  Phases and
directions are interleaved in a random order.  Every 0x159 frame is checked against config/joint_lock.yaml.
"""
import argparse
import csv
import math
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
ap = argparse.ArgumentParser()
ap.add_argument("iface")
ap.add_argument("out")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--amp", type=float, default=0.5)
ap.add_argument("--hz", type=float, default=6.0)
ap.add_argument("--hold", type=float, default=1.3)
a = ap.parse_args()
bus = can.Bus(channel=a.iface, interface="socketcan")
fb, rows = {}, []


def poll(dt):
    end = time.time() + dt
    while (left := end - time.time()) > 0:
        m = bus.recv(left)
        if m is not None and m.arbitration_id == 0x2A8:
            fb["w"] = struct.unpack(">ih", bytes(m.data[:6]))[0] / 1000


def hold(layer, target_mm, secs, label, phase=None):
    t0 = time.time()
    nxt = t0
    tr = []
    while (t := time.time() - t0) < secs:
        if phase is not None:                                   # force the sine phase at the instant of the step
            layer._phase = math.radians(phase) - 2 * math.pi * layer.hz * DT
            phase = None
        check(a.iface, 0x159)
        bus.send(can.Message(arbitration_id=0x159, is_extended_id=False, data=struct.pack(
            ">ihBB", int(round(layer.update(target_mm / 1000, DT) * 1e6)), 1000, 0x01, 0)))
        nxt += DT
        poll(max(0.0, nxt - time.time()))
        tr.append((t, fb["w"]))
        rows.append([f"{time.time():.4f}", label, f"{target_mm:.1f}", f"{fb['w']:.2f}"])
    return tr


poll(0.4)
if "w" not in fb:
    sys.exit(f"no gripper feedback on {a.iface}")
PH = list(range(0, 360, 30))
res = {(p, d): [] for p in PH for d in ("open", "close")}
rng = random.Random(4)
layer = GripperLayer(slew_mm_s=0, dither_mm=a.amp, dither_hz=a.hz)
layer.reset(fb["w"] / 1000)
for rep in range(a.reps):
    order = list(res)
    rng.shuffle(order)
    for ph, d in order:
        tgt, back = (70.0, 0.0) if d == "open" else (0.0, 70.0)
        hold(layer, back, a.hold, f"prep {d} ph{ph}")             # go to the far end first (random sine phase)
        m0 = fb["w"]
        tr = hold(layer, tgt, a.hold, f"{d} ph{ph} rep{rep}", phase=ph)
        sgn = 1 if tgt > m0 else -1
        st = next((t for t, m in tr if (m - m0) * sgn > 2.0), None)
        res[(ph, d)].append(st)
        print(f"rep {rep + 1}/{a.reps} {d:5s} phase {ph:3d} deg: start {'never' if st is None else f'{st:.2f} s'}", flush=True)
bus.shutdown()
print(f"\nstart-moving delay by sine phase at the step (median s of {a.reps}; 'never' = did not move within {a.hold} s)")
print("phase deg  " + " ".join(f"{p:5d}" for p in PH))
for d in ("open", "close"):
    cells = []
    for p in PH:
        v = [x for x in res[(p, d)] if x is not None]
        cells.append("never" if not v else f"{statistics.median(v):5.2f}")
    print(f"{d:9s}  " + " ".join(f"{c:>5s}" for c in cells))
allv = {d: [x for p in PH for x in res[(p, d)]] for d in ("open", "close")}
for d, v in allv.items():
    ok = [x for x in v if x is not None]
    print(f"{d}: {len(ok)}/{len(v)} moved; delay median {statistics.median(ok):.2f} s, worst {max(ok):.2f} s")
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t", "pattern", "target_mm", "measured_mm"])
    w.writerows(rows)
