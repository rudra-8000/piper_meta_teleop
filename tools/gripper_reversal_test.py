#!/usr/bin/env python3
"""Does a QUICK re-open (soon after a close command) stall the gripper?  (gripper frames only)

    python tools/gripper_reversal_test.py can_right logs/gripper_reversal_right.csv [--reps 3] [--dwells 0.15,0.3,0.6,1.0]

Trial: open (70 mm) 1.0 s -> close (0 mm) for DWELL s -> open (70 mm) 1.5 s.  Measured on the final open: time until the
opening has moved 2 mm toward 70 (start) and until it is within 3 mm (arrive).  Settings x dwells are interleaved in a
random order every rep.  Every 0x159 frame is checked against config/joint_lock.yaml.
"""
import argparse
import csv
import itertools
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
    "step + 1mm@6Hz": (0, 1.0, 6),
    "step + 2mm@6Hz": (0, 2.0, 6),
    "ramp 600 mm/s + 0.5mm@6Hz": (600, 0.5, 6),
}
ap = argparse.ArgumentParser()
ap.add_argument("iface")
ap.add_argument("out")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--dwells", default="0.15,0.3,0.6,1.0")
ap.add_argument("--settings", help='JSON {"name": [slew, amp, hz]}')
a = ap.parse_args()
if a.settings:
    SETTINGS = {k: tuple(v) for k, v in json.loads(a.settings).items()}
DWELLS = [float(x) for x in a.dwells.split(",")]
bus = can.Bus(channel=a.iface, interface="socketcan")
fb, rows = {}, []


def poll(dt):
    end = time.time() + dt
    while (left := end - time.time()) > 0:
        m = bus.recv(left)
        if m is not None and m.arbitration_id == 0x2A8:
            fb["w"] = struct.unpack(">ih", bytes(m.data[:6]))[0] / 1000


def hold(layer, target_mm, secs, label):
    t0 = time.time()
    nxt = t0
    tr = []
    while (t := time.time() - t0) < secs:
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
res = {k: [] for k in itertools.product(SETTINGS, DWELLS)}
rng = random.Random(9)
for rep in range(a.reps):
    combos = list(res)
    rng.shuffle(combos)
    for name, dwell in combos:
        slew, amp, hz = SETTINGS[name]
        layer = GripperLayer(slew_mm_s=slew, dither_mm=amp, dither_hz=hz)
        layer.reset(fb["w"] / 1000)
        lab = f"{name} dwell{dwell} rep{rep}"
        hold(layer, 70.0, 1.0, lab + " open1")
        hold(layer, 0.0, dwell, lab + " close")
        m0 = fb["w"]
        tr = hold(layer, 70.0, 1.5, lab + " reopen")
        st = next((t for t, m in tr if m - m0 > 2.0), None)
        ar = next((t for t, m in tr if abs(m - 70) < 3.0), None)
        res[(name, dwell)].append((st, ar))
        print(f"rep {rep + 1}/{a.reps} {name:28s} dwell {dwell:4.2f}s  closed to {m0:4.1f} mm  reopen start "
              f"{'never' if st is None else f'{st:.2f}'}  arrive {'never' if ar is None else f'{ar:.2f}'}", flush=True)
bus.shutdown()
f = lambda v: f"{statistics.median(v):4.2f}/{max(v):4.2f}" if v else "never"
print(f"\nre-open delay: median/worst seconds  (start moving | arrive within 3 mm);  'x/n' = never arrived")
print(f"{'setting':28s} " + " ".join(f"{'dwell ' + str(d) + ' s':>26s}" for d in DWELLS))
for name in SETTINGS:
    cells = []
    for d in DWELLS:
        r = res[(name, d)]
        s = [x[0] for x in r if x[0] is not None]
        ar = [x[1] for x in r if x[1] is not None]
        cells.append(f"{f(s)} | {f(ar)}  {len(r) - len(ar)}/{len(r)}")
    print(f"{name:28s} " + " ".join(f"{c:>26s}" for c in cells))
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t", "pattern", "target_mm", "measured_mm"])
    w.writerows(rows)
