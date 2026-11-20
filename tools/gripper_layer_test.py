#!/usr/bin/env python3
"""Score command-layer settings on the cases the gripper fails: step targets held for a while.

    python tools/gripper_layer_test.py can_left logs/gripper_layer_test.csv

Command = ramp(target, slew) + A * sin(2*pi*f*t).  For each setting the same hold sequence is run; per hold the score is
the mean |measured - target| over the hold's last 0.6 s ("reached" if < 3 mm).  The no-dither baseline runs first AND
last, because the gripper is not deterministic.  Raw 0x159 frames only, each checked against the joint lock.
"""
import csv
import math
import struct
import sys
import time
from pathlib import Path

import can

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop.joint_lock import check  # noqa: E402

import json
iface, out = sys.argv[1], sys.argv[2]
HOLDS = [40.0, 10.0, 60.0, 25.0, 65.0, 5.0]
if len(sys.argv) > 3:                                   # optional JSON: {"holds": [...], "settings": [[label, slew, amp, f], ...]}
    _o = json.loads(sys.argv[3])
    HOLDS = _o.get("holds", HOLDS)
HOLD_S = 2.0
SETTINGS = [  # (label, slew mm/s, amplitude mm, frequency Hz)
    ("baseline ramp only", 150, 0.0, 0.0),
    ("ramp + 1 mm @ 3 Hz", 150, 1.0, 3.0),
    ("ramp + 2 mm @ 3 Hz", 150, 2.0, 3.0),
    ("ramp + 3 mm @ 3 Hz", 150, 3.0, 3.0),
    ("ramp + 2 mm @ 1.5 Hz", 150, 2.0, 1.5),
    ("ramp + 2 mm @ 6 Hz", 150, 2.0, 6.0),
    ("no ramp + 2 mm @ 3 Hz", 0, 2.0, 3.0),
    ("baseline ramp only (again)", 150, 0.0, 0.0),
]
if len(sys.argv) > 3 and "settings" in _o:
    SETTINGS = [tuple(s) for s in _o["settings"]]
bus = can.Bus(channel=iface, interface="socketcan")
fb = {}


def poll(dt):
    end = time.time() + dt
    while (left := end - time.time()) > 0:
        m = bus.recv(left)
        if m is not None and m.arbitration_id == 0x2A8:
            w, _ = struct.unpack(">ih", bytes(m.data[:6]))
            fb["w"] = w / 1000


poll(0.3)
rows, summary = [], []
for label, slew, amp, freq in SETTINGS:
    ramp = fb["w"]
    t_set = time.time()
    errs, jig, settle = [], [], []
    for tgt in HOLDS:
        t0 = time.time()
        nxt = t0
        tail = []
        t_in = None
        while (t := time.time() - t0) < HOLD_S:
            ramp = tgt if slew <= 0 else ramp + max(-slew * 0.005, min(slew * 0.005, tgt - ramp))
            cmd = min(70.0, max(0.0, ramp + amp * math.sin(2 * math.pi * freq * (time.time() - t_set))))
            check(iface, 0x159)
            bus.send(can.Message(arbitration_id=0x159, data=struct.pack(">ihBB", int(round(cmd * 1000)), 1000, 0x01, 0),
                                 is_extended_id=False))
            nxt += 0.005
            poll(max(0.0, nxt - time.time()))
            rows.append([f"{time.time():.4f}", label, f"{tgt:.1f}", f"{cmd:.3f}", f"{fb['w']:.2f}"])
            if t_in is None and abs(fb["w"] - tgt) < 3.0:
                t_in = t                                  # response time: first time within 3 mm of the new target
            if t > HOLD_S - 0.6:
                tail.append(fb["w"])
        errs.append(sum(abs(x - tgt) for x in tail) / len(tail))
        jig.append(max(tail) - min(tail))                # peak-to-peak finger motion while "holding"
        settle.append(t_in if t_in is not None else HOLD_S)
    reached = sum(e < 3.0 for e in errs)
    summary.append((label, reached, errs))
    print(f"  {label:28s} reached {reached}/{len(HOLDS)}   mean error {sum(errs) / len(errs):5.1f} mm   "
          f"jiggle p-p {sum(jig) / len(jig):4.2f} mm   response {sum(settle) / len(settle):4.2f} s   "
          f"per hold: {' '.join('%4.1f' % e for e in errs)}", flush=True)
bus.shutdown()
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t", "pattern", "target_mm", "cmd_mm", "measured_mm"])
    w.writerows(rows)
