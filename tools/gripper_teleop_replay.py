#!/usr/bin/env python3
"""Replay what the Quest teleop server sent to the gripper, with wild input patterns, and log target vs measured.

    python tools/gripper_teleop_replay.py can_left logs/can/replay.csv [--effort 1000] [--no-mode-frame]

Per 5 ms tick, like the teleop server: the 0x151 mode frame (CAN control, MOVE_J, 100 %) then the 0x159 gripper target
(effort 1000 = teleop default, status 0x01).  Joint frames (0x155-0x157) and joint enable are NOT sent: the left arm is
joint-locked.  The mode frame passes the lock only while all six joint motors are proven disabled (0x261-0x266) - the
run aborts the instant any joint motor reports enabled or any joint angle drifts.  Gripper-side patterns: hold at the
current opening (teleop's enable), slow/fast sweeps over 0-70 mm, square waves, random jumps, small dithers, ramps.
"""
import argparse
import csv
import math
import random
import struct
import sys
import time
from pathlib import Path

import can

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop.joint_lock import check  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("can")
ap.add_argument("out_csv")
ap.add_argument("--effort", type=int, default=1000)
ap.add_argument("--no-mode-frame", action="store_true")
ap.add_argument("--slew", type=float, default=0.0, help="rate-limit the commanded target to this many mm/s (0 = off)")
ap.add_argument("--only", nargs="*", help="run only patterns whose name starts with one of these words")
a = ap.parse_args()

bus = can.Bus(channel=a.can, interface="socketcan")
motor, angles, st = {}, {}, {}


def poll(dt):
    end = time.time() + dt
    while (left := end - time.time()) > 0:
        m = bus.recv(left)
        if m is None:
            continue
        cid, d = m.arbitration_id, bytes(m.data)
        if 0x261 <= cid <= 0x266:
            motor[cid - 0x260] = (time.time(), bool(d[5] & 0x40))
        elif 0x2A5 <= cid <= 0x2A7:
            x, y = struct.unpack(">ii", d)
            j = 1 + 2 * (cid - 0x2A5)
            angles[j], angles[j + 1] = x / 1000, y / 1000
        elif cid == 0x2A1:
            st["mode"] = d[0]
        elif cid == 0x2A8:
            w, e = struct.unpack(">ih", d[:6])
            st.update(w=w / 1000, e=e, gs=d[6])


def disabled():
    now = time.time()
    return len(motor) == 6 and all(now - t < 0.2 and not en for t, en in motor.values())


poll(0.6)
if not disabled():
    sys.exit(f"ABORT: joint motors not all disabled: {motor}")
ref = dict(angles)
cur = st["w"]
rng = random.Random(7)
rand_targets = [rng.uniform(0, 70) for _ in range(40)]

# (name, seconds, f(t) -> target mm)
patterns = [
    ("hold at current (teleop enable)", 2.0, lambda t: cur),
    ("slow sweep 0-70 mm, 6 s period", 9.0, lambda t: 35 - 35 * math.cos(2 * math.pi * t / 6)),
    ("fast sweep 10-60 mm, 1.5 s period", 6.0, lambda t: 35 - 25 * math.cos(2 * math.pi * t / 1.5)),
    ("square 5 <-> 65 mm every 1 s", 6.0, lambda t: 5 if int(t) % 2 == 0 else 65),
    ("random jumps every 0.4 s", 8.0, lambda t: rand_targets[int(t / 0.4) % 40]),
    ("dither +-2 mm around current", 4.0, lambda t: max(0.0, cur + 2 * math.sin(2 * math.pi * t * 2))),
    ("ramp 0 -> 70 mm over 6 s", 6.0, lambda t: min(70.0, t / 6 * 70)),
    ("trigger-like: release/squeeze pulses", 6.0, lambda t: 70 if (t % 1.2) < 0.6 else 0),
    ("end: target 10 mm", 2.0, lambda t: 10.0),
]
if a.only:
    patterns = [patterns[0]] + [p for p in patterns[1:] if any(p[0].startswith(w) for w in a.only)]
mode_frame = bytes([0x01, 0x01, 100, 0x00, 0, 0, 0, 0])
cmd = cur                                              # slew-limited command actually sent
rows, seg_stats = [], []
t_run = time.time()
try:
    for name, secs, f in patterns:
        t0 = time.time()
        meas = []
        nxt = t0
        while (t := time.time() - t0) < secs:
            if not disabled():
                raise SystemExit(f"ABORT: a joint motor reports enabled / stale status: {motor}")
            if max(abs(angles[j] - ref[j]) for j in ref) > 0.5:
                raise SystemExit(f"ABORT: joint angle drift: {angles}")
            raw = float(min(70.0, max(0.0, f(t))))
            if a.slew > 0:
                step = a.slew * 0.005
                cmd = cmd + max(-step, min(step, raw - cmd))
            else:
                cmd = raw
            tgt = cmd
            if not a.no_mode_frame:
                check(a.can, 0x151, data=mode_frame, joints_disabled=True)
                bus.send(can.Message(arbitration_id=0x151, data=mode_frame, is_extended_id=False))
            check(a.can, 0x159)
            bus.send(can.Message(arbitration_id=0x159, data=struct.pack(">ihBB", int(tgt * 1000), a.effort, 0x01, 0),
                                 is_extended_id=False))
            nxt += 0.005
            poll(max(0.0, nxt - time.time()))
            rows.append([f"{time.time():.4f}", name, f"{tgt:.2f}", f"{st['w']:.2f}", st["e"], st["gs"], st.get("mode")])
            meas.append(st["w"])
        seg_stats.append((name, min(meas), max(meas)))
        print(f"  {name:40s} measured {min(meas):6.2f} .. {max(meas):6.2f} mm   (moved {max(meas) - min(meas):5.2f} mm)",
              flush=True)
finally:
    bus.shutdown()
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "pattern", "target_mm", "measured_mm", "effort_fb", "gripper_status", "arm_mode"])
        w.writerows(rows)
    print(f"joint angle max drift {max(abs(angles[j] - ref[j]) for j in ref):.3f} deg | joints disabled at end: "
          f"{disabled()} | {len(rows)} ticks -> {a.out_csv}")
