#!/usr/bin/env python3
"""Gripper test battery: every known command variant, in sequence, with a timestamped step log.

    python tools/gripper_battery.py can_left logs/can/battery_steps.json

Bare python-can; sends ONLY 0x159 gripper frames, each checked against config/joint_lock.yaml.  Run tools/candump.py
and tools/xtion_snap.py alongside it to get the CAN timeline and video for the same period.  Keep fingers clear.
"""
import json
import struct
import sys
import time
from pathlib import Path

import can

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop.joint_lock import check  # noqa: E402

iface, out = sys.argv[1], sys.argv[2]
bus = can.Bus(channel=iface, interface="socketcan")
fb = {}


def poll(dt):
    end = time.time() + dt
    while (left := end - time.time()) > 0:
        m = bus.recv(left)
        if m is not None and m.arbitration_id == 0x2A8:
            w, e = struct.unpack(">ih", bytes(m.data[:6]))
            fb.update(width_mm=w / 1000, effort=e, status=m.data[6], mode=m.data[7])


def frame(val, effort, code):
    check(iface, 0x159)
    bus.send(can.Message(arbitration_id=0x159, data=struct.pack(">ihBB", int(val), int(effort), code, 0), is_extended_id=False))


def stream(val, effort, code, secs):
    end = time.time() + secs
    while time.time() < end:
        frame(val, effort, code)
        poll(0.01)


steps = []


def step(name, fn):
    poll(0.05)
    before = dict(fb)
    t0 = time.time()
    fn()
    t1 = time.time()
    poll(0.1)
    steps.append({"name": name, "t0": t0, "t1": t1, "before": before, "after": dict(fb)})
    a = fb
    print(f"{t0:.2f}  {name:44s} {before.get('width_mm', float('nan')):6.2f} -> {a['width_mm']:6.2f} mm  "
          f"effort {a['effort']:5d}  status 0x{a['status']:02X}  mode {a['mode']}", flush=True)


poll(0.3)
cur = max(0, int(fb.get("width_mm", 0) * 1000))
step("baseline (nothing sent)", lambda: poll(1.5))
step("enable 0x01 at current opening, eff 1000", lambda: stream(cur, 1000, 0x01, 1.5))
step("width 10 mm, eff 1000", lambda: stream(10000, 1000, 0x01, 2.0))
step("width 30 mm, eff 1000", lambda: stream(30000, 1000, 0x01, 2.0))
step("clear error 0x03 (enable+clear) 0.5 s", lambda: stream(cur, 1000, 0x03, 0.5))
step("width 20 mm, eff 3000", lambda: stream(20000, 3000, 0x01, 2.0))
step("disable 0x00 1 s", lambda: stream(0, 1000, 0x00, 1.0))
step("fresh enable: width 20 mm, eff 3000", lambda: stream(20000, 3000, 0x01, 2.0))
step("width 40 mm, eff 5000", lambda: stream(40000, 5000, 0x01, 2.0))
step("width 0 mm, eff 5000", lambda: stream(0, 5000, 0x01, 2.0))
step("angle mode: 0x07 enable+clear (angle)", lambda: stream(0, 1000, 0x07, 0.5))
step("angle 30 deg, eff 3000 (0x05)", lambda: stream(30000, 3000, 0x05, 2.0))
step("back to width mode 0x01 at 0", lambda: stream(0, 1000, 0x01, 1.0))
step("single frame 20 mm eff 3000 (ArmRobotUA-style)", lambda: (frame(20000, 3000, 0x01), poll(2.0)))
step("final width 0, eff 1000", lambda: stream(0, 1000, 0x01, 1.5))
bus.shutdown()
json.dump(steps, open(out, "w"), indent=1)
