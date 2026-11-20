#!/usr/bin/env python3
"""Drive the gripper through the REAL teleop controller (TeleopArm + the configured backend) with a synthetic trigger.

    python tools/gripper_demo.py left logs/gripper_demo.csv

The trigger is the only input (GRIP is never pressed, so the arm itself never engages).  Patterns: slow squeeze and
release, fast mashing, random jumps, full-range pulses.  Logs per tick: trigger, gripper target, command actually sent
(after the gripper_slew_mm_s ramp) and measured opening.  On a joint-locked arm the backend sends gripper frames only.
"""
import csv
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop import config as cfgmod  # noqa: E402
from piper_teleop.controller import HandSample, TeleopArm  # noqa: E402

arm_name, out = sys.argv[1], sys.argv[2]
raw = cfgmod.load()
section = raw["arms"][arm_name]
backend = cfgmod.make_backend(section, raw, False)
backend.connect()
if not backend.enable():
    sys.exit("enable failed: " + getattr(backend, "error", ""))
for n in getattr(backend, "notes", []):
    print(f"[{arm_name}] {n}")
arm = TeleopArm(cfgmod.arm_config(arm_name, raw, section), backend)
arm.start()

rng = random.Random(3)
jumps = [rng.choice([0.0, 1.0, rng.random()]) for _ in range(60)]
patterns = [                                   # (name, seconds, trigger(t) in 0..1; 1 = squeezed = closed)
    ("released (opens)", 2.5, lambda t: 0.0),
    ("slow squeeze + release", 6.0, lambda t: 0.5 - 0.5 * math.cos(2 * math.pi * t / 6)),
    ("fast mashing 0 <-> 1", 6.0, lambda t: 1.0 if (t % 0.8) < 0.4 else 0.0),
    ("random trigger jumps", 8.0, lambda t: jumps[int(t / 0.3) % 60]),
    ("half squeeze, hold", 3.0, lambda t: 0.5),
    ("full squeeze, hold", 3.0, lambda t: 1.0),
    ("release", 3.0, lambda t: 0.0),
]
rows = []
t_run = time.perf_counter()
seq = 0
pos = [0.2, 1.1, -0.3]
for name, secs, trig in patterns:
    t0 = time.perf_counter()
    nxt = t0
    while (t := time.perf_counter() - t0) < secs:
        now = time.perf_counter()
        seq += 1
        tr = float(trig(t))
        arm.step(HandSample(t_recv=now, seq=seq, pos=list(pos), squeeze=0.0, trigger=tr), now, 0.005)
        rows.append([f"{time.time():.4f}", name, f"{tr:.3f}", f"{arm.grip_target * 1000:.2f}",
                     f"{arm.grip_cmd * 1000:.2f}", f"{arm.grip_meas * 1000:.2f}", arm.state])
        nxt += 0.005
        while time.perf_counter() < nxt:
            time.sleep(0.0005)
    seg = [float(r[5]) for r in rows if r[1] == name]
    print(f"  {name:24s} measured {min(seg):5.1f} .. {max(seg):5.1f} mm", flush=True)
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t", "pattern", "trigger", "target_mm", "cmd_mm", "measured_mm", "arm_state"])
    w.writerows(rows)
backend.close()
print(f"{len(rows)} ticks -> {out}")
