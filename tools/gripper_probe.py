#!/usr/bin/env python3
"""Gripper probe: does the gripper accept a move right after power-up, and then latch?

    python tools/gripper_probe.py can_left            # run IMMEDIATELY after power-cycling that arm

Gripper only (no arm joint is commanded).  Keep fingers out of the gripper.  It sends raw 0x159 frames exactly as
AgileX's ArmRobotUA encodes them (int32 width um, int16 effort, status 0x01, zero 0x00) - effort up to 5000, the SDK
maximum - and prints the measured opening after each target, so a "moves once, then ignores everything" pattern is visible.
Finding so far (left arm): after `params --range 100` the gripper moved 0 -> 9.1 mm for a 10 mm target at effort 5000,
then ignored every later target (0-40 mm, effort 1000-30000, clear-error, disable/enable, parameter re-writes).
"""
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop import gripper as G  # noqa: E402
from piper_teleop.arm import PiperArm  # noqa: E402

can = sys.argv[1] if len(sys.argv) > 1 else "can_left"
a = PiperArm(can)
a.connect()
time.sleep(0.6)
p = a._p
mm = lambda: G.status(p)["value_m"] * 1000  # noqa: E731
print(f"{can}: start {mm():.1f} mm  status {G.status(p)}")
input("Fingers clear of the gripper? Enter to start... ")
for target, effort in ((10, 5000), (25, 5000), (5, 5000), (30, 5000), (0, 5000)):
    v0 = mm()
    t0 = time.time()
    while time.time() - t0 < 2.0:
        p._arm_can.SendCanMessage(0x159, struct.pack(">ihBB", target * 1000, effort, 0x01, 0))
        time.sleep(0.01)
    v = mm()
    print(f"  target {target:3d} mm  effort {effort}: {v0:5.1f} -> {v:5.1f} mm   {'MOVED' if abs(v - v0) > 1 else 'no motion'}")
