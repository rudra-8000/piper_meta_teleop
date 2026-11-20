#!/usr/bin/env python3
"""Planner-style control: drive the tool tip through waypoints with tool-pose targets (IK), no policy.

    .venv/bin/python examples/cartesian_demo.py --sim             # simulated arm
    .venv/bin/python examples/cartesian_demo.py                   # the real left arm: it moves! clear the area

Draws a 6 cm square in the air, 10 cm above the rest pose.  Position only (rotation='free'): holding the rest orientation
fixed forces strange wrist configurations for sideways targets this close to the base, and send_tcp_pose refuses those.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop.robot import IKError, PiperRobot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sim", action="store_true")
ap.add_argument("--arm")
a = ap.parse_args()

with PiperRobot(a.arm, sim=a.sim, cameras=False) as robot:
    robot.go_rest()
    _, p0 = robot.tcp_pose()
    start = p0 + np.array([0.0, 0.0, 0.10])
    corners = [start + np.array(d) for d in ([0, 0, 0], [0.06, 0, 0], [0.06, 0.06, 0], [0, 0.06, 0], [0, 0, 0])]
    for i, p in enumerate(corners):
        try:
            robot.send_tcp_pose(p, "free", speed=0.4)
        except (IKError, ValueError) as e:                       # unreachable / outside the workspace: nothing was sent
            print(f"waypoint {i} skipped: {e}")
            continue
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0 and np.linalg.norm(robot.tcp_pose()[1] - p) > 0.004:
            time.sleep(0.02)
        print(f"waypoint {i}: tool tip {np.round(robot.tcp_pose()[1], 3).tolist()} m (wanted {np.round(p, 3).tolist()})")
