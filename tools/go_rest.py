#!/usr/bin/env python3
"""Move Piper arms slowly to the folded rest (zero) pose.  Never touches the gripper.

    python tools/go_rest.py can_left can_right --plan        # read-only: show pose, mode, planned move
    python tools/go_rest.py can_left can_right               # do it (0.3 rad/s); arms STAY ENABLED and holding
    python tools/go_rest.py can_left --release               # ...and then deliberately cut motor power (wrist droops)

Safety: refuses to switch control mode unless the arm is already folded (J2, J3 near 0), moves all
joints together at a capped speed, never touches the gripper, and NEVER disables the motors unless you
pass --release (disabling lets the wrist fall by 20-30 degrees).
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from piper_teleop.arm import ArmBackend, PiperArm, SimArm, is_folded, status_text  # noqa: E402
from piper_teleop.kinematics import JOINT_HI, JOINT_LO  # noqa: E402


def fmt(q): return "[" + ", ".join(f"{math.degrees(v):7.2f}" for v in q) + "]"


def move_to(backend: ArmBackend, goal, speed=0.3, rate=100.0, settle_tol=math.radians(1.0), timeout=12.0):
    """Linear joint-space move with all joints arriving together.  Returns the final measured joints."""
    goal = np.clip(np.asarray(goal, float), JOINT_LO, JOINT_HI)
    q = backend.read().q.copy()
    dt = 1.0 / rate
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        d = goal - q
        m = float(np.max(np.abs(d)))
        if m > 1e-9:
            q = q + d * min(1.0, speed * dt / m)
        backend.command(q, None)                 # None: do not touch the gripper
        time.sleep(dt)
        if m < 1e-4 and float(np.max(np.abs(goal - backend.read().q))) < settle_tol:
            break
    time.sleep(0.5)
    return backend.read().q


def run_arm(name, backend, plan, speed, release, goal) -> bool:
    print(f"\n=== {name}")
    backend.connect()
    time.sleep(0.4)
    st = backend.read()
    if not st.fresh:
        print("  [FAIL] no joint feedback"); return False
    print(f"  now      : {fmt(st.q)} deg")
    if hasattr(backend, "_p"):
        print(f"  arm says : {status_text(backend._p, 6)}")
    print(f"  rest goal: {fmt(goal)} deg   (largest joint move: {math.degrees(np.max(np.abs(goal - st.q))):.1f} deg, "
          f"~{np.max(np.abs(goal - st.q)) / speed:.1f} s at {speed} rad/s)")
    print(f"  folded now: {is_folded(st.q)}  (J2/J3 within 6 deg of zero)")
    if plan:
        print("  (plan only - nothing was sent)"); return True
    if not backend.enable(gripper=False):
        print("  [FAIL] enable failed: " + getattr(backend, "error", "")); return False
    for note in getattr(backend, "notes", []):
        print(f"  enable   : {note}")
    final = move_to(backend, goal, speed)
    err = np.max(np.abs(final - goal))
    print(f"  reached  : {fmt(final)} deg   (worst joint off goal: {math.degrees(err):.2f} deg)")
    ok = err < math.radians(2.0)
    print("  [ OK ] at rest pose" if ok else "  [WARN] did not settle on the goal")
    if hasattr(backend, "_p"):
        print(f"  arm says : {status_text(backend._p, 6)}")
    if release and ok and is_folded(final):
        backend.disable()
        print("  motors RELEASED on request (the wrist will droop)")
    else:
        print("  motors left ENABLED and holding the rest pose (nothing was powered down)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("can", nargs="+")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--speed", type=float, default=0.3, help="rad/s, max joint speed (default 0.3)")
    ap.add_argument("--release", action="store_true", help="deliberately disable the motors afterwards (wrist droops)")
    ap.add_argument("--sim", action="store_true")
    a = ap.parse_args()
    goal = np.zeros(6)                            # the manual's zero point == folded rest pose
    results = []
    for name in a.can:
        be = SimArm(name, q0=np.radians([5, 3, -2, -5, 23, 12])) if a.sim else PiperArm(name)
        results.append(run_arm(name, be, a.plan, min(a.speed, 0.6), a.release, goal))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
