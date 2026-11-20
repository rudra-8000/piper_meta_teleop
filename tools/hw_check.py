#!/usr/bin/env python3
"""Step-by-step hardware check for one or more Piper arms.  Read-only unless you ask.

    python tools/hw_check.py can_left can_right            # CAN + feedback + firmware + status
    python tools/hw_check.py can_left --enable             # also enable the motors (no motion)
    python tools/hw_check.py can_left --enable --nudge     # also move joint 1 by +3 deg and back

Exit code is non-zero if any arm fails a check.  Run this before anything else.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from piper_teleop.arm import PiperArm, _make_interface, is_folded, status_text  # noqa: E402


def ok(m): print(f"  [ OK ] {m}")
def bad(m): print(f"  [FAIL] {m}"); return False
def info(m): print(f"  [info] {m}")


def check_can(name) -> bool:
    r = subprocess.run(["ip", "-details", "link", "show", name], capture_output=True, text=True)
    if r.returncode != 0:
        return bad(f"interface '{name}' does not exist. Plug in the USB-CAN adapter and run scripts/can_setup.sh")
    up = "UP" in r.stdout.split("\n")[0] and "state DOWN" not in r.stdout
    br = re.search(r"bitrate (\d+)", r.stdout)
    good = True
    if not up:
        good = bad(f"{name} is down: sudo ip link set {name} up type can bitrate 1000000")
    else:
        ok(f"{name} is up")
    if not br or int(br.group(1)) != 1_000_000:
        good = bad(f"bitrate is {br.group(1) if br else 'unknown'}, must be 1000000")
    else:
        ok("bitrate 1 Mbit/s")
    return good


def check_arm(name, enable, nudge, release=False) -> bool:
    print(f"\n=== {name}")
    if not check_can(name):
        return False
    p = _make_interface(name, 1, True)
    p.ConnectPort()
    time.sleep(0.4)
    j = p.GetArmJointMsgs()
    hz = getattr(j, "Hz", 0)
    if hz < 10:
        return bad("no joint feedback frames. Check: arm powered (24 V, status light on)? aviation plug seated "
                   "(red dots)? CAN_H/CAN_L not swapped? Is this arm set as LEADER (0xFA)? A leader sends no "
                   "feedback - run tools/set_role.py follower")
    ok(f"joint feedback at {hz:.0f} Hz")
    s = j.joint_state
    q = np.array([s.joint_1, s.joint_2, s.joint_3, s.joint_4, s.joint_5, s.joint_6]) / 1000.0
    info("joints (deg): " + ", ".join(f"{v:7.2f}" for v in q))
    try:
        info(f"firmware: {p.GetPiperFirmwareVersion()}")
    except Exception as e:
        info(f"firmware: unreadable ({e})")
    info(f"CAN rx rate: {p.GetCanFps()} fps (about 3000 expected for one arm)")
    info("arm says: " + status_text(p, 6))
    g = p.GetArmGripperMsgs().gripper_state
    info(f"gripper feedback: {g.grippers_angle / 1000:.1f} mm, effort {g.grippers_effort}")
    if g.grippers_angle == 0 and q is not None:
        info("gripper reads exactly 0: normal if closed. If it NEVER changes, run tools/gripper_setup.py")
    info(f"motors enabled: {p.GetArmEnableStatus()}")
    good = True
    if enable:
        arm = PiperArm(name, iface=p)
        if not arm.enable(timeout=8):
            return bad("enable failed: " + (arm.error or "unknown"))
        for note in arm.notes:
            info("enable: " + note)
        info("after enable, arm says: " + status_text(p, 6))
        ok(f"motors enabled: {p.GetArmEnableStatus()}")
        if nudge:
            input("\n>>> NUDGE TEST: joint 1 will rotate +3 deg and return, slowly. Clear the area, "
                  "keep a hand near the power switch. Enter to go... ")
            p.MotionCtrl_2(0x01, 0x01, 20, 0x00)
            q0 = arm.read().q
            for k in range(0, 101):
                tgt = q0.copy(); tgt[0] += np.radians(3) * min(k, 100) / 100
                arm.command(tgt, arm.read().gripper); time.sleep(0.02)
            time.sleep(0.5)
            moved = np.degrees(arm.read().q[0] - q0[0])
            for k in range(100, -1, -1):
                tgt = q0.copy(); tgt[0] += np.radians(3) * k / 100
                arm.command(tgt, arm.read().gripper); time.sleep(0.02)
            time.sleep(0.5)
            (ok if 1.5 < moved < 4.5 else bad)(f"joint 1 moved {moved:+.2f} deg (expected about +3)")
            good = 1.5 < moved < 4.5
            if not good:
                info("arm says now: " + status_text(p, 6))
                info("The arm accepted enable but ignored joint commands. Paste the two 'arm says' lines above. "
                     "Things to check: teach-button light OFF; Control Mode must read CAN (0x1); "
                     "Arm Status must be NORMAL; try power-cycling the arm with the light off.")
        if release and is_folded(arm.read().q):
            arm.disable()
            info("motors RELEASED on request (the wrist will droop)")
        else:
            info("motors left ENABLED and holding - nothing was powered down. "
                 "(--release disables them, only at the folded pose, if you really want that)")
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("can", nargs="+")
    ap.add_argument("--enable", action="store_true")
    ap.add_argument("--nudge", action="store_true")
    ap.add_argument("--release", action="store_true", help="disable the motors at the end (default: leave holding)")
    a = ap.parse_args()
    results = [check_arm(c, a.enable or a.nudge, a.nudge, a.release) for c in a.can]
    print("\nRESULT:", "all good" if all(results) else "problems found (see FAIL lines)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
