#!/usr/bin/env python3
"""Set a Piper's linkage role (stored in the arm).  See guide "Mode A / Mode B".

    python tools/set_role.py can_leader leader      # 0xFA: teaching/input arm - broadcasts its pose, free to drag
    python tools/set_role.py can_follower follower  # 0xFC: motion/output arm (the default; also use to UNDO leader)

A leader sends no joint feedback and ignores commands, so *always* run `follower` on an arm
before using it in Quest teleop.  Send once, then power-cycle the arm if it does not take effect.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop.arm import _make_interface  # noqa: E402

CODES = {"leader": 0xFA, "follower": 0xFC}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("can")
ap.add_argument("role", choices=CODES)
a = ap.parse_args()
p = _make_interface(a.can, 1, True)
p.ConnectPort()
time.sleep(0.2)
for _ in range(5):                       # repeat: CAN frames are cheap, a dropped one is not
    p.MasterSlaveConfig(CODES[a.role], 0, 0, 0)
    time.sleep(0.05)
print(f"sent MasterSlaveConfig({CODES[a.role]:#04x}) to {a.can}: arm should now act as {a.role.upper()}.")
print("Power-cycle the arm if hw_check.py still disagrees.")
