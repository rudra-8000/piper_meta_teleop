#!/usr/bin/env python3
"""From-scratch gripper control: bare python-can, no AgileX library in the path.

    python tools/gripper_raw.py can_left 0 20 0 --effort 5000      # targets in mm, 2 s each

Sends ONLY 0x159 (gripper control): int32 BE width in um, int16 BE effort (0.001 N*m), status 0x01 (enable, width
mode), byte 7 = 0.  Decodes 0x2A8 (gripper feedback) itself: int32 BE width um, int16 BE effort, status byte
(bit6 = driver enabled, bits0-5 = faults), mode byte.  Checks config/joint_lock.yaml before every frame.
"""
import argparse
import struct
import sys
import time
from pathlib import Path

import can

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop.joint_lock import check  # noqa: E402

GRIP_CTRL, GRIP_FB = 0x159, 0x2A8


def decode_fb(data):
    width_um, effort = struct.unpack(">ih", bytes(data[:6]))
    return width_um / 1000.0, effort, data[6], data[7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("can")
    ap.add_argument("targets_mm", nargs="+", type=float)
    ap.add_argument("--effort", type=int, default=1000)
    ap.add_argument("--hold", type=float, default=2.0)
    a = ap.parse_args()
    bus = can.Bus(channel=a.can, interface="socketcan")
    fb = None

    def poll(dt):
        nonlocal fb
        end = time.monotonic() + dt
        while (left := end - time.monotonic()) > 0:
            m = bus.recv(left)
            if m is not None and m.arbitration_id == GRIP_FB:
                fb = decode_fb(m.data)

    poll(0.3)
    print("start:", fb)
    for tgt in a.targets_mm:
        frame = struct.pack(">ihBB", int(tgt * 1000), a.effort, 0x01, 0x00)
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.hold:
            check(a.can, GRIP_CTRL)
            bus.send(can.Message(arbitration_id=GRIP_CTRL, data=frame, is_extended_id=False))
            poll(0.01)
        w, e, st, mode = fb
        print(f"  target {tgt:5.1f} mm -> measured {w:6.2f} mm  effort {e:5d}  status 0x{st:02X} "
              f"(enabled={bool(st & 0x40)}, faults=0x{st & 0x3F:02X})  mode {mode}")
    bus.shutdown()


if __name__ == "__main__":
    main()
