#!/usr/bin/env python3
"""Does the gripper only follow position targets while the arm controller is in CAN control mode?

    python tools/gripper_mode_test.py can_left logs/can/mode_test_steps.json

1. Proves from the arm's own motor-status frames (0x261-0x266, driver-enable bit 6 of byte 5) that all six joint
   motors are DISABLED; aborts otherwise.
2. Sends the 0x151 mode frame STANDBY -> CAN control (MOVE_J, 10 %), passing that proof to the joint lock (the only way
   the lock's mode-frame exception can be used), then checks the arm reports CAN mode and that no joint angle changed.
3. Gripper sequence (0x159 only), aborting the instant any joint motor reports enabled or any joint angle drifts.
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
motor_t = {}            # joint index -> (time, enabled)
angles = {}             # joint index -> deg
st = {}                 # latest arm status + gripper


def poll(dt):
    end = time.time() + dt
    while (left := end - time.time()) > 0:
        m = bus.recv(left)
        if m is None:
            continue
        cid, d = m.arbitration_id, bytes(m.data)
        if 0x261 <= cid <= 0x266:
            motor_t[cid - 0x260] = (time.time(), bool(d[5] & 0x40))
        elif 0x2A5 <= cid <= 0x2A7:
            a, b = struct.unpack(">ii", d)
            j = 1 + 2 * (cid - 0x2A5)
            angles[j], angles[j + 1] = a / 1000, b / 1000
        elif cid == 0x2A1:
            st["ctrl_mode"], st["arm_status"] = d[0], d[1]
        elif cid == 0x2A8:
            w, e = struct.unpack(">ih", d[:6])
            st.update(width_mm=w / 1000, effort=e, gstatus=d[6], gmode=d[7])


def joints_disabled_now():
    now = time.time()
    return len(motor_t) == 6 and all(now - t < 0.2 and not en for t, en in motor_t.values())


class Abort(Exception):
    pass


ref_angles = {}


def guard():
    if not joints_disabled_now():
        raise Abort(f"a joint motor reports ENABLED (or status is stale): {motor_t}")
    drift = max(abs(angles[j] - ref_angles[j]) for j in ref_angles)
    if drift > 0.5:
        raise Abort(f"a joint angle drifted {drift:.2f} deg: {angles}")


def send(cid, data, joints_disabled=None):
    check(iface, cid, data=data, joints_disabled=joints_disabled)
    bus.send(can.Message(arbitration_id=cid, data=data, is_extended_id=False))


steps = []


def grip_step(name, mm, secs, eff=1000):
    before, t0 = st["width_mm"], time.time()
    frame = struct.pack(">ihBB", int(mm * 1000), eff, 0x01, 0)
    trace = []
    while time.time() - t0 < secs:
        guard()
        send(0x159, frame)
        poll(0.01)
        if not trace or time.time() - trace[-1][0] > 0.5:
            trace.append((time.time(), st["width_mm"]))
    steps.append(dict(name=name, t0=t0, t1=time.time(), target=mm, before=before, after=st["width_mm"],
                      ctrl_mode=st["ctrl_mode"]))
    print(f"{t0:.2f} {name:28s} target {mm:5.1f}: {before:6.2f} -> {st['width_mm']:6.2f} mm  "
          f"[{' '.join('%.1f' % w for _, w in trace)}]  gripper st 0x{st['gstatus']:02X}  arm mode {st['ctrl_mode']}", flush=True)
    return st["width_mm"]


try:
    poll(0.6)
    if not joints_disabled_now():
        raise Abort(f"joint motors not all disabled - refusing to send the mode frame: {motor_t}")
    ref_angles = dict(angles)
    print(f"joints disabled: OK | angles {[round(ref_angles[j], 2) for j in sorted(ref_angles)]} | arm mode "
          f"{st['ctrl_mode']} | gripper {st['width_mm']:.2f} mm st 0x{st['gstatus']:02X}")
    mode_frame = bytes([0x01, 0x01, 10, 0x00, 0, 0, 0, 0])      # CAN control, MOVE_J, 10 %, position-velocity
    for _ in range(5):
        poll(0.03)
        guard()
        send(0x151, mode_frame, joints_disabled=joints_disabled_now())
    poll(0.5)
    guard()
    print(f"after mode frame: arm mode {st['ctrl_mode']} (1 = CAN control), joints still disabled, "
          f"max angle drift {max(abs(angles[j] - ref_angles[j]) for j in ref_angles):.3f} deg")
    cur = st["width_mm"]
    grip_step("hold at current", cur, 1.5)
    w = grip_step("close to 6 mm", 6.0, 2.5)
    if abs(w - cur) > 1.0:
        grip_step("close to 3 mm", 3.0, 2.5)
        grip_step("open to 12 mm", 12.0, 2.5)
    else:
        grip_step("open to 15 mm", 15.0, 2.5)
        grip_step("close to 0 mm", 0.0, 2.5)
    print(f"end: joints disabled {joints_disabled_now()}, max angle drift "
          f"{max(abs(angles[j] - ref_angles[j]) for j in ref_angles):.3f} deg")
except Abort as e:
    print("ABORTED:", e)
finally:
    bus.shutdown()
    json.dump(steps, open(out, "w"), indent=1)
