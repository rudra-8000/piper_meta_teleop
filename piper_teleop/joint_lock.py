"""Time-limited joint lock: while active for an arm, ONLY gripper and read-only query frames may be sent to it.

Configured in config/joint_lock.yaml:

    can_left:
      until: "2026-10-28 13:00"      # local time
      reason: "gripper-only testing"

Enforced at the lowest layer every program in this project shares - the function that puts a frame on the CAN bus
(piper_sdk's SendCanMessage and pyAgxArm's comm.send are both wrapped).  A blocked frame raises JointLockError, so a
locked arm cannot be moved by the teleop server, leader/follower, go_rest, hw_check, or an ad-hoc script that goes
through piper_teleop.  It does NOT cover programs outside this project (AgileX ArmRobotUA, a bare piper_sdk script,
your own pyAgxArm venv).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import yaml

LOCK_FILE = Path(__file__).resolve().parent.parent / "config" / "joint_lock.yaml"

# Frames that cannot move a joint: gripper control, gripper/pendant parameters, read-only queries.
ALLOWED_WHILE_LOCKED = {
    0x159,   # gripper control
    0x47D,   # gripper / teach-pendant parameters
    0x477,   # parameter ENQUIRY (read request)
    0x472,   # motor angle/speed limit query
    0x4AF,   # firmware version request
}


class JointLockError(RuntimeError):
    pass


def _load(path=None) -> dict:
    p = Path(path or LOCK_FILE)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def lock_until(can_name: str, now: _dt.datetime | None = None, path=None) -> _dt.datetime | None:
    """End time of an ACTIVE lock on this CAN interface, or None."""
    entry = _load(path).get(can_name)
    if not entry:
        return None
    until = _dt.datetime.fromisoformat(str(entry["until"]))
    return until if (now or _dt.datetime.now()) < until else None


MODE_FRAME = 0x151


def check(can_name: str, can_id: int, now=None, path=None, data=None, joints_disabled=None) -> None:
    """Raise JointLockError unless the frame may be sent.

    Optional narrow exception (lock entry `allow_mode_frame_when_joints_disabled: true`): a 0x151 mode frame whose
    control-mode byte is STANDBY (0x00) or CAN (0x01) passes, but ONLY when the caller passes ``joints_disabled=True``
    proven from the arm's own motor-status frames just before sending.  Unpowered joint motors cannot move a joint.
    """
    until = lock_until(can_name, now, path)
    if until is None or can_id in ALLOWED_WHILE_LOCKED:
        return
    entry = _load(path).get(can_name) or {}
    if (can_id == MODE_FRAME and entry.get("allow_mode_frame_when_joints_disabled") and joints_disabled is True
            and data is not None and len(data) >= 1 and data[0] in (0x00, 0x01)):
        return
    raise JointLockError(f"{can_name} is JOINT-LOCKED until {until:%Y-%m-%d %H:%M} (config/joint_lock.yaml): "
                         f"frame 0x{can_id:03X} refused - only the gripper may be commanded")


def guard_piper_sdk(iface, can_name: str):
    """Wrap a piper_sdk interface's CAN sender."""
    can = getattr(iface, "_arm_can", None)
    if can is None or getattr(can, "_joint_lock_wrapped", False):
        return iface
    raw = can.SendCanMessage

    def send(arbitration_id, data, *a, **kw):
        check(can_name, int(arbitration_id))
        return raw(arbitration_id, data, *a, **kw)

    can.SendCanMessage = send
    can._joint_lock_wrapped = True
    return iface


def guard_pyagxarm(robot, can_name: str):
    """Wrap a connected pyAgxArm robot's CAN comm."""
    ctx = getattr(robot, "_ctx", None)
    comm = ctx.get_comm() if ctx is not None else None
    if comm is None:
        if lock_until(can_name):
            raise JointLockError(f"{can_name} is joint-locked but its CAN sender could not be guarded - refusing to run")
        return robot
    if getattr(comm, "_joint_lock_wrapped", False):
        return robot
    raw = comm.send

    def send(message, *a, **kw):
        check(can_name, int(message.arbitration_id))
        return raw(message, *a, **kw)

    comm.send = send
    comm._joint_lock_wrapped = True
    return robot
