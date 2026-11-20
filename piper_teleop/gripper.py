"""Gripper helpers: status, zeroing and a follow-test, all verified through the gripper's OWN firmware feedback.

Real-hardware findings (two Pipers, firmware S-V1.8-9, V3 interface):
* The arm ACKNOWLEDGES a set-zero frame (0x476, instruction index 0x75, is_set_zero_successfully == 1) every time.  That
  ACK is the success signal AgileX's own pyAgxArm library uses.  The `homing_status` bit in the gripper feedback stayed
  False throughout and is NOT a reliable "zeroed" indicator - an earlier version of this tool trusted it and was wrong.
* The gripper nevertheless holds torque when enabled but ignores new position targets (no motion, ~0 effort) with
  byte-identical frames to AgileX's current move_gripper_m().  Cause not found; see the guide.
"""
from __future__ import annotations

import time

ERROR_FLAGS = ("voltage_too_low", "motor_overheating", "driver_overcurrent", "driver_overheating",
               "sensor_status", "driver_error_status")
VARIANTS = ("official", "enabled", "handshake_first")


def arm_motors_enabled(p):
    """List of 6 booleans (joint motors enabled), or None if unreadable."""
    try:
        return list(p.GetArmEnableStatus())
    except Exception:
        return None


def disable_arm(p, *, consent: bool, sleep=time.sleep, timeout=4.0) -> bool:
    """Disable the arm's JOINT motors (AgileX's gripper-setup procedure starts with 'disable the arm').

    Does NOTHING unless ``consent=True`` - the caller must have asked the user.  Only call this on a folded, resting
    arm: a disabled Piper lets its wrist sag by ~25 degrees.  Returns True once the motors report disabled."""
    if consent:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            en = arm_motors_enabled(p)
            if en is not None and not any(en):
                return True
            p.DisablePiper()
            sleep(0.15)
        en = arm_motors_enabled(p)
        return en is not None and not any(en)
    return False


def status(p) -> dict | None:
    """{value_m, effort, enabled, zeroed, error} from the V3 gripper feedback, or None if unreadable."""
    try:
        gs = p.GetArmGripperMsgs_V3().gripper_state
        foc = gs.foc_status
        return {"value_m": gs.grippers_val / 1e6, "effort": gs.grippers_effort,
                "enabled": bool(foc.driver_enable_status), "zeroed": bool(foc.homing_status),
                "error": any(bool(getattr(foc, k, False)) for k in ERROR_FLAGS)}
    except Exception:
        return None


def _send(p, secs, val_mm, code, zero=0, effort=1000, sleep=time.sleep, rate=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        p.GripperCtrl(int(val_mm * 1000), effort, code, zero)
        sleep(rate)


def zero_variant(p, name: str, sleep=time.sleep):
    """Run one zeroing sequence.  The gripper must already be closed by hand."""
    if name == "official":                       # the AgileX V3 demo: disable, wait, ONE set-zero frame, then enable
        _send(p, 1.5, 0, 0x00, sleep=sleep)
        p.GripperCtrl(0, 1000, 0x00, 0xAE)
        sleep(0.5)
        _send(p, 1.0, 0, 0x01, sleep=sleep)
    elif name == "enabled":                      # set-zero while the gripper is enabled
        _send(p, 0.6, 0, 0x01, sleep=sleep)
        p.GripperCtrl(0, 1000, 0x01, 0xAE)
        sleep(0.5)
        _send(p, 1.0, 0, 0x01, sleep=sleep)
    elif name == "handshake_first":              # clear-error + enable handshake, then the official sequence
        _send(p, 0.4, 0, 0x02, sleep=sleep)
        _send(p, 0.4, 0, 0x01, sleep=sleep)
        _send(p, 0.8, 0, 0x00, sleep=sleep)
        p.GripperCtrl(0, 1000, 0x00, 0xAE)
        sleep(0.5)
        _send(p, 1.0, 0, 0x01, sleep=sleep)
    else:
        raise ValueError(name)


def zero_ack(p):
    """(timestamp, ok) of the arm's latest set-instruction response, or None.  ok means index 0x75 and success flag 1."""
    try:
        r = p.GetRespInstruction()
        ir = r.instruction_response
        return r.time_stamp, (ir.instruction_index == 0x75 and ir.is_set_zero_successfully == 1)
    except Exception:
        return None


def try_zero(p, sleep=time.sleep, log=print) -> dict:
    """Send the zero command; success = the ARM acknowledges it (the gripper's status bit is not trusted)."""
    tried = []
    for v in VARIANTS:
        before = zero_ack(p)
        zero_variant(p, v, sleep)
        sleep(0.3)
        after = zero_ack(p)
        acked = bool(after and after[1] and (before is None or after[0] > before[0]))
        st = status(p)
        tried.append((v, acked))
        log(f"    variant '{v}': arm ACK {'OK (zero accepted)' if acked else 'missing / failed'};  gripper homing bit = "
            f"{None if st is None else st['zeroed']} (not reliable)")
        if acked:
            return {"ok": True, "variant": v, "status": st, "tried": tried}
    return {"ok": False, "variant": None, "status": status(p), "tried": tried}


def follow_test(p, delta_mm: float = 10.0, sleep=time.sleep) -> dict:
    """Move the gripper delta_mm open and back THROUGH the gripper input layer (ramp + slow sine - a plain step target
    freezes these grippers, see gripper_layer.py); report whether the measured opening followed."""
    from .gripper_layer import GripperLayer
    st0 = status(p)
    if st0 is None:
        return {"followed": False, "moved_mm": 0.0, "note": "gripper feedback unreadable"}
    v0 = st0["value_m"]
    layer = GripperLayer()
    layer.reset(v0)

    def drive(target_m, secs):
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            p.GripperCtrl(int(round(layer.update(target_m, 0.01) * 1e6)), 1000, 0x01, 0)
            sleep(0.01)

    d = delta_mm / 1000
    tgt = v0 + d if v0 + d <= 0.065 else v0 - d          # open if there is room, otherwise close
    drive(tgt, 1.5)
    st1 = status(p)
    drive(v0, 1.5)
    moved = ((st1["value_m"] - v0) * 1000) if st1 else 0.0
    return {"followed": abs(moved) >= 0.5 * delta_mm, "moved_mm": round(moved, 2), "note": ""}


def diagnose(p) -> list[str]:
    """Read-only facts for a support request: firmware, arm mode, joint enable state, gripper flags and stroke setting."""
    out = []
    try:
        out.append(f"arm firmware        : {p.GetPiperFirmwareVersion()}")
    except Exception as e:
        out.append(f"arm firmware        : unreadable ({e})")
    try:
        out.append("arm status          : " + "; ".join(l.strip() for l in str(p.GetArmStatus().arm_status).splitlines()[:5]))
    except Exception as e:
        out.append(f"arm status          : unreadable ({e})")
    out.append(f"joint motors enabled: {arm_motors_enabled(p)}")
    ack = zero_ack(p)
    out.append(f"last set-instruction ACK: {'none seen' if ack is None else ('zero accepted' if ack[1] else 'not a zero ACK')}")
    st = status(p)
    out.append("gripper             : " + ("unreadable" if st is None else
               f"opening {st['value_m'] * 1000:.1f} mm, effort {st['effort']}, driver enabled={st['enabled']}, "
               f"zeroed flag={st['zeroed']}, fault={st['error']}"))
    try:
        p.ArmParamEnquiryAndConfig(4)
        time.sleep(0.5)
        fb = p.GetGripperTeachingPendantParamFeedback().arm_gripper_teaching_param_feedback
        out.append(f"gripper parameters  : travel coefficient {fb.teaching_range_per}, stroke setting {fb.max_range_config}, friction {fb.teaching_friction}")
    except Exception as e:
        out.append(f"gripper parameters  : unreadable ({e})")
    return out
