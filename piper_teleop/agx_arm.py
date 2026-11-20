"""Arm backend on AgileX's current library, pyAgxArm (github.com/agilexrobotics/pyAgxArm).

Drop-in replacement for ``arm.PiperArm`` (same ArmBackend interface, SI units: rad and metres), selected with
``piper.driver: pyagxarm`` in config/teleop.yaml.  pyAgxArm has a driver profile per firmware (``v189`` for the
S-V1.8-9 arms this was developed on); ``piper.firmware: auto`` asks the arm.

Safety rules carried over from PiperArm, because pyAgxArm itself does not enforce them:
* ``reset()`` (CAN MotionCtrl 0x02) and ``disable()`` POWER THE ARM OFF - a raised arm falls.  reset() is only ever
  sent to a folded arm to clear a latched e-stop; disable() only on an explicit release request.
* An arm that is already enabled, in CAN mode and fault-free is ADOPTED: nothing is sent.
* Any state change (leaving standby/teaching, clearing an e-stop) is refused while the arm is raised.
"""
from __future__ import annotations

import time

import numpy as np

from .arm import GRIPPER_MAX_M, ArmBackend, ArmState, is_folded
from .kinematics import JOINT_HI, JOINT_LO

_BENIGN = (0x00, 0x02, 0x03, 0x04)         # arm_status values that are not latched faults
_CTRL_NAMES = {0: "STANDBY", 1: "CAN_CTRL", 2: "TEACHING", 6: "LINKAGE_TEACHING_INPUT", 7: "OFFLINE_TRAJECTORY"}


FALLBACK_FIRMWARE = "v189"                  # pyAgxArm profile for firmware S-V1.8-9


def _i(v):
    try:
        return int(getattr(v, "value", v))
    except (TypeError, ValueError):
        return None


def _create_robot(can_name: str, firmware: str):
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config, resolve_firmware_profile
    if firmware == "auto":
        probe = AgxArmFactory.create_arm(create_agx_arm_config(robot=ArmModel.PIPER, channel=can_name))
        probe.connect()
        from .joint_lock import guard_pyagxarm
        guard_pyagxarm(probe, can_name)
        fw = None
        t0 = time.monotonic()
        while fw is None and time.monotonic() - t0 < 3.0:
            fw = probe.get_firmware()
            time.sleep(0.1)
        probe.disconnect()
        if not fw:
            raise RuntimeError(f"{can_name}: could not read the arm firmware (set piper.firmware explicitly, e.g. v189)")
        firmware = resolve_firmware_profile(ArmModel.PIPER, fw["software_version"])
    robot = AgxArmFactory.create_arm(create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=firmware,
                                                           channel=can_name))
    return robot, firmware


class AgxPiperArm(ArmBackend):
    def __init__(self, can_name: str, *, firmware: str = "auto", speed_pct: int = 100, follow_mode: str = "standard",
                 gripper_effort: int = 1000, robot=None, effector=None):
        self.name = self.can_name = can_name
        self.firmware = firmware
        self.speed_pct = int(speed_pct)
        self.fast = follow_mode == "high_follow"          # move_js: no smoothing (AgileX: "extremely high risk")
        self.gripper_force_n = gripper_effort / 1000.0    # our effort units are 0.001 N*m; pyAgxArm takes newtons
        self._r, self._eff = robot, effector
        self.error = ""
        self.notes: list[str] = []

    # -- lifecycle -----------------------------------------------------------
    def connect(self):
        if self._r is None:
            self._r, self.firmware = _create_robot(self.can_name, self.firmware)
            self._eff = self._r.init_effector(self._r.OPTIONS.EFFECTOR.AGX_GRIPPER)   # must precede connect()
        self._r.connect()
        from .joint_lock import guard_pyagxarm
        guard_pyagxarm(self._r, self.can_name)
        time.sleep(0.2)

    def _status(self):
        st = self._r.get_arm_status()
        if st is None:
            return None, None
        return _i(st.msg.ctrl_mode), _i(st.msg.arm_status)

    def status_text(self) -> str:
        mode, state = self._status()
        return f"control mode {_CTRL_NAMES.get(mode, mode)}, arm status {state} (0 = normal)"

    def _enabled(self) -> bool:
        try:
            return all(self._r.get_joints_enable_status_list())
        except Exception:
            return False

    def _fresh(self) -> bool:
        ja = self._r.get_joint_angles()
        return ja is not None and (ja.hz or 0) > 10

    def enable(self, timeout=10.0, gripper=True) -> bool:
        r = self._r
        self.error, self.notes = "", []
        from .joint_lock import lock_until
        self.joint_locked = lock_until(self.can_name)
        if self.joint_locked:          # gripper-only: no enable, mode or joint frame may reach this arm
            self.notes.append(f"JOINT LOCK until {self.joint_locked:%Y-%m-%d %H:%M}: gripper only, no joint will be "
                              f"commanded (config/joint_lock.yaml)")
            return True
        t0 = time.monotonic()
        while not self._fresh():
            if time.monotonic() - t0 > 3.0:
                self.error = "no joint feedback: wrong CAN, arm off, or arm set as leader. " + self.status_text()
                return False
            time.sleep(0.02)
        # move_j()/move_js() re-send pyAgxArm's cached mode frame (default speed 50 %); set our speed in that cache
        # WITHOUT transmitting, so adopting an arm still sends nothing until the first real command.
        if hasattr(r, "_msg_mode"):
            r._msg_mode.move_spd_rate_ctrl = self.speed_pct
        mode, state = self._status()
        enabled, raised = self._enabled(), not is_folded(self.read().q)
        if mode == 0x01 and enabled and state in _BENIGN + (None,):
            self.notes.append("already enabled, in CAN mode, no latched fault: ADOPTED as-is (sent nothing)")
            return True
        if raised and (mode not in (0x01, None) or state == 0x01):
            self.error = ("REFUSING to change the arm's state while it is raised (it could drop): " + self.status_text()
                          + ". Lower it by hand to the folded rest pose (or power-cycle it there), then retry.")
            return False
        if state == 0x01:
            r.reset()                                     # powers the arm off - only reached for a FOLDED arm
            self.notes.append("emergency-stop state cleared (reset, arm folded)")
            time.sleep(0.3)
        if not self._enabled():
            while not r.enable():
                if time.monotonic() - t0 > timeout:
                    self.error = "motors did not enable. " + self.status_text()
                    return False
                time.sleep(0.01)
            self.notes.append("motors enabled")
        r.set_speed_percent(self.speed_pct)               # also puts the arm in CAN control mode
        t1 = time.monotonic()
        while True:
            r.set_motion_mode("js" if self.fast else "j")
            mode, state = self._status()
            if mode in (0x01, None) and state in _BENIGN + (None,):
                break
            if time.monotonic() - t1 > 2.0:
                why = " (teaching mode: teach-button light must be OFF)" if mode == 0x02 else ""
                self.error = f"arm will not enter CAN control mode{why}. " + self.status_text()
                return False
            time.sleep(0.05)
        if gripper and self._eff is not None:             # enable the gripper where it is, so it does not snap shut
            self._eff.move_gripper_m(value=self._grip(), force=self.gripper_force_n)
        return True

    # -- I/O -------------------------------------------------------------------
    def _grip(self) -> float:
        if self._eff is None:
            return 0.0
        gs = self._eff.get_gripper_status()
        return max(0.0, float(gs.msg.value)) if gs is not None else 0.0

    def read(self) -> ArmState:
        ja = self._r.get_joint_angles()
        q = np.array(ja.msg if ja is not None else [0.0] * 6, dtype=float)
        return ArmState(q, self._grip(), ja is not None and (ja.hz or 0) > 10, self._enabled(), "")

    def command(self, q, gripper_m):
        if not getattr(self, "joint_locked", None):
            qs = [float(x) for x in np.clip(q, JOINT_LO, JOINT_HI)]
            (self._r.move_js if self.fast else self._r.move_j)(qs)
        if gripper_m is not None and self._eff is not None:
            self._eff.move_gripper_m(value=float(np.clip(gripper_m, 0.0, GRIPPER_MAX_M)), force=self.gripper_force_n)

    def gripper_status(self):
        if self._eff is None:
            return None
        gs = self._eff.get_gripper_status()
        if gs is None:
            return None
        f = gs.msg.foc_status
        err = any(bool(getattr(f, k, False)) for k in ("voltage_too_low", "motor_overheating", "driver_overcurrent",
                                                       "driver_overheating", "sensor_status", "driver_error_status"))
        return {"value_m": float(gs.msg.value), "effort": round(float(gs.msg.force) * 1000), "enabled": bool(f.driver_enable_status),
                "zeroed": bool(f.homing_status), "error": err}

    def emergency_stop(self):
        self._r.electronic_emergency_stop()               # damped stop (AgileX: descends slowly, does not drop)

    def resume(self):
        # Leaving a firmware e-stop needs reset(), which powers the arm off: never do that automatically.
        self.error = "firmware e-stop latched: fold the arm (support it), then restart the server to re-enable"

    def disable(self):
        self._r.disable()

    def close(self):
        try:
            self._r.disconnect()
        except Exception:
            pass


class AgxLeader:
    """Read-only leader arm on pyAgxArm (the arm is in leader / linkage-teaching mode, 0xFA).  Never enabled or commanded."""

    def __init__(self, can_name: str, firmware: str = "auto", robot=None, effector=None, source: str = "ctrl"):
        self.can_name, self.firmware = can_name, firmware
        self.source = source           # ctrl = broadcast command frames of a 0xFA leader; msgs = plain joint FEEDBACK (drag-teach)
        self._r, self._eff = robot, effector

    def connect(self):
        if self._r is None:
            try:
                self._r, self.firmware = _create_robot(self.can_name, self.firmware)
            except RuntimeError:
                # A leader (0xFA) answers no firmware query, so auto-detect times out.  We only read its broadcast, so use
                # the profile of the firmware these arms were measured on (S-V1.8-9) instead of refusing to start.
                self._r, self.firmware = _create_robot(self.can_name, FALLBACK_FIRMWARE)
                print(f"[lf] {self.can_name}: a leader answers no firmware query - using profile {FALLBACK_FIRMWARE}", flush=True)
            self._eff = self._r.init_effector(self._r.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._r.connect()
        from .joint_lock import guard_pyagxarm
        guard_pyagxarm(self._r, self.can_name)
        time.sleep(0.2)

    def read(self) -> ArmState:
        if self.source == "msgs":                         # a normal arm in drag-teach: its measured joints and gripper
            ja = self._r.get_joint_angles()
            q = np.array(ja.msg if ja is not None else [0.0] * 6, dtype=float)
            gs = self._eff.get_gripper_status() if self._eff is not None else None
            g = max(0.0, float(gs.msg.value)) if gs is not None else 0.0
            return ArmState(q, g, ja is not None and (ja.hz or 0) > 10, True, "pyagxarm-msgs")
        la = self._r.get_leader_joint_angles()            # the leader's broadcast joint commands, rad
        q = np.array(la.msg if la is not None else [0.0] * 6, dtype=float)
        g = 0.0
        if self._eff is not None:
            gc = self._eff.get_gripper_ctrl_states()      # the pendant's broadcast gripper command, m
            g = max(0.0, float(gc.msg.value)) if gc is not None else 0.0
        return ArmState(q, g, la is not None and (la.hz or 0) > 10, True, "pyagxarm")
