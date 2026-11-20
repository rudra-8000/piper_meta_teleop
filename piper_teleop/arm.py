"""Arm backends: ``SimArm`` (no hardware) and ``PiperArm`` (real, via piper_sdk).

Both expose the same tiny interface so the teleop logic never touches CAN:

    connect() -> enable() -> [read(), command(q, gripper_m)] ... -> park/disable()

Units at this boundary are SI: joints in rad, gripper opening in metres.
The Piper CAN protocol uses 0.001 deg and 0.001 mm integers; conversion lives here.
"""
from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .kinematics import JOINT_HI, JOINT_LO

RAD_TO_MDEG = 180.0 / math.pi * 1000.0      # rad -> 0.001 deg
GRIPPER_MAX_M = 0.070                         # 0-70 mm two-finger gripper / pendant


@dataclass
class ArmState:
    q: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gripper: float = 0.0          # opening [m]
    fresh: bool = False           # feedback is arriving
    enabled: bool = False
    info: str = ""


class ArmBackend(ABC):
    name = "arm"

    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def enable(self, timeout: float = 10.0, gripper: bool = True) -> bool: ...
    @abstractmethod
    def read(self) -> ArmState: ...
    @abstractmethod
    def command(self, q: np.ndarray, gripper_m: float | None) -> None: ...
    @abstractmethod
    def emergency_stop(self) -> None: ...
    @abstractmethod
    def resume(self) -> None: ...
    @abstractmethod
    def disable(self) -> None: ...
    def close(self) -> None:
        pass


# --------------------------------------------------------------------- simulator
class SimArm(ArmBackend):
    """Kinematic stand-in for a Piper: first-order lag + speed limit.

    ``lag_s`` imitates the firmware's own smoothing, ``max_speed`` the joint
    speed limit (manual: 180-225 deg/s).
    """

    def __init__(self, name="sim", q0=None, lag_s=0.03, max_speed=3.0):
        self.name = name
        self._q = np.zeros(6) if q0 is None else np.array(q0, float)
        self._target = self._q.copy()
        self._grip = 0.035
        self._grip_t = self._grip
        self._lag = lag_s
        self._vmax = max_speed
        self._t = time.perf_counter()
        self._enabled = False
        self._stopped = False
        self.commands: list = []   # (t, q, g) when ``record`` is set
        self.record = False

    def connect(self): pass

    def enable(self, timeout=10.0, gripper=True):
        self._enabled = True
        return True

    def _advance(self):
        now = time.perf_counter()
        dt = min(now - self._t, 0.1)
        self._t = now
        if self._stopped or not self._enabled:
            return
        a = 1.0 - math.exp(-dt / self._lag) if self._lag > 0 else 1.0
        dq = (self._target - self._q) * a
        lim = self._vmax * dt
        m = float(np.max(np.abs(dq)))
        if m > lim:
            dq *= lim / m
        self._q = self._q + dq
        self._grip += (self._grip_t - self._grip) * a

    def read(self) -> ArmState:
        self._advance()
        return ArmState(self._q.copy(), self._grip, True, self._enabled, "sim")

    def command(self, q, gripper_m):
        if self.record:
            self.commands.append((time.perf_counter(), np.array(q), gripper_m))
        if self._stopped:
            return
        self._target = np.clip(q, JOINT_LO, JOINT_HI)
        if gripper_m is not None:
            self._grip_t = float(np.clip(gripper_m, 0.0, GRIPPER_MAX_M))

    def emergency_stop(self): self._stopped = True
    def resume(self): self._stopped = False
    def disable(self): self._enabled = False


# -------------------------------------------------------------------- real arm
def _enum_val(x):
    return int(getattr(x, "value", x))


def status_text(p, lines: int = 5) -> str:
    """The arm's own decode of its state, e.g. 'Control Mode: STANDBY(0x0); Arm Status: NORMAL(0x0); ...'."""
    try:
        st = p.GetArmStatus().arm_status
        return "; ".join(l.strip() for l in str(st).splitlines()[:lines])
    except Exception as e:                     # never let diagnostics break the caller
        return f"status unreadable ({e})"


def ctrl_mode_of(p):
    """0 standby, 1 CAN command control, 2 teaching ... or None if unreadable."""
    try:
        return _enum_val(p.GetArmStatus().arm_status.ctrl_mode)
    except Exception:
        return None


def arm_state_of(p):
    """0 = NORMAL, 1 = emergency stop, ... (see SDK GetArmStatus) or None."""
    try:
        return _enum_val(p.GetArmStatus().arm_status.arm_status)
    except Exception:
        return None


def is_folded(q_rad, tol_deg: float = 6.0) -> bool:
    """True when the arm is in its folded rest pose (J2 and J3 at their zero stops).

    The quick-start manual requires the zero pose before switching out of teach/link mode, because the
    switch can make a raised arm drop.  The wrist (J4-J6) rests a bit off zero, so only J2/J3 are judged.
    """
    return abs(math.degrees(q_rad[1])) < tol_deg and abs(math.degrees(q_rad[2])) < tol_deg


# arm_status values that are NOT latched faults: normal / no IK solution / singularity / target beyond limit
_BENIGN_STATES = (0x00, 0x02, 0x03, 0x04)


def _make_interface(can_name: str, dh_is_offset: int, judge_flag: bool):
    """Prefer the V3 interface (firmware >= 1.8-8); fall back to V2."""
    import piper_sdk
    cls = getattr(piper_sdk, "C_PiperInterface_V3", None) or piper_sdk.C_PiperInterface_V2
    from .joint_lock import guard_piper_sdk
    return guard_piper_sdk(cls(can_name=can_name, judge_flag=judge_flag, can_auto_init=True,
               dh_is_offset=dh_is_offset, start_sdk_joint_limit=True,
               start_sdk_gripper_limit=True), can_name)


class PiperArm(ArmBackend):
    def __init__(self, can_name: str, *, dh_is_offset: int = 1, speed_pct: int = 100,
                 follow_mode: str = "standard", gripper_effort: int = 1000,
                 judge_flag: bool = True, iface=None):
        self.name = can_name
        self.can_name = can_name
        self.speed_pct = int(speed_pct)
        self.mit_flag = 0xAD if follow_mode == "high_follow" else 0x00
        self.gripper_effort = int(gripper_effort)
        self._dh = dh_is_offset
        self._judge = judge_flag
        self._p = iface
        self._lock = threading.Lock()
        self._grip_last = 0.0
        self.error = ""
        self.notes: list[str] = []

    # -- lifecycle ---------------------------------------------------------
    def connect(self):
        if self._p is None:
            self._p = _make_interface(self.can_name, self._dh, self._judge)
        from .joint_lock import guard_piper_sdk
        guard_piper_sdk(self._p, self.can_name)
        self._p.ConnectPort()
        time.sleep(0.1)

    def _feedback_live(self) -> bool:
        j = self._p.GetArmJointMsgs()
        return getattr(j, "Hz", 0) > 10

    def _motors_enabled(self) -> bool:
        try:
            return all(self._p.GetArmEnableStatus())
        except Exception:
            return False

    def enable(self, timeout=10.0, gripper=True) -> bool:
        """Bring the arm to CAN joint mode with motors on - sending ONLY the frames the arm's real state needs.

        An arm that is already enabled, in CAN mode and has no latched fault is ADOPTED untouched: no reset, no
        enable, no mode frame.  (Every extra state-changing frame risks a hold-torque glitch, and a raised arm
        drops.)  On failure ``self.error`` says what the arm reported; ``self.notes`` lists what was done.
        """
        p = self._p
        self.error = ""
        self.notes = []
        from .joint_lock import lock_until
        self.joint_locked = lock_until(self.can_name)
        if self.joint_locked:          # gripper-only: no enable, mode or joint frame may reach this arm
            self.notes.append(f"JOINT LOCK until {self.joint_locked:%Y-%m-%d %H:%M}: gripper only, no joint will be "
                              f"commanded (config/joint_lock.yaml)")
            return True
        t0 = time.monotonic()
        while not self._feedback_live():
            if time.monotonic() - t0 > 3.0:
                self.error = ("no joint feedback: wrong CAN, arm off, or arm set as leader (0xFA). "
                              + status_text(p))
                return False
            time.sleep(0.02)
        mode, state = ctrl_mode_of(p), arm_state_of(p)
        enabled = self._motors_enabled()
        raised = not is_folded(self.read().q)
        if mode == 0x01 and enabled and state in _BENIGN_STATES + (None,):
            self.notes.append("already enabled, in CAN mode, no latched fault: ADOPTED as-is "
                              "(sent no reset, enable or mode frame)")
            return True
        # Anything that would change the arm's state while it is raised can make it DROP: refuse.
        if raised and (mode not in (0x01, None) or state == 0x01):
            self.error = ("REFUSING to change the arm's state while it is raised (it could drop): "
                          + status_text(p, 2) + ". Support the arm and lower it by hand to the folded rest pose "
                          "(or power-cycle it there; teach-button light off; clear any emergency-stop state), then retry.")
            return False
        if state == 0x01:                       # latched emergency stop: resume it (folded arms only, see above)
            p.MotionCtrl_1(0x02, 0, 0)
            self.notes.append("emergency-stop state cleared (resume frame sent)")
        if mode not in (0x01, None):            # standby before CAN mode (quick-start manual 2.2)
            p.MotionCtrl_2(0x00, 0x00, 0, 0x00)
            self.notes.append("went through standby to leave teaching/other mode")
            time.sleep(0.2)
        if not enabled:
            while not p.EnablePiper():
                if time.monotonic() - t0 > timeout:
                    self.error = "motors did not enable. " + status_text(p)
                    return False
                time.sleep(0.01)
            self.notes.append("motors enabled")
        # the arm must actually *report* CAN mode, otherwise it will silently ignore every JointCtrl
        t1 = time.monotonic()
        while True:
            p.MotionCtrl_2(0x01, 0x01, self.speed_pct, self.mit_flag)
            mode, state = ctrl_mode_of(p), arm_state_of(p)
            if mode in (0x01, None) and state in _BENIGN_STATES + (None,):
                break
            if time.monotonic() - t1 > 2.0:
                why = ""
                if mode == 0x02:
                    why = " (teaching mode: make sure the teach-button light is OFF)"
                elif state == 0x01:
                    why = " (emergency-stop state)"
                self.error = f"arm will not enter CAN control mode{why}. " + status_text(p)
                return False
            time.sleep(0.05)
        if gripper:    # enable the gripper at its *current* opening so it does not snap shut
            g = self._read_gripper()
            p.GripperCtrl(int(g * 1e6), self.gripper_effort, 0x01, 0)
        return True

    def read(self) -> ArmState:
        p = self._p
        js = p.GetArmJointMsgs()
        s = js.joint_state
        q = np.array([s.joint_1, s.joint_2, s.joint_3, s.joint_4, s.joint_5, s.joint_6],
                     dtype=float) / RAD_TO_MDEG
        fresh = getattr(js, "Hz", 0) > 10
        en = all(p.GetArmEnableStatus())
        return ArmState(q, self._read_gripper(), fresh, en, "")

    def _read_gripper(self) -> float:
        g = self._p.GetArmGripperMsgs().gripper_state.grippers_angle
        self._grip_last = max(0.0, g / 1e6)
        return self._grip_last

    def command(self, q, gripper_m):
        p = self._p
        if gripper_m is not None:
            gi = int(round(float(np.clip(gripper_m, 0.0, GRIPPER_MAX_M)) * 1e6))
        if getattr(self, "joint_locked", None):
            if gripper_m is not None:
                p.GripperCtrl(gi, self.gripper_effort, 0x01, 0)
            return
        qi = [int(round(x * RAD_TO_MDEG)) for x in np.clip(q, JOINT_LO, JOINT_HI)]
        p.MotionCtrl_2(0x01, 0x01, self.speed_pct, self.mit_flag)
        p.JointCtrl(*qi)
        if gripper_m is not None:
            gi = int(round(float(np.clip(gripper_m, 0.0, GRIPPER_MAX_M)) * 1e6))
            p.GripperCtrl(gi, self.gripper_effort, 0x01, 0)

    def gripper_status(self):
        """The gripper's own report: {value_m, effort, enabled, zeroed, error} or None (V2 interface / unreadable)."""
        from .gripper import status
        return status(self._p) if hasattr(self._p, "GetArmGripperMsgs_V3") else None

    def emergency_stop(self):
        self._p.MotionCtrl_1(0x01, 0, 0)

    def resume(self):
        self._p.MotionCtrl_1(0x02, 0, 0)

    def disable(self):
        self._p.DisablePiper()
