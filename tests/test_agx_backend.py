"""AgxPiperArm (pyAgxArm backend) against a fake pyAgxArm robot that records every call."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from piper_teleop.agx_arm import AgxLeader, AgxPiperArm
from piper_teleop.arm import is_folded

FOLDED = [0.0, 0.0, 0.0, math.radians(-5), math.radians(23), math.radians(12)]
RAISED = [0.0, math.radians(35), math.radians(-34), 0.0, math.radians(37), 0.0]


class FakeRobot:
    """Mimics the pyAgxArm Piper driver calls AgxPiperArm uses; every state-changing call is logged."""

    def __init__(self, q=FOLDED, mode=1, state=0, enabled=True):
        self.q, self.mode, self.state, self.en = list(q), mode, state, enabled
        self.calls = []
        self._msg_mode = NS(move_spd_rate_ctrl=50)
        self.frames = []                                          # raw CAN IDs that reached the "bus"
        comm = NS(send=lambda m: self.frames.append(m.arbitration_id))
        self._ctx = NS(get_comm=lambda: comm)

    def connect(self): self.calls.append("connect")
    def disconnect(self): self.calls.append("disconnect")
    def get_joint_angles(self): return NS(msg=self.q, hz=200.0)
    def get_arm_status(self): return NS(msg=NS(ctrl_mode=self.mode, arm_status=self.state), hz=200.0)
    def get_joints_enable_status_list(self): return [self.en] * 6
    def get_leader_joint_angles(self): return NS(msg=[0.1] * 6, hz=200.0)

    def enable(self):
        self.calls.append("enable"); was = self.en; self.en = True; return was

    def disable(self): self.calls.append("disable"); self.en = False
    def reset(self): self.calls.append("reset"); self.state = 0; self.en = False
    def set_speed_percent(self, p): self.calls.append(("speed", p)); self.mode = 1
    def set_motion_mode(self, m): self.calls.append(("mode", m)); self.mode = 1
    def move_j(self, q): self.calls.append(("move_j", list(q)))
    def move_js(self, q): self.calls.append(("move_js", list(q)))
    def electronic_emergency_stop(self): self.calls.append("estop")


class FakeEff:
    def __init__(self, value=0.027):
        self.value, self.calls = value, []

    def get_gripper_status(self):
        foc = NS(driver_enable_status=True, homing_status=False, voltage_too_low=False, motor_overheating=False,
                 driver_overcurrent=False, driver_overheating=False, sensor_status=False, driver_error_status=False)
        return NS(msg=NS(value=self.value, force=0.12, foc_status=foc))

    def get_gripper_ctrl_states(self): return NS(msg=NS(value=0.04))
    def move_gripper_m(self, value, force): self.calls.append((round(value, 6), force))


def arm(robot, eff=None, **kw):
    a = AgxPiperArm("can_x", robot=robot, effector=eff if eff is not None else FakeEff(), **kw)
    a.connect()
    return a


DANGEROUS = ("reset", "disable")


def test_healthy_raised_arm_is_adopted_and_nothing_is_sent():
    r = FakeRobot(q=RAISED, mode=1, state=0, enabled=True)
    a = arm(r)
    assert a.enable() and "ADOPTED" in a.notes[0]
    assert r.calls == ["connect"]                        # not one frame
    assert r._msg_mode.move_spd_rate_ctrl == 100         # speed set in the cache only


def test_raised_arm_in_teaching_mode_is_refused_without_sending_anything():
    r = FakeRobot(q=RAISED, mode=2, enabled=True)
    a = arm(r)
    assert not a.enable() and "REFUSING" in a.error
    assert r.calls == ["connect"]


def test_raised_arm_with_latched_estop_is_never_reset():
    r = FakeRobot(q=RAISED, mode=1, state=1)
    a = arm(r)
    assert not a.enable()
    assert not any(c in DANGEROUS for c in r.calls)


def test_folded_estop_arm_is_reset_then_enabled_and_put_in_joint_mode():
    r = FakeRobot(q=FOLDED, mode=0, state=1, enabled=False)
    a = arm(r)
    assert a.enable(), a.error
    assert r.calls.index("reset") < r.calls.index("enable") and ("speed", 100) in r.calls and ("mode", "j") in r.calls


def test_high_follow_maps_to_move_js():
    r = FakeRobot(q=FOLDED, mode=0, enabled=False)
    a = arm(r, follow_mode="high_follow")
    assert a.enable() and ("mode", "js") in r.calls
    a.command(np.zeros(6), None)
    assert r.calls[-1][0] == "move_js"


def test_command_units_clamping_and_gripper_force():
    r, e = FakeRobot(), FakeEff()
    a = arm(r, e, gripper_effort=1000)
    a.command(np.radians([200, -50, 20, 150, 90, 180]), 0.5)
    q = r.calls[-1][1]
    assert np.allclose(np.degrees(q), [150, 0, 0, 100, 70, 100])   # clipped to our joint limits (rad in, rad out)
    assert e.calls[-1] == (0.07, 1.0)                               # metres, 1000 -> 1.0 N, clipped to 70 mm
    n = len(e.calls)
    a.command(np.zeros(6), None)
    assert len(e.calls) == n                                        # None = gripper untouched


def test_read_gives_si_units_and_status_dict():
    a = arm(FakeRobot(q=RAISED), FakeEff(0.0273))
    st = a.read()
    assert np.allclose(st.q, RAISED) and st.gripper == pytest.approx(0.0273) and st.fresh and st.enabled
    g = a.gripper_status()
    assert g == {"value_m": 0.0273, "effort": 120, "enabled": True, "zeroed": False, "error": False}


def test_resume_after_firmware_estop_never_powers_the_arm_off():
    r = FakeRobot(q=RAISED)
    a = arm(r)
    a.emergency_stop()
    a.resume()
    assert "estop" in r.calls and not any(c in DANGEROUS for c in r.calls) and "fold" in a.error


def test_leader_reads_broadcast_joints_and_pendant():
    ld = AgxLeader("can_l", robot=FakeRobot(), effector=FakeEff())
    ld.connect()
    st = ld.read()
    assert np.allclose(st.q, 0.1) and st.gripper == pytest.approx(0.04) and st.fresh


def test_config_selects_pyagxarm_by_default_and_piper_sdk_on_request():
    from piper_teleop import config as C
    from piper_teleop.arm import PiperArm
    raw = C.load()
    assert type(C.make_backend({"can": "can_x"}, raw, False)).__name__ == "AgxPiperArm"
    raw["piper"]["driver"] = "piper_sdk"
    assert isinstance(C.make_backend({"can": "can_x"}, raw, False), PiperArm)


def test_the_real_pyagxarm_driver_has_every_method_we_call():
    pytest.importorskip("pyAgxArm")
    from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
    r = AgxArmFactory.create_arm(create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V189,
                                                       channel="none", enable_check_can=False, auto_connect=False))
    e = r.init_effector(r.OPTIONS.EFFECTOR.AGX_GRIPPER)
    for n in ("connect", "disconnect", "get_joint_angles", "get_arm_status", "get_joints_enable_status_list", "enable",
              "reset", "set_speed_percent", "set_motion_mode", "move_j", "move_js", "electronic_emergency_stop",
              "get_leader_joint_angles", "get_firmware", "disable"):
        assert callable(getattr(r, n)), n
    assert hasattr(r, "_msg_mode") and hasattr(r._msg_mode, "move_spd_rate_ctrl")
    for n in ("get_gripper_status", "get_gripper_ctrl_states", "move_gripper_m"):
        assert callable(getattr(e, n)), n
