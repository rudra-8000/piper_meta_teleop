import csv
import time

import numpy as np
import pytest

from piper_teleop import gripper as G
from piper_teleop.arm import SimArm
from piper_teleop.controller import ArmConfig, HandSample, TeleopArm
from piper_teleop.paths import log_dir
from piper_teleop.system import TeleopSystem
from test_core import Rig, HOME  # noqa: F401  (rig that drives one arm with a virtual hand)


# ----------------------------------------------------------------- fake V3 gripper
class FakeGripper:
    """Behaves like the real ones did: enabled + readable, but it IGNORES position commands until its own firmware
    flag 'zeroed' is set - which only a particular zeroing sequence achieves."""

    class _O:
        def __init__(self, **k): self.__dict__.update(k)

    def __init__(self, works_with=None, start_mm=27.3, needs_arm_disabled=False):
        self.works_with = works_with          # name of the variant that flips the flag, or None = never
        self.needs_arm_disabled = needs_arm_disabled
        self.ack = None
        self.arm_enabled = True
        self.val, self.enabled, self.zeroed = start_mm, True, False
        self.frames = []
        self._variant_guess = None

    def GripperCtrl(self, val, eff, code, zero):
        self.frames.append((val, eff, code, zero))
        self.enabled = code in (0x01, 0x03)
        if zero == 0xAE:
            # remember how the set-zero frame was sent: disabled (0x00) vs enabled (0x01) and whether a handshake preceded
            hist = [f[2] for f in self.frames[:-1]]
            self._variant_guess = ("enabled" if code == 0x01 else
                                   "handshake_first" if 0x02 in hist else "official")
            if self.works_with == self._variant_guess and not (self.needs_arm_disabled and self.arm_enabled):
                self.zeroed, self.val = True, 0.0
                self.ack = (time.time(), True)
        if self.zeroed and self.enabled:
            self.val = val / 1000.0

    def GetRespInstruction(self):
        ts, ok = self.ack if self.ack else (0.0, False)
        return self._O(time_stamp=ts, instruction_response=self._O(instruction_index=0x75 if self.ack else -1,
                                                                   is_set_zero_successfully=1 if ok else -1))

    def GetArmEnableStatus(self):
        return [self.arm_enabled] * 6

    def DisablePiper(self):
        self.arm_enabled = False

    def GetArmGripperMsgs_V3(self):
        foc = self._O(driver_enable_status=self.enabled, homing_status=self.zeroed, voltage_too_low=False,
                      motor_overheating=False, driver_overcurrent=False, driver_overheating=False,
                      sensor_status=False, driver_error_status=False)
        return self._O(gripper_state=self._O(grippers_val=int(self.val * 1000), grippers_effort=0, foc_status=foc))


NOSLEEP = lambda s: None  # noqa: E731


def test_status_reads_the_grippers_own_flags():
    st = G.status(FakeGripper())
    assert st["zeroed"] is False and st["enabled"] is True and st["value_m"] == pytest.approx(0.0273)


@pytest.mark.parametrize("works_with", ["official", "enabled", "handshake_first"])
def test_try_zero_finds_whichever_sequence_the_gripper_accepts(works_with):
    g = FakeGripper(works_with=works_with)
    res = G.try_zero(g, sleep=NOSLEEP, log=lambda *_: None)
    assert res["ok"] and res["variant"] == works_with and res["status"]["zeroed"]
    assert res["tried"][-1] == (works_with, True)


def test_try_zero_reports_failure_honestly_when_nothing_works():
    g = FakeGripper(works_with=None)
    res = G.try_zero(g, sleep=NOSLEEP, log=lambda *_: None)
    assert not res["ok"] and [t[1] for t in res["tried"]] == [False, False, False]


def test_follow_test_distinguishes_a_gripper_that_ignores_commands():
    dead = FakeGripper(works_with=None)
    assert G.follow_test(dead, sleep=NOSLEEP)["followed"] is False
    live = FakeGripper(works_with="official")
    G.zero_variant(live, "official", NOSLEEP)
    r = G.follow_test(live, 3.0, sleep=NOSLEEP)
    assert r["followed"] and abs(r["moved_mm"]) >= 1.5


# -------------------------------------------------------- gripper monitor in teleop
class StuckGripperArm(SimArm):
    def command(self, q, gripper_m):                       # joints follow, gripper never does
        super().command(q, None)


def test_a_gripper_that_does_not_follow_raises_a_warning_but_a_following_one_does_not():
    for be, expect_warn in ((StuckGripperArm(q0=HOME, lag_s=0.0, max_speed=100), True), (SimArm(q0=HOME, lag_s=0.0, max_speed=100), False)):
        be.enable()
        be._grip = be._grip_t = 0.027
        arm = TeleopArm(ArmConfig(name="t", hand="right"), be)
        arm.start()
        t = 100.0
        for i in range(int(4.5 / 0.005)):
            t += 0.005
            arm.step(None, t, 0.005)
        assert bool(arm.grip_warn) == expect_warn
        if expect_warn:
            assert "not reaching its commanded opening" in arm.grip_warn and any(e[0] == "gripper_warn" for e in arm.events)
            assert arm.snapshot()["grip_warn"] and arm.snapshot()["grip_meas"] == pytest.approx(0.027)


# ------------------------------------------------------------------ control feel
def _overshoot_then_return(**kw):
    kw.setdefault("workspace", {"x": [0.05, 0.32], "y": [-0.30, 0.30], "z": [0.05, 0.40]})     # a small box to overshoot
    cfg = ArmConfig(name="t", hand="right", **kw)
    be = SimArm(q0=HOME, lag_s=0.0, max_speed=100); be.enable()
    arm = TeleopArm(cfg, be); arm.start()
    rig = Rig.__new__(Rig); rig.arm_be, rig.cfg, rig.arm, rig.t, rig.seq, rig.send = be, cfg, arm, 100.0, 0, True
    rig.hand = dict(pos=np.array([0.2, 1.1, -0.3]), quat=np.array([0, 0, 0, 1.0]), squeeze=0.0, trigger=0.0,
                    a=False, b=False, tracked=True, emulated=False)
    rig.run(0.05); rig.hand["squeeze"] = 1.0; rig.run(0.1)
    out = np.array([0, 0, -1.0])
    for _ in range(100):
        rig.hand["pos"] = rig.hand["pos"] + out * 0.30 / 100; rig.run(0.01)       # 30 cm past what the arm can reach
    rig.run(0.6)
    x_far, msg = arm.kin.fk_tcp(arm.q_cmd)[1][0], arm.msg
    travelled = None
    for k in range(150):
        rig.hand["pos"] = rig.hand["pos"] - out * 0.002; rig.run(0.012)
        if arm.kin.fk_tcp(arm.q_cmd)[1][0] < x_far - 0.01:
            travelled = (k + 1) * 0.002
            break
    return x_far, msg, travelled


def test_anti_windup_removes_the_dead_zone_after_overshooting_reach():
    _, _, off = _overshoot_then_return(anti_windup=0.0)
    _, _, on = _overshoot_then_return()
    assert off is not None and off > 0.15                 # old behaviour: ~21 cm of hand travel before the arm responds
    assert on is not None and on < 0.06                   # new: a couple of cm


def test_limit_messages_say_which_way_and_unreachable_targets_do_not_freeze_the_arm():
    x, msg, _ = _overshoot_then_return(anti_windup=0.0)
    assert "too far out" in msg                            # direction named
    r = Rig()
    r.hand["squeeze"] = 1.0; r.run(0.1)
    for _ in range(60):
        r.hand["pos"] = r.hand["pos"] + np.array([0, 0, 0.004]); r.run(0.01)     # toward the base: ~10 cm of travel only
    assert "IK rejected" not in r.arm.msg and "holding" not in r.arm.msg


def test_haptic_tick_when_controllers_are_tracked_again_after_a_real_gap():
    r = Rig()
    r.run(0.2)
    r.arm.events.clear()
    r.send = False
    r.run(1.5)                                             # controllers vanish for 1.5 s
    r.send = True
    r.arm.events.clear()
    r.run(0.1)
    assert any(e[0] == "haptic" for e in r.arm.events)
    r.arm.events.clear()
    r.run(0.3)
    assert not any(e[0] == "haptic" for e in r.arm.events)      # only once


# --------------------------------------------------------------- pose recorder
def test_engaged_arms_are_recorded_to_a_csv_and_idle_ones_are_not():
    be = SimArm(q0=HOME, lag_s=0.0, max_speed=100); be.enable()
    arm = TeleopArm(ArmConfig(name="left", hand="left"), be); arm.start()
    sys_ = TeleopSystem([arm], rate_hz=200, pose_log=True)
    t = 100.0
    pos = np.array([-0.2, 1.1, -0.3])
    for i in range(60):                                    # released: nothing to record
        t += 0.011; sys_.submit("left", HandSample(t_recv=t, seq=i, pos=pos.copy(), squeeze=0.0)); sys_._tick(t, 0.005)
    before = set(log_dir().glob("teleop_*.csv"))
    for i in range(200):
        t += 0.011; pos = pos + np.array([0, 0, -0.001])
        sys_.submit("left", HandSample(t_recv=t, seq=100 + i, pos=pos.copy(), squeeze=1.0)); sys_._tick(t, 0.005)
    sys_._pose_file.flush()
    new = sorted(set(log_dir().glob("teleop_*.csv")) - before)
    assert new
    rows = list(csv.DictReader(open(new[-1])))
    assert rows and rows[0]["arm"] == "left" and float(rows[-1]["tcp_x"]) > 0 and "ik_err_mm" in rows[0]


def test_zeroing_is_ignored_while_the_arm_is_enabled_and_works_once_it_is_disabled():
    """The real finding: the gripper firmware ignored zero writes while the arm's joint motors were enabled."""
    g = FakeGripper(works_with="official", needs_arm_disabled=True)
    assert not G.try_zero(g, sleep=NOSLEEP, log=lambda *_: None)["ok"]          # what happened on the real arms
    assert G.arm_motors_enabled(g) == [True] * 6
    assert G.disable_arm(g, consent=False, sleep=NOSLEEP) is False and G.arm_motors_enabled(g) == [True] * 6   # no consent: nothing happens
    assert G.disable_arm(g, consent=True, sleep=NOSLEEP) and G.arm_motors_enabled(g) == [False] * 6
    g.frames.clear()                                                         # the fake infers the variant from history
    res = G.try_zero(g, sleep=NOSLEEP, log=lambda *_: None)
    assert res["ok"] and res["variant"] == "official"


def test_the_tool_refuses_to_disable_an_arm_that_is_not_folded():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("gs", pathlib.Path(__file__).parent.parent / "tools" / "gripper_setup.py")
    gs = importlib.util.module_from_spec(spec); spec.loader.exec_module(gs)
    class Raised(FakeGripper):
        def GetArmJointMsgs(self):
            o = self._O(Hz=200.0)
            o.joint_state = self._O(joint_1=0, joint_2=40000, joint_3=-40000, joint_4=0, joint_5=0, joint_6=0)   # J2 40 deg: raised
            return o
        def GetArmGripperMsgs(self):
            return self._O(gripper_state=self._O(grippers_angle=0))
    g = Raised()
    assert gs.prepare_arm("can_x", g, consent=True) is False and g.arm_enabled is True       # nothing was disabled


def test_gripper_can_be_switched_off_completely():
    be = StuckGripperArm(q0=HOME, lag_s=0.0, max_speed=100); be.enable(); be.record = True
    be._grip = be._grip_t = 0.027
    arm = TeleopArm(ArmConfig(name="t", hand="right", gripper_enabled=False), be); arm.start()
    t = 100.0
    for _ in range(int(4.5 / 0.005)):
        t += 0.005
        arm.step(None, t, 0.005)
    assert be.commands and all(c[2] is None for c in be.commands)        # joints commanded, gripper never
    assert arm.grip_warn == "" and not any(e[0] == "gripper_warn" for e in arm.events)


def test_gripper_switch_reaches_the_real_arm_enable_call():
    from piper_teleop import config as C
    raw = C.load()
    raw["defaults"]["gripper_enabled"] = False
    assert C.arm_config("left", raw, raw["arms"]["left"]).gripper_enabled is False
    assert C.arm_config("left", C.load(), C.load()["arms"]["left"]).gripper_enabled is True


def test_per_arm_workspace_overrides_only_the_axes_it_names():
    from piper_teleop import config as cfgmod
    raw = {"defaults": {"workspace": {"x": [0.05, 1.0], "y": [-1.0, 1.0], "z": [0.02, 1.0]}}}
    cfg = cfgmod.arm_config("left", raw, {"workspace": {"y": [-0.3, 1.0]}})
    assert cfg.workspace == {"x": [0.05, 1.0], "y": [-0.3, 1.0], "z": [0.02, 1.0]}


def test_rotation_axis_gains_scale_the_rotation_vector_not_euler_angles():
    from piper_teleop.controller import scale_rotation_axes
    from piper_teleop.kinematics import so3_exp, so3_log
    k = (0.5, 1.0, 0.3)
    for axis, gain in enumerate(k):                              # single-axis rotations scale exactly by that axis' gain
        w = np.zeros(3)
        w[axis] = 0.8
        assert np.allclose(so3_log(scale_rotation_axes(so3_exp(w), k)), w * gain, atol=1e-9)
    w = np.array([0.4, -0.7, 1.1])                               # a combined twist: per-axis gains on the rotation vector
    assert np.allclose(so3_log(scale_rotation_axes(so3_exp(w), k)), w * np.array(k), atol=1e-9)
    w = np.array([0.0, np.radians(89.0), 0.0])                   # near the Euler singularity nothing blows up
    assert np.allclose(so3_log(scale_rotation_axes(so3_exp(w), k)), w, atol=1e-9)
