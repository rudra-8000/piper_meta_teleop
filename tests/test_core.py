import math

import numpy as np
import pytest

from piper_teleop.arm import PiperArm, SimArm
from piper_teleop.controller import ArmConfig, HandSample, TeleopArm, ENGAGED, ESTOP, HOMING, IDLE
from piper_teleop.kinematics import PiperKinematics, so3_exp, so3_log, quat_to_mat, mat_to_quat
from piper_teleop import protocol

HOME = np.radians(ArmConfig().home_deg)      # the default start pose (rest +5 cm forward, +10 cm up)


# ---------------------------------------------------------------- kinematics
def test_fk_matches_sdk():
    sdk = pytest.importorskip("piper_sdk")
    ref = sdk.C_PiperForwardKinematics(1)
    kin = PiperKinematics(1)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(kin.lo, kin.hi)
        want = np.array(ref.CalFK(list(q))[-1][:3]) * 1e-3
        assert np.allclose(kin.fk(q)[1], want, atol=1e-9)


def test_jacobian_finite_difference():
    kin = PiperKinematics(1)
    rng = np.random.default_rng(1)
    q = rng.uniform(kin.lo, kin.hi)
    R, p, J = kin.fk_jac(q)
    for i in range(6):
        dq = np.zeros(6); dq[i] = 1e-6
        R2, p2 = kin.fk(q + dq)
        assert np.allclose((p2 - p) / 1e-6, J[:3, i], atol=1e-4)
        assert np.allclose(so3_log(R2 @ R.T) / 1e-6, J[3:, i], atol=1e-4)


def test_ik_roundtrip_and_limits():
    kin = PiperKinematics(1, (0, 0, 0.12))
    R, p = kin.fk_tcp(HOME)
    q, ep, er = kin.ik_tcp(HOME + 0.15, R, p, iters=150)   # near-folded pose: damped, converges slowly
    assert ep < 1e-3 and er < 1e-2
    q, *_ = kin.ik_tcp(HOME, R, p + np.array([5.0, 0, 0]), iters=10)   # absurd target
    assert np.all(q >= kin.lo - 1e-9) and np.all(q <= kin.hi + 1e-9)


def test_rotation_helpers():
    rng = np.random.default_rng(2)
    for _ in range(20):
        w = rng.normal(size=3)
        w *= rng.uniform(0.1, 3.0) / np.linalg.norm(w)
        R = so3_exp(w)
        assert np.allclose(so3_log(R), w, atol=1e-8)
        assert np.allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-8)


# ------------------------------------------------------------------ protocol
def test_protocol_roundtrip():
    hands = [dict(hand="left", pos=(1, 2, 3), quat=(0, 0, 0, 1), ray_quat=(0, 0, 0, 1), trigger=0.25,
                  squeeze=1.0, stick=(0.1, -0.2), a=True, tracked=True),
             dict(hand="right", pos=(-1, 0.5, 0), quat=(0, 0.7071068, 0, 0.7071068), trigger=1, squeeze=0,
                  b=True, stick_click=True, emulated=True)]
    out = protocol.decode_pose(protocol.encode_pose(7, 12.5, hands), 99.0)
    assert set(out) == {"left", "right"}
    l, r = out["left"], out["right"]
    assert l.seq == 7 and l.btn_a and not l.btn_b and abs(l.trigger - 0.25) < 1e-6 and l.stick[1] == pytest.approx(-0.2)
    assert r.btn_b and r.stick_click and r.emulated and np.allclose(r.pos, (-1, 0.5, 0))
    with pytest.raises(ValueError):
        protocol.decode_pose(b"\x01\x02", 0)


# ---------------------------------------------------------------- controller
class Rig:
    """Drive one TeleopArm with a virtual 200 Hz clock and a virtual hand."""

    def __init__(self, **cfg):
        self.arm_be = SimArm(q0=HOME, lag_s=0.0, max_speed=100)
        self.arm_be.enable()
        self.cfg = ArmConfig(name="t", hand="right", **cfg)
        self.arm = TeleopArm(self.cfg, self.arm_be)
        self.arm.start()
        self.t = 100.0
        self.seq = 0
        self.hand = dict(pos=np.array([0.2, 1.1, -0.3]), quat=np.array([0, 0, 0, 1.0]), squeeze=0.0,
                         trigger=0.0, a=False, b=False, tracked=True, emulated=False)
        self.send = True
        self.run(0.05)            # a few released-GRIP samples first (the arm requires seeing one)

    def run(self, seconds, rate=200, hand_hz=90):
        n = int(seconds * rate)
        dt = 1.0 / rate
        every = max(1, round(rate / hand_hz))
        for i in range(n):
            self.t += dt
            if self.send and i % every == 0:
                self.seq += 1
                self.last = HandSample(
                    t_recv=self.t, seq=self.seq, tracked=self.hand["tracked"], emulated=self.hand["emulated"],
                    pos=self.hand["pos"].copy(), quat=self.hand["quat"], ray_quat=self.hand["quat"],
                    trigger=self.hand["trigger"], squeeze=self.hand["squeeze"],
                    btn_a=self.hand["a"], btn_b=self.hand["b"])
            self.arm.step(getattr(self, "last", None), self.t, dt)

    def tcp(self):
        return self.arm.kin.fk_tcp(self.arm.q_cmd)[1]


def test_no_motion_without_grip():
    r = Rig()
    q0 = r.arm.q_cmd.copy()
    r.hand["pos"] = np.array([0.5, 1.5, -0.8])
    r.run(0.5)
    assert np.allclose(r.arm.q_cmd, q0) and r.arm.state == IDLE


def test_engage_without_jump_and_forward_mapping():
    r = Rig(smoothing_ms=0, filter_min_cutoff=50, filter_beta=0)
    p0 = r.tcp().copy()
    r.hand["squeeze"] = 1.0
    r.run(0.1)
    assert r.arm.state == ENGAGED
    assert np.linalg.norm(r.tcp() - p0) < 1e-3            # no jump on grab
    # move the hand 8 cm FORWARD (-z in WebXR) -> robot +x by 8 cm
    for k in range(40):
        r.hand["pos"] = r.hand["pos"] + np.array([0, 0, -0.002])
        r.run(0.01)
    r.run(0.3)
    d = r.tcp() - p0
    assert d[0] == pytest.approx(0.08, abs=0.006) and abs(d[1]) < 0.006 and abs(d[2]) < 0.006
    # move hand LEFT (-x in WebXR) -> robot +y
    for k in range(25):
        r.hand["pos"] = r.hand["pos"] + np.array([-0.002, 0, 0])
        r.run(0.01)
    r.run(0.3)
    assert (r.tcp() - p0)[1] == pytest.approx(0.05, abs=0.006)
    # up -> +z
    for k in range(20):
        r.hand["pos"] = r.hand["pos"] + np.array([0, 0.002, 0])
        r.run(0.01)
    r.run(0.3)
    assert (r.tcp() - p0)[2] == pytest.approx(0.04, abs=0.006)


def test_release_holds_and_regrab_reanchors():
    r = Rig(smoothing_ms=0)
    r.hand["squeeze"] = 1.0
    r.run(0.1)
    r.hand["pos"] = r.hand["pos"] + np.array([0, 0, -0.05])
    r.run(0.3)
    held = r.arm.q_cmd.copy()
    r.hand["squeeze"] = 0.0
    r.run(0.1)
    assert r.arm.state == IDLE
    r.hand["pos"] = r.hand["pos"] + np.array([0, 0.3, 0])   # move hand while released
    r.run(0.3)
    assert np.allclose(r.arm.q_cmd, held)
    r.hand["squeeze"] = 1.0
    r.run(0.1)
    assert np.allclose(r.arm.q_cmd, held, atol=2e-3)         # re-grab: no jump


def test_stale_stream_holds_and_requires_regrab():
    r = Rig(smoothing_ms=0)
    r.hand["squeeze"] = 1.0
    r.run(0.1)
    r.send = False
    r.run(0.5)                       # stream dies
    assert r.arm.state == IDLE and "hold" in r.arm.msg
    held = r.arm.q_cmd.copy()
    r.send = True                    # stream resumes with GRIP *still held*
    r.hand["pos"] = r.hand["pos"] + np.array([0, 0, -0.1])
    r.run(0.3)
    assert r.arm.state == IDLE and np.allclose(r.arm.q_cmd, held)
    r.hand["squeeze"] = 0.0          # must release...
    r.run(0.05)
    r.hand["squeeze"] = 1.0          # ...and grab again
    r.run(0.05)
    assert r.arm.state == ENGAGED


def test_grip_held_at_startup_does_not_engage():
    arm_be = SimArm(q0=HOME); arm_be.enable()
    arm = TeleopArm(ArmConfig(name="t", hand="right"), arm_be); arm.start()
    t = 10.0
    for i in range(100):
        t += 0.005
        arm.step(HandSample(t_recv=t, seq=i, squeeze=1.0), t, 0.005)
    assert arm.state == IDLE


def test_emulated_tracking_is_treated_as_lost():
    r = Rig()
    r.hand["squeeze"] = 1.0
    r.run(0.1)
    assert r.arm.state == ENGAGED
    r.hand["emulated"] = True
    r.run(0.05)
    assert r.arm.state == IDLE


def test_joint_speed_limit_and_workspace_clamp():
    r = Rig(smoothing_ms=0, max_joint_speed=1.0)
    r.hand["squeeze"] = 1.0
    r.run(0.1)
    prev = r.arm.q_cmd.copy()
    worst = 0.0
    r.hand["pos"] = r.hand["pos"] + np.array([0, 0, -3.0])      # teleport hand 3 m forward
    for _ in range(400):
        r.run(0.005)
        worst = max(worst, np.max(np.abs(r.arm.q_cmd - prev)) / 0.005)
        prev = r.arm.q_cmd.copy()
    assert worst <= 1.0 + 1e-6
    ws = r.cfg.workspace
    tcp = r.tcp()
    assert tcp[0] <= ws["x"][1] + 0.02                          # box (+ small IK slack)


def test_homing_and_cancel():
    r = Rig(home_speed=1.0)
    r.arm.q_cmd = np.radians([0, 20, -20, 0, 10, 0])
    r.arm.go_home()
    assert r.arm.state == HOMING
    r.run(4.0)
    assert r.arm.state == IDLE
    assert np.allclose(r.arm.q_cmd, np.radians(r.cfg.home_deg), atol=1e-6)


def test_estop_blocks_everything():
    r = Rig()
    r.arm.estop()
    q = r.arm.q_cmd.copy()
    r.hand["squeeze"] = 1.0
    r.run(0.3)
    assert r.arm.state == ESTOP and np.allclose(r.arm.q_cmd, q)


SETTLE = 0.6     # long enough for any gripper layer setting used in these tests (an optional 150 mm/s ramp needs 0.47 s)


def test_gripper_trigger_mapping():
    r = Rig()
    r.hand["trigger"] = 1.0
    r.run(0.05)
    assert r.arm.grip_target == pytest.approx(0.0)          # the target follows the trigger at once
    r.run(SETTLE)
    assert r.arm.grip_cmd == pytest.approx(0.0)             # the sent command gets there via the ramp
    r.hand["trigger"] = 0.0
    r.run(SETTLE)
    assert r.arm.grip_cmd == pytest.approx(0.07)


# -------------------------------------------------------------- PiperArm I/O
class FakeIface:
    """Minimal stand-in for piper_sdk.C_PiperInterface_V3 to test unit conversions."""

    class _Obj:
        def __init__(self, **kw): self.__dict__.update(kw)

    def __init__(self):
        self.sent = []
        self.joint_mdeg = [10000, 20000, -30000, 0, 5000, -1000]
        self.grip_um = 35000

    def ConnectPort(self): pass
    def GetArmJointMsgs(self):
        o = self._Obj(Hz=200.0)
        o.joint_state = self._Obj(**{f"joint_{i + 1}": v for i, v in enumerate(self.joint_mdeg)})
        return o
    def GetArmGripperMsgs(self):
        return self._Obj(gripper_state=self._Obj(grippers_angle=self.grip_um))
    def GetArmEnableStatus(self): return [True] * 6
    def EnablePiper(self): return True
    def MotionCtrl_1(self, *a): self.sent.append(("MC1", a))
    def MotionCtrl_2(self, *a): self.sent.append(("MC2", a))
    def JointCtrl(self, *a): self.sent.append(("JC", a))
    def GripperCtrl(self, *a): self.sent.append(("GC", a))
    def DisablePiper(self): self.sent.append(("disable",))


def test_piper_arm_units_and_enable_sequence():
    f = FakeIface()
    arm = PiperArm("can_x", iface=f, follow_mode="high_follow", gripper_effort=800)
    arm.connect()
    assert arm.enable()
    # gripper enabled at its *current* opening (35 mm), not at 0
    assert ("GC", (35000, 800, 0x01, 0)) in f.sent
    st = arm.read()
    assert np.allclose(np.degrees(st.q), [10, 20, -30, 0, 5, -1], atol=1e-6)
    assert st.gripper == pytest.approx(0.035)
    f.sent.clear()
    arm.command(np.radians([10, 20, -30, 0, 5, -1]), 0.0215)
    assert ("MC2", (0x01, 0x01, 100, 0xAD)) in f.sent
    assert ("JC", (10000, 20000, -30000, 0, 5000, -1000)) in f.sent
    assert ("GC", (21500, 800, 0x01, 0)) in f.sent
    # out-of-range commands are clamped to joint limits / gripper stroke
    f.sent.clear()
    arm.command(np.radians([200, -50, 20, 150, 90, 180]), 0.5)
    jc = [a for k, *a in f.sent if k == "JC"][0][0]
    assert jc == (150000, 0, 0, 100000, 70000, 100000)
    assert [a for k, *a in f.sent if k == "GC"][0][0][0] == 70000


# ------------------------------------------------- enable() must verify CAN mode
class StatusFake(FakeIface):
    """FakeIface whose reported control mode reacts to MotionCtrl_2 like the real arm should."""

    def __init__(self, mode=0, state=0, stuck=False, enabled=True):
        super().__init__()
        self.mode, self.state, self.stuck, self.enabled = mode, state, stuck, enabled

    def GetArmEnableStatus(self):
        return [self.enabled] * 6

    def EnablePiper(self):                    # like the SDK: returns the status from BEFORE it sends the enable
        was = self.enabled
        self.sent.append(("EN",))
        self.enabled = True
        return was

    def MotionCtrl_1(self, *a):
        super().MotionCtrl_1(*a)
        if a[0] == 0x02 and not self.stuck:
            self.state = 0

    def MotionCtrl_2(self, ctrl_mode, *a):
        super().MotionCtrl_2(ctrl_mode, *a)
        if not self.stuck and ctrl_mode in (0x00, 0x01):
            self.mode = ctrl_mode

    def GetArmStatus(self):
        o = self._Obj()
        o.arm_status = self._Obj(ctrl_mode=self.mode, arm_status=self.state)
        return o


FOLDED_MDEG = [500, 0, 0, -5000, 23000, 12000]


def _enable(fake, **kw):
    arm = PiperArm("can_x", iface=fake, **kw)
    arm.connect()
    return arm, arm.enable(timeout=3)


def test_enable_from_standby_reaches_can_mode_and_goes_via_standby():
    f = StatusFake(mode=2)                    # e.g. was left in teaching mode
    f.joint_mdeg = FOLDED_MDEG
    arm, ok = _enable(f)
    assert ok and f.mode == 1
    modes = [x[1][0] for x in f.sent if x[0] == "MC2"]            # ctrl_mode of each 0x151 frame sent
    assert modes[0] == 0x00 and 0x01 in modes[1:]                 # standby first, then CAN control


def test_enable_already_in_can_mode_skips_standby():
    f = StatusFake(mode=1)
    arm, ok = _enable(f)
    assert ok
    assert not any(x[0] == "MC2" and x[1][0] == 0x00 for x in f.sent)


def test_enable_refuses_mode_change_when_arm_is_raised():
    f = StatusFake(mode=2)                    # teaching mode AND raised (J2 20 deg, J3 -30 deg in the fake default)
    arm, ok = _enable(f)
    assert not ok and "REFUSING" in arm.error and "drop" in arm.error
    assert f.sent == []                       # not a single frame sent: no standby, no CAN-mode switch


def test_enable_raised_arm_already_in_can_mode_is_allowed():
    f = StatusFake(mode=1)                    # already CAN mode: no mode switch needed
    arm, ok = _enable(f)
    assert ok


def test_enable_fails_loudly_when_arm_ignores_can_mode():
    f = StatusFake(mode=2, stuck=True)        # teaching mode, will not leave it
    f.joint_mdeg = FOLDED_MDEG
    arm, ok = _enable(f)
    assert not ok and "teach" in arm.error.lower()


def test_enable_fails_in_emergency_stop_state():
    f = StatusFake(mode=1, state=1, stuck=True)
    arm, ok = _enable(f)
    assert not ok and "emergency" in arm.error.lower()


# ------------------------------------------------ default home pose + facing arms
REST = np.radians([0, 0, 0, -5, 23, 12])          # measured folded rest pose (wrist sags to ~(-5, 23, 12))


def test_default_home_is_rest_plus_5cm_forward_10cm_up():
    cfg = ArmConfig()
    kin = PiperKinematics(1, cfg.tcp_offset)
    d = kin.fk_tcp(np.radians(cfg.home_deg))[1] - kin.fk_tcp(REST)[1]
    assert d == pytest.approx([0.05, 0.0, 0.10], abs=0.006)
    assert cfg.home_deg[0] == 0                                   # base joint stays at zero
    assert np.max(np.abs(np.degrees(np.radians(cfg.home_deg) - REST))) < 40   # a small move from rest


@pytest.mark.parametrize("d", [[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0], [0, -1, 0]])
def test_every_direction_is_trackable_from_default_home(d):
    """Regression: a tool tilted only like the rest pose is near a wrist singularity and cannot move sideways."""
    cfg = ArmConfig(name="t", hand="right", workspace={"x": [0, 0.6], "y": [-0.4, 0.4], "z": [0, 0.6]})
    be = SimArm(q0=np.radians(cfg.home_deg), lag_s=0.0, max_speed=100); be.enable()
    arm = TeleopArm(cfg, be); arm.start()
    rig = Rig.__new__(Rig)
    rig.arm_be, rig.cfg, rig.arm, rig.t, rig.seq, rig.send = be, cfg, arm, 100.0, 0, True
    rig.hand = dict(pos=np.array([0.2, 1.1, -0.3]), quat=np.array([0, 0, 0, 1.0]), squeeze=0.0, trigger=0.0,
                    a=False, b=False, tracked=True, emulated=False)
    rig.run(0.05)
    rig.hand["squeeze"] = 1.0
    rig.run(0.1)
    _, p0 = arm.kin.fk_tcp(arm.q_cmd)
    dv = np.array(d, float)
    for _ in range(80):
        rig.hand["pos"] = rig.hand["pos"] + dv * 0.001
        rig.run(0.01)
    rig.run(0.4)
    got = arm.kin.fk_tcp(arm.q_cmd)[1] - p0
    want = np.array([-dv[2], -dv[0], dv[1]]) * 0.08
    assert np.linalg.norm(got - want) < 0.008 and "reject" not in arm.msg


def test_base_yaw_rotates_arm_frame_for_facing_arms():
    # arm whose forward axis points 90 deg ccw of the operator's forward (left arm of a facing pair is -90)
    for yaw, expect in ((90, [0.0, -0.08, 0.0]), (-90, [0.0, 0.08, 0.0]), (0, [0.08, 0.0, 0.0])):
        r = Rig(smoothing_ms=0, filter_min_cutoff=50, filter_beta=0, base_yaw_deg=yaw)
        p0 = r.tcp().copy()
        r.hand["squeeze"] = 1.0
        r.run(0.1)
        for _ in range(40):
            r.hand["pos"] = r.hand["pos"] + np.array([0, 0, -0.002])   # operator pushes hands FORWARD 8 cm
            r.run(0.01)
        r.run(1.5)                                   # sideways motion from the folded start is joint-speed limited
        assert r.tcp() - p0 == pytest.approx(expect, abs=0.008), yaw


def test_go_rest_moves_slowly_without_touching_gripper():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("go_rest", pathlib.Path(__file__).parent.parent / "tools" / "go_rest.py")
    gr = importlib.util.module_from_spec(spec); spec.loader.exec_module(gr)
    be = SimArm(q0=np.radians([5, 3, -2, -5, 23, 12]), lag_s=0.0, max_speed=100); be.enable(); be.record = True
    final = gr.move_to(be, np.zeros(6), speed=2.0, rate=200)       # fast speed only to keep the test short
    assert np.max(np.abs(final)) < np.radians(1.0)
    qs = np.array([c[1] for c in be.commands])
    assert np.max(np.abs(np.diff(qs, axis=0))) <= 2.0 / 200 + 1e-9  # never faster than the cap
    assert all(c[2] is None for c in be.commands)                   # gripper never commanded


def test_command_without_gripper_sends_no_gripper_frame():
    f = FakeIface()
    arm = PiperArm("can_x", iface=f); arm.connect()
    arm.command(np.zeros(6), None)
    assert [k for k, *_ in f.sent] == ["MC2", "JC"]


# ---------------------------------------------- deadman + freeze (E-STOP) behaviour
def _lag_rig(lag=0.08, **cfg):
    """Rig whose simulated arm trails its command (like the real firmware's smoothing)."""
    r = Rig(smoothing_ms=0, **cfg)
    r.arm_be._lag = lag
    r.arm_be._vmax = 3.0
    return r


def _engage_and_move(r, secs=0.4):
    r.hand["squeeze"] = 1.0
    r.run(0.1)
    for _ in range(int(secs * 100)):
        r.hand["pos"] = r.hand["pos"] + np.array([0, 0, -0.003])    # fast forward motion
        r.run(0.01)


def test_release_stops_where_the_arm_physically_is():
    r = _lag_rig()
    _engage_and_move(r)
    gap = np.max(np.abs(r.arm.q_cmd - r.arm_be._q))
    assert gap > 0.02                                    # the arm really is trailing its command
    at_release = r.arm_be._q.copy()
    r.hand["squeeze"] = 0.0
    r.run(1.0)
    assert r.arm.state == IDLE
    assert np.max(np.abs(r.arm_be._q - at_release)) < 0.02   # it did not coast on to the old command
    assert np.max(np.abs(r.arm_be._q - r.arm.q_cmd)) < 1e-3


def test_freeze_holds_pose_keeps_commanding_and_never_touches_power_or_gripper():
    r = _lag_rig()
    r.hand["trigger"] = 1.0                              # gripper being squeezed shut
    _engage_and_move(r)
    grip_before = r.arm.grip_cmd
    at_freeze = r.arm_be._q.copy()
    n_before = len(r.arm_be.commands) if r.arm_be.record else 0
    r.arm_be.record = True
    r.arm.estop()                                        # the SPACE key
    r.hand["pos"] = r.hand["pos"] + np.array([0, 0, -0.5])   # operator keeps flailing
    r.run(1.0)
    assert r.arm.state == ESTOP and "FROZEN" in r.arm.msg
    assert np.max(np.abs(r.arm_be._q - at_freeze)) < 0.02
    assert not r.arm_be._stopped and r.arm_be._enabled   # no firmware stop, no disable
    assert len(r.arm_be.commands) > 50                   # still actively holding the pose
    assert all(c[2] is None or abs(c[2] - grip_before) <= 0.0005 + 1e-9 for c in r.arm_be.commands)  # gripper held (only the 0.5 mm layer sine)


def test_hard_estop_is_opt_in():
    r = Rig()
    r.arm.estop(hard=True)
    assert r.arm_be._stopped
    r.arm.resume()
    assert not r.arm_be._stopped and r.arm.state == IDLE


def test_resume_needs_a_fresh_grab():
    r = _lag_rig()
    _engage_and_move(r)
    r.arm.estop(); r.run(0.2)
    r.arm.resume()
    r.run(0.2)
    assert r.arm.state == IDLE                           # GRIP still held from before: must not re-engage
    r.hand["squeeze"] = 0.0; r.run(0.05)
    r.hand["squeeze"] = 1.0; r.run(0.05)
    assert r.arm.state == ENGAGED


def test_grip_is_the_deadman_and_trigger_only_works_the_gripper():
    r = Rig()
    q0 = r.arm.q_cmd.copy()
    r.hand["trigger"] = 1.0                              # trigger alone, hand moving
    r.hand["pos"] = r.hand["pos"] + np.array([0, 0, -0.1])
    r.run(SETTLE)
    assert r.arm.state == IDLE and np.allclose(r.arm.q_cmd, q0)    # arm did not move
    assert r.arm.grip_cmd == pytest.approx(0.0)                    # but the gripper closed
    r.hand["trigger"] = 0.0
    r.run(SETTLE)
    assert r.arm.grip_cmd == pytest.approx(0.07)                   # and opens again, arm still idle


def test_console_space_is_freeze_and_q_cannot_quit_by_accident():
    from piper_teleop.quest_server import console_command
    assert console_command(" ") == ("cmd", "estop")
    assert console_command("e") == ("cmd", "estop")
    assert console_command("q") is None and console_command("Q") == ("quit", None)
    assert console_command("z") is None and console_command("x") == ("cmd", "calibrate_cancel")


# ----------------------- never disturb an arm that is already holding (the raised-arm drop)
def test_healthy_enabled_can_mode_arm_is_adopted_without_sending_anything():
    f = StatusFake(mode=1, state=0, enabled=True)         # raised (fake default), enabled, CAN mode, healthy
    arm, ok = _enable(f)
    assert ok and f.sent == [] and "ADOPTED" in arm.notes[0]


def test_benign_status_codes_do_not_stop_adoption():
    for state in (0x02, 0x03, 0x04):                      # no IK solution / singularity / target beyond limit
        f = StatusFake(mode=1, state=state, enabled=True)
        arm, ok = _enable(f)
        assert ok and f.sent == []


def test_disabled_folded_arm_gets_enable_but_no_reset_or_standby():
    f = StatusFake(mode=1, state=0, enabled=False)
    f.joint_mdeg = FOLDED_MDEG
    arm, ok = _enable(f)
    kinds = [x[0] for x in f.sent]
    assert ok and "EN" in kinds and "MC1" not in kinds
    assert not any(x[0] == "MC2" and x[1][0] == 0x00 for x in f.sent)


def test_latched_estop_is_resumed_only_when_folded():
    f = StatusFake(mode=1, state=1, enabled=True)
    f.joint_mdeg = FOLDED_MDEG
    arm, ok = _enable(f)
    assert ok and ("MC1", (0x02, 0, 0)) in f.sent
    f = StatusFake(mode=1, state=1, enabled=True)         # raised + latched e-stop: do nothing, tell the user
    arm, ok = _enable(f)
    assert not ok and f.sent == [] and "REFUSING" in arm.error and "emergency" in arm.error.lower()


# ------------------------------------------------------------------ grippers
def test_gripper_starts_open_even_if_the_measured_opening_reads_zero():
    be = SimArm(q0=HOME, lag_s=0.0, max_speed=100); be.enable(); be.record = True
    be._grip = be._grip_t = 0.0                           # gripper reading 0 mm (closed)
    arm = TeleopArm(ArmConfig(name="t", hand="right"), be); arm.start()
    assert arm.grip_target == pytest.approx(0.07) and arm.grip_cmd == pytest.approx(0.0)   # ramps from where it IS
    t = 100.0
    for _ in range(int(SETTLE / 0.005)):                 # no controller data at all: it still opens, never closes
        t += 0.005
        arm.step(None, t, 0.005)
    g = [c[2] for c in be.commands]
    assert g[-1] == pytest.approx(0.07, abs=0.0005)                    # open (top of the stroke, minus the layer sine)
    assert min(g[1:]) >= -1e-12 and arm.gripper_layer.ramp == pytest.approx(0.07)
    c = arm.cfg
    ramp_ok = all(b >= a - c.gripper_dither_mm * 1e-3 * 2 * np.pi * c.gripper_dither_hz * 0.005 - 1e-12 for a, b in zip(g, g[1:]))
    assert ramp_ok                                                     # never closes, apart from the sine wobble


def test_gripper_default_is_open_and_trigger_closes_proportionally():
    r = Rig()
    r.run(SETTLE)
    assert r.arm.grip_cmd == pytest.approx(0.07)          # no trigger: open
    for trig, want in ((0.5, 0.035), (1.0, 0.0), (0.0, 0.07)):
        r.hand["trigger"] = trig
        r.run(SETTLE)
        assert r.arm.grip_cmd == pytest.approx(want)      # proportional; released: back to open


def test_gripper_command_never_jumps_faster_than_the_ramp():
    """With the optional ramp configured, every command step must respect gripper_slew_mm_s."""
    r = Rig(gripper_slew_mm_s=150.0)
    r.arm_be.record = True
    for trig in (1.0, 0.0, 1.0, 0.3, 1.0, 0.0):          # a trigger mashed between extremes
        r.hand["trigger"] = trig
        r.run(0.15)
    g = np.array([c[2] for c in r.arm_be.commands if c[2] is not None])
    dt = 0.005
    sine_rate = 2 * np.pi * r.cfg.gripper_dither_hz * r.cfg.gripper_dither_mm * 1e-3   # max slope of the layer sine
    assert np.max(np.abs(np.diff(g))) <= (r.cfg.gripper_slew_mm_s * 1e-3 + sine_rate) * dt + 1e-9
    assert g.max() - g.min() > 0.02                       # and it still moved a useful amount


def test_freeze_also_freezes_the_gripper_ramp():
    r = Rig(gripper_slew_mm_s=150.0)
    r.run(SETTLE)
    r.hand["trigger"] = 1.0
    r.run(0.1)                                            # half-way through closing
    r.arm.estop()
    g0 = r.arm.grip_cmd
    r.run(0.5)
    assert r.arm.grip_cmd == pytest.approx(g0)


def test_gripper_keeps_its_last_command_when_tracking_is_lost():
    r = Rig()
    r.hand["trigger"] = 1.0
    r.run(SETTLE)
    assert r.arm.grip_cmd == pytest.approx(0.0)           # holding something
    r.send = False
    r.run(0.5)                                            # stream dies
    assert r.arm.grip_cmd == pytest.approx(0.0)           # NOT released on a network hiccup


def test_gripper_status_reports_zeroed_flag_and_server_warns():
    class V3Fake(FakeIface):
        def __init__(self, zeroed): super().__init__(); self.zeroed = zeroed
        def GetArmGripperMsgs_V3(self):
            foc = self._Obj(driver_enable_status=True, homing_status=self.zeroed, voltage_too_low=False, motor_overheating=False,
                            driver_overcurrent=False, driver_overheating=False, sensor_status=False, driver_error_status=False)
            return self._Obj(gripper_state=self._Obj(grippers_val=64500, grippers_effort=120, foc_status=foc))
    st = PiperArm("c", iface=V3Fake(False)).gripper_status()
    assert st == {"value_m": 0.0645, "effort": 120, "enabled": True, "zeroed": False, "error": False}
    assert PiperArm("c", iface=V3Fake(True)).gripper_status()["zeroed"] is True
    assert PiperArm("c", iface=FakeIface()).gripper_status() is None       # V2 interface: unknown, no false alarm


def test_wrist_stays_calm_with_a_naturally_rotating_hand():
    """Regression for the 'arms twist a lot' complaint: J4/J6 used to swing ~100+ deg against each other."""
    import math
    from piper_teleop.kinematics import mat_to_quat, so3_exp
    cfg = ArmConfig(name="t", hand="right")
    be = SimArm(q0=HOME, lag_s=0.0, max_speed=100); be.enable()
    arm = TeleopArm(cfg, be); arm.start()
    t, dt, seq, Q = 100.0, 0.005, 0, []
    for i in range(int(14 / dt)):
        t += dt
        tt = max(0.0, i * dt - 1.0)
        pos = np.array([0.2 - 0.10 * math.sin(2 * math.pi * 0.25 * tt + 0.6), 1.1 + 0.04 * math.sin(2 * math.pi * 0.4 * tt + 1.0),
                        -0.3 - 0.06 * math.sin(2 * math.pi * 0.35 * tt)])
        R = (so3_exp(np.array([0, math.radians(25 * math.sin(2 * math.pi * 0.25 * tt + 1.0)), 0]))
             @ so3_exp(np.array([math.radians(10 * math.sin(2 * math.pi * 0.4 * tt)), 0, 0]))
             @ so3_exp(np.array([0, 0, math.radians(12 * math.sin(2 * math.pi * 0.55 * tt + 2.0))])))
        q = mat_to_quat(R)
        if i % 2 == 0:
            seq += 1
        arm.step(HandSample(t_recv=t, seq=seq, pos=pos, quat=q, ray_quat=q, squeeze=0.0 if i * dt < 0.5 else 1.0), t, dt)
        if i * dt > 1.5:
            Q.append(arm.q_cmd.copy())
    Q = np.degrees(np.array(Q))
    rng = Q.max(0) - Q.min(0)
    assert rng[3] < 25 and rng[5] < 25                    # J4 / J6 swing (was ~100 / ~120)
    assert Q[:, 4].min() >= 19.5                          # J5 never enters the wrist singularity


@pytest.mark.parametrize("axis", [1, 2])
def test_small_hand_twist_does_not_swing_the_base(axis):
    """Regression: with follow_base the anti-windup anchor slide fed the sideways shortfall back into the heading, so a
    15-20 deg hand twist about the vertical (or the sideways) axis ran J1 away to ~85 deg."""
    r = Rig()
    r.hand["squeeze"] = 1.0
    r.run(0.3)
    q0 = r.arm_be.read().q.copy()
    w = np.zeros(3)
    w[axis] = math.radians(20.0)
    r.hand["quat"] = mat_to_quat(so3_exp(w))
    r.run(1.5)
    dq = np.degrees(r.arm_be.read().q - q0)
    assert abs(dq[0]) < 15.0, dq                         # J1: a 20 deg twist x yaw gain 0.3 is ~6 deg of heading
    assert np.max(np.abs(dq)) < 15.0, dq


def test_tool_tip_never_sinks_below_the_floor_even_when_the_hand_drives_it_down():
    r = Rig()
    floor = r.cfg.workspace["z"][0]
    r.hand["squeeze"] = 1.0
    r.run(0.3)
    kin = PiperKinematics()
    lowest = 9.0
    for dz in (-0.05, -0.15, -0.30, -0.6):                 # hand goes down, and twists while it does
        r.hand["pos"] = np.array([0.2, 1.1 + dz, -0.3])
        r.hand["quat"] = mat_to_quat(so3_exp(np.array([0.3, 0.0, 0.0])))
        for _ in range(40):
            r.run(0.05)
            lowest = min(lowest, float(kin.fk_tcp(r.arm_be.read().q)[1][2]))
    assert lowest >= floor - 0.015, lowest


def test_limit_buzz_is_immediate_strong_and_repeats_while_pushing():
    r = Rig()
    r.hand["squeeze"] = 1.0
    r.run(0.3)
    r.arm.events.clear()
    first = None
    for i in range(60):                                      # push the hand back, 2 cm per 25 ms (0.8 m/s), past the rear plane
        r.hand["pos"] = np.array([0.2, 1.1, -0.3 + 0.02 * (i + 1)])
        r.run(0.025)
        if first is None and any(e[0] == "haptic" for e in r.arm.events):
            first = i
    buzz = [e for e in r.arm.events if e[0] == "haptic"]
    assert buzz and buzz[0][2] >= 0.7, buzz                  # the first buzz is strong ...
    assert first is not None and first < 25, first           # ... and comes as soon as the limit is reached (not seconds later)
    assert len(buzz) >= 3, buzz                              # keeps pulsing while you keep pushing


def test_free_motion_inside_the_limits_never_buzzes():
    r = Rig()
    r.hand["squeeze"] = 1.0
    r.run(0.3)
    r.arm.events.clear()
    for i in range(40):                                      # 8 cm forward and 4 cm sideways, comfortably reachable
        r.hand["pos"] = np.array([0.2 + 0.001 * i, 1.1, -0.3 - 0.002 * i])
        r.run(0.025)
    assert not [e for e in r.arm.events if e[0] == "haptic"], r.arm.msg


def test_tool_tip_speed_is_capped_and_slower_near_the_desk():
    from piper_teleop.kinematics import PiperKinematics
    kin = PiperKinematics()
    for cfg, zlow in (({}, False), ({"slow_zone_z": 0.5}, True)):      # second case: the whole workspace counts as "near the desk"
        r = Rig(**cfg)
        r.hand["squeeze"] = 1.0
        r.run(0.3)
        prev = kin.fk_tcp(r.arm_be.read().q)[1]
        peak = 0.0
        for i in range(60):                                  # hand slams 0.4 m forward in 0.3 s (1.3 m/s)
            r.hand["pos"] = np.array([0.2, 1.1, -0.3 - 0.4 * min(1.0, (i + 1) / 30)])
            r.run(0.01)
            p = kin.fk_tcp(r.arm_be.read().q)[1]
            peak = max(peak, float(np.linalg.norm(p - prev)) / 0.01)
            prev = p
        assert peak < (0.25 if zlow else 0.5) * 1.15, (peak, zlow)


@pytest.mark.parametrize("q0_deg", [(-20, 85, -70, 5, 45, 5), (-20, 90, -90, 0, 60, 5)])
@pytest.mark.parametrize("pitch_deg", [-45, 45])
def test_pitching_the_tool_with_a_still_hand_does_not_fling_the_tip(q0_deg, pitch_deg):
    """Regression (live log): pitching the gripper 45 deg with the arm stretched out pinned J5 at its limit; the old
    weighted IK then dragged the tip 10-25 cm away to chase the unreachable orientation, and the anti-windup slide counted
    the same shortfall again on every hand sample (the target retreated 17 cm while the hand sat still)."""
    import math
    from piper_teleop.kinematics import mat_to_quat, so3_exp
    cfg = ArmConfig(name="t", hand="right", pos_scale=0.8, rot_scale=0.8)
    be = SimArm(q0=np.radians(q0_deg), lag_s=0.0, max_speed=100); be.enable()
    arm = TeleopArm(cfg, be); arm.start()
    tcp0 = arm.kin.fk_tcp(arm.q_cmd)[1]
    t, dt, seq, worst = 100.0, 0.005, 0, 0.0
    for i in range(int(5 / dt)):
        t += dt
        tt = max(0.0, i * dt - 1.0)
        q = mat_to_quat(so3_exp(np.array([math.radians(pitch_deg) * min(1.0, tt / 1.5), 0, 0])))
        if i % 2 == 0:
            seq += 1
        arm.step(HandSample(t_recv=t, seq=seq, pos=np.array([0.1, 1.3, -0.25]), quat=q, ray_quat=q,
                            squeeze=0.0 if i * dt < 0.5 else 1.0), t, dt)
        worst = max(worst, float(np.linalg.norm(arm.kin.fk_tcp(arm.q_cmd)[1] - tcp0)))
    assert worst < 0.05, worst


def test_ik_keeps_position_when_the_orientation_is_out_of_the_wrists_reach():
    kin = PiperKinematics(1, (0, 0, 0.12))
    q0 = np.radians([-20, 85, -70, 5, 45, 5])
    R0, p0 = kin.fk_tcp(q0)
    R_bad = so3_exp(np.array([math.radians(-80), 0, 0])) @ R0               # pitch the tool far past J5's limit
    q, ep, er = kin.ik_tcp(q0, R_bad, p0, iters=40)
    assert np.linalg.norm(kin.fk_tcp(q)[1] - p0) < 0.02                       # position is priority 1
    assert er > 0.1                                                           # orientation is what gives
    assert np.all(q <= kin.hi + 1e-9) and np.all(q >= kin.lo - 1e-9)
