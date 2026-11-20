import time

import numpy as np
import pytest

from piper_teleop.arm import SimArm
from piper_teleop.leader_follower import LeaderFollower, PiperLeader, SimLeader
from piper_teleop.arm import ArmState


class ScriptedLeader:
    def __init__(self, q):
        self.q = np.array(q, float); self.g = 0.03; self.fresh = True

    def connect(self): pass

    def read(self):
        return ArmState(self.q.copy(), self.g, self.fresh, True, "t")


def run(rig, seconds):
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        rig.step(0.005)
        time.sleep(0.005)


def test_aligns_slowly_then_follows_fast():
    leader = ScriptedLeader(np.radians([0, 40, -40, 0, 20, 0]))
    fol = SimArm(q0=np.zeros(6), lag_s=0.0, max_speed=100); fol.enable()
    rig = LeaderFollower(leader, fol, max_joint_speed=3.0, sync_speed=0.5)
    rig.start()
    prev = rig.q_cmd.copy()
    worst = 0.0
    for _ in range(200):                      # 1 s of virtual time at 200 Hz
        rig.step(0.005)
        worst = max(worst, np.max(np.abs(rig.q_cmd - prev)) / 0.005)
        prev = rig.q_cmd.copy()
    assert worst <= 0.5 + 1e-6 and not rig.synced       # alignment obeys the slow speed
    for _ in range(400):
        rig.step(0.005)
    assert rig.synced and np.allclose(rig.q_cmd, leader.q, atol=1e-6)
    leader.q = leader.q + np.radians([5, -4, 3, 2, 1, -6])     # leader moves: follower tracks
    rig.step(0.005)
    for _ in range(40):
        rig.step(0.005)
    assert np.allclose(rig.q_cmd, leader.q, atol=1e-6)


def test_speed_cap_and_silent_leader_hold():
    leader = ScriptedLeader(np.zeros(6))
    fol = SimArm(q0=np.zeros(6), lag_s=0.0, max_speed=100); fol.enable()
    rig = LeaderFollower(leader, fol, max_joint_speed=2.0, stale_s=0.05)
    rig.start(); rig.synced = True
    leader.q = np.radians([100, 90, -90, 80, 60, 90])          # leader slammed far away
    prev = rig.q_cmd.copy()
    for _ in range(50):
        rig.step(0.005)
        assert np.max(np.abs(rig.q_cmd - prev)) / 0.005 <= 2.0 + 1e-6
        prev = rig.q_cmd.copy()
    held = rig.q_cmd.copy()
    leader.fresh = False                                         # leader dies mid-move
    time.sleep(0.08)
    rig.step(0.005); rig.step(0.005)
    assert np.allclose(rig.q_cmd, held) and "silent" in rig.msg


def test_gripper_scale_and_clip():
    leader = ScriptedLeader(np.zeros(6)); leader.g = 0.05
    fol = SimArm(q0=np.zeros(6)); fol.enable()
    rig = LeaderFollower(leader, fol, gripper_scale=2.0)
    rig.start(); rig.synced = True
    for _ in range(200):
        rig.step(0.005)
    assert rig.g_cmd == pytest.approx(0.07)                      # 0.05*2 clipped to the 70 mm stroke (via the ramp)


class FakeIface:
    class _O:
        def __init__(self, **k): self.__dict__.update(k)

    def GetArmJointCtrl(self):
        return self._O(Hz=200.0, joint_ctrl=self._O(joint_1=10000, joint_2=20000, joint_3=-30000, joint_4=0, joint_5=5000, joint_6=-1000))

    def GetArmGripperCtrl(self):
        return self._O(Hz=200.0, gripper_ctrl=self._O(grippers_angle=42000))

    def ConnectPort(self): pass


def test_piper_leader_reads_command_frames_in_si_units():
    st = PiperLeader("can_l", "ctrl", iface=FakeIface()).read()
    assert np.allclose(np.degrees(st.q), [10, 20, -30, 0, 5, -1]) and st.gripper == pytest.approx(0.042) and st.fresh


def _follow(mode, q_leader, **kw):
    from piper_teleop.leader_follower import JointMap
    leader = ScriptedLeader(q_leader)
    fol = SimArm(q0=np.zeros(6), lag_s=0.0, max_speed=100); fol.enable()
    rig = LeaderFollower(leader, fol, joint_map=JointMap(mode, **kw), sync_speed=100.0)
    rig.start()
    for _ in range(400):
        rig.step(0.005)
    return rig


Q = np.radians([30, 60, -50, 20, 40, -25])


def test_copy_mode_reproduces_the_leader_joints():
    assert np.allclose(_follow("copy", Q).q_cmd, Q, atol=1e-6)


def test_mirror_mode_flips_the_swing_and_roll_joints_only():
    got = _follow("mirror", Q).q_cmd
    assert np.allclose(got, Q * np.array([-1, 1, 1, -1, 1, -1]), atol=1e-6)      # J1 / J4 / J6 flip; J2 / J3 / J5 do not


def test_mirror_is_a_true_reflection_of_the_tool_tip():
    from piper_teleop.kinematics import PiperKinematics
    k = PiperKinematics(1, (0, 0, 0.12))
    pl = k.fk_tcp(Q)[1]
    pf = k.fk_tcp(_follow("mirror", Q).q_cmd)[1]
    assert np.allclose(pf, pl * np.array([1, -1, 1]), atol=1e-6)               # same reach and height, left/right swapped


def test_base_offset_is_added_after_mapping_and_clipped_to_the_joint_limit():
    got = _follow("mirror", Q, offset_deg=20.0).q_cmd
    assert got[0] == pytest.approx(-Q[0] + np.radians(20.0), abs=1e-6)
    far = _follow("copy", np.radians([120, 60, -50, 0, 40, 0]), offset_deg=40.0).q_cmd
    assert far[0] == pytest.approx(np.radians(150.0), abs=1e-6)                 # J1 stops at its limit, never beyond


def test_unknown_mapping_is_refused():
    from piper_teleop.leader_follower import JointMap
    with pytest.raises(ValueError):
        JointMap("sideways")


def test_a_leader_that_answers_no_firmware_query_falls_back_to_the_known_profile(monkeypatch, capsys):
    from types import SimpleNamespace as NS
    from piper_teleop import agx_arm
    calls = []

    def fake_create(can, fw):
        calls.append(fw)
        if fw == "auto":
            raise RuntimeError("could not read the arm firmware")
        return NS(OPTIONS=NS(EFFECTOR=NS(AGX_GRIPPER=1)), init_effector=lambda e: None, connect=lambda: None), fw

    monkeypatch.setattr(agx_arm, "_create_robot", fake_create)
    monkeypatch.setattr("piper_teleop.joint_lock.guard_pyagxarm", lambda r, c: None)
    ld = agx_arm.AgxLeader("can_x", "auto")
    ld.connect()
    assert calls == ["auto", agx_arm.FALLBACK_FIRMWARE] and ld.firmware == agx_arm.FALLBACK_FIRMWARE
    assert "using profile" in capsys.readouterr().out


def test_msgs_source_reads_the_leaders_joint_feedback_and_gripper_status():
    from types import SimpleNamespace as NS
    from piper_teleop.agx_arm import AgxLeader
    q = [0.1, 0.2, -0.3, 0.0, 0.4, 0.5]
    robot = NS(get_joint_angles=lambda: NS(msg=q, hz=200), get_leader_joint_angles=lambda: None)
    eff = NS(get_gripper_status=lambda: NS(msg=NS(value=0.04)), get_gripper_ctrl_states=lambda: None)
    st = AgxLeader("can_x", robot=robot, effector=eff, source="msgs").read()
    assert np.allclose(st.q, q) and st.gripper == pytest.approx(0.04) and st.fresh
    assert not AgxLeader("can_x", robot=robot, effector=eff, source="ctrl").read().fresh    # ctrl reads the (silent) broadcast


def test_a_180_degree_base_flip_wraps_instead_of_clipping():
    from piper_teleop.leader_follower import JointMap
    m = JointMap("copy", offset_deg=180.0)
    for l1, f1 in ((-150, 30), (-120, 60), (-30, 150), (30, -150), (150, -30)):
        q = np.radians([l1, 20, -10, 5, 30, 8])
        got = np.degrees(m(q))
        assert got[0] == pytest.approx(f1, abs=1e-6) and np.allclose(got[1:], [20, -10, 5, 30, 8], atol=1e-6)
    # straight behind is unreachable: the follower waits at whichever limit it is already on, and never jumps across
    for side in (150.0, -150.0):
        for l1 in (-8, 0, 8):
            ref = np.radians([side, 0, 0, 0, 0, 0])
            assert np.degrees(m(np.radians([l1, 0, 0, 0, 0, 0]), ref)[0]) == pytest.approx(side)


def test_leader_gripper_range_is_mapped_onto_the_followers_70mm_stroke():
    leader = ScriptedLeader(np.zeros(6))
    fol = SimArm(q0=np.zeros(6), lag_s=0.0, max_speed=100); fol.enable()
    rig = LeaderFollower(leader, fol, leader_gripper_mm=(16.0, 105.5), sync_speed=100.0, gripper_dither_mm=0.0)
    rig.start()
    for leader_mm, want_mm in ((16.0, 0.0), (60.75, 35.0), (105.5, 70.0), (130.0, 70.0), (5.0, 0.0)):
        leader.g = leader_mm * 1e-3
        for _ in range(5):
            rig.step(0.005)
        assert rig.glayer.ramp * 1e3 == pytest.approx(want_mm, abs=0.05), (leader_mm, rig.glayer.ramp)
    assert rig.lg_seen == [5.0, 130.0]


class StuckFollower:
    """Gripper wedged at 54 mm pushing at full effort, whatever it is asked (until it is asked to relax)."""

    def __init__(self):
        self.cmds, self.q = [], np.zeros(6)
        self.stuck_mm = 54.0

    def read(self):
        return ArmState(self.q.copy(), self.stuck_mm * 1e-3, True, True, "t")

    def command(self, q, g):
        self.cmds.append(g)

    def gripper_status(self):
        return {"value_m": self.stuck_mm * 1e-3, "effort": -1040, "enabled": True, "zeroed": False, "error": False}


def test_a_gripper_pushing_at_full_effort_without_moving_is_told_to_relax_then_retried():
    leader = ScriptedLeader(np.zeros(6)); leader.g = 0.0305
    fol = StuckFollower()
    rig = LeaderFollower(leader, fol, leader_gripper_mm=(16.0, 105.5), gripper_dither_mm=0.0, unstick=True)
    rig.start()
    rig.synced = True
    for _ in range(6):
        time.sleep(0.12)                         # unstick works on wall-clock time; 0.7 s of "wedged"
        for _ in range(8):
            rig.step(0.005)
    assert rig.relaxes >= 1
    relaxed = [g for g in fol.cmds if abs(g - 0.054) < 1e-6]
    assert relaxed                                   # it asked for where the gripper already is
    assert any(abs(g - 0.0114) < 2e-3 for g in fol.cmds[-3:]) or fol.cmds[-1] != 0.054   # and went back to the target


def test_a_gripper_that_is_free_is_never_interrupted():
    leader = ScriptedLeader(np.zeros(6)); leader.g = 0.0305
    fol = StuckFollower(); fol.gripper_status = lambda: {"value_m": 0.054, "effort": -100, "enabled": True, "zeroed": False, "error": False}
    rig = LeaderFollower(leader, fol, leader_gripper_mm=(16.0, 105.5), gripper_dither_mm=0.0, unstick=True)
    rig.start(); rig.synced = True
    for _ in range(6):
        time.sleep(0.12)
        for _ in range(8):
            rig.step(0.005)
    assert rig.relaxes == 0


def test_holding_an_object_is_never_released_by_default():
    leader = ScriptedLeader(np.zeros(6)); leader.g = 0.0305            # operator over-closes the pendant
    fol = StuckFollower()                                               # follower wedged on an object at full effort
    rig = LeaderFollower(leader, fol, leader_gripper_mm=(16.0, 105.5), gripper_dither_mm=0.0)
    rig.start(); rig.synced = True
    for _ in range(8):
        time.sleep(0.12)
        for _ in range(8):
            rig.step(0.005)
    assert rig.relaxes == 0 and all(g < 0.02 for g in fol.cmds[-20:])      # still squeezing toward the target


def test_arms_are_chosen_by_physical_side_through_the_hardware_mapping():
    from piper_teleop import config as C
    raw = C.load()
    assert raw["hardware"]["can"] == {"left": "can_right", "right": "can_left"}        # the adapters are swapped vs their names
    assert C.can_for(raw, "left") == "can_right" and C.can_for(raw, "right") == "can_left"
    assert C.can_for(raw, "can_x") == "can_x"                                          # a literal CAN name passes through
    assert raw["leader_follower"]["leader_arm"] == "right" and raw["leader_follower"]["follower_arm"] == "left"
    assert raw["arms"]["left"]["can"] == "left" and raw["arms"]["right"]["can"] == "right"
    assert C.can_for({}, "left") == "can_left"                                          # no mapping = the natural names
    swapped_back = {"hardware": {"can": {"left": "can_left", "right": "can_right"}}}
    assert C.can_for(swapped_back, "left") == "can_left"                                # fixing the udev rule = changing the 2 lines


def test_the_backend_for_an_arm_gets_the_mapped_can_name():
    from piper_teleop import config as C
    raw = C.load()
    be = C.make_backend({"can": "left", "backend": "sim"}, raw, False)
    assert be.name == "can_right"                                                       # the physical left arm is on can_right


def test_record_config_names_the_cameras_after_the_arms_and_the_view():
    from piper_teleop import config as C
    cams = C.load()["record"]["cameras"]
    assert set(cams) == {"cam_high", "cam_left_wrist"}
    assert cams["cam_left_wrist"]["serial"] == "CC1WC5200WW" and cams["cam_high"]["serial"] == "CC1365303LL"


HOME = np.radians([0, 35, -34, -4, 37, 10])


def _homing_rig(leader_q, follower_q0=np.zeros(6), tol_deg=10.0):
    leader = ScriptedLeader(leader_q)
    fol = SimArm(q0=follower_q0, lag_s=0.0, max_speed=100); fol.enable()
    rig = LeaderFollower(leader, fol, max_joint_speed=3.0, sync_speed=0.5)
    rig.home_q, rig.home_speed, rig.home_tol = HOME.copy(), 0.5, np.radians(tol_deg)
    rig.start()
    return rig, leader, fol


def test_homing_drives_the_follower_home_at_the_capped_speed_then_waits_for_the_leader():
    far = np.radians([30, 60, -60, 20, 40, 50])                       # the leader is nowhere near home
    rig, leader, fol = _homing_rig(far)
    rig.go_home()
    assert rig.homing and not rig.synced
    prev, worst = rig.q_cmd.copy(), 0.0
    for _ in range(600):                                               # 3 s of virtual time at 200 Hz: home is 37 deg away = ~1.3 s
        rig.step(0.005)
        worst = max(worst, np.max(np.abs(rig.q_cmd - prev)) / 0.005)
        prev = rig.q_cmd.copy()
        assert not rig.last_tick_ok                                    # never a demonstration frame while homing
    assert worst <= 0.5 + 1e-6                                         # home_speed is respected
    assert np.allclose(rig.q_cmd, HOME, atol=1e-6) and rig.homing      # arrived, and holds: it does NOT chase the far leader
    assert "move the leader to its home pose" in rig.msg and not rig.synced


def test_homing_ends_when_the_leader_is_within_tolerance_and_following_resumes():
    rig, leader, fol = _homing_rig(HOME + np.radians([25, 0, 0, 0, 0, 0]))
    rig.go_home()
    for _ in range(400):
        rig.step(0.005)
    assert rig.homing and np.allclose(rig.q_cmd, HOME, atol=1e-6)
    leader.q = HOME + np.radians([6, -4, 3, 0, 0, 0])                  # the person brings it close enough (<= 10 deg)
    for _ in range(300):
        rig.step(0.005)
    assert not rig.homing and rig.synced and rig.last_tick_ok
    leader.q = HOME + np.radians([6, 15, 3, 0, 0, 0])                  # now it follows normally
    for _ in range(200):
        rig.step(0.005)
    assert np.allclose(rig.q_cmd, leader.q, atol=1e-3)


def test_a_silent_leader_never_ends_homing_and_the_gripper_is_left_alone():
    rig, leader, fol = _homing_rig(HOME)
    leader.fresh = False
    rig.go_home()
    g0 = rig.g_cmd
    for _ in range(400):
        rig.step(0.005)
    assert rig.homing and "silent" in rig.msg and rig.g_cmd == g0
    leader.fresh = True
    for _ in range(10):
        rig.step(0.005)
    assert not rig.homing


def test_homing_without_a_home_pose_does_nothing_and_the_sim_person_can_be_asked_to_arrive():
    rig, leader, fol = _homing_rig(HOME)
    rig.home_q = None
    rig.go_home()
    assert not rig.homing
    rig.home_q = HOME.copy()
    arrived = []
    rig.on_home_reached = lambda: arrived.append(1)
    rig.go_home()
    for _ in range(600):
        rig.step(0.005)
    assert arrived == [1]                                              # once per homing, not every tick
