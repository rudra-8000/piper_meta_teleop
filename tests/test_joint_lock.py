import datetime as dt
from types import SimpleNamespace as NS

import numpy as np
import pytest

from piper_teleop import joint_lock as JL
from piper_teleop.arm import PiperArm, SimArm
from piper_teleop.agx_arm import AgxPiperArm
from piper_teleop.controller import ArmConfig, HandSample, TeleopArm
from test_agx_backend import FakeEff, FakeRobot, RAISED
from test_core import FakeIface

NOW = dt.datetime(2026, 10, 27, 22, 0)


@pytest.fixture()
def lockfile(tmp_path, monkeypatch):
    f = tmp_path / "joint_lock.yaml"
    f.write_text('can_x:\n  until: "2099-01-01 00:00"\n')     # far future: guards read the real clock
    monkeypatch.setattr(JL, "LOCK_FILE", f)
    return f


def test_lock_window_and_allowed_frames(lockfile):
    lockfile.write_text('can_x:\n  until: "2026-10-28 13:00"\n')                  # exact expiry, explicit clock
    assert JL.lock_until("can_x", NOW) == dt.datetime(2026, 10, 28, 13, 0)
    assert JL.lock_until("can_x", dt.datetime(2026, 10, 28, 13, 0)) is None          # expires exactly at 13:00
    assert JL.lock_until("can_other", NOW) is None
    for ok in (0x159, 0x47D, 0x477, 0x472, 0x4AF):
        JL.check("can_x", ok, NOW)
    for bad in (0x150, 0x151, 0x152, 0x155, 0x156, 0x157, 0x15A, 0x470, 0x471, 0x475, 0x191):
        with pytest.raises(JL.JointLockError):
            JL.check("can_x", bad, NOW)


def test_the_real_lock_file_locks_can_left_until_1pm_tomorrow():
    assert JL._load()["can_left"]["until"] == "2026-10-28 13:00"


def test_piper_sdk_sender_is_wrapped_and_blocks_joint_frames(lockfile):
    sent = []
    iface = NS(_arm_can=NS(SendCanMessage=lambda cid, d: sent.append(cid)))
    JL.guard_piper_sdk(iface, "can_x")
    iface._arm_can.SendCanMessage(0x159, b"\0" * 8)
    with pytest.raises(JL.JointLockError):
        iface._arm_can.SendCanMessage(0x155, b"\0" * 8)
    assert sent == [0x159]


def test_pyagxarm_comm_is_wrapped_and_blocks_joint_frames(lockfile):
    sent = []
    comm = NS(send=lambda m: sent.append(m.arbitration_id))
    robot = NS(_ctx=NS(get_comm=lambda: comm))
    JL.guard_pyagxarm(robot, "can_x")
    comm.send(NS(arbitration_id=0x159))
    with pytest.raises(JL.JointLockError):
        comm.send(NS(arbitration_id=0x151))
    assert sent == [0x159]


def test_locked_backends_enable_gripper_only_and_never_send_joints(lockfile):
    r = FakeRobot(q=RAISED, mode=0, enabled=False)
    a = AgxPiperArm("can_x", robot=r, effector=FakeEff())
    a.connect()
    assert a.enable() and "JOINT LOCK" in a.notes[0]
    a.command(np.ones(6), 0.03)
    assert r.calls == ["connect"] and a._eff.calls == [(0.03, 1.0)]       # no enable/mode/move_j, gripper yes
    f = FakeIface()
    p = PiperArm("can_x", iface=f)
    p.connect()
    assert p.enable() and "JOINT LOCK" in p.notes[0]
    p.command(np.ones(6), 0.03)
    assert [c[0] for c in f.sent] == ["GC"]


def test_teleop_on_a_locked_arm_keeps_the_joint_target_on_the_real_pose(lockfile):
    be = SimArm(name="can_x", q0=np.radians([0, 35, -34, -4, 37, 10]), lag_s=0.0, max_speed=100)
    be.enable()
    be.joint_locked = dt.datetime(2026, 10, 28, 13, 0)
    be.record = True
    arm = TeleopArm(ArmConfig(name="t", hand="right"), be); arm.start()
    q_real = be._q.copy()
    t, pos = 100.0, np.array([0.2, 1.1, -0.3])
    for i in range(400):
        t += 0.005
        if i > 20:
            pos = pos + np.array([0, 0, -0.001])
        arm.step(HandSample(t_recv=t, seq=i, pos=pos.copy(), squeeze=0.0 if i < 20 else 1.0, trigger=0.5), t, 0.005)
    assert all(np.allclose(c[1], q_real) for c in be.commands)           # joint target never left the real pose
    assert be.commands[-1][2] == pytest.approx(0.035)                    # gripper still follows the trigger


def test_mode_frame_exception_needs_flag_proof_of_disabled_joints_and_a_safe_mode(tmp_path, monkeypatch):
    f = tmp_path / "joint_lock.yaml"
    f.write_text('can_x:\n  until: "2026-10-28 13:00"\n  allow_mode_frame_when_joints_disabled: true\n')
    monkeypatch.setattr(JL, "LOCK_FILE", f)
    can_mode = bytes([0x01, 0x01, 10, 0, 0, 0, 0, 0])
    JL.check("can_x", 0x151, NOW, data=can_mode, joints_disabled=True)                    # allowed
    for kw in (dict(data=can_mode, joints_disabled=None), dict(data=can_mode, joints_disabled=False),
               dict(data=None, joints_disabled=True), dict(data=bytes([0x02]) + can_mode[1:], joints_disabled=True)):
        with pytest.raises(JL.JointLockError):
            JL.check("can_x", 0x151, NOW, **kw)
    with pytest.raises(JL.JointLockError):                                                   # other frames unaffected
        JL.check("can_x", 0x471, NOW, data=can_mode, joints_disabled=True)
    f.write_text('can_x:\n  until: "2026-10-28 13:00"\n')                                  # without the flag: blocked
    with pytest.raises(JL.JointLockError):
        JL.check("can_x", 0x151, NOW, data=can_mode, joints_disabled=True)


def test_library_wrappers_never_get_the_exception(lockfile):
    lockfile.write_text('can_x:\n  until: "2099-01-01 00:00"\n  allow_mode_frame_when_joints_disabled: true\n')
    iface = NS(_arm_can=NS(SendCanMessage=lambda cid, d: None))
    JL.guard_piper_sdk(iface, "can_x")
    with pytest.raises(JL.JointLockError):
        iface._arm_can.SendCanMessage(0x151, bytes([1, 1, 10, 0, 0, 0, 0, 0]))
