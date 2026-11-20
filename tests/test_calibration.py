import math

import numpy as np
import pytest

from piper_teleop.arm import SimArm
from piper_teleop.calibration import GuidedCalibration, azimuth_deg, wrap_deg
from piper_teleop.controller import ArmConfig, HandSample, TeleopArm, ENGAGED, IDLE
from piper_teleop.system import TeleopSystem
from piper_teleop.kinematics import mat_to_quat, so3_exp


def fwd(phi_deg):            # tracking-frame forward vector of an operator whose forward has azimuth phi
    a = math.radians(phi_deg)
    return np.array([math.sin(a), 0.0, -math.cos(a)])


def right(phi_deg):          # operator's right: azimuth phi + 90
    return fwd(phi_deg + 90.0)


UP = np.array([0.0, 1.0, 0.0])


class Operator:
    """Drives a GuidedCalibration with synthetic hand data at 90 Hz."""

    def __init__(self, cal, t0=100.0):
        self.cal, self.t = cal, t0
        self.pos = {"left": np.array([-0.2, 1.1, -0.3]), "right": np.array([0.2, 1.1, -0.3])}
        self.events = []
        self.noise = 0.0
        self.rng = np.random.default_rng(1)

    def tick(self, grips, dt=1 / 90):
        self.t += dt
        samples = {h: HandSample(t_recv=self.t, seq=0, pos=p + self.rng.normal(0, self.noise, 3) if self.noise else p.copy(),
                                 squeeze=1.0 if h in grips else 0.0)
                   for h, p in self.pos.items()}
        self.events += self.cal.feed(samples, self.t)

    def idle(self, secs, grips=()):
        for _ in range(int(secs * 90)):
            self.tick(grips)

    def move(self, direction, dist=0.2, hands=("left", "right"), wobble=0.0, secs=0.7, arc=0.0):
        """wobble: tracking jitter (m, measurement noise).  arc: sideways bulge of a human arm path (m)."""
        self.idle(0.3)                                   # grips released, then squeeze
        self.noise = wobble
        self.tick(hands)
        n = int(secs * 90)
        side = np.cross(direction, UP) if abs(direction[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        for k in range(n):
            bump = arc * (math.sin(math.pi * (k + 1) / n) - math.sin(math.pi * k / n))
            for h in hands:
                self.pos[h] = self.pos[h] + direction * dist / n + side * bump
            self.tick(hands)
        self.noise = 0.0
        self.idle(0.3)                                   # release


def run_calibration(phi, **kw):
    cal = GuidedCalibration()
    op = Operator(cal)
    op.events += cal.start(op.t)
    op.idle(1.5)
    op.move(fwd(phi)); op.idle(1.2)
    op.move(right(phi)); op.idle(1.2)
    op.move(UP)
    return cal, op


@pytest.mark.parametrize("phi", [0, 30, -75, 120, 180, -150])
def test_heading_is_recovered_for_any_standing_direction(phi):
    cal, op = run_calibration(phi)
    assert cal.status == "done", cal.msg
    assert wrap_deg(cal.result["yaw_deg"] - phi) == pytest.approx(0, abs=1.0)
    assert cal.result["axis_error_deg"] < 2 and cal.result["up_tilt_deg"] < 2


def test_noisy_hand_motion_still_calibrates():
    cal = GuidedCalibration(); op = Operator(cal)
    op.events += cal.start(op.t); op.idle(1.5)
    for d in (fwd(40), right(40)):
        op.move(d, wobble=0.002); op.idle(1.2)          # 2 mm tracking jitter
    op.move(UP, wobble=0.002)
    assert cal.status == "done" and wrap_deg(cal.result["yaw_deg"] - 40) == pytest.approx(0, abs=2)


def test_a_naturally_curved_human_arm_path_still_gives_the_right_heading():
    cal = GuidedCalibration(); op = Operator(cal)
    op.events += cal.start(op.t); op.idle(1.5)
    for d in (fwd(-20), right(-20)):
        op.move(d, arc=0.02); op.idle(1.2)              # 2 cm bulge on a 20 cm move (a typical arm arc)
    op.move(UP, arc=0.02)
    assert cal.status == "done" and wrap_deg(cal.result["yaw_deg"] + 20) == pytest.approx(0, abs=3)


def test_one_hand_is_enough():
    cal = GuidedCalibration(); op = Operator(cal)
    op.events += cal.start(op.t); op.idle(1.5)
    for d in (fwd(10), right(10)):
        op.move(d, hands=("right",)); op.idle(1.2)
    op.move(UP, hands=("right",))
    assert cal.status == "done"


def test_short_move_is_rejected_with_the_distance_and_can_be_retried():
    cal = GuidedCalibration(); op = Operator(cal)
    op.events += cal.start(op.t); op.idle(1.5)
    op.move(fwd(0), dist=0.05)
    assert cal.step == 0 and cal.active and "5 centimeters" in cal.msg
    assert any(e[0] == "haptic" and e[2] == 1.0 for e in op.events)          # strong buzz = rejected
    op.idle(1.2); op.move(fwd(0))
    assert cal.step == 1                                                      # now accepted


def test_curved_path_is_rejected():
    cal = GuidedCalibration(); op = Operator(cal)
    op.events += cal.start(op.t); op.idle(1.5)
    op.idle(0.3); op.tick(("left", "right"))
    for k in range(90):                                    # out 20 cm and back 15 cm: path 35, displacement 5... then sideways
        for h in ("left", "right"):
            op.pos[h] = op.pos[h] + np.array([0.0, 0.0, -0.4 / 90 if k < 45 else 0.3 / 90])
        op.tick(("left", "right"))
    op.idle(0.3)
    assert cal.step == 0 and cal.active


def test_moving_left_instead_of_right_is_detected():
    cal = GuidedCalibration(); op = Operator(cal)
    op.events += cal.start(op.t); op.idle(1.5)
    op.move(fwd(0)); op.idle(1.2)
    op.move(-right(0)); op.idle(1.2)                       # LEFT by mistake
    op.move(UP)
    assert cal.status == "running" and cal.step == 0 and "left instead of right" in cal.msg   # restarted from forward


def test_non_perpendicular_axes_are_detected_and_three_failures_give_up():
    cal = GuidedCalibration(); op = Operator(cal)
    op.events += cal.start(op.t); op.idle(1.5)
    for attempt in range(3):
        op.move(fwd(0)); op.idle(1.2)
        op.move(fwd(20)); op.idle(1.2)                     # "right" only 20 deg from forward
        op.move(UP); op.idle(1.5)
    assert cal.status == "failed" and "failed three times" in cal.msg


def test_hands_moving_in_different_directions_are_rejected():
    cal = GuidedCalibration(); op = Operator(cal)
    op.events += cal.start(op.t); op.idle(1.5)
    op.idle(0.3); op.tick(("left", "right"))
    for _ in range(63):
        op.pos["left"] = op.pos["left"] + fwd(0) * 0.2 / 63
        op.pos["right"] = op.pos["right"] + right(0) * 0.2 / 63
        op.tick(("left", "right"))
    op.idle(0.3)
    assert cal.step == 0 and "different directions" in cal.msg


def test_grip_still_held_from_before_does_not_count_as_a_move():
    cal = GuidedCalibration(); op = Operator(cal)
    op.events += cal.start(op.t)
    op.idle(2.0, grips=("left", "right"))                  # operator was already holding grips
    assert cal.phase == "intro" and cal.step == 0          # waits until they are released


def test_timeout_and_cancel():
    cal = GuidedCalibration(step_timeout=3.0); op = Operator(cal)
    op.events += cal.start(op.t); op.idle(4.0)
    assert cal.status == "failed" and "timed out" in cal.msg
    cal.start(op.t); ev = cal.cancel()
    assert not cal.active and ev and ev[0][0] == "speak"


def test_prompts_are_spoken_in_order():
    cal, op = run_calibration(0)
    spoken = [e[1] for e in op.events if e[0] == "speak"]
    assert "forward" in spoken[0].lower() and any("right" in t.lower() for t in spoken) and any("up" in t.lower() for t in spoken)
    assert "complete" in spoken[-1].lower()


# ------------------------------------------------------------- system integration
HOME = np.radians(ArmConfig().home_deg)


def make_system(auto=False):
    arms = []
    for name, yaw in (("left", 0.0), ("right", 0.0)):
        be = SimArm(q0=HOME, lag_s=0.0, max_speed=100); be.enable()
        arms.append(TeleopArm(ArmConfig(name=name, hand=name, base_yaw_deg=yaw), be))
    ev = []
    sys_ = TeleopSystem(arms, rate_hz=200, on_event=lambda *a: ev.append(a), auto_calibrate=auto)
    for a in arms:
        a.start()
    return sys_, arms, ev


def feed_system(sys_, t, hands, grips=()):
    for h, p in hands.items():
        sys_.submit(h, HandSample(t_recv=t, seq=int(t * 1000), pos=p.copy(), squeeze=1.0 if h in grips else 0.0))
    sys_._tick(t, 0.005)


def test_arms_hold_still_during_calibration_and_need_a_fresh_grab_afterwards():
    sys_, arms, ev = make_system()
    pos = {"left": np.array([-0.2, 1.1, -0.3]), "right": np.array([0.2, 1.1, -0.3])}
    t = 100.0
    for _ in range(40):                                   # released samples, so the arms are armed
        t += 0.011; feed_system(sys_, t, pos)
    sys_.request("calibrate")
    q0 = [a.q_cmd.copy() for a in arms]
    for _ in range(60):
        t += 0.011; feed_system(sys_, t, pos)
    assert sys_.calib.active
    # operator squeezes and pushes forward during calibration: arms must not move
    for k in range(120):
        t += 0.011
        for h in pos: pos[h] = pos[h] + np.array([0, 0, -0.002])
        feed_system(sys_, t, pos, grips=("left", "right"))
    assert all(a.state != ENGAGED for a in arms)
    assert all(np.allclose(a.q_cmd, q, atol=1e-6) for a, q in zip(arms, q0))
    assert any(e[0] == "speak" for e in ev)


def test_completed_calibration_sets_the_heading_for_both_arms():
    sys_, arms, ev = make_system()
    phi = 55.0
    pos = {"left": np.array([-0.2, 1.1, -0.3]), "right": np.array([0.2, 1.1, -0.3])}
    t = [100.0]

    def step(grips, n=1):
        for _ in range(n):
            t[0] += 1 / 90; feed_system(sys_, t[0], pos, grips)

    step((), 5); sys_.request("calibrate"); step((), 120)
    for d in (fwd(phi), right(phi), UP):
        step((), 30); step(("left", "right"))
        for _ in range(63):
            for h in pos: pos[h] = pos[h] + d * 0.2 / 63
            step(("left", "right"))
        step((), 120)
    assert sys_.calibrated and sys_.calib.status == "done", sys_.calib.msg
    assert wrap_deg(sys_.frame_yaw_deg - phi) == pytest.approx(0, abs=1.0)
    # an arm with base_yaw 0 now maps "push forward" (in the operator's real forward) to its own +x
    arm = arms[0]
    p0 = arm.kin.fk_tcp(arm.q_cmd)[1].copy()
    step((), 20)
    step(("left",))
    for _ in range(40):
        pos["left"] = pos["left"] + fwd(phi) * 0.002
        step(("left",))
    step(("left",), 40)
    d = arm.kin.fk_tcp(arm.q_cmd)[1] - p0
    assert d[0] > 0.06 and abs(d[1]) < 0.02


def test_auto_start_after_both_hands_seen_and_both_sticks_cancels():
    sys_, arms, ev = make_system(auto=True)
    pos = {"left": np.array([-0.2, 1.1, -0.3]), "right": np.array([0.2, 1.1, -0.3])}
    t = 100.0
    for _ in range(170):                                   # 1.9 s with both hands tracked (needs > 1.5 s)
        t += 0.011; feed_system(sys_, t, pos)
    assert sys_.calib.active                               # started by itself
    for h in pos:                                          # both thumbsticks: cancel
        sys_.submit(h, HandSample(t_recv=t + 0.011, pos=pos[h].copy(), stick_click=True))
    sys_._tick(t + 0.011, 0.005)
    assert not sys_.calib.active
    for _ in range(300):                                   # and it must not restart by itself
        t += 0.011; feed_system(sys_, t, pos)
    assert not sys_.calib.active


# ---------------------------------------------------------------- orientation mode
def _sweep(mode):
    from piper_teleop.controller import yaw_matrix  # noqa
    cfg = ArmConfig(name="t", hand="right", orientation_mode=mode)
    be = SimArm(q0=HOME, lag_s=0.0, max_speed=100); be.enable()
    arm = TeleopArm(cfg, be); arm.start()
    t = 100.0
    pos = np.array([0.2, 1.1, -0.3])
    for i in range(10):
        t += 0.01; arm.step(HandSample(t_recv=t, seq=i, pos=pos.copy(), squeeze=0.0), t, 0.01)
    q0 = arm.q_cmd.copy()
    for i in range(300):
        t += 0.005
        if i < 150: pos = pos + np.array([-0.001, 0, 0])             # hand to the operator's LEFT: arm's sideways axis
        arm.step(HandSample(t_recv=t, seq=100 + i // 2, pos=pos.copy(), squeeze=1.0), t, 0.005)
    return np.degrees(np.abs(arm.q_cmd - q0))


def test_follow_base_needs_far_less_wrist_motion_for_sideways_moves():
    world, follow = _sweep("world"), _sweep("follow_base")
    assert follow[0] > 5                                   # base joint does the sweeping
    assert follow[3] + follow[5] < 10                      # wrist barely moves
    assert world[3] + world[5] > 3 * (follow[3] + follow[5])


def test_arms_stay_frozen_until_calibrated_or_skipped_when_auto_calibration_is_on():
    sys_, arms, ev = make_system(auto=True)
    pos = {"left": np.array([-0.2, 1.1, -0.3]), "right": np.array([0.2, 1.1, -0.3])}
    t = 100.0
    for _ in range(20):                                    # operator grabs right away, before calibration could start
        t += 0.011; feed_system(sys_, t, pos)
    q0 = arms[0].q_cmd.copy()
    for _ in range(40):
        t += 0.011
        for h in pos: pos[h] = pos[h] + np.array([0, 0, -0.002])
        feed_system(sys_, t, pos, grips=("left", "right"))
    assert all(a.state != ENGAGED for a in arms) and np.allclose(arms[0].q_cmd, q0)
    assert "waiting for axis calibration" in arms[0].msg or "calibration" in arms[0].msg
    assert sys_.snapshot()["calibration_required"] is True
    sys_.request("calibrate_cancel")                       # deliberate skip
    for _ in range(5):
        t += 0.011; feed_system(sys_, t, pos)
    for _ in range(5):                                     # release, then a fresh grab is allowed to engage
        t += 0.011; feed_system(sys_, t, pos)
    for _ in range(40):
        t += 0.011
        for h in pos: pos[h] = pos[h] + np.array([0, 0, -0.002])
        feed_system(sys_, t, pos, grips=("left",))
    assert arms[0].state == ENGAGED
    assert sys_.snapshot()["calibration_required"] is False


def test_both_sticks_recalibrate_every_time_from_where_the_controllers_point():
    sys_, arms, ev = make_system(auto=False)
    pos = {"left": np.array([-0.2, 1.1, -0.3]), "right": np.array([0.2, 1.1, -0.3])}
    t = 100.0
    yaws = []
    for ang in (0.0, 40.0, -25.0):                         # face a different way each time and press both sticks again
        q = mat_to_quat(so3_exp(np.array([0.0, math.radians(ang), 0.0])))
        t += 0.5
        for h in pos:                                      # sticks released, then pressed together
            sys_.submit(h, HandSample(t_recv=t, pos=pos[h].copy(), quat=q, ray_quat=q, stick_click=False))
        sys_._tick(t, 0.005)
        t += 0.011
        for h in pos:
            sys_.submit(h, HandSample(t_recv=t, pos=pos[h].copy(), quat=q, ray_quat=q, stick_click=True))
        sys_._tick(t, 0.005)
        assert sys_.calibrated
        yaws.append(sys_.frame_yaw_deg)
    assert abs(yaws[0]) < 1.0 and abs(yaws[1] - yaws[0]) > 30 and abs(yaws[2] - yaws[1]) > 30, yaws
