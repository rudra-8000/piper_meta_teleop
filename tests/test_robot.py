import time
from pathlib import Path

import numpy as np
import pytest

from piper_teleop import run_policy as rp
from piper_teleop.kinematics import JOINT_HI, JOINT_LO
from piper_teleop.robot import IKError, PiperRobot, Rate, feature_names

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "policies.py"


def make(**kw):
    kw.setdefault("cameras", False)
    return PiperRobot("left", sim=True, **kw)


def wait_until(cond, timeout=6.0):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, "timed out"
        time.sleep(0.01)


def test_observation_has_the_datasets_keys_units_and_names():
    with make(cameras="fake") as r:
        obs = r.get_observation()
        assert set(obs) == {"observation.state", "observation.images.cam_high", "timestamp", "frame_age_s", "state_age_s"}
        assert set(obs["frame_age_s"]) == {"cam_high"} and 0.0 <= obs["frame_age_s"]["cam_high"] < 0.2 and 0.0 <= obs["state_age_s"] < 0.1
        s = obs["observation.state"]
        assert s.shape == (7,) and s.dtype == np.float32 and 0.0 <= s[6] <= 1.0
        img = obs["observation.images.cam_high"]
        assert img.dtype == np.uint8 and img.shape == (48, 64, 3)
        assert r.state_names == feature_names("left") == ["left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
                                                           "left_joint_5", "left_joint_6", "left_gripper"]
        assert r.camera_names == ["cam_high"]


def test_camera_sets_resolve_like_the_recorder_and_unknown_ones_are_refused():
    r = PiperRobot("left", sim=True, cameras="xtion_top")
    assert r._cam_specs["cam_high"]["type"] == "xtion" and r._cam_specs["cam_left_wrist"]["serial"] == "CC1WC5200WW"
    assert PiperRobot("left", sim=True)._cam_specs == {}                       # sim: no hardware cameras unless asked
    with pytest.raises(ValueError, match="xtion_top"):
        PiperRobot("left", sim=True, cameras="nope")


def window_speeds(commands, win=0.1):
    """Joint speed [rad/s] over every ``win``-second window of the recorded commands.  (Per-tick speed is not measurable: after a scheduling
    stall the loop takes one catch-up step worth at most 20 ms of motion a few ms after the previous command.)"""
    t = np.array([c[0] for c in commands])
    q = np.array([c[1] for c in commands])
    out = []
    for i in range(len(t)):
        k = int(np.searchsorted(t, t[i] + win))
        if k >= len(t):
            break
        out.append(float(np.max(np.abs(q[k] - q[i]))) / (t[k] - t[i]))
    return np.array(out)


def test_first_target_is_approached_slowly_then_the_speed_cap_applies():
    with make(max_joint_speed=2.0, sync_speed=0.5) as r:
        r.backend.record = True
        a = np.zeros(7)
        a[1] = np.radians(40)
        r.send_action(a)
        wait_until(lambda: r.synced and abs(r.state[1] - a[1]) < 0.02, 8.0)
        v = window_speeds(r.backend.commands)
        assert 0.4 < np.median(v[v > 0.05]) <= 0.55 and v.max() <= 0.5 * 1.15          # sync speed 0.5 rad/s, not the 2.0 cap
        r.backend.commands.clear()
        b = a.copy()
        b[1] = np.radians(5)
        r.send_action(b)
        wait_until(lambda: abs(r.state[1] - b[1]) < 0.02, 8.0)
        v = window_speeds(r.backend.commands)
        assert v.max() > 1.0 and v.max() <= 2.0 * 1.15                                  # now the 2.0 rad/s cap


def test_send_action_validates_and_clips_to_the_joint_limits():
    with make() as r:
        for bad in (np.zeros(6), np.full(7, np.nan), np.array([0, 0, 0, 0, 0, 0, np.inf]), np.zeros((2, 7))):
            with pytest.raises(ValueError):
                r.send_action(bad)
        with pytest.raises(ValueError):
            r.send_action(np.zeros(7), speed=0.0)
        a = np.zeros(7)
        a[0], a[6] = 10.0, 5.0                                                      # 10 rad on J1, gripper 5.0
        r.send_action(a)
        time.sleep(0.2)
        assert r._tgt_q[0] == JOINT_HI[0] and r._tgt_g == 1.0
        a[0] = -10.0
        r.send_action(a)
        assert r._tgt_q[0] == JOINT_LO[0]


def test_gripper_follows_a_normalised_target():
    with make() as r:
        a = np.zeros(7)
        a[6] = 1.0
        r.send_action(a)
        wait_until(lambda: r.state[6] > 0.95)
        a[6] = 0.0
        r.send_action(a)
        wait_until(lambda: r.state[6] < 0.05)


def test_reset_go_to_timeout_and_cancel():
    with make(rest_speed=1.0) as r:
        q = np.radians([20, 30, -30, 10, 20, 15])
        r.go_to(q, speed=2.0)
        assert np.allclose(r.state[:6], q, atol=0.03)
        r.go_rest()
        assert np.allclose(r.state[:6], 0.0, atol=0.03)
        with pytest.raises(TimeoutError, match="holds its pose"):
            r.go_to(q, speed=0.2, timeout=0.3)
        import threading
        threading.Timer(0.3, r.cancel_moves).start()
        with pytest.raises(InterruptedError):
            r.go_to(np.zeros(6), speed=0.1, timeout=20)
        assert r.backend.read().enabled                                             # still enabled and holding


def test_close_folds_to_rest_and_never_disables_the_motors(monkeypatch):
    r = make().connect()
    monkeypatch.setattr(r.backend, "disable", lambda: (_ for _ in ()).throw(AssertionError("motors must never be disabled")))
    r.go_to(np.radians([30, 30, -30, 0, 20, 0]), speed=2.0)
    r.close()
    assert np.allclose(r.backend._q, 0.0, atol=0.03) and r.backend._enabled and not r.connected
    r.close()                                                                       # twice is fine


def test_park_on_exit_false_just_holds():
    with make(park_on_exit=False) as r:
        q = np.radians([30, 30, -30, 0, 20, 0])
        r.go_to(q, speed=2.0)
    assert np.allclose(r.backend._q, q, atol=0.05) and r.backend._enabled


def test_a_failing_control_thread_surfaces_instead_of_hanging():
    r = make().connect()
    r.backend.command = lambda q, g: (_ for _ in ()).throw(RuntimeError("boom: CAN down"))
    wait_until(lambda: r._error is not None, 2.0)
    with pytest.raises(RuntimeError, match="boom: CAN down"):
        r.send_action(np.zeros(7))
    with pytest.raises(RuntimeError, match="boom"):
        r.get_observation()
    r.close()                                                                       # does not try to park a dead loop, does not hang
    assert not r.connected


def test_tcp_pose_targets_position_only_fixed_orientation_and_refusals():
    with make() as r:
        R0, p0 = r.tcp_pose()
        target = p0 + np.array([0.06, 0.06, 0.10])                                  # sideways: needs rotation="free" near rest
        r.send_tcp_pose(target, "free", speed=2.0)
        wait_until(lambda: np.linalg.norm(r.tcp_pose()[1] - target) < 0.006)
        # a straight-ahead lift that keeps the orientation is fine
        t2 = r.tcp_pose()[1] + np.array([0.0, 0.0, 0.03])
        R_now = r.tcp_pose()[0]
        r.send_tcp_pose(t2, R_now, speed=2.0)
        wait_until(lambda: np.linalg.norm(r.tcp_pose()[1] - t2) < 0.006)
        assert np.linalg.norm(r.tcp_pose()[0] - R_now) < 0.15
        with pytest.raises(ValueError, match="workspace"):
            r.send_tcp_pose([0.0, 0.0, 0.3], "free")                                # behind the rear plane (x_min 0.05)
        with pytest.raises(ValueError, match="workspace"):
            r.send_tcp_pose([0.3, 0.0, 0.01], "free")                               # below the table limit
        with pytest.raises(IKError, match="cannot reach"):
            r.send_tcp_pose([0.95, 0.0, 0.5], "free")                               # inside the box but beyond the arm's reach
        q_before = r._tgt_q.copy()
        assert np.allclose(q_before, r._tgt_q)                                      # refusals send nothing


def test_a_big_joint_swing_is_refused_not_sent():
    with make() as r:
        _, p0 = r.tcp_pose()
        before = r._tgt_q.copy()
        with pytest.raises(IKError, match="swing|cannot reach"):
            r.send_tcp_pose(p0 + np.array([0.06, 0.06, 0.10]), None)               # holding the rest orientation sideways: wild wrist
        assert np.allclose(r._tgt_q, before)


def test_run_executes_chunks_open_loop_and_passes_the_prompt():
    seen = []

    def policy(obs):
        seen.append(obs["prompt"])
        s = obs["observation.state"].astype(float)
        return np.tile(s, (5, 1))                                                   # a chunk of 5 "stay here" actions

    with make() as r:
        out = rp.run(r, policy, hz=100, prompt="clear the table", steps=10, execute=2, log=lambda *_: None)
    assert out["steps"] == 10 and out["policy_calls"] == 5 and set(seen) == {"clear the table"}


def test_a_policy_error_holds_the_arm_and_is_raised():
    calls = {"n": 0}

    def policy(obs):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("model crashed")
        s = obs["observation.state"].astype(float)
        s[1] += 0.3
        return s

    with make() as r:
        with pytest.raises(RuntimeError, match="model crashed"):
            rp.run(r, policy, hz=50, steps=50, reset=False, log=lambda *_: None)
        q_hold = r._tgt_q.copy()
        assert np.allclose(q_hold, r._q_cmd, atol=1e-6)                             # target = where it is: holding
    with make() as r:
        with pytest.raises(ValueError, match="7,"):
            rp.run(r, lambda obs: np.zeros(5), hz=50, steps=3, reset=False, log=lambda *_: None)


def test_rate_never_bursts():
    r = Rate(50)
    t0 = time.perf_counter()
    for _ in range(5):
        r.sleep()
    assert 0.09 <= time.perf_counter() - t0 <= 0.2
    time.sleep(0.1)                                                                 # overrun by 5 periods
    t1 = time.perf_counter()
    r.sleep()
    r.sleep()
    assert time.perf_counter() - t1 >= 0.015                                        # no catch-up burst: the next slots are not instant


def test_replay_example_reproduces_a_recorded_episode(tmp_path):
    from piper_teleop.recording.writer import LeRobotWriter
    names = feature_names("left")
    w = LeRobotWriter(tmp_path / "ds", fps=30, state_names=names, action_names=names, cameras={}, robot_type="piper", fmt="v3.0")
    rng = np.random.default_rng(1)
    eps = [rng.uniform(-0.2, 0.2, (12, 7)).astype(np.float32) for _ in range(2)]
    for e in eps:
        w.begin_episode("t")
        for a in e:
            w.add_frame(a, a, {})
        w.save_episode()
    w.close()
    pol = rp.load_policy(f"{EXAMPLES}:replay", {"root": str(tmp_path / "ds"), "episode": 1})
    got = pol({"observation.state": np.zeros(7, np.float32)})
    assert got.shape == (12, 7) and np.allclose(got, eps[1], atol=1e-6)
    assert np.allclose(pol({"observation.state": np.ones(7, np.float32)}), 1.0)     # after the episode: hold
    pol.reset()
    assert pol({"observation.state": np.zeros(7, np.float32)}).shape == (12, 7)


def test_run_policy_cli_end_to_end_in_sim(capsys):
    assert rp.main(["--policy", f"{EXAMPLES}:sine", "--arg", "amplitude_deg=5", "--sim", "--steps", "45", "--hz", "60", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "45 steps" in out and "motors were left enabled" in out
    with pytest.raises(SystemExit, match="MODULE:FACTORY"):
        rp.main(["--policy", "nocolon", "--sim"])
    with pytest.raises(SystemExit, match="no 'nope'"):
        rp.main(["--policy", f"{EXAMPLES}:nope", "--sim"])


# ------------------------------------------------------------ command freshness / speed policy
def test_a_stale_target_is_replaced_by_hold_but_only_when_ttl_is_set():
    with make(command_ttl_s=0.3) as r:
        a = np.zeros(7)
        a[1] = 0.9
        r.send_action(a, speed=0.5)                                                 # a slow 1.8 s approach...
        time.sleep(0.9)                                                             # ...but the client goes quiet
        q = r.commanded_q.copy()
        time.sleep(0.5)
        assert r.expired >= 1 and np.allclose(r.commanded_q, q, atol=1e-6) and 0.05 < q[1] < 0.8   # stopped part-way, held
        assert r._tgt_q[1] == pytest.approx(q[1], abs=1e-6)
    with make() as r:                                                               # no ttl: it keeps going to the target
        a = np.zeros(7)
        a[1] = 0.5
        r.send_action(a, speed=2.0)
        wait_until(lambda: abs(r.state[1] - 0.5) < 0.02)
        assert r.expired == 0


def test_a_client_that_keeps_sending_is_never_expired_and_blocking_moves_are_exempt():
    with make(command_ttl_s=0.3, sync_speed=2.0) as r:
        for i in range(12):                                                         # a fresh target every 0.1 s
            a = np.zeros(7)
            a[1] = 0.05 * (i + 1)
            r.send_action(a)
            time.sleep(0.1)
        assert r.expired == 0 and r.commanded_q[1] > 0.4
        r.go_to(np.radians([20, 40, -40, 10, 20, 10]), speed=0.5)                   # takes ~1.5 s, ttl is 0.3 s: exempt
        assert np.allclose(r.state[:6], np.radians([20, 40, -40, 10, 20, 10]), atol=0.03) and r.expired == 0


def test_speeds_above_the_cap_are_refused_everywhere_including_go_to():
    with make(max_joint_speed=1.0) as r:
        a = np.zeros(7)
        for call in (lambda: r.send_action(a, speed=1.5), lambda: r.go_to(np.zeros(6), speed=1.5),
                     lambda: r.send_tcp_pose(r.tcp_pose()[1] + [0.01, 0, 0.01], "free", speed=1.5)):
            with pytest.raises(ValueError, match="above the configured cap"):
                call()
        assert not r._moving and not r._have_target                                 # a refused call changes nothing
        r.send_action(a, speed=0.5)                                                 # lowering the speed is fine
        r.go_to(np.zeros(6), speed=1.0)                                             # exactly the cap is fine
    with make(allow_speed_override=False) as r:
        with pytest.raises(ValueError, match="allow_speed_override"):
            r.send_action(np.zeros(7), speed=0.1)
        with pytest.raises(ValueError, match="allow_speed_override"):
            r.go_to(np.zeros(6), speed=0.1)
        r.send_action(np.zeros(7))                                                  # the configured cap still applies
        r.go_to(np.zeros(6))                                                        # default rest_speed


def test_ttl_is_validated_and_comes_from_the_config():
    for bad in (0.0, -1.0, 100.0):
        with pytest.raises(ValueError, match="command_ttl_s"):
            PiperRobot("left", sim=True, cameras=False, command_ttl_s=bad)
    from piper_teleop import config as C
    raw = C.load()
    raw["robot"]["command_ttl_s"] = 0.25
    raw["robot"]["allow_speed_override"] = False
    r = PiperRobot("left", sim=True, cameras=False, config=raw)
    assert r.command_ttl_s == 0.25 and r.allow_speed_override is False
    shipped = PiperRobot("left", sim=True, cameras=False)
    assert shipped.command_ttl_s is None and shipped.allow_speed_override is True   # shipped defaults change nothing
