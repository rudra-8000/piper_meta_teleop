import json
import time

import numpy as np
import pyarrow.parquet as pq
import pytest

from piper_teleop import record
from piper_teleop.arm import ArmState


class ScriptedKeys:
    """Stand-in for the terminal: (seconds since start, key) pairs."""
    script: list = []

    def __init__(self, *a, **k):
        self.t0 = None
        self.pending = list(self.script)

    def start(self):
        self.t0 = time.monotonic()
        return True

    def poll(self):
        out = []
        while self.pending and time.monotonic() - self.t0 >= self.pending[0][0]:
            out.append(self.pending.pop(0)[1])
        return out

    def stop(self):
        pass


@pytest.mark.parametrize("fmt", ["v3.0", "v2.1"])
def test_sim_session_end_to_end(tmp_path, monkeypatch, fmt):
    monkeypatch.setenv("PIPER_TELEOP_LOG_DIR", str(tmp_path / "logs"))
    T = 2.6                                                                   # the simulated follower needs ~1.5 s to align first
    ScriptedKeys.script = [(0.3, "\n"),                                       # too early: refused, nothing starts
                           (T, "\n"), (T + 1.0, "n"), (T + 1.1, "k"),         # episode 0: ~1 s
                           (T + 1.4, "\n"), (T + 1.9, "p"), (T + 2.5, "p"), (T + 3.0, "n"), (T + 3.1, "r"),   # paused 0.6 s, retaken
                           (T + 3.4, "\n"), (T + 4.1, "n"), (T + 4.2, "k")]  # retake kept -> --num-episodes 2 reached
    monkeypatch.setattr(record, "KeyReader", ScriptedKeys)
    root = tmp_path / "ds"
    assert record.main(["--sim", "--yes", "--fake-camera", "--root", str(root), "--format", fmt, "--task", "wave",
                        "--num-episodes", "2"]) == 0
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == fmt and info["total_episodes"] == 2 and info["fps"] == 30
    assert 40 <= info["total_frames"] <= 90
    feats = info["features"]
    assert feats["observation.state"]["shape"] == [7] and feats["action"]["names"] == record.feature_names("left") and feats["observation.state"]["names"][0] == "left_joint_1"
    assert feats["observation.images.fake"]["shape"] == [48, 64, 3]
    f = root / ("data/chunk-000/file-000.parquet" if fmt == "v3.0" else "data/chunk-000/episode_000000.parquet")
    t = pq.read_table(f).to_pydict()
    st, ac = np.array(t["observation.state"]), np.array(t["action"])
    assert st.shape[1] == 7 and np.all((st[:, 6] >= 0) & (st[:, 6] <= 1)) and np.all((ac[:, 6] >= 0) & (ac[:, 6] <= 1))
    assert np.abs(ac[:, :6] - st[:, :6]).max() < 0.5                          # the follower tracks the leader
    tasks = json.loads((root / "meta/tasks.jsonl").read_text().splitlines()[0])["task"] if fmt == "v2.1" else \
        pq.read_table(root / "meta/tasks.parquet").to_pylist()[0]["task"]
    assert tasks == "wave"


def test_a_second_run_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPER_TELEOP_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(record, "KeyReader", ScriptedKeys)
    root = tmp_path / "ds"
    for _ in range(2):
        ScriptedKeys.script = [(2.6, "\n"), (3.2, "n"), (3.3, "k")]
        record.main(["--sim", "--yes", "--no-cameras", "--root", str(root), "--num-episodes", "1"])
    assert json.loads((root / "meta/info.json").read_text())["total_episodes"] == 2


def test_arm_to_arm_source_gives_none_while_the_leader_is_silent_and_normalises_the_gripper():
    class Rig:
        synced = True
        last_tick_ok = True
        last_goal = np.arange(6) * 0.1
        last_g_target = 0.035
        last_follower = ArmState(np.ones(6), 0.07, True, True, "")

    s = record.ArmToArmSource(Rig())
    state, action = s.sample()
    assert state.dtype == np.float32 and state.shape == (7,) and state[6] == pytest.approx(1.0)
    assert action[6] == pytest.approx(0.5) and np.allclose(action[:6], Rig.last_goal)
    Rig.last_tick_ok = False
    assert s.sample() is None
    Rig.last_tick_ok, Rig.synced = True, False                                  # still aligning to the leader
    assert s.sample() is None
    Rig.synced = True
    Rig.last_tick_ok = True
    Rig.last_follower = ArmState(np.ones(6), 0.07, False, True, "")
    assert s.sample() is None


def test_names_follow_the_physical_follower_arm():
    assert record.feature_names("left") == ["left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5",
                                            "left_joint_6", "left_gripper"]
    assert record.feature_names("right")[-1] == "right_gripper"


def test_a_session_leaves_a_record_of_how_it_was_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPER_TELEOP_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(record, "KeyReader", ScriptedKeys)
    root = tmp_path / "ds"
    for task in ("first", "second"):
        ScriptedKeys.script = [(2.6, "\n"), (3.2, "n"), (3.3, "k")]
        record.main(["--sim", "--yes", "--fake-camera", "--root", str(root), "--task", task, "--num-episodes", "1"])
    setup = json.loads((root / "meta/recording_setup.json").read_text())["sessions"]
    assert [s["task"] for s in setup] == ["first", "second"] and setup[0]["episodes_kept"] == 1
    assert setup[0]["follower_arm"] == "left" and setup[0]["leader_arm"] == "right" and setup[0]["fps"] == 30
    assert setup[0]["cameras"]["fake"]["type"] == "fake" and setup[0]["video"]["crf"] == 20


def test_orbbec_camera_settings_come_from_the_config():
    from piper_teleop import config as C
    from piper_teleop.recording.cameras import make_camera
    cams = C.load()["record"]["cameras"]
    cam = make_camera("cam_high", cams["cam_high"])
    assert cam.settings == {"exposure": 200, "gain": 40} and cam.serial == "CC1365303LL"
    assert make_camera("w", cams["cam_left_wrist"]).settings == {"exposure": 200, "gain": 50}


def test_rgb_gain_turns_a_measured_white_card_neutral_and_reaches_the_helper():
    from piper_teleop.recording.cameras import OrbbecCamera, suggest_rgb_gain
    g = suggest_rgb_gain((200.0, 180.0, 160.0))                    # a reddish, yellowish card
    assert g == [0.9, 1.0, 1.125]
    assert np.allclose(np.array([200.0, 180.0, 160.0]) * g, 180.0)
    with pytest.raises(ValueError):
        suggest_rgb_gain((0.0, 3.0, 3.0))
    import subprocess
    cmds = []

    class Boom(Exception):
        pass

    def fake_popen(cmd, **kw):
        cmds.append(cmd)
        raise Boom

    cam = OrbbecCamera("c", "S1", settings={"exposure": 200, "gain": 40, "rgb_gain": [1.25, 1.0, 0.9]})
    orig = subprocess.Popen
    subprocess.Popen = fake_popen
    try:
        with pytest.raises(Boom):
            cam.connect()
    finally:
        subprocess.Popen = orig
        cam.disconnect()
    c = cmds[0]
    assert c[c.index("--rgb-gain") + 1] == "1.2500,1.0000,0.9000"
    assert json.loads(c[c.index("--settings") + 1]) == {"exposure": 200, "gain": 40}      # the software gain is not an SDK setting


def test_xtion_top_camera_set_keeps_the_names_and_the_left_wrist_unchanged():
    from piper_teleop import config as C
    from piper_teleop.recording.cameras import OrbbecCamera, XtionCamera, make_camera
    rc = C.load()["record"]
    a = record.parse(["--root", "x", "--cameras", "xtion_top"])
    specs = record.camera_specs(rc, a)
    assert list(specs) == list(rc["cameras"]) == ["cam_high", "cam_left_wrist"]          # same feature keys as the default set
    assert specs["cam_left_wrist"] == rc["cameras"]["cam_left_wrist"]                      # wrist camera untouched
    high = make_camera("cam_high", specs["cam_high"])
    assert isinstance(high, XtionCamera) and (high.width, high.height, high.fps) == (640, 480, 30) and high.serial is None
    assert isinstance(make_camera("w", specs["cam_left_wrist"]), OrbbecCamera) and not isinstance(make_camera("w", specs["cam_left_wrist"]), XtionCamera)
    default = record.camera_specs(rc, record.parse(["--root", "x"]))
    assert default["cam_high"]["serial"] == "CC1365303LL"                                  # the default set is unchanged


def test_unknown_camera_set_is_refused_with_the_choices():
    from piper_teleop import config as C
    with pytest.raises(SystemExit, match="xtion_top"):
        record.camera_specs(C.load()["record"], record.parse(["--root", "x", "--cameras", "nope"]))


def test_xtion_helper_command_line(monkeypatch):
    from piper_teleop.recording import cameras as cm
    seen = {}

    import io

    class P:
        stderr = io.StringIO("no ASUS Xtion found (test)\n")
        def poll(self): return 1
    monkeypatch.setattr(cm.subprocess, "Popen", lambda cmd, **k: seen.update(cmd=cmd, env=k["env"]) or P())
    cam = cm.make_camera("cam_high", {"type": "xtion", "width": 640, "height": 480, "fps": 30, "settings": {"gain": 20, "rgb_gain": [1.1, 1, 0.9]}})
    with pytest.raises(RuntimeError, match="ASUS Xtion"):
        cam.connect()
    cmd = seen["cmd"]
    assert cmd[1].endswith("tools/xtion_stream.py") and cmd[3:6] == ["640", "480", "30"] and "--serial" not in cmd
    assert cmd[cmd.index("--settings") + 1] == '{"gain": 20}' and cmd[cmd.index("--rgb-gain") + 1] == "1.1000,1.0000,0.9000"
    assert "orbbec_v1_legacy" not in seen["env"].get("PYTHONPATH", "")


def test_session_goes_home_before_every_episode_and_not_with_no_home(tmp_path, monkeypatch):
    """Homing is called on entering WAIT (first episode, after keep, after retake); --no-home never calls it."""
    from piper_teleop import leader_follower as lf
    calls = []
    orig = lf.LeaderFollower.go_home
    monkeypatch.setattr(lf.LeaderFollower, "go_home", lambda self: (calls.append(1), orig(self))[1])
    monkeypatch.setenv("PIPER_TELEOP_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(record, "KeyReader", ScriptedKeys)
    ScriptedKeys.script = [(3.2, "\n"), (3.8, "n"), (3.9, "k"),          # episode 0 kept   -> homing #2
                           (4.2, "\n"), (4.8, "n"), (4.9, "r"),          # episode 1 retaken -> homing #3
                           (5.2, "\n"), (5.8, "n"), (5.9, "k")]          # episode 1 kept, num-episodes 2 reached
    assert record.main(["--sim", "--yes", "--fake-camera", "--root", str(tmp_path / "a"), "--num-episodes", "2"]) == 0
    assert json.loads((tmp_path / "a/meta/info.json").read_text())["total_episodes"] == 2
    assert len(calls) == 3                                                 # start + after each of the two ends that return to WAIT
    calls.clear()
    ScriptedKeys.script = [(2.6, "\n"), (3.2, "n"), (3.3, "k")]
    assert record.main(["--sim", "--yes", "--fake-camera", "--no-home", "--root", str(tmp_path / "b"), "--num-episodes", "1"]) == 0
    assert calls == []


def test_episode_home_is_the_rest_pose_unless_the_config_names_another():
    from piper_teleop import config as C
    raw = C.load()
    assert record.episode_home_deg(raw) == raw["defaults"]["park_deg"] == [0, 0, 0, 0, 0, 0]     # not the raised home_deg
    raw["record"]["home"]["deg"] = [0, 35, -34, -4, 37, 10]
    assert record.episode_home_deg(raw) == [0, 35, -34, -4, 37, 10]
    raw["record"]["home"]["deg"] = [0, 1, 2]
    with pytest.raises(SystemExit, match="6 joint angles"):
        record.episode_home_deg(raw)
