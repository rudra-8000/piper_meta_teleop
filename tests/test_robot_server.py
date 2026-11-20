import json
import time
from pathlib import Path

import numpy as np
import pytest
from websockets.sync.client import connect

from piper_teleop import run_policy as rp
from piper_teleop.errors import IKError
from piper_teleop.robot import PiperRobot
from piper_teleop.robot_client import PiperClient
from piper_teleop.robot_server import RobotServer

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "policies.py"


@pytest.fixture
def served():
    robot = PiperRobot("left", sim=True, cameras="fake").connect()
    server = RobotServer(robot, "127.0.0.1", 0, log=lambda *_: None).start()
    yield robot, server
    server.stop()
    robot.close()


def wait_until(cond, timeout=6.0):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, "timed out"
        time.sleep(0.02)


def raw_call(ws, op, **kw):
    ws.send(json.dumps({"op": op, "id": 1, **kw}))
    return json.loads(ws.recv(timeout=10))


def test_info_and_observation_round_trip(served):
    robot, srv = served
    with PiperClient("127.0.0.1", srv.port) as c:
        assert c.arm == "left" and c.camera_names == ["cam_high"] and c.state_names[-1] == "left_gripper"
        assert c.info["image_shapes"] == {"cam_high": [48, 64]} and len(c.info["joint_lo"]) == 6 and c.info["you_control"]
        for mode in ("jpeg", "raw", "none"):
            c.images = mode
            obs = c.get_observation()
            assert obs["observation.state"].shape == (7,) and obs["observation.state"].dtype == np.float32
            if mode == "none":
                assert set(obs) == {"observation.state", "timestamp"}
            else:
                im = obs["observation.images.cam_high"]
                assert im.shape == (48, 64, 3) and im.dtype == np.uint8
        st = c.get_state()
        assert len(st["tcp"]["p"]) == 3 and np.asarray(st["tcp"]["R"]).shape == (3, 3) and st["you_control"]
        R, p = c.tcp_pose()
        assert np.allclose(p, robot.tcp_pose()[1], atol=1e-3)


def test_raw_images_are_lossless_and_match_the_servers_frame(served):
    robot, srv = served
    robot.cameras["cam_high"].read = lambda: np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    with PiperClient("127.0.0.1", srv.port, images="raw") as c:
        assert np.array_equal(c.get_observation()["observation.images.cam_high"], np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3))


def test_send_action_moves_the_arm_and_errors_come_back_as_the_same_exceptions(served):
    robot, srv = served
    with PiperClient("127.0.0.1", srv.port) as c:
        a = np.zeros(7)
        a[1] = 0.3
        c.send_action(a, speed=2.0)
        wait_until(lambda: abs(robot.state[1] - 0.3) < 0.02)
        assert abs(c.state[1] - 0.3) < 0.03
        with pytest.raises(ValueError, match="7"):
            c.send_action(np.zeros(5))
        with pytest.raises(ValueError, match="finite"):
            c.send_action(np.full(7, np.nan))
        with pytest.raises(IKError, match="cannot reach"):
            c.send_tcp_pose([0.95, 0.0, 0.5], "free")
        with pytest.raises(ValueError, match="workspace"):
            c.send_tcp_pose([0.0, 0.0, 0.3], "free")
        q = c.send_tcp_pose(c.tcp_pose()[1] + [0.04, 0.0, 0.05], "free", speed=2.0)
        assert q.shape == (6,)
        with pytest.raises(TimeoutError):
            c.go_to(np.radians([40, 60, -60, 30, 30, 30]), speed=0.1, timeout=0.3)
        c.go_rest()
        assert np.allclose(c.state[:6], 0.0, atol=0.03)
        c.run_chunk(np.tile(c.state.astype(float), (3, 1)), hz=100)


def test_only_one_client_controls_but_anyone_can_stop_it(served):
    robot, srv = served
    a = PiperClient("127.0.0.1", srv.port).connect()
    with pytest.raises(PermissionError, match="already controls"):
        PiperClient("127.0.0.1", srv.port).connect()
    b = PiperClient("127.0.0.1", srv.port, control=False).connect()                 # read-only is always fine
    assert b.state.shape == (7,) and not b.get_state()["you_control"]
    with pytest.raises(PermissionError, match="not in control"):
        b.send_action(np.zeros(7))
    with pytest.raises(PermissionError, match="not in control"):
        b.go_rest()
    tgt = np.zeros(7)
    tgt[1] = 0.8
    a.send_action(tgt, speed=0.3)
    time.sleep(0.5)
    b.hold()                                                                        # a read-only client can STOP it
    q_stop = robot.commanded_q.copy()
    time.sleep(0.4)
    assert np.allclose(robot.commanded_q, q_stop, atol=1e-6) and q_stop[1] < 0.75
    a.close()                                                                       # controller leaves: the arm holds, lease frees up
    b.acquire()
    assert b.get_state()["you_control"]
    b.close()


def test_a_controller_that_disconnects_mid_move_leaves_the_arm_holding(served):
    robot, srv = served
    c = PiperClient("127.0.0.1", srv.port).connect()
    t = np.zeros(7)
    t[1] = 1.2
    c.send_action(t, speed=0.5)
    time.sleep(0.4)
    c._cm.__exit__(None, None, None)                                                # vanish without releasing
    c._ws = None
    wait_until(lambda: srv.controller is None)
    q = robot.commanded_q.copy()
    time.sleep(0.5)
    assert np.allclose(robot.commanded_q, q, atol=1e-6) and q[1] < 1.0              # stopped where it was, not at the 1.2 target


def test_token_is_required_when_set():
    robot = PiperRobot("left", sim=True, cameras=False).connect()
    server = RobotServer(robot, "127.0.0.1", 0, token="s3cret", log=lambda *_: None).start()
    try:
        with connect(f"ws://127.0.0.1:{server.port}") as ws:
            r = raw_call(ws, "state")
            assert not r["ok"] and r["type"] == "PermissionError" and "token" in r["error"]
            assert not raw_call(ws, "hello", token="nope")["ok"]
            assert not raw_call(ws, "get_observation")["ok"]
            assert raw_call(ws, "hello", token="s3cret")["ok"] and raw_call(ws, "state")["ok"]
        with pytest.raises(PermissionError):
            PiperClient("127.0.0.1", server.port, token="wrong").connect()
        with PiperClient("127.0.0.1", server.port, token="s3cret") as c:
            assert c.state.shape == (7,)
    finally:
        server.stop()
        robot.close()


def test_bad_requests_get_an_error_reply_and_the_connection_survives(served):
    robot, srv = served
    with connect(f"ws://127.0.0.1:{srv.port}") as ws:
        for bad in ("not json", "[1, 2]", json.dumps({"no": "op"}), json.dumps({"op": "fly"})):
            ws.send(bad)
            r = json.loads(ws.recv(timeout=5))
            assert not r["ok"] and r["type"] == "ValueError"
        ws.send(b"\x00\x01")                                                         # binary frame
        assert not json.loads(ws.recv(timeout=5))["ok"]
        assert not raw_call(ws, "acquire") is None and raw_call(ws, "send_action")["type"] == "ValueError"   # missing field
        assert raw_call(ws, "get_observation", images="png")["type"] == "ValueError"
        r = raw_call(ws, "ping")
        assert r["ok"] and r["id"] == 1


def test_oversized_messages_are_refused(served):
    robot, srv = served
    with connect(f"ws://127.0.0.1:{srv.port}") as ws:
        ws.send("x" * (2 << 20))
        with pytest.raises(Exception):                                               # the server closes the connection (1009)
            ws.recv(timeout=5)
    with PiperClient("127.0.0.1", srv.port) as c:                                    # and keeps serving others
        assert c.state.shape == (7,)


def test_the_server_refuses_to_listen_on_the_network_without_a_token():
    from piper_teleop import robot_server
    with pytest.raises(SystemExit, match="--token"):
        robot_server.main(["--sim", "--host", "0.0.0.0", "--port", "0"])


def test_run_policy_works_through_the_server(served, capsys):
    robot, srv = served
    out = rp.main(["--policy", f"{EXAMPLES}:sine", "--arg", "amplitude_deg=5", "--remote", f"127.0.0.1:{srv.port}", "--steps", "40",
                   "--hz", "60", "--yes"])
    assert out == 0 and "40 steps" in capsys.readouterr().out
    assert np.allclose(robot.state[:6], 0.0, atol=0.05)                              # folded to rest at the end
    assert srv.controller is None                                                    # lease released
    with pytest.raises(SystemExit, match="SERVER's settings"):
        rp.main(["--policy", f"{EXAMPLES}:hold", "--remote", f"127.0.0.1:{srv.port}", "--cameras", "xtion_top"])


def test_the_client_module_does_not_pull_in_robot_code():
    import subprocess
    import sys
    code = "import sys, piper_teleop.robot_client; print(sorted(m for m in ('piper_teleop.robot', 'piper_teleop.arm', 'piper_teleop.config') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
