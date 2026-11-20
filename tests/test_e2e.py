"""Real WebSocket server + simulated arms + fake Quest client, all in-process."""
import asyncio
import pathlib
import json
import threading
import time
import urllib.request

import numpy as np
import pytest
from websockets.sync.client import connect

from piper_teleop import config as cfgmod
from piper_teleop.controller import TeleopArm
from piper_teleop.arm import SimArm
from piper_teleop.protocol import encode_pose
from piper_teleop.quest_server import QuestServer
from piper_teleop.system import TeleopSystem

HOME = np.radians(cfgmod.arm_config('x', cfgmod.load(), {}).home_deg)


@pytest.fixture()
def server():
    raw = cfgmod.load()
    arms = []
    for name in ("left", "right"):
        be = SimArm(q0=HOME, lag_s=0.0, max_speed=100)
        be.enable()
        be.record = True
        arms.append(TeleopArm(cfgmod.arm_config(name, raw, {**raw["arms"][name], "base_yaw_deg": 0}), be))
    system = TeleopSystem(arms, rate_hz=200)
    srv = QuestServer(system, "127.0.0.1", 0)
    system.on_event = srv.on_event
    system.start()
    ready = threading.Event()
    port_box = {}

    async def run():
        from websockets.asyncio.server import serve
        srv.loop = asyncio.get_running_loop()
        async with serve(srv.handler, "127.0.0.1", 0, process_request=srv.process_request,
                         compression=None) as s:
            port_box["port"] = s.sockets[0].getsockname()[1]
            asyncio.ensure_future(srv.status_task())
            ready.set()
            await asyncio.Future()

    th = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
    th.start()
    assert ready.wait(5)
    yield srv, system, port_box["port"]
    system.shutdown(park=False)


def hands(x, grip, trig=0.0):
    mk = lambda n, ox: dict(hand=n, pos=(ox, 1.1, -0.3 + x), quat=(0, 0, 0, 1), ray_quat=(0, 0, 0, 1),
                            trigger=trig, squeeze=grip)
    return [mk("left", -0.2), mk("right", 0.2)]


def test_pages_served(server):
    _, _, port = server
    for path, needle in (("/", b"Piper Quest Teleop"), ("/dashboard", b"Piper Teleop"), ("/quest.js", b"WebSocket")):
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}").read()
        assert needle in body
    with pytest.raises(Exception):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/etc/passwd")


def test_pose_to_command_latency_and_motion(server):
    srv, system, port = server
    with connect(f"ws://127.0.0.1:{port}/ws", compression=None) as ws:
        ws.send(json.dumps({"t": "hello", "role": "controller"}))
        seq = 0
        for _ in range(30):                                   # 0.33 s released
            ws.send(encode_pose(seq, 0, hands(0.0, 0.0))); seq += 1; time.sleep(1 / 90)
        for _ in range(20):                                   # grab
            ws.send(encode_pose(seq, 0, hands(0.0, 1.0))); seq += 1; time.sleep(1 / 90)
        be = system.arms[0].backend
        n0 = len(be.commands)
        q_before = be.commands[-1][1].copy()
        t_step = time.perf_counter()                          # hands step 6 cm forward
        react = None
        for _ in range(60):
            ws.send(encode_pose(seq, 0, hands(-0.06, 1.0))); seq += 1
            time.sleep(1 / 90)
            if react is None:
                for t, q, g in be.commands[n0:]:
                    if t >= t_step and np.max(np.abs(q - q_before)) > 2e-3:
                        react = t - t_step
                        break
        assert react is not None, "arm never reacted"
        print(f"\nPC-side pose->command reaction time: {react * 1e3:.1f} ms")
        assert react < 0.040
        time.sleep(0.4)
        tcp0 = system.arms[0].kin.fk_tcp(HOME)[1]
        tcp1 = system.arms[0].kin.fk_tcp(be.commands[-1][1])[1]
        assert tcp1[0] - tcp0[0] == pytest.approx(0.06 * system.arms[0].cfg.pos_scale, abs=0.01)   # hand moved 6 cm
        assert system.loop_hz > 150


def test_viewer_gets_status_and_can_estop(server):
    srv, system, port = server
    with connect(f"ws://127.0.0.1:{port}/ws", compression=None) as ws:
        ws.send(json.dumps({"t": "hello", "role": "viewer"}))
        for _ in range(40):
            m = json.loads(ws.recv(timeout=2))
            if m.get("t") == "status":
                break
        assert set(m["arms"]) == {"left", "right"} and len(m["arms"]["left"]["pts"]) == 8
        ws.send(json.dumps({"t": "cmd", "cmd": "estop"}))
        time.sleep(0.2)
        assert all(a.state == "estop" for a in system.arms)
        assert not any(a.backend._stopped for a in system.arms)    # freeze = software hold, NOT the firmware stop
        assert all(a.backend._enabled for a in system.arms)        # and never a power cut


def test_malformed_packets_are_ignored(server):
    srv, system, port = server
    with connect(f"ws://127.0.0.1:{port}/ws", compression=None) as ws:
        for junk in (b"", b"\x01", b"\x01\x02\x00\x00" + b"\x00" * 50, b"\xff" * 200, b"{not json"):
            ws.send(junk)
        ws.send("not json at all")
        ws.send(encode_pose(0, 0, hands(0, 0)))
        time.sleep(0.2)
        assert system.error is None


def test_port_check_detects_a_taken_port_and_a_free_one():
    import socket
    from piper_teleop.quest_server import _port_free
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0)); sk.listen()
        taken = sk.getsockname()[1]
        assert not _port_free("0.0.0.0", taken)
        assert not _port_free("127.0.0.1", taken)
    assert _port_free("127.0.0.1", taken)             # released again


def test_server_refuses_to_start_before_touching_arms_when_port_is_busy():
    import socket
    from piper_teleop import quest_server
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0)); sk.listen()
        busy = sk.getsockname()[1]
        with pytest.raises(SystemExit) as e:
            quest_server.main(["--sim", "--port", str(busy), "--host", "127.0.0.1"])
        assert "already in use" in str(e.value) and "no arm was touched" in str(e.value)


def test_headset_page_state_reaches_the_dashboard(server):
    srv, system, port = server
    with connect(f"ws://127.0.0.1:{port}/ws", compression=None) as headset, \
         connect(f"ws://127.0.0.1:{port}/ws", compression=None) as dash:
        dash.send(json.dumps({"t": "hello", "role": "viewer"}))
        headset.send(json.dumps({"t": "hello", "role": "controller"}))
        headset.send(json.dumps({"t": "xr", "state": "error", "msg": "requestSession(immersive-ar) failed: NotAllowedError",
                                 "inputs": [{"h": "left", "p": "oculus-touch-v3", "g": True, "hand": False}], "tracked": "none"}))
        q = None
        for _ in range(60):
            m = json.loads(dash.recv(timeout=2))
            if m.get("t") == "status" and m["net"]["quest"] and m["net"]["quest"][0].get("state") == "error":
                q = m["net"]["quest"][0]
                break
        assert q and "NotAllowedError" in q["msg"] and q["inputs"][0]["h"] == "left"


def test_abrupt_client_disconnect_is_quiet(server, caplog):
    """A Quest that falls asleep drops the socket without a close frame: must not log a handler crash."""
    import logging
    srv, system, port = server
    with caplog.at_level(logging.ERROR, logger="websockets"):
        ws = connect(f"ws://127.0.0.1:{port}/ws", compression=None)
        ws.send(json.dumps({"t": "hello", "role": "controller"}))
        time.sleep(0.2)
        ws.socket.close()                         # abrupt: no closing handshake
        time.sleep(0.6)
    assert not [r for r in caplog.records if "handler failed" in r.getMessage()]
    assert system.error is None


def test_status_exposes_live_hand_values_and_session_log_records_grip(server):
    srv, system, port = server
    with connect(f"ws://127.0.0.1:{port}/ws", compression=None) as cam, connect(f"ws://127.0.0.1:{port}/ws", compression=None) as dash:
        dash.send(json.dumps({"t": "hello", "role": "viewer"}))
        for _ in range(20):
            cam.send(encode_pose(0, 0, hands(0.0, 1.0, trig=0.4)))
            time.sleep(0.02)
        got = None
        for _ in range(60):
            m = json.loads(dash.recv(timeout=2))
            if m.get("t") == "status" and m["hands"]:
                got = m["hands"]; break
        assert got and got["left"]["squeeze"] == pytest.approx(1.0) and got["right"]["trigger"] == pytest.approx(0.4, abs=0.01)
    from piper_teleop.paths import log_dir
    log = (log_dir() / "session.log").read_text()
    assert "GRIP PRESSED" in log
