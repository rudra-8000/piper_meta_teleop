"""A small network server in front of the robot, so ANY client (any language, any machine) can read it and command it.

    python -m piper_teleop.robot_server --cameras xtion_top                  # real left arm, listens on 127.0.0.1:8800
    python -m piper_teleop.robot_server --sim --cameras fake                 # simulated arm
    python -m piper_teleop.robot_server --host 0.0.0.0 --token SECRET        # reachable from other machines (token REQUIRED)

The server owns the arm and the cameras (one `PiperRobot`, see robot.py).  It does NOT move the arm by itself: on start it enables
it and holds where it is; it moves only when a client with control tells it to.  Wire protocol: docs/remote_control.md.

    transport   WebSocket, one JSON text message per request, one JSON reply (same "id" echoed back)
    request     {"op": "send_action", "id": 7, "action": [j1..j6 rad, gripper 0..1]}
    reply       {"ok": true, "id": 7, ...}   or   {"ok": false, "id": 7, "type": "ValueError", "error": "..."}

    ops         hello  info  state  get_observation  acquire  release  send_action  send_tcp_pose  go_to  go_rest  hold  cancel_moves  ping

Many clients may connect and read (info / state / get_observation); only ONE at a time has control (`acquire`) and may move the arm.
When the controlling client disconnects (or its connection dies - ping timeout ~10 s) the arm HOLDS where it is.  `hold` and
`cancel_moves` are accepted from ANY client: stopping is always allowed.  Nothing here ever disables the motors.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import signal
import threading
import time

import numpy as np
from PIL import Image
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .arm import GRIPPER_MAX_M
from .kinematics import JOINT_HI, JOINT_LO
from .robot import PiperRobot

PROTOCOL = 1
MAX_MESSAGE = 1 << 20                      # 1 MiB: requests are tiny; refuse anything bigger
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def encode_image(img: np.ndarray, mode: str, quality: int) -> dict:
    h, w = img.shape[:2]
    if mode == "raw":
        return {"w": w, "h": h, "format": "raw_rgb8", "data": base64.b64encode(np.ascontiguousarray(img).tobytes()).decode()}
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(quality))
    return {"w": w, "h": h, "format": "jpeg", "data": base64.b64encode(buf.getvalue()).decode()}


class _Conn:
    def __init__(self, ws, authed: bool):
        self.ws, self.authed = ws, authed
        self.peer = str(getattr(ws, "remote_address", "?"))


class RobotServer:
    """Serve one ``PiperRobot`` over WebSocket.  ``start()`` runs it in a background thread (``port=0`` picks a free port: read
    ``server.port``); ``stop()`` stops it (the robot is NOT closed - you own that)."""

    def __init__(self, robot: PiperRobot, host: str = "127.0.0.1", port: int = 8800, token: str | None = None, log=print):
        self.robot, self.host, self.port, self.token, self.log = robot, host, port, token or None, log
        self.controller: _Conn | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self.ops = {"ping": self._ping, "info": self._info, "state": self._state, "get_observation": self._get_observation,
                    "acquire": self._acquire, "release": self._release, "hold": self._hold, "cancel_moves": self._cancel,
                    "send_action": self._send_action, "send_tcp_pose": self._send_tcp_pose, "go_to": self._go_to,
                    "go_rest": self._go_rest}
        self.needs_control = {"send_action", "send_tcp_pose", "go_to", "go_rest"}

    # --------------------------------------------------------------- lifecycle
    def start(self) -> "RobotServer":
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), name="piper-server", daemon=True)
        self._thread.start()
        self._ready.wait(10.0)
        if self._error is not None:
            raise self._error
        return self

    def stop(self):
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    async def _main(self):
        self._loop, self._stop = asyncio.get_running_loop(), asyncio.Event()
        try:
            async with serve(self._handler, self.host, self.port, max_size=MAX_MESSAGE, ping_interval=5, ping_timeout=5) as srv:
                self.port = srv.sockets[0].getsockname()[1]
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:                                  # noqa: BLE001  (port in use ...): report to start()
            self._error = e
            self._ready.set()

    # ----------------------------------------------------------------- session
    async def _handler(self, ws):
        conn = _Conn(ws, authed=self.token is None)
        self.log(f"[server] client connected: {conn.peer}")
        try:
            async for message in ws:
                reply = await self._dispatch(conn, message)
                await ws.send(json.dumps(reply))
        except ConnectionClosed:
            pass
        finally:
            if self.controller is conn:
                self.controller = None
                await asyncio.to_thread(self.robot.hold)             # the controller is gone: stop where it is
                self.log(f"[server] controller {conn.peer} left - the arm holds its pose")
            else:
                self.log(f"[server] client {conn.peer} left")

    async def _dispatch(self, conn: _Conn, message) -> dict:
        rid = None
        try:
            if isinstance(message, bytes):
                raise ValueError("send JSON text frames, not binary")
            try:
                req = json.loads(message)
            except json.JSONDecodeError as e:
                raise ValueError(f"not valid JSON: {e}") from None
            if not isinstance(req, dict) or "op" not in req:
                raise ValueError('a request is a JSON object with an "op"')
            rid, op = req.get("id"), req["op"]
            if op == "hello":
                if self.token is not None and req.get("token") != self.token:
                    raise PermissionError("wrong or missing token")
                conn.authed = True
                return {"ok": True, "id": rid, "protocol": PROTOCOL}
            if not conn.authed:
                raise PermissionError('this server needs a token: send {"op": "hello", "token": ...} first')
            if op not in self.ops:
                raise ValueError(f"unknown op {op!r}; ops: {sorted(self.ops)} + hello")
            if op in self.needs_control and self.controller is not conn:
                raise PermissionError('not in control: send {"op": "acquire"} first (only one client controls the arm at a time)')
            out = await self.ops[op](conn, req)
            return {"ok": True, "id": rid, **(out or {})}
        except Exception as e:                                       # noqa: BLE001  every failure becomes a reply, never a crash
            if isinstance(e, KeyError):
                e = ValueError(f"missing field {e}")
            return {"ok": False, "id": rid, "type": type(e).__name__, "error": str(e)}

    @staticmethod
    def _vec(req, key, n=None):
        if key not in req:
            raise ValueError(f"missing field {key!r}")
        v = np.asarray(req[key], dtype=float)
        if n is not None and v.shape != (n,):
            raise ValueError(f"{key!r} must be {n} numbers, got shape {v.shape}")
        return v

    # --------------------------------------------------------------------- ops
    async def _ping(self, conn, req):
        return {"time": time.time()}

    async def _info(self, conn, req):
        r = self.robot
        return {"protocol": PROTOCOL, "arm": r.arm, "sim": r.sim, "state_names": r.state_names, "camera_names": r.camera_names,
                "image_shapes": {n: [c.height, c.width] for n, c in r.cameras.items()},
                "joint_lo": JOINT_LO.tolist(), "joint_hi": JOINT_HI.tolist(), "gripper_max_m": GRIPPER_MAX_M,
                "rest_q": r.rest_q.tolist(), "max_joint_speed": r.max_joint_speed, "sync_speed": r.sync_speed,
                "rest_speed": r.rest_speed, "control_hz": r.control_hz, "you_control": self.controller is conn,
                "controller_present": self.controller is not None,
                "workspace": {k: list(v) for k, v in r.acfg.workspace.items()}}

    async def _state(self, conn, req):
        r = self.robot
        R, p = r.tcp_pose()
        return {"state": r.state.tolist(), "commanded_q": r.commanded_q.tolist(), "tcp": {"p": p.tolist(), "R": R.tolist()},
                "synced": r.synced, "loop_hz": r.loop_hz, "stale_frames": dict(r.stale_frames), "you_control": self.controller is conn,
                "timestamp": time.time()}

    async def _get_observation(self, conn, req):
        mode = req.get("images", "jpeg")
        if mode not in ("jpeg", "raw", "none"):
            raise ValueError('"images" must be "jpeg", "raw" or "none"')
        obs = await asyncio.to_thread(self.robot.get_observation)
        images = {}
        if mode != "none":
            for k, v in obs.items():
                if k.startswith("observation.images."):
                    images[k[len("observation.images."):]] = await asyncio.to_thread(encode_image, v, mode, req.get("quality", 90))
        return {"state": obs["observation.state"].tolist(), "images": images, "timestamp": obs["timestamp"]}

    async def _acquire(self, conn, req):
        if self.controller is not None and self.controller is not conn:
            raise PermissionError(f"another client ({self.controller.peer}) already controls the arm")
        self.controller = conn
        self.log(f"[server] {conn.peer} has control")
        return {"you_control": True}

    async def _release(self, conn, req):
        if self.controller is conn:
            self.controller = None
            await asyncio.to_thread(self.robot.hold)
            self.log(f"[server] {conn.peer} released control - the arm holds its pose")
        return {"you_control": False}

    async def _hold(self, conn, req):                                # any client, any time
        await asyncio.to_thread(self.robot.hold)
        return {}

    async def _cancel(self, conn, req):                              # any client, any time
        self.robot.cancel_moves()
        await asyncio.to_thread(self.robot.hold)
        return {}

    async def _send_action(self, conn, req):
        self.robot.send_action(self._vec(req, "action", 7), speed=req.get("speed"))
        return {}

    async def _send_tcp_pose(self, conn, req):
        rot = req.get("rotation")
        q = await asyncio.to_thread(self.robot.send_tcp_pose, self._vec(req, "position", 3), rot, req.get("gripper"),
                                    speed=req.get("speed"), **{k: req[k] for k in ("tol_pos", "tol_rot", "max_joint_change") if k in req})
        return {"q": np.asarray(q).tolist()}

    async def _go_to(self, conn, req):
        await asyncio.to_thread(self.robot.go_to, self._vec(req, "q", 6), req.get("gripper"), speed=req.get("speed"),
                                timeout=float(req.get("timeout", 30.0)))
        return {"state": self.robot.state.tolist()}

    async def _go_rest(self, conn, req):
        await asyncio.to_thread(self.robot.go_rest, req.get("gripper"), timeout=float(req.get("timeout", 40.0)))
        return {"state": self.robot.state.tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", help="physical arm (default robot.arm in the config)")
    ap.add_argument("--sim", action="store_true", help="simulated arm, no hardware")
    ap.add_argument("--cameras", metavar="SET", help="a record.camera_sets entry (e.g. xtion_top), or 'fake'; default: record.cameras")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--host", help="interface to listen on (default robot.server_host = 127.0.0.1: this machine only)")
    ap.add_argument("--port", type=int, help="default robot.server_port = 8800")
    ap.add_argument("--token", default=os.environ.get("PIPER_TOKEN"), help="shared secret clients must send in 'hello' (or env PIPER_TOKEN)")
    ap.add_argument("--insecure", action="store_true", help="allow a non-local --host WITHOUT a token (anyone on the network can move the arm)")
    ap.add_argument("--max-joint-speed", type=float, help="rad/s cap (default robot.max_joint_speed)")
    ap.add_argument("--no-park", action="store_true", help="do not fold to rest when the server stops (the arm just holds)")
    ap.add_argument("--config")
    a = ap.parse_args(argv)

    rc = (__import__("piper_teleop.config", fromlist=["load"]).load(a.config)).get("robot", {})
    host = a.host or rc.get("server_host", "127.0.0.1")
    port = a.port if a.port is not None else rc.get("server_port", 8800)
    if host not in LOOPBACK and not a.token and not a.insecure:
        raise SystemExit(f"refusing to listen on {host} without --token (or env PIPER_TOKEN): anyone who can reach this port could move "
                         "the arm. Pass --token SECRET, or --insecure if you really mean it.")
    cams = False if a.no_cameras else a.cameras
    robot = PiperRobot(a.arm, sim=a.sim, cameras=cams, config=a.config, max_joint_speed=a.max_joint_speed,
                       park_on_exit=False if a.no_park else None)
    stop = threading.Event()
    n = {"c": 0}

    def on_sigint(*_):
        n["c"] += 1
        stop.set()
        if n["c"] >= 2:
            robot.park_on_exit = False
            robot.cancel_moves()
            print("\n[server] second Ctrl-C: not folding to rest - the arm holds its pose.", flush=True)

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)
    robot.connect()
    server = RobotServer(robot, host, port, a.token).start()
    print(f"[server] {robot.arm} arm{' (sim)' if a.sim else ''} ready on ws://{host}:{server.port}  "
          f"cameras {robot.camera_names or 'none'}  token {'required' if a.token else 'none'}\n"
          "[server] the arm moves only when a client with control commands it; Ctrl-C stops (folds to rest, motors stay enabled).", flush=True)
    try:
        stop.wait()
    finally:
        server.stop()
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
