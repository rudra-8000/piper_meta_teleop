"""Python client for piper_teleop.robot_server: the same methods as `PiperRobot`, over the network.

    from piper_teleop.robot_client import PiperClient

    with PiperClient("192.168.1.20", 8800, token="SECRET") as robot:      # this machine: PiperClient() = 127.0.0.1:8800
        obs = robot.get_observation()                                      # same dict as PiperRobot.get_observation()
        robot.send_action(my_policy(obs))                                  # (7,) joints [rad] + gripper 0..1

Connecting takes CONTROL of the arm (only one client controls at a time; ``control=False`` connects read-only).  Calls are
synchronous: each sends one request and waits for its reply.  Errors on the server come back as the same Python exceptions
(``ValueError``, ``IKError``, ``TimeoutError``, ``PermissionError``...); if the server has a control lease and you went quiet
for longer than it, control is revoked (the arm holds) and the next command raises ``PermissionError`` - ``acquire()`` again.  Closing releases control and the arm holds where it is.
It needs only numpy, Pillow and websockets - no robot code runs on the client machine.

Command line (handy for checking a server or stopping it from another terminal):
    python -m piper_teleop.robot_client HOST:PORT info | state | hold | rest [--token T]
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time

import numpy as np
from PIL import Image
from websockets.sync.client import connect

from .errors import IKError

_ERRORS = {"ValueError": ValueError, "IKError": IKError, "TimeoutError": TimeoutError, "InterruptedError": InterruptedError,
           "PermissionError": PermissionError, "RuntimeError": RuntimeError}


class PiperClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8800, *, token: str | None = None, control: bool = True,
                 images: str = "jpeg", quality: int = 90, timeout: float = 60.0):
        self.host, self.port, self.token, self.want_control = host, int(port), token, control
        self.images, self.quality, self.timeout = images, quality, timeout
        self._ws = None
        self._id = 0
        self.info: dict = {}

    # --------------------------------------------------------------- plumbing
    def connect(self) -> "PiperClient":
        if self._ws is not None:
            return self
        # entered as a context manager by hand: websockets >= 17.1 deprecates a bare connect(), older versions accept this too
        self._cm = connect(f"ws://{self.host}:{self.port}", max_size=1 << 26, open_timeout=10, ping_interval=5, ping_timeout=10)
        self._ws = self._cm.__enter__()
        try:
            if self.token is not None:
                self._call("hello", token=self.token)
            if self.want_control:
                self._call("acquire")                                # fails with PermissionError if another client has control
            self.info = self._call("info")
        except BaseException:
            self._cm.__exit__(None, None, None)
            self._ws = None
            raise
        return self

    def __enter__(self) -> "PiperClient":
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def _call(self, op: str, **fields) -> dict:
        if self._ws is None:
            raise RuntimeError("not connected: use `with PiperClient(...) as robot:` or call connect()")
        self._id += 1
        self._ws.send(json.dumps({"op": op, "id": self._id, **fields}))
        reply = json.loads(self._ws.recv(timeout=self.timeout))
        if not reply.get("ok"):
            raise _ERRORS.get(reply.get("type"), RuntimeError)(reply.get("error", "request failed"))
        return reply

    def close(self, park: bool = False):
        """Release control (the arm holds where it is) and disconnect.  ``park=True`` folds it to rest first."""
        if self._ws is None:
            return
        try:
            if park:
                self.go_rest()
            elif self.want_control:
                self._call("release")
        except Exception as e:                                       # noqa: BLE001  (server gone): still disconnect
            print(f"[client] could not finish cleanly: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            self._cm.__exit__(None, None, None)
            self._ws = None

    # ------------------------------------------------------------ read access
    @property
    def arm(self) -> str:
        return self.info.get("arm", "?")

    @property
    def state_names(self) -> list[str]:
        return self.info.get("state_names", [])

    @property
    def camera_names(self) -> list[str]:
        return self.info.get("camera_names", [])

    @property
    def max_joint_speed(self) -> float:
        return self.info.get("max_joint_speed", 0.0)

    park_on_exit = False                                            # the SERVER parks when it stops; closing a client only releases

    def get_observation(self) -> dict:
        """Like ``PiperRobot.get_observation()``: ``observation.state`` (7,) float32, ``observation.images.<cam>`` (H, W, 3) uint8 RGB
        (JPEG on the wire by default; ``images="raw"`` for lossless, ``"none"`` for state only), ``timestamp``."""
        r = self._call("get_observation", images=self.images, quality=self.quality)
        obs = {"observation.state": np.asarray(r["state"], dtype=np.float32)}
        for name, im in r["images"].items():
            data = base64.b64decode(im["data"])
            if im["format"] == "jpeg":
                arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            else:
                arr = np.frombuffer(data, np.uint8).reshape(im["h"], im["w"], 3).copy()
            obs[f"observation.images.{name}"] = arr
        obs["timestamp"] = r["timestamp"]
        if "frame_age_s" in r:                                      # how old each camera frame / the arm feedback was when read
            obs["frame_age_s"], obs["state_age_s"] = r["frame_age_s"], r.get("state_age_s")
        return obs

    def get_state(self) -> dict:
        """The cheap status reply: state (7,), commanded_q, tcp {p, R}, synced, loop_hz, stale_frames."""
        return self._call("state")

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self._call("state")["state"], dtype=np.float32)

    @property
    def commanded_q(self) -> np.ndarray:
        return np.asarray(self._call("state")["commanded_q"])

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        t = self._call("state")["tcp"]
        return np.asarray(t["R"]), np.asarray(t["p"])

    # --------------------------------------------------------------- commands
    def acquire(self):
        self._call("acquire")
        self.want_control = True

    def release(self):
        self._call("release")
        self.want_control = False

    def send_action(self, action, *, speed: float | None = None):
        self._call("send_action", action=np.asarray(action, float).reshape(-1).tolist(), speed=speed)

    def run_chunk(self, actions, hz: float = 30.0, on_step=None):
        actions = np.asarray(actions, float)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"an action chunk is (N, 7), got {actions.shape}")
        period, nxt = 1.0 / hz, time.perf_counter()
        for i, a in enumerate(actions):
            if on_step is not None:
                on_step(i)
            self.send_action(a)
            nxt += period
            if nxt > time.perf_counter():
                time.sleep(nxt - time.perf_counter())
            else:
                nxt = time.perf_counter()                            # overran: no catch-up burst

    def send_tcp_pose(self, position, rotation=None, gripper: float | None = None, *, speed: float | None = None, **kw) -> np.ndarray:
        rot = rotation if rotation is None or isinstance(rotation, str) else np.asarray(rotation, float).tolist()
        r = self._call("send_tcp_pose", position=np.asarray(position, float).tolist(), rotation=rot, gripper=gripper, speed=speed, **kw)
        return np.asarray(r["q"])

    def go_to(self, q, gripper: float | None = None, *, speed: float | None = None, timeout: float = 30.0):
        self._call("go_to", q=np.asarray(q, float).tolist(), gripper=gripper, speed=speed, timeout=timeout)

    def go_rest(self, gripper: float | None = None, **kw):
        self._call("go_rest", gripper=gripper, **kw)

    def keepalive(self) -> float | None:
        """Tell the server this controller is alive (needed only when the server runs with a control lease and you are not sending
        commands for a while).  Returns the lease length in seconds, or None if the server has no lease."""
        return self._call("keepalive").get("lease_s")

    def hold(self):
        self._call("hold")

    def cancel_moves(self):
        self._call("cancel_moves")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("server", help="HOST:PORT of a running robot_server")
    ap.add_argument("cmd", choices=["info", "state", "hold", "rest"], help="info/state: read only; hold: stop the arm where it is (any client "
                    "may); rest: take control and fold the arm to the rest pose")
    ap.add_argument("--token")
    a = ap.parse_args(argv)
    host, _, port = a.server.rpartition(":")
    with PiperClient(host or "127.0.0.1", int(port), token=a.token, control=(a.cmd == "rest")) as c:
        if a.cmd == "info":
            print(json.dumps({k: v for k, v in c.info.items()}, indent=1))
        elif a.cmd == "state":
            print(json.dumps(c.get_state(), indent=1))
        elif a.cmd == "hold":
            c.hold()
            print("holding")
        else:
            c.go_rest()
            print("at rest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
