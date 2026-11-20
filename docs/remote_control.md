# Robot server: a lightweight interface so you can write your own client

`python -m piper_teleop.robot_server` puts a small WebSocket server in front of the arm (and its cameras). Any program on this PC,
or on another machine, in any language, can then read the robot and command it with plain JSON. It is a thin layer over
[`PiperRobot`](policy_control.md): same observation / action format as the recorded datasets, same speed cap and safety.

    # on the machine with the arm and cameras
    python -m piper_teleop.robot_server --cameras xtion_top                       # real left arm, 127.0.0.1:8800 (this machine only)
    python -m piper_teleop.robot_server --sim --cameras fake                      # simulated, no hardware
    python -m piper_teleop.robot_server --host 0.0.0.0 --token SECRET             # reachable from other machines; a token is REQUIRED

    # a client
    python -m piper_teleop.robot_client HOST:PORT info|state|hold|rest            # quick check / stop from another terminal
    python examples/remote_client_minimal.py HOST:PORT [TOKEN]                    # the protocol with only `websockets` + `json`

The server does **not** move the arm by itself. On start it enables the arm and holds where it is; it moves only when a client that
has control tells it to. Ctrl-C stops it: the arm folds to rest (slowly), the motors stay **enabled**; a second Ctrl-C skips the folding.

## Python client

    from piper_teleop.robot_client import PiperClient

    with PiperClient("192.168.1.20", 8800, token="SECRET") as robot:              # PiperClient() = 127.0.0.1:8800
        obs = robot.get_observation()          # {"observation.state": (7,) f32, "observation.images.<cam>": (H,W,3) u8, "timestamp"}
        robot.send_action(my_policy(obs))      # (7,): 6 joints [rad] + gripper 0..1
        robot.send_tcp_pose([0.25, 0.0, 0.30], "free")
        robot.go_rest()

It has the same methods as `PiperRobot` (`get_observation`, `state`, `tcp_pose`, `send_action`, `run_chunk`, `send_tcp_pose`, `go_to`,
`go_rest`, `hold`, `cancel_moves`) and raises the same exceptions (`ValueError`, `IKError`, `TimeoutError`, `PermissionError`). It
needs only numpy, Pillow and websockets (it does not import any robot code), so `piper_teleop/robot_client.py` + `errors.py` can be
copied to another machine. `run_policy --remote HOST:PORT [--token T]` runs a policy through it.

## Wire protocol (version 1)

WebSocket, **text frames**, one JSON object per message. A request is `{"op": ..., "id": <any>, ...fields}`; the reply echoes `id`:
`{"ok": true, "id": ..., ...}` or `{"ok": false, "id": ..., "type": "ValueError", "error": "message"}`. Requests on one connection are
answered in order, one at a time. A bad request gets an error reply and the connection stays open. Requests over 1 MiB close it.

| op | needs control | fields -> reply |
|---|---|---|
| `hello` | | `token` -> `{protocol}`. **First message when the server has a token** (every other op is refused until then) |
| `info` | | -> `arm, sim, state_names, camera_names, image_shapes {cam: [h, w]}, joint_lo, joint_hi, gripper_max_m, rest_q, max_joint_speed, sync_speed, rest_speed, workspace, you_control, controller_present` |
| `state` | | -> `state` (7), `commanded_q` (6), `tcp {p, R}`, `synced`, `loop_hz`, `stale_frames`, `you_control`, `timestamp` (cheap, no images) |
| `get_observation` | | `images`: `"jpeg"` (default) / `"raw"` / `"none"`, `quality` (JPEG, 90) -> `state` (7), `images {cam: {w, h, format, data}}`, `timestamp`. `data` is base64: a JPEG, or `raw_rgb8` = h*w*3 bytes, row-major RGB |
| `acquire` / `release` | | take / give back control (see below) |
| `send_action` | yes | `action` (7: joints 1..6 [rad], gripper 0..1), optional `speed` [rad/s] -> `{}` immediately; the arm moves there at the speed cap |
| `send_tcp_pose` | yes | `position` (3, m, arm base frame: x forward, y left, z up), `rotation` (null = keep / `"free"` = position only / 3x3 / quaternion xyzw), optional `gripper`, `speed` -> `q` (6) |
| `go_to` | yes | `q` (6), optional `gripper`, `speed`, `timeout` (30 s) -> blocks until there, `state` |
| `go_rest` | yes | optional `gripper`, `timeout` (40 s) -> blocks until at the folded rest pose |
| `hold` | **no** | stop where it is (the target becomes the current commanded pose) |
| `cancel_moves` | **no** | abort a blocking `go_to` / `go_rest` and hold |
| `ping` | | -> `time` |

Errors: `ValueError` (bad or missing field, outside the workspace box, ...), `IKError` (pose not reachable, or needs a joint to swing
more than 57 deg), `TimeoutError` (`go_to` did not arrive), `PermissionError` (no token / not in control), `InterruptedError`
(`cancel_moves`), `RuntimeError` (robot-side failure, e.g. the CAN link). Units: radians, metres, seconds, gripper 0..1 (1 = open, 70 mm).

## Control, safety, network

* **One controller at a time.** Any number of clients may connect and read. `acquire` takes control (refused with `PermissionError`
  if another client has it); the Python client does it on connect. Commands from a client without control are refused.
* **`hold` and `cancel_moves` work from any client**, so a second terminal (`robot_client HOST:PORT hold`) can always stop the arm
  even while the controller is in a blocking move.
* **A vanished controller stops the arm**: when the controlling client disconnects, or its connection dies (ping timeout ~10 s), the
  arm holds where it is and the lease is freed.
* **Same safety as local control**: joint limits, the speed cap (`robot.max_joint_speed`, 2 rad/s), the workspace box for tool-pose
  targets, the joint lock. The motors are never disabled.
* **Network**: it listens on `127.0.0.1` by default. A non-local `--host` is refused without `--token` (or `PIPER_TOKEN`) unless you pass
  `--insecure`. The token only keeps strangers off: traffic is **unencrypted** `ws://`, so use it on a trusted LAN or through an SSH tunnel
  (`ssh -L 8800:localhost:8800 robot-pc`, then connect to `localhost:8800`). Defaults live in `robot.server_host / server_port`.
* **Latency**: a request/reply round trip plus the JPEG encode is a few ms on a LAN. Observations are JPEG (quality 90) by default;
  the dataset's videos are h264, so this is comparable, and `images: "raw"` is lossless (about 0.9 MB per 640x480 frame).
* There is no push/stream mode: poll `get_observation` at your rate. Everything is tested against the simulated arm only.
