#!/usr/bin/env python3
"""The wire protocol with nothing but `websockets` + `json` - copy this to write your own client (any language works the same way).

    python -m piper_teleop.robot_server --sim --cameras fake          # terminal 1
    python examples/remote_client_minimal.py                          # terminal 2   (HOST:PORT [TOKEN] as arguments for a real server)

One JSON text message per request, one JSON reply.  Fields: docs/remote_control.md.
"""
import base64
import json
import sys

from websockets.sync.client import connect

host_port = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8800"
token = sys.argv[2] if len(sys.argv) > 2 else None

with connect(f"ws://{host_port}", max_size=None) as ws:
    n = 0

    def call(op, **fields):
        global n
        n += 1
        ws.send(json.dumps({"op": op, "id": n, **fields}))
        reply = json.loads(ws.recv())
        if not reply["ok"]:
            raise RuntimeError(f"{reply['type']}: {reply['error']}")
        return reply

    if token:
        call("hello", token=token)
    info = call("info")
    print("arm:", info["arm"], "| cameras:", info["camera_names"], "| state:", info["state_names"])
    call("acquire")                                           # take control: required before any command

    obs = call("get_observation", images="jpeg")
    print("state:", [round(x, 3) for x in obs["state"]])
    for name, im in obs["images"].items():                    # base64 JPEG (or "raw_rgb8" with images="raw")
        open(f"/tmp/{name}.jpg", "wb").write(base64.b64decode(im["data"]))
        print(f"  {name}: {im['w']}x{im['h']} {im['format']} -> /tmp/{name}.jpg")

    target = list(obs["state"])
    target[1] += 0.2                                          # joint 2 up by 0.2 rad; the server rate-limits the motion
    call("send_action", action=target)
    print("moving... state now:", [round(x, 3) for x in call("state")["state"]])
    call("go_rest")                                           # blocking: returns when it is at the rest pose
    call("release")
