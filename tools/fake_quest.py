#!/usr/bin/env python3
"""Pretend to be the Quest page: stream synthetic controller poses to the server.

    python tools/fake_quest.py                 # circles for 15 s against a --sim server
    python tools/fake_quest.py --url ws://localhost:8788/ws --seconds 30

Use it to see the dashboard move without a headset, or to test a config change.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from websockets.sync.client import connect  # noqa: E402

from piper_teleop.protocol import encode_pose  # noqa: E402


def hand(name, t, squeeze, trigger, offset):
    ph = 0.0 if name == "left" else 1.7
    return dict(hand=name, tracked=True,
                pos=(offset + 0.06 * math.sin(1.1 * t + ph), 1.1 + 0.04 * math.sin(0.8 * t + ph),
                     -0.35 + 0.06 * math.cos(1.1 * t + ph)),
                quat=(0, 0, 0, 1), ray_quat=(0, 0, 0, 1), trigger=trigger, squeeze=squeeze)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8788/ws")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--hz", type=float, default=90)
    a = ap.parse_args()
    with connect(a.url, compression=None) as ws:
        ws.send(json.dumps({"t": "hello", "role": "controller", "ua": "fake_quest"}))
        t0 = time.perf_counter()
        seq = 0
        print("released for 1 s, then GRIP held and hands drawing circles...")
        while (t := time.perf_counter() - t0) < a.seconds:
            grip = 1.0 if t > 1.0 else 0.0
            trig = 0.5 + 0.5 * math.sin(t)
            ws.send(encode_pose(seq, t * 1000, [hand("left", t, grip, trig, -0.2), hand("right", t, grip, 1 - trig, 0.2)]))
            seq += 1
            time.sleep(max(0, t0 + seq / a.hz - time.perf_counter()))
        print("done")


if __name__ == "__main__":
    main()
