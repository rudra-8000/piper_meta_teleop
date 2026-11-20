#!/usr/bin/env python3
"""Grab colour + depth from the ASUS Xtion (PrimeSense, USB 1d27:0601) via OpenNI2.

    .venv_cam/bin/python tools/xtion_snap.py OUT_DIR [--frames N]

Runs in the separate Python 3.10 environment .venv_cam (the `openni` binding does not work on Python 3.13).
Writes color.png, depth.png (near = bright) and depth.npy (millimetres, 0 = no reading).  Read-only: it never talks to the arms.
"""
import argparse
import os

import numpy as np
from openni import openni2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--color-only", action="store_true", help="skip the depth stream (faster, less disk)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
# libOpenNI2.so + OpenNI2/Drivers/*.so symlinks to the system OpenNI2 (Ubuntu ships them as *.so.0)
openni2.initialize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv_cam", "openni_lib"))
dev = openni2.Device.open_any()
print("device:", dev.get_device_info())
streams = {}
for kind, make in (("color", dev.create_color_stream),) + (() if a.color_only else (("depth", dev.create_depth_stream),)):
    try:
        s = make()
        s.start()
        streams[kind] = s
        print(kind, s.get_video_mode())
    except Exception as e:
        print(kind, "unavailable:", e)
import time
times = open(f"{a.out}/times.txt", "w")              # wall-clock time of each frame, to align with CAN logs
for i in range(a.frames):
    tag = "" if a.frames == 1 else f"_{i:03d}"
    times.write(f"{i} {time.time():.4f}\n")
    for kind, s in streams.items():
        for _ in range(3 if i == 0 else 1):
            f = s.read_frame()
        if kind == "color":
            img = np.frombuffer(f.get_buffer_as_uint8(), np.uint8).reshape(f.height, f.width, 3)
            Image.fromarray(img).save(f"{a.out}/color{tag}.png")
        else:
            d = np.frombuffer(f.get_buffer_as_uint16(), np.uint16).reshape(f.height, f.width).astype(float)
            np.save(f"{a.out}/depth{tag}.npy", d)
            v = d[d > 0]
            lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if len(v) else (0, 1)
            vis = np.where(d > 0, 255 - np.clip((d - lo) / (hi - lo + 1e-9) * 255, 0, 255), 0).astype(np.uint8)
            Image.fromarray(vis).save(f"{a.out}/depth{tag}.png")
            if i == 0:
                print(f"depth: {100 * len(v) / d.size:.0f}% valid, median {np.median(v):.0f} mm" if len(v) else "depth: no valid pixels")
for s in streams.values():
    s.stop()
openni2.unload()
