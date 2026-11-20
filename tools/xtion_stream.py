#!/usr/bin/env python3
"""Stream the ASUS Xtion's colour camera into a memory-mapped file, for the dataset recorder.

    .venv_cam/bin/python tools/xtion_stream.py MMAP_FILE W H FPS [--settings JSON] [--rgb-gain R,G,B]

Same job and the same file layout as tools/orbbec_stream.py (uint64 frame counter, float64 capture time, then H*W*3 RGB bytes;
the counter is written LAST), so piper_teleop/recording/cameras.py:XtionCamera reads it exactly like an Orbbec.  Runs in the
Python 3.10 camera venv through OpenNI2 (see tools/xtion_snap.py for the library setup).  Colour only - the depth stream is
never started.  Read-only: never talks to the arms.
"""
import argparse
import json
import mmap
import os
import struct
import sys
import time

import numpy as np
from openni import openni2

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("width", type=int)
ap.add_argument("height", type=int)
ap.add_argument("fps", type=int)
ap.add_argument("--rgb-gain", default="1,1,1", help="software white balance: per-channel multipliers R,G,B")
ap.add_argument("--settings", default="{}", help='JSON, any of: auto_exposure, exposure, gain, auto_white_balance (OpenNI2 camera settings)')
a = ap.parse_args()

openni2.initialize(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv_cam", "openni_lib"))
try:
    dev = openni2.Device.open_any()
except Exception as e:                                           # noqa: BLE001
    sys.exit(f"no ASUS Xtion found ({e}) - is it plugged in?")
stream = dev.create_color_stream()
modes = stream.get_sensor_info().videoModes
want = [m for m in modes if (m.resolutionX, m.resolutionY, m.fps) == (a.width, a.height, a.fps) and m.pixelFormat == openni2.PIXEL_FORMAT_RGB888]
if not want:
    have = sorted({f"{m.resolutionX}x{m.resolutionY}@{m.fps}" for m in modes if m.pixelFormat == openni2.PIXEL_FORMAT_RGB888})
    sys.exit(f"the Xtion has no RGB colour mode {a.width}x{a.height} @ {a.fps} fps (it offers: {', '.join(have) or 'none'})")
stream.set_video_mode(want[0])

settings = json.loads(a.settings)
known = {"auto_exposure", "exposure", "gain", "auto_white_balance"}
if set(settings) - known:
    sys.exit(f"unknown Xtion setting(s) {sorted(set(settings) - known)}; known: {sorted(known)}")
if settings:
    cam = stream.camera
    if cam is None:
        sys.exit("this Xtion colour stream has no camera settings")
    if "exposure" in settings and "auto_exposure" not in settings:
        settings["auto_exposure"] = False
    for key in ("auto_exposure", "auto_white_balance", "exposure", "gain"):          # a manual value after its auto switch is off
        if key not in settings:
            continue
        try:
            setattr(cam, key, settings[key])
            got = getattr(cam, key)
        except Exception as e:                                   # noqa: BLE001
            sys.exit(f"could not set Xtion setting {key}={settings[key]}: {e}")
        if got != settings[key]:
            sys.exit(f"Xtion setting {key}: asked for {settings[key]}, camera reports {got}")

gains = [float(x) for x in a.rgb_gain.split(",")]
if len(gains) != 3 or min(gains) <= 0:
    sys.exit("--rgb-gain needs three positive numbers R,G,B")
LUT = None if gains == [1.0, 1.0, 1.0] else np.stack([np.clip(np.arange(256) * g + 0.5, 0, 255).astype(np.uint8) for g in gains])

stream.start()
size = 16 + a.width * a.height * 3
with open(a.path, "r+b") as f:
    mm = mmap.mmap(f.fileno(), size)
    n = 0
    try:
        while True:
            fr = stream.read_frame()                             # blocks until the next frame
            rgb = np.frombuffer(fr.get_buffer_as_uint8(), np.uint8)
            if rgb.size != a.width * a.height * 3:
                continue
            rgb = rgb.reshape(a.height, a.width, 3)
            if LUT is not None:
                rgb = np.stack([LUT[c][rgb[..., c]] for c in range(3)], axis=-1)
            mm[16:size] = rgb.tobytes()
            mm[8:16] = struct.pack("<d", time.time())
            n += 1
            mm[0:8] = struct.pack("<Q", n)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        openni2.unload()
