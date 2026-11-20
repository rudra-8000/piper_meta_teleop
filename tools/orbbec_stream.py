#!/usr/bin/env python3
"""Stream an Orbbec colour camera into a memory-mapped file, for the dataset recorder.

    PYTHONPATH=$HOME/orbbec_v1_legacy/install/lib .venv_cam/bin/python tools/orbbec_stream.py MMAP_FILE W H FPS [--serial S]

Runs in the Python 3.10 camera venv (pyorbbecsdk v1.x); the recorder runs in the main venv and reads the file
(piper_teleop/recording/cameras.py: OrbbecCamera).  File layout: uint64 frame counter, float64 capture time
(time.time()), then H*W*3 RGB bytes.  The counter is written LAST, so a reader that sees the same counter before and after
copying the frame has a whole one.  Read-only: never talks to the arms.
"""
import argparse
import io
import mmap
import struct
import sys
import time

import numpy as np
from PIL import Image
import json

from pyorbbecsdk import Config, Context, OBFormat, OBLogLevel, OBPropertyID, OBSensorType, Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("width", type=int)
ap.add_argument("height", type=int)
ap.add_argument("fps", type=int)
ap.add_argument("--serial")
ap.add_argument("--rgb-gain", default="1,1,1", help="software white balance: per-channel multipliers R,G,B (tools/camera_tune.py --white-card)")
ap.add_argument("--settings", default="{}", help='JSON, e.g. {"exposure": 100, "gain": 0, "white_balance": 5000}')
a = ap.parse_args()

try:
    Context.set_logger_level(OBLogLevel.NONE)
except Exception:
    pass
dl = Context().query_devices()
devs = [dl.get_device_by_index(i) for i in range(dl.get_count())]
dev = next((d for d in devs if a.serial in (None, d.get_device_info().get_serial_number())), None)
if dev is None:
    sys.exit(f"no Orbbec camera{' with serial ' + a.serial if a.serial else ''} found ({len(devs)} device(s) visible)")


def apply_settings(dev, s: dict):
    """Colour controls of the DaBai.  A manual value switches the matching auto mode off first.  Fails loudly: a setting that
    silently did not apply would make a dataset's exposure differ from what the config says."""
    P = OBPropertyID
    # Manual values PERSIST inside the camera between runs (a tuning session would leave it dark forever), so every start
    # begins from the same defined state - auto exposure + auto white balance, constant frame rate - and only then applies
    # what the config asks for.
    todo = {"auto_exposure": True, "auto_white_balance": True, "ae_priority": 0}
    todo.update(s)
    known = {"auto_exposure", "ae_priority", "exposure", "gain", "auto_white_balance", "white_balance", "hue"}
    if set(s) - known:
        sys.exit(f"unknown camera setting(s) {sorted(set(s) - known)}; known: {sorted(known)}")
    if "exposure" in s and "auto_exposure" not in s:
        todo["auto_exposure"] = False
    if "white_balance" in s and "auto_white_balance" not in s:
        todo["auto_white_balance"] = False
    plan = [("auto_exposure", P.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, "bool"), ("auto_white_balance", P.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, "bool"),
            ("ae_priority", P.OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT, "int"), ("exposure", P.OB_PROP_COLOR_EXPOSURE_INT, "int"),
            ("gain", P.OB_PROP_COLOR_GAIN_INT, "int"), ("white_balance", P.OB_PROP_COLOR_WHITE_BALANCE_INT, "int"),
            ("hue", P.OB_PROP_COLOR_HUE_INT, "int")]
    for key, pid, kind in plan:
        if key not in todo:
            continue
        val = todo[key]
        try:
            if kind == "bool":
                dev.set_bool_property(pid, bool(val))
                got = dev.get_bool_property(pid)
            else:
                r = dev.get_int_property_range(pid)
                if not r.min <= int(val) <= r.max:
                    sys.exit(f"camera setting {key}={val} is outside what this camera accepts ({r.min}..{r.max})")
                dev.set_int_property(pid, int(val))
                got = dev.get_int_property(pid)
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(f"could not set camera setting {key}={val}: {e}")
        if got != (bool(val) if kind == "bool" else int(val)):
            sys.exit(f"camera setting {key}: asked for {val}, camera reports {got}")


apply_settings(dev, json.loads(a.settings))
gains = [float(x) for x in a.rgb_gain.split(",")]
if len(gains) != 3 or min(gains) <= 0:
    sys.exit("--rgb-gain needs three positive numbers R,G,B")
LUT = None if gains == [1.0, 1.0, 1.0] else np.stack([np.clip(np.arange(256) * g + 0.5, 0, 255).astype(np.uint8) for g in gains])
pipe = Pipeline(dev)
pl = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
prof = None
for fmt in (OBFormat.RGB, OBFormat.MJPG):                       # RGB needs no decoding; MJPG is what the camera offers at 30 fps
    try:
        prof = pl.get_video_stream_profile(a.width, a.height, fmt, a.fps)
        break
    except Exception:
        continue
if prof is None:
    sys.exit(f"the camera has no colour mode {a.width}x{a.height} @ {a.fps} fps (RGB or MJPG)")
cfg = Config()
cfg.enable_stream(prof)
pipe.start(cfg)

size = 16 + a.width * a.height * 3
with open(a.path, "r+b") as f:
    mm = mmap.mmap(f.fileno(), size)
    n = 0
    try:
        while True:
            fs = pipe.wait_for_frames(200)
            if fs is None:
                continue
            fr = fs.get_color_frame()
            if fr is None:
                continue
            w, h = fr.get_width(), fr.get_height()
            raw = np.frombuffer(fr.get_data(), dtype=np.uint8)
            if fr.get_format() == OBFormat.MJPG:
                rgb = np.asarray(Image.open(io.BytesIO(raw.tobytes())).convert("RGB"))
            else:
                rgb = raw.reshape(h, w, 3)
            if rgb.shape != (a.height, a.width, 3):
                continue
            if LUT is not None:
                rgb = np.stack([LUT[c][rgb[..., c]] for c in range(3)], axis=-1)
            mm[16:size] = rgb.tobytes()
            mm[8:16] = struct.pack("<d", time.time())
            n += 1
            mm[0:8] = struct.pack("<Q", n)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
