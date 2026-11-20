#!/usr/bin/env python3
"""List an Orbbec camera's sensors and every supported stream mode, then save one colour (and depth) frame.

    .venv_cam/bin/python tools/orbbec_probe.py OUT_DIR [--frames N]

Runs in the Python 3.10 camera venv (pyorbbecsdk 1.3.2).  Read-only: it never talks to the arms.
"""
import argparse
import os

import numpy as np
from pyorbbecsdk import Config, Context, OBFormat, OBLogLevel, OBSensorType, Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--serial", help="pick the camera with this serial number (default: the first one)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

try:
    Context.set_logger_level(OBLogLevel.NONE)
except Exception:
    pass
ctx = Context()
dl = ctx.query_devices()
print("devices found:", dl.get_count())
if dl.get_count() == 0:
    raise SystemExit("no Orbbec device visible to the SDK (permissions? see the udev rule)")
devs = [dl.get_device_by_index(i) for i in range(dl.get_count())]
for d in devs:
    i_ = d.get_device_info()
    print(f"  found: {i_.get_name()}  serial {i_.get_serial_number()}  pid 0x{i_.get_pid():04x}")
dev = next((d for d in devs if a.serial in (None, d.get_device_info().get_serial_number())), None)
if dev is None:
    raise SystemExit(f"no camera with serial {a.serial}")
info = dev.get_device_info()
print(f"{info.get_name()}  serial {info.get_serial_number()}  firmware {info.get_firmware_version()}  "
      f"USB {info.get_connection_type()}  pid 0x{info.get_pid():04x}")

pipe = Pipeline(dev)
modes = {}
for kind in (OBSensorType.COLOR_SENSOR, OBSensorType.DEPTH_SENSOR, OBSensorType.IR_SENSOR):
    try:
        pl = pipe.get_stream_profile_list(kind)
    except Exception as e:
        print(f"{kind.name}: not available ({e})")
        continue
    seen = []
    for i in range(pl.get_count()):
        p = pl.get_stream_profile_by_index(i).as_video_stream_profile()
        seen.append((p.get_width(), p.get_height(), p.get_fps(), p.get_format().name))
    modes[kind] = pl
    print(f"{kind.name}: {len(seen)} modes")
    for w, h, fps, fmt in sorted(set(seen), key=lambda x: (-x[0] * x[1], -x[2], x[3])):
        print(f"    {w:5d} x {h:<5d} @ {fps:3d} fps  {fmt}")



def grab(kind, name):
    """Start ONE stream on its own (so a failing sensor does not hide the others) and save a frame."""
    if kind not in modes:
        return
    cfg = Config()
    cfg.enable_stream(modes[kind].get_default_video_stream_profile())
    p = Pipeline(dev)
    try:
        p.start(cfg)
    except Exception as e:
        print(f"{name}: could not start: {e}")
        return
    n = 0
    for _ in range(100):
        fs = p.wait_for_frames(300)
        if fs is None:
            continue
        f = {"color": fs.get_color_frame, "depth": fs.get_depth_frame, "ir": fs.get_ir_frame}[name]()
        if f is None:
            continue
        n += 1
        if n < 4 and a.frames == 1:                                       # let auto-exposure settle
            continue
        w, h = f.get_width(), f.get_height()
        raw = np.frombuffer(f.get_data(), dtype=np.uint8)
        if name == "color":
            import io
            from PIL import Image
            if f.get_format() == OBFormat.MJPG:
                img = Image.open(io.BytesIO(raw.tobytes())).convert("RGB")
            else:
                img = Image.fromarray(raw.reshape(h, w, 3))
            img.save(f"{a.out}/color.png")
            print(f"colour frame saved: {w}x{h} {f.get_format().name}")
        else:
            d = np.frombuffer(f.get_data(), dtype=np.uint16).reshape(h, w).astype(float)
            if name == "depth":
                d = d * f.get_depth_scale()
            np.save(f"{a.out}/{name}.npy", d)
            v = d[d > 0]
            print(f"{name} frame saved: {w}x{h}, {100 * len(v) / d.size:.0f}% non-zero"
                  + (f", median {np.median(v):.0f}" if len(v) else ""))
            if name == "depth":
                from PIL import Image
                lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if len(v) else (0, 1)
                vis = np.where(d > 0, 255 - np.clip((d - lo) / (hi - lo + 1e-9) * 255, 0, 255), 0).astype(np.uint8)
                Image.fromarray(vis).save(f"{a.out}/depth.png")
        break
    else:
        print(f"{name}: started but no frame arrived")
    p.stop()


for kind, name in ((OBSensorType.DEPTH_SENSOR, "depth"), (OBSensorType.COLOR_SENSOR, "color"), (OBSensorType.IR_SENSOR, "ir")):
    grab(kind, name)
