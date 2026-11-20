#!/usr/bin/env python3
"""Live preview of the recorder's cameras in a browser, for aiming and mounting them.

    .venv/bin/python tools/camera_preview.py [--cameras xtion_top] [--port 8765]     then open  http://localhost:8765

Opens exactly the cameras the recorder would (`record.cameras`, or `record.camera_sets.NAME` with --cameras), through the same
code and settings, so what you see is what gets recorded.  Each tile shows the camera name, its delivered frame rate and the
brightness / blown-out percentage; `--grid` adds a rule-of-thirds grid.  Serves on localhost only.  Ctrl-C to stop - and stop it
before recording: a camera can only be opened by one program at a time.  Read-only: never talks to the arms.
"""
import argparse
import io
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop import config as cfgmod  # noqa: E402
from piper_teleop.recording.cameras import make_camera  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cameras", metavar="SET", help="record.camera_sets.SET instead of record.cameras")
ap.add_argument("--config")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--grid", action="store_true", help="draw a rule-of-thirds grid")
ap.add_argument("--quality", type=int, default=85, help="JPEG quality of the preview (the recording itself is h264)")
a = ap.parse_args()

rc = cfgmod.load(a.config).get("record", {})
specs = (rc.get("camera_sets") or {}).get(a.cameras) if a.cameras else rc.get("cameras")
if not specs:
    sys.exit(f"no cameras: record.{'camera_sets.' + a.cameras if a.cameras else 'cameras'} is empty or missing")
cams = {n: make_camera(n, s) for n, s in specs.items()}
state = {n: {"jpg": b"", "n": 0, "fps": 0.0, "mean": 0.0, "clip": 0.0, "err": None} for n in cams}
stop = threading.Event()


def pump(name, cam):
    t_fps, frames = time.time(), 0
    last = None
    while not stop.is_set():
        try:
            f = cam.read()
        except Exception as e:                                   # noqa: BLE001
            state[name]["err"] = str(e)
            time.sleep(0.2)
            continue
        if f[::37, ::37].tobytes() == last:                      # nothing new yet
            time.sleep(0.004)
            continue
        last = f[::37, ::37].tobytes()
        frames += 1
        now = time.time()
        if now - t_fps >= 1.0:
            state[name]["fps"], t_fps, frames = frames / (now - t_fps), now, 0
        luma = f.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        img = Image.fromarray(f)
        d = ImageDraw.Draw(img)
        if a.grid:
            w, h = img.size
            for k in (1, 2):
                d.line([(w * k // 3, 0), (w * k // 3, h)], fill=(255, 255, 255), width=1)
                d.line([(0, h * k // 3), (w, h * k // 3)], fill=(255, 255, 255), width=1)
        s = state[name]
        s["mean"], s["clip"] = float(luma.mean()), 100 * float((luma >= 250).mean())
        d.rectangle([0, 0, 330, 14], fill=(0, 0, 0))
        d.text((3, 2), f"{name}  {s['fps']:.0f} fps  brightness {s['mean']:.0f}  blown {s['clip']:.1f}%", fill=(255, 255, 0))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=a.quality)
        s["jpg"], s["n"], s["err"] = buf.getvalue(), s["n"] + 1, None


PAGE = ("<!doctype html><meta charset=utf-8><title>camera preview</title><body style='background:#111;color:#ddd;font:14px sans-serif;margin:8px'>"
        "<div style='display:flex;flex-wrap:wrap;gap:8px'>" + "".join(f"<div><img src='/cam/{n}.mjpg' style='max-width:48vw;min-width:320px'></div>" for n in cams) +
        "</div><p>Ctrl-C in the terminal to stop (frees the cameras for recording).</p>")


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        name = self.path.removeprefix("/cam/").removesuffix(".mjpg")
        if not self.path.startswith("/cam/") or name not in state:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last = -1
        try:
            while not stop.is_set():
                s = state[name]
                if s["n"] != last and s["jpg"]:
                    last = s["n"]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(s["jpg"]) + s["jpg"] + b"\r\n")
                else:
                    time.sleep(0.005)
        except (BrokenPipeError, ConnectionResetError):
            pass


signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))     # a plain `kill` stops it cleanly too (releases the cameras)
try:
    for n, c in cams.items():
        print(f"starting camera '{n}' ...", flush=True)
        c.connect()
        threading.Thread(target=pump, args=(n, c), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"\nlive preview:  http://localhost:{a.port}     (Ctrl-C to stop)", flush=True)
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    for c in cams.values():
        c.disconnect()
