"""Cameras for the dataset recorder.  Every camera keeps its LATEST frame ready, so the 30 Hz loop never waits on a device.

    cam = make_camera("wrist", {"type": "orbbec", "serial": "CC1365303LL", "width": 640, "height": 480})
    cam.connect();  rgb = cam.read();  cam.disconnect()

Types: ``fake`` (moving test pattern), ``opencv`` (UVC webcam: index or /dev/videoN), ``orbbec`` (Orbbec colour stream
through a helper process in the Python 3.10 camera venv - the v1 SDK that supports the DaBai needs it) and ``xtion`` (the ASUS
Xtion, same helper-process scheme through OpenNI2).
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class Camera:
    name = "camera"
    width = 640
    height = 480
    fps = 30

    def connect(self): ...
    def disconnect(self): ...

    def read(self) -> np.ndarray:
        """Latest RGB uint8 frame (height, width, 3)."""
        raise NotImplementedError

    def frame_age(self) -> float:
        """Seconds since the latest frame was captured (inf if none yet)."""
        return 0.0


class FakeCamera(Camera):
    def __init__(self, name="fake", width=64, height=48, fps=30):
        self.name, self.width, self.height, self.fps = name, int(width), int(height), int(fps)
        self.n = 0

    def read(self):
        self.n += 1
        img = np.zeros((self.height, self.width, 3), np.uint8)
        img[..., 0] = (self.n * 5) % 255
        x = (self.n * 3) % max(1, self.width - 8)
        img[10:20, x:x + 8, 1] = 200
        return img


class OpenCVCamera(Camera):
    def __init__(self, name, index=0, width=640, height=480, fps=30):
        self.name, self.index, self.width, self.height, self.fps = name, index, int(width), int(height), int(fps)
        self._frame, self._t, self._stop, self._thread = None, 0.0, threading.Event(), None

    def connect(self):
        import cv2
        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera '{self.name}': cannot open {self.index!r}")
        ok, f = self._cap.read()
        if not ok or f.shape[:2] != (self.height, self.width):
            raise RuntimeError(f"camera '{self.name}': gave {None if not ok else f.shape[:2]}, wanted {(self.height, self.width)}")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import cv2
        while not self._stop.is_set():
            ok, f = self._cap.read()
            if ok:
                self._frame, self._t = cv2.cvtColor(f, cv2.COLOR_BGR2RGB), time.time()

    def read(self):
        if self._frame is None:
            raise RuntimeError(f"camera '{self.name}': no frame yet")
        return self._frame.copy()

    def frame_age(self):
        return time.time() - self._t if self._frame is not None else float("inf")

    def disconnect(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._cap.release()


class OrbbecCamera(Camera):
    """Orbbec colour stream, produced by tools/orbbec_stream.py in the camera venv and shared through a mmap'd file."""

    HELPER = "orbbec_stream.py"
    LABEL = "Orbbec"

    def __init__(self, name, serial=None, width=640, height=480, fps=30, python=None, sdk_path=None, settings=None):
        self.settings = dict(settings or {})                 # exposure / gain / white_balance ... (tools/orbbec_stream.py)
        self.name, self.serial, self.width, self.height, self.fps = name, serial, int(width), int(height), int(fps)
        self.python = str(python or ROOT / ".venv_cam" / "bin" / "python")
        self.sdk_path = os.path.expanduser(str(sdk_path or "~/orbbec_v1_legacy/install/lib"))
        self._proc = self._mm = None
        self._path = None

    def connect(self):
        size = 16 + self.width * self.height * 3
        fd, self._path = tempfile.mkstemp(prefix=f"piper_cam_{self.name}_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        os.ftruncate(fd, size)
        self._mm = mmap.mmap(fd, size)
        os.close(fd)
        env = self._env()
        cmd = [self.python, str(ROOT / "tools" / self.HELPER), self._path, str(self.width), str(self.height), str(self.fps)]
        if self.serial:
            cmd += ["--serial", self.serial]
        sdk = {k: v for k, v in self.settings.items() if k != "rgb_gain"}
        if sdk:
            cmd += ["--settings", json.dumps(sdk)]
        if self.settings.get("rgb_gain"):
            cmd += ["--rgb-gain", ",".join(f"{float(g):.4f}" for g in self.settings["rgb_gain"])]
        self._proc = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True)
        t0 = time.time()
        while struct.unpack("<Q", self._mm[0:8])[0] == 0:
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() or "").strip().splitlines()
                self.disconnect()
                raise RuntimeError(f"camera '{self.name}' ({self.LABEL} {self.serial or 'first found'}) did not start: "
                                   f"{err[-1] if err else 'helper exited'}")
            if time.time() - t0 > 10.0:
                self.disconnect()
                raise RuntimeError(f"camera '{self.name}' ({self.LABEL} {self.serial or ''}): no frame within 10 s "
                                   "(on a 12 Mb/s hub port? plug it straight into the laptop)")
            time.sleep(0.02)

    def _env(self):
        return dict(os.environ, PYTHONPATH=self.sdk_path + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def read(self):
        for _ in range(5):
            seq = struct.unpack("<Q", self._mm[0:8])[0]
            frame = np.frombuffer(self._mm[16:16 + self.width * self.height * 3], np.uint8).reshape(self.height, self.width, 3).copy()
            if struct.unpack("<Q", self._mm[0:8])[0] == seq:
                return frame
        return frame

    def frame_age(self):
        return time.time() - struct.unpack("<d", self._mm[8:16])[0]

    def disconnect(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._path:
            try:
                os.unlink(self._path)
            except OSError:
                pass
            self._path = None


class XtionCamera(OrbbecCamera):
    """ASUS Xtion (PrimeSense) colour stream through OpenNI2 - tools/xtion_stream.py in the camera venv, same shared-file
    protocol as the Orbbec.  No serial (one Xtion per machine: it opens the first it finds).  ``settings`` may hold
    auto_exposure / exposure / gain / auto_white_balance (OpenNI2's own controls) and the software ``rgb_gain``."""

    HELPER = "xtion_stream.py"
    LABEL = "ASUS Xtion"

    def __init__(self, name, width=640, height=480, fps=30, python=None, settings=None):
        super().__init__(name, None, width, height, fps, python=python, settings=settings)

    def _env(self):
        return dict(os.environ)


def suggest_rgb_gain(mean_rgb) -> list[float]:
    """Per-channel multipliers that turn a white card's measured mean colour into neutral grey (green is the reference)."""
    r, g, b = (float(x) for x in mean_rgb)
    if min(r, g, b) < 1:
        raise ValueError("the white card is too dark to measure")
    gains = [g / r, 1.0, g / b]
    return [round(x, 3) for x in gains]


def make_camera(name: str, cfg: dict) -> Camera:
    kind = cfg.get("type", "opencv")
    common = {k: cfg[k] for k in ("width", "height", "fps") if k in cfg}
    if kind == "fake":
        return FakeCamera(name, **common)
    if kind == "opencv":
        return OpenCVCamera(name, cfg.get("index", 0), **common)
    if kind == "orbbec":
        return OrbbecCamera(name, cfg.get("serial"), python=cfg.get("python"), sdk_path=cfg.get("sdk_path"),
                            settings=cfg.get("settings"), **common)
    if kind == "xtion":
        return XtionCamera(name, python=cfg.get("python"), settings=cfg.get("settings"), **common)
    raise ValueError(f"camera '{name}': unknown type '{kind}' (fake | opencv | orbbec | xtion)")
