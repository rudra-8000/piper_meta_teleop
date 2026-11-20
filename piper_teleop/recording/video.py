"""Streaming mp4 encoding (PyAV) and lossless concatenation of episode videos.

Every episode is encoded to its own mp4 while it is recorded (a background thread per camera, so the 30 Hz loop never
waits for the encoder).  v2.1 keeps one mp4 per episode; v3.0 appends it to the current rolling file with `concat_mp4`
(packet copy, no re-encode).  The stream time base is 1/fps and frame i has pts i, so a frame's timestamp is exactly
i / fps - what LeRobot's `from_timestamp` / `to_timestamp` columns assume.  B-frames are off and keyframes are frequent
(gop 2, like LeRobot) so random access during training is cheap.
"""
from __future__ import annotations

import queue
import threading
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


class StreamEncoder:
    def __init__(self, path, width: int, height: int, fps: int, codec: str = "libx264", crf: int = 23, gop: int = 2,
                 preset: str = "veryfast"):
        self.path = str(path)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(self.path, "w")
        self.stream = self.container.add_stream(codec, rate=self.fps)
        self.stream.width, self.stream.height, self.stream.pix_fmt = self.width, self.height, "yuv420p"
        self.stream.time_base = Fraction(1, self.fps)
        opts = {"g": str(gop), "bf": "0"}
        if codec in ("libx264", "libx265", "h264", "hevc"):
            opts.update({"crf": str(crf), "preset": preset})
        elif codec in ("libaom-av1", "libsvtav1"):
            opts.update({"crf": str(crf)})
        self.stream.options = opts
        self.n = 0
        self.error: Exception | None = None
        self._q: queue.Queue = queue.Queue(maxsize=90)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def put(self, rgb: np.ndarray):
        if self.error:
            raise RuntimeError(f"video encoder failed: {self.error}")
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"frame {rgb.shape} does not match the camera's {(self.height, self.width, 3)}")
        self._q.put(np.ascontiguousarray(rgb, dtype=np.uint8))

    def _run(self):
        i = 0
        try:
            while True:
                arr = self._q.get()
                if arr is None:
                    break
                fr = av.VideoFrame.from_ndarray(arr, format="rgb24")
                fr.pts = i
                fr.time_base = Fraction(1, self.fps)
                i += 1
                for pkt in self.stream.encode(fr):
                    self.container.mux(pkt)
            for pkt in self.stream.encode():
                self.container.mux(pkt)
            self.container.close()
            self.n = i
        except Exception as e:                      # surfaced by put() / close()
            self.error = e

    def close(self) -> int:
        self._q.put(None)
        self._t.join()
        if self.error:
            raise RuntimeError(f"video encoder failed: {self.error}")
        return self.n


def concat_mp4(dst, srcs) -> None:
    """Join mp4 files of identical encoding into `dst` by copying packets (no re-encode).

    The running offset is kept in seconds and converted to each input's own time base (a file that was itself produced
    by a concatenation can have a different one), so any number of joins keeps frame i at exactly i / fps."""
    out = av.open(str(dst), "w")
    ostream = None
    offset_s = Fraction(0)
    for src in srcs:
        with av.open(str(src)) as inp:
            ins = inp.streams.video[0]
            if ostream is None:
                ostream = out.add_stream(template=ins)
            off = int(round(offset_s / ins.time_base))
            n = 0
            for pkt in inp.demux(ins):
                if pkt.dts is None:
                    continue
                pkt.stream = ostream
                pkt.pts += off
                pkt.dts += off
                out.mux(pkt)
                n += 1
            offset_s += Fraction(n) / ins.average_rate
    out.close()


def count_frames(path) -> int:
    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(video=0))
