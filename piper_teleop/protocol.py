"""Binary pose packet sent by the Quest page (little-endian).

header  : <B type=1> <B n_hands> <H seq> <d t_client_ms>                (12 bytes)
per hand: <B hand 0=left 1=right> <B flags>
          pos(3f) grip_quat xyzw(4f) ray_quat xyzw(4f)
          trigger squeeze stick_x stick_y (4f)                          (62 bytes)
flags   : bit0 tracked, bit1 emulated, bit2 A/X, bit3 B/Y, bit4 stick click

Keep this in sync with ``web/quest.js``.
"""
from __future__ import annotations

import struct

import numpy as np

from .controller import HandSample

HEADER = struct.Struct("<BBHd")
HAND = struct.Struct("<BB15f")
TYPE_POSE = 1
HAND_NAMES = ("left", "right")

F_TRACKED, F_EMULATED, F_A, F_B, F_STICK = 1, 2, 4, 8, 16


def encode_pose(seq: int, t_client_ms: float, hands: list[dict]) -> bytes:
    out = [HEADER.pack(TYPE_POSE, len(hands), seq & 0xFFFF, t_client_ms)]
    for h in hands:
        flags = ((F_TRACKED if h.get("tracked", True) else 0)
                 | (F_EMULATED if h.get("emulated") else 0)
                 | (F_A if h.get("a") else 0) | (F_B if h.get("b") else 0)
                 | (F_STICK if h.get("stick_click") else 0))
        out.append(HAND.pack(
            0 if h["hand"] == "left" else 1, flags, *h["pos"], *h["quat"], *h.get("ray_quat", (0, 0, 0, 1)),
            h.get("trigger", 0.0), h.get("squeeze", 0.0), *h.get("stick", (0.0, 0.0))))
    return b"".join(out)


def decode_pose(buf: bytes, t_recv: float) -> dict[str, HandSample]:
    """Returns {'left'|'right': HandSample}.  Raises ValueError on malformed input."""
    if len(buf) < HEADER.size:
        raise ValueError("short packet")
    typ, n, seq, t_client = HEADER.unpack_from(buf, 0)
    if typ != TYPE_POSE or n > 2 or len(buf) != HEADER.size + n * HAND.size:
        raise ValueError("bad packet")
    out = {}
    off = HEADER.size
    for _ in range(n):
        v = HAND.unpack_from(buf, off)
        off += HAND.size
        hid, flags = v[0], v[1]
        f = np.array(v[2:], dtype=float)
        if not np.all(np.isfinite(f)):
            continue
        quat = f[3:7]
        rq = f[7:11]
        if np.linalg.norm(quat) < 0.5 or np.linalg.norm(rq) < 0.5:
            continue
        name = HAND_NAMES[hid & 1]
        out[name] = HandSample(
            t_recv=t_recv, t_client=t_client, seq=seq,
            tracked=bool(flags & F_TRACKED), emulated=bool(flags & F_EMULATED),
            pos=f[0:3], quat=quat / np.linalg.norm(quat), ray_quat=rq / np.linalg.norm(rq),
            trigger=float(np.clip(f[11], 0, 1)), squeeze=float(np.clip(f[12], 0, 1)),
            stick=(float(f[13]), float(f[14])),
            btn_a=bool(flags & F_A), btn_b=bool(flags & F_B), stick_click=bool(flags & F_STICK))
    return out
