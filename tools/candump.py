#!/usr/bin/env python3
"""Minimal candump: log every frame on a SocketCAN interface (listen-only; it never transmits).

    python tools/candump.py can_left 10 logs/can_left.log      # 10 s, candump -L format: (ts) iface ID#DATA

Frames sent by other programs on this PC appear too (SocketCAN local loopback), so a test's own commands and the arm's
responses end up in one timeline.  Prints a per-ID summary at the end.
"""
import collections
import sys
import time

import can

iface, secs, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cnt, first, last = collections.Counter(), {}, {}
errors = 0


def open_bus():
    """(Re)open the interface; it disappears when the arm is power-cycled (its USB-CAN adapter resets)."""
    while time.time() < t_end:
        try:
            with open(f"/sys/class/net/{iface}/operstate") as fs:
                up = fs.read().strip() in ("up", "unknown")
        except OSError:
            up = False                                   # adapter not (re)enumerated yet
        if up:
            try:
                return can.Bus(channel=iface, interface="socketcan")
            except (OSError, can.CanError):
                pass
        time.sleep(0.05)
    return None


t_end = time.time() + secs
bus = open_bus()
with open(out, "w") as f:
    while bus is not None and (left := t_end - time.time()) > 0:
        try:
            m = bus.recv(min(left, 0.2))
        except (OSError, can.CanError) as e:
            f.write(f"({time.time():.6f}) {iface} # interface lost: {e}\n")
            print(f"{time.strftime('%H:%M:%S')} interface lost ({e}) - waiting for it to come back", flush=True)
            try:
                bus.shutdown()
            except Exception:
                pass
            bus = open_bus()
            f.write(f"({time.time():.6f}) {iface} # interface back\n")
            print(f"{time.strftime('%H:%M:%S')} interface back", flush=True)
            continue
        if m is None:
            continue
        if m.is_error_frame:
            errors += 1
            f.write(f"({m.timestamp:.6f}) {iface} ERROR#{bytes(m.data).hex().upper()}\n")
            continue
        cid = m.arbitration_id
        f.write(f"({m.timestamp:.6f}) {iface} {cid:03X}#{bytes(m.data).hex().upper()}\n")
        cnt[cid] += 1
        first.setdefault(cid, bytes(m.data))
        last[cid] = bytes(m.data)
if bus is not None:
    bus.shutdown()
print(f"{iface}: {sum(cnt.values())} frames, {len(cnt)} IDs, {errors} error frames in {secs:.0f} s -> {out}")
for cid in sorted(cnt):
    ch = "" if first[cid] == last[cid] else "  (changed)"
    print(f"  0x{cid:03X} {cnt[cid] / secs:7.1f} Hz  last {last[cid].hex(' ')}{ch}")
