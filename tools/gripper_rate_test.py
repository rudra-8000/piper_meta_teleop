#!/usr/bin/env python3
"""How fast does the gripper converge, and can it follow a policy at 30 / 50 / 100 Hz?  (gripper frames only)

    python tools/gripper_rate_test.py can_left converge logs/gripper_converge.csv
    python tools/gripper_rate_test.py can_left policy   logs/gripper_policy.csv

converge : step targets through the gripper layer (no ramp) for several sine frequencies; per step: dead time (first
           1 mm of motion toward the target), time to within 3 mm and 1 mm.  Two passes in different orders.
policy   : a smooth full-stroke target trajectory sampled at 30 / 50 / 100 / 200 Hz (zero-order hold, like a policy
           or a dataset replay), sent through the layer at 200 Hz; plus random step targets every 0.5 s at 30 Hz.
           Reports RMS error, lag (best time shift) and RMS error after removing that lag.
Every 0x159 frame is checked against config/joint_lock.yaml.
"""
import csv
import math
import random
import statistics
import struct
import sys
import time
from pathlib import Path

import can

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop.gripper_layer import GripperLayer  # noqa: E402
from piper_teleop.joint_lock import check  # noqa: E402

iface, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
bus = can.Bus(channel=iface, interface="socketcan")
fb = {}
rows = []
DT = 0.005


def poll(dt):
    end = time.time() + dt
    while (left := end - time.time()) > 0:
        m = bus.recv(left)
        if m is not None and m.arbitration_id == 0x2A8:
            w, _ = struct.unpack(">ih", bytes(m.data[:6]))
            fb["w"] = w / 1000


def send_mm(mm):
    check(iface, 0x159)
    bus.send(can.Message(arbitration_id=0x159, data=struct.pack(">ihBB", int(round(mm * 1000)), 1000, 0x01, 0),
                         is_extended_id=False))


def drive(layer, target_fn, secs, label):
    """target_fn(t) -> target mm; layer output sent every 5 ms.  Returns [(t, target, measured)]."""
    t0 = time.time()
    nxt = t0
    trace = []
    while (t := time.time() - t0) < secs:
        tgt = target_fn(t)
        send_mm(layer.update(tgt / 1000, DT) * 1000)
        nxt += DT
        poll(max(0.0, nxt - time.time()))
        trace.append((t, tgt, fb["w"]))
        rows.append([f"{time.time():.4f}", label, f"{tgt:.2f}", f"{fb['w']:.2f}"])
    return trace


poll(0.3)
if mode == "converge":
    holds = [50, 5, 65, 20, 0, 70, 30, 60, 15]
    freqs = [2, 3, 4, 6, 8, 10, 12]
    res = {f: [] for f in freqs}
    for p, order in enumerate((freqs, freqs[::-1])):
        for f in order:
            layer = GripperLayer(slew_mm_s=0, dither_mm=0.5, dither_hz=f)
            layer.reset(fb["w"] / 1000)
            for tgt in (holds if p == 0 else holds[::-1]):
                start = fb["w"]
                tr = drive(layer, lambda t, g=tgt: g, 1.5, f"{f} Hz pass{p + 1} -> {tgt}")
                dist = abs(tgt - start)
                if dist < 4:                                   # too small a step to time
                    continue
                sign = 1 if tgt > start else -1
                dead = next((t for t, _, m in tr if (m - start) * sign > 1.0), None)
                t3 = next((t for t, _, m in tr if abs(m - tgt) < 3.0), None)
                t1 = next((t for t, _, m in tr if abs(m - tgt) < 1.0), None)
                res[f].append((dist, dead, t3, t1))
    print(f"{'sine':>6s}  steps  reached<3mm  dead time    to 3 mm     to 1 mm     (median s; worst in brackets)")
    for f in freqs:
        r = res[f]
        ok3 = [x for x in r if x[2] is not None]
        med = lambda xs: statistics.median(xs) if xs else float("nan")
        worst = lambda xs: max(xs) if xs else float("nan")
        d = [x[1] for x in r if x[1] is not None]
        a3 = [x[2] for x in r if x[2] is not None]
        a1 = [x[3] for x in r if x[3] is not None]
        print(f"{f:4d} Hz  {len(r):4d}   {len(ok3):3d}/{len(r):<3d}    {med(d):5.3f} ({worst(d):4.2f})  {med(a3):5.3f} ({worst(a3):4.2f})"
              f"  {med(a1):5.3f} ({worst(a1):4.2f})  1mm reached {len(a1)}/{len(r)}", flush=True)
else:
    def smooth(t):                                             # full-stroke, policy-like motion (0.25 + 0.7 Hz content)
        return min(68.0, max(2.0, 35 + 30 * (0.6 * math.sin(2 * math.pi * 0.25 * t) + 0.4 * math.sin(2 * math.pi * 0.7 * t + 1))))

    def sampled(fn, rate):                                     # zero-order hold at the policy rate
        return lambda t: fn(math.floor(t * rate) / rate)

    print(f"{'policy rate':>12s}  {'RMS err':>8s}  {'lag':>6s}  {'RMS err after lag':>17s}  {'max err after lag':>17s}")
    for rate in (30, 50, 100, 200):
        layer = GripperLayer(slew_mm_s=0, dither_mm=0.5, dither_hz=6)
        layer.reset(fb["w"] / 1000)
        drive(layer, sampled(smooth, rate), 1.0, f"{rate} Hz settle")
        tr = drive(layer, sampled(smooth, rate), 16.0, f"policy {rate} Hz smooth")
        tg = [x[1] for x in tr]
        ms = [x[2] for x in tr]
        rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(tg, ms)) / len(tg))
        best = min(range(0, 120), key=lambda k: sum((tg[i] - ms[i + k]) ** 2 for i in range(len(tg) - k)) / (len(tg) - k))
        sh = [(tg[i] - ms[i + best]) for i in range(len(tg) - best)]
        print(f"{rate:9d} Hz  {rms:6.2f} mm  {best * DT:4.2f} s  {math.sqrt(sum(x * x for x in sh) / len(sh)):12.2f} mm"
              f"  {max(abs(x) for x in sh):14.2f} mm", flush=True)
    rng = random.Random(11)
    steps = [rng.uniform(2, 68) for _ in range(40)]
    layer = GripperLayer(slew_mm_s=0, dither_mm=0.5, dither_hz=6)
    layer.reset(fb["w"] / 1000)
    tr = drive(layer, sampled(lambda t: steps[int(t / 0.5) % 40], 30), 12.0, "policy 30 Hz random steps every 0.5 s")
    ends = [tr[i][2] - tr[i][1] for i in range(len(tr) - 1) if int(tr[i + 1][0] / 0.5) != int(tr[i][0] / 0.5)]
    print(f"  random steps every 0.5 s @ 30 Hz: error at the end of each 0.5 s step: median {statistics.median(map(abs, ends)):.2f} mm,"
          f" worst {max(map(abs, ends)):.2f} mm, within 3 mm {sum(abs(e) < 3 for e in ends)}/{len(ends)}")
bus.shutdown()
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t", "pattern", "target_mm", "measured_mm"])
    w.writerows(rows)
