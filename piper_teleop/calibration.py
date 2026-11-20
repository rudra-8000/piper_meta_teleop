"""Guided axis calibration: the operator pushes the controllers FORWARD, RIGHT and UP.

Why: the Quest's tracking frame has an arbitrary heading (whichever way the headset faced when the VR session
started), and a mirrored or skewed idea of "right"/"up" produces exactly the kind of confusing arm behaviour that
is hard to diagnose.  Three short, checked moves give us the operator's real axes:

* forward + right  -> heading of the operator in the tracking frame (the two must agree, 90 degrees apart)
* up               -> sanity check against gravity (the tracking frame's +y is exactly up)

Pure state machine: ``feed(samples, now)`` is called every control tick and returns UI events
(``("speak", text)`` and ``("haptic", hand, intensity, ms)``).  No hardware, no network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

INTRO = ("Axis calibration. Stand where you will work, facing the table, and hold both controllers in front of your "
         "chest. For each move, squeeze the grip buttons, move, then let go.")
STEPS = (
    ("forward", "Push both controllers straight forward, away from your body, about twenty centimeters, then let go."),
    ("right", "Now move both controllers to your right, about twenty centimeters, then let go."),
    ("up", "Now lift both controllers straight up, about twenty centimeters, then let go."),
)


def azimuth_deg(v) -> float:
    """Horizontal direction of a tracking-frame vector: degrees clockwise (seen from above) from tracking-forward
    (-z) toward tracking-right (+x)."""
    return math.degrees(math.atan2(v[0], -v[2]))


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class _Rec:
    """One hand's recording.  Endpoints are averages of a few samples and the path length is measured in 1 cm
    steps, so tracking jitter cannot make a straight move look crooked."""

    def __init__(self, pos):
        self.first = [pos.copy()]
        self.recent = [pos.copy()]
        self.anchor = pos.copy()
        self.path = 0.0

    def add(self, pos):
        if len(self.first) < 5:
            self.first.append(pos.copy())
        self.recent = (self.recent + [pos.copy()])[-5:]
        d = float(np.linalg.norm(pos - self.anchor))
        if d >= 0.01:
            self.path += d
            self.anchor = pos.copy()

    @property
    def start(self):
        return np.mean(self.first, axis=0)

    @property
    def last(self):
        return np.mean(self.recent, axis=0)

    @property
    def total_path(self):
        return self.path + float(np.linalg.norm(self.recent[-1] - self.anchor))


@dataclass
class GuidedCalibration:
    min_move: float = 0.10            # m: a move shorter than this is rejected
    min_straight: float = 0.75        # displacement / path length
    min_horizontal: float = 0.70      # forward/right moves must be mostly horizontal
    max_axis_error_deg: float = 35.0  # forward-vs-right disagreement
    step_timeout: float = 90.0
    max_attempts: int = 3
    grip_on: float = 0.6
    active: bool = False
    status: str = "idle"              # idle | running | done | failed
    msg: str = ""
    step: int = 0
    phase: str = "idle"               # intro | wait | record
    result: dict = field(default_factory=dict)
    _t_phase: float = 0.0
    _t_step: float = 0.0
    _recs: dict = field(default_factory=dict)
    _vec: dict = field(default_factory=dict)
    _attempt: int = 0

    # ------------------------------------------------------------------ control
    def start(self, now: float) -> list:
        self.active, self.status, self.step, self.phase = True, "running", 0, "intro"
        self._t_phase = self._t_step = now
        self._recs, self._vec, self._attempt, self.result = {}, {}, 1, {}
        self.msg = STEPS[0][1]
        return [("speak", INTRO + " " + STEPS[0][1])]

    def cancel(self, text: str = "Calibration cancelled.") -> list:
        if not self.active:
            return []
        self.active, self.status, self.msg, self.phase = False, "idle", text, "idle"
        return [("speak", text)]

    def snapshot(self) -> dict:
        return {"active": self.active, "status": self.status, "step": self.step, "of": len(STEPS),
                "name": STEPS[min(self.step, len(STEPS) - 1)][0], "phase": self.phase, "msg": self.msg,
                "result": self.result}

    # --------------------------------------------------------------------- tick
    def _held(self, samples: dict, now: float) -> dict:
        return {h: s for h, s in samples.items()
                if s is not None and now - s.t_recv < 0.25 and s.tracked and not s.emulated and s.squeeze > self.grip_on}

    def feed(self, samples: dict, now: float) -> list:
        if not self.active:
            return []
        ev: list = []
        if now - self._t_step > self.step_timeout:
            return self._fail("Calibration timed out. Press both thumbsticks to try again.")
        held = self._held(samples, now)
        if self.phase == "intro":
            if not held and now - self._t_phase > 1.0:          # grips must be released before a move can start
                self.phase = "wait"
        elif self.phase == "wait":
            if held:
                self.phase = "record"
                self._recs = {h: _Rec(s.pos) for h, s in held.items()}
                ev += [("haptic", h, 0.3, 30) for h in held]
        elif self.phase == "record":
            for h, s in held.items():
                r = self._recs.get(h)
                if r is None:
                    self._recs[h] = _Rec(s.pos)
                else:
                    r.add(s.pos)
            if not held:                                         # all grips released: judge the move
                ev += self._judge(now)
        return ev

    # ------------------------------------------------------------------ judging
    def _retry(self, now: float, text: str) -> list:
        self.phase, self._t_phase, self.msg = "intro", now, text + " " + STEPS[self.step][1]
        return [("speak", self.msg)] + [("haptic", h, 1.0, 250) for h in ("left", "right")]

    def _judge(self, now: float) -> list:
        name = STEPS[self.step][0]
        cands = []
        for h, r in self._recs.items():
            d = r.last - r.start
            length = float(np.linalg.norm(d))
            cands.append((h, d, length, length / max(r.total_path, 1e-6)))
        self._recs = {}
        good = [c for c in cands if c[2] >= self.min_move and c[3] >= self.min_straight]
        if name != "up":
            good = [c for c in good if math.hypot(c[1][0], c[1][2]) / c[2] >= self.min_horizontal]
        if not good:
            best = max(cands, key=lambda c: c[2]) if cands else None
            if best is None or best[2] < self.min_move:
                cm = 0 if best is None else round(best[2] * 100)
                return self._retry(now, f"That move was only {cm} centimeters. Please move about twenty.")
            if best[3] < self.min_straight:
                return self._retry(now, "That move was not straight. Move in a straight line.")
            return self._retry(now, "That move should be level, not up or down.")
        if len(good) == 2:
            u0, u1 = good[0][1] / good[0][2], good[1][1] / good[1][2]
            if float(np.dot(u0, u1)) < 0.6:
                return self._retry(now, "Your two hands moved in different directions.")
        v = sum(c[1] / c[2] for c in good)
        self._vec[name] = v / float(np.linalg.norm(v))
        self.step += 1
        ev = [("haptic", h, 0.6, 60) for h in ("left", "right")]
        if self.step >= len(STEPS):
            return ev + self._finish(now)
        self.phase, self._t_phase = "intro", now
        self.msg = STEPS[self.step][1]
        return ev + [("speak", self.msg)]

    def _restart(self, now: float, text: str) -> list:
        self._attempt += 1
        if self._attempt > self.max_attempts:
            return self._fail("Calibration failed three times. Check where you stand and press both thumbsticks to try again.")
        self.step, self._vec, self.phase, self._t_phase = 0, {}, "intro", now
        self.msg = text + " Let's start again. " + STEPS[0][1]
        return [("speak", self.msg)] + [("haptic", h, 1.0, 250) for h in ("left", "right")]

    def _finish(self, now: float) -> list:
        f, r, u = self._vec["forward"], self._vec["right"], self._vec["up"]
        az_f, az_r = azimuth_deg(f), azimuth_deg(r)
        right_vs_forward = wrap_deg(az_r - az_f)               # +90 when "right" really is right of "forward"
        if abs(wrap_deg(right_vs_forward + 90.0)) < self.max_axis_error_deg:
            return self._restart(now, "It looks like you moved left instead of right.")
        err = abs(wrap_deg(right_vs_forward - 90.0))
        if err > self.max_axis_error_deg:
            return self._restart(now, "Forward and right were not at right angles to each other.")
        heading = math.degrees(math.atan2(
            math.sin(math.radians(az_f)) + math.sin(math.radians(az_r - 90.0)),
            math.cos(math.radians(az_f)) + math.cos(math.radians(az_r - 90.0))))
        up_tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(u[1])))))
        self.result = {"yaw_deg": round(heading, 1), "axis_error_deg": round(err, 1), "up_tilt_deg": round(up_tilt, 1),
                       "forward": [round(float(x), 3) for x in f], "right": [round(float(x), 3) for x in r],
                       "up": [round(float(x), 3) for x in u]}
        self.active, self.status, self.phase = False, "done", "idle"
        warn = " Your up move was tilted; keep it vertical next time." if up_tilt > 30 else ""
        self.msg = f"Calibration complete. Heading {heading:.0f} degrees.{warn} Let go, then squeeze a grip to start moving the arms."
        return [("speak", self.msg), ("done", dict(self.result))] + [("haptic", h, 0.8, 120) for h in ("left", "right")]

    def _fail(self, text: str) -> list:
        self.active, self.status, self.phase, self.msg = False, "failed", "idle", text
        return [("speak", text)] + [("haptic", h, 1.0, 400) for h in ("left", "right")]
