"""One-Euro style adaptive low-pass filters for position and orientation.

Low cutoff when the hand is slow (kills tracking jitter), high cutoff when it
moves fast (almost no added lag).  See Casiez et al., "1 Euro Filter", CHI 2012.
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(dt: float, fc: float) -> float:
    tau = 1.0 / (2.0 * math.pi * fc)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 3.0, beta: float = 12.0, d_cutoff: float = 4.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = None
        self._dx = None

    def __call__(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or dt <= 0:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            return self._x.copy()
        dx = (x - self._x) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        fc = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = _alpha(dt, fc)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()


class OneEuroQuat:
    """Same idea on SO(3): slerp toward the new sample with an adaptive gain."""

    def __init__(self, min_cutoff: float = 3.0, beta: float = 0.6, d_cutoff: float = 4.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._q = None
        self._w = 0.0  # filtered angular speed [rad/s]

    def __call__(self, q, dt: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)
        if self._q is None or dt <= 0:
            self._q = q.copy()
            self._w = 0.0
            return q.copy()
        d = float(np.dot(self._q, q))
        if d < 0.0:
            q, d = -q, -d
        ang = 2.0 * math.acos(min(1.0, d))
        w = ang / dt
        a_d = _alpha(dt, self.d_cutoff)
        self._w = a_d * w + (1.0 - a_d) * self._w
        a = _alpha(dt, self.min_cutoff + self.beta * self._w)
        self._q = slerp(self._q, q, a)
        return self._q.copy()


def slerp(q0, q1, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    th = math.acos(d)
    s = math.sin(th)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / s
