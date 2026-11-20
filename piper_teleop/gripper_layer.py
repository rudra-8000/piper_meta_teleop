"""Input layer between gripper targets (teleop trigger, leader pendant, a policy) and raw Piper gripper commands.

Why (measured on the real arm, firmware S-V1.8-9, both grippers): the gripper only adopts a new position target while
the command is varying slowly and continuously.  A target that jumps and then stays constant freezes it wherever it
happens to be (holds 2/9 step targets reached); frame-to-frame flicker of the command (1 um - 1 mm at 100 Hz) does not
wake it.  A slow sine on top of the command does: +-0.25 mm @ 3 Hz already reached 9/9 holds, +-0.5 mm @ 1-6 Hz
reached 9/9 with mean error 0.4 mm and no measurable finger jiggle (0.00 mm peak-to-peak), +-1 mm jiggles ~0.5 mm.
Frequency: 6 Hz was the fastest that stayed perfect (27/27 holds over 3 runs, response ~0.44 s vs ~0.48 s at 3 Hz);
10-15 Hz worked once but slipped to 7-8/9 on repeat; 25 Hz reached 4/9, 50 Hz 3/9.

Step alignment: how soon the gripper reacts to a step depends on where in the sine the step lands (measured on the right
gripper, 12 phases x 3 reps: start delay 0.06 s at the best phase, 0.5 s typical worst, up to 0.9 s; closing is the exact
mirror of opening, shifted by 180 deg).  So when the target jumps by >= ``step_mm`` in one tick, the sine phase is set to
270 deg for an opening step and 90 deg for a closing step.  Smooth targets (a trigger sweep) are untouched.

Ramp: with the sine on, NO ramp was the fastest and most reliable (27/27 holds over 3 runs, ~0.25-0.3 s to arrive);
a 150 mm/s ramp took 0.52 s, and 300 / 600 mm/s ramps occasionally failed.  ``slew_mm_s`` stays available for gentler
closing.

    layer = GripperLayer()                 # no ramp + 0.5 mm @ 6 Hz sine
    layer.reset(measured_opening_m)        # once, from where the gripper really is
    cmd_m = layer.update(target_m, dt)     # every control tick; send cmd_m to the gripper
"""
from __future__ import annotations

import math


OPEN_PHASE = math.radians(270.0)               # sine phase that gets an opening step moving soonest (measured)
CLOSE_PHASE = math.radians(90.0)               # ... and a closing step (the mirror image)


class GripperLayer:
    def __init__(self, slew_mm_s: float = 0.0, dither_mm: float = 0.5, dither_hz: float = 6.0,
                 max_m: float = 0.070, step_mm: float = 3.0, align: bool = True):
        self.slew = slew_mm_s * 1e-3            # m/s; 0 = no ramp
        self.amp = dither_mm * 1e-3             # m;   0 = no sine
        self.hz = dither_hz
        self.max = max_m
        self.step = step_mm * 1e-3              # m; a one-tick target change at least this big is a "step"
        self.align = align                      # line the sine up with a step (see the module docstring)
        self._last_target: float | None = None
        self.ramp = 0.0                         # rate-limited target (what the gripper should end up at)
        self.cmd = 0.0                          # last raw command (ramp + sine, clipped to the stroke)
        self._phase = 0.0

    def reset(self, opening_m: float):
        self.ramp = self.cmd = min(self.max, max(0.0, float(opening_m)))
        self._phase = 0.0
        self._last_target = None

    def update(self, target_m: float, dt: float) -> float:
        target_m = min(self.max, max(0.0, float(target_m)))
        dphi = 2.0 * math.pi * self.hz * dt
        if self.align and self._last_target is not None and abs(target_m - self._last_target) >= self.step:
            self._phase = (OPEN_PHASE if target_m > self._last_target else CLOSE_PHASE) - dphi   # the advance lands on it
        self._last_target = target_m
        if self.slew > 0:
            step = self.slew * dt
            self.ramp += max(-step, min(step, target_m - self.ramp))
        else:
            self.ramp = target_m
        self._phase = (self._phase + dphi) % (2.0 * math.pi)
        self.cmd = min(self.max, max(0.0, self.ramp + self.amp * math.sin(self._phase)))
        return self.cmd
