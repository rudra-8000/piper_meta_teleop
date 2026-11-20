import math

import numpy as np
import pytest

from piper_teleop.gripper_layer import GripperLayer

DT = 0.005


def run(layer, target, secs):
    return np.array([layer.update(target, DT) for _ in range(int(secs / DT))])


def test_ramp_then_sine_around_the_target():
    g = GripperLayer(150, 0.5, 3.0)
    g.reset(0.010)
    out = run(g, 0.040, 1.0)
    assert g.ramp == pytest.approx(0.040)                       # 30 mm at 150 mm/s = 0.2 s, then settled
    tail = out[-int(1 / 3 / DT):]                               # one full sine period at the end
    assert tail.max() == pytest.approx(0.0405, abs=2e-5) and tail.min() == pytest.approx(0.0395, abs=2e-5)
    assert np.max(np.abs(np.diff(out))) <= (0.150 + 2 * math.pi * 3 * 0.0005) * DT + 1e-12


def test_sine_frequency_is_what_was_asked():
    g = GripperLayer(0, 0.5, 3.0)
    g.reset(0.030)
    out = run(g, 0.030, 2.0) - 0.030
    crossings = np.sum((out[:-1] < 0) & (out[1:] >= 0))          # upward zero crossings in 2 s
    assert crossings in (5, 6)


def test_clipped_to_the_stroke_and_targets_clamped():
    g = GripperLayer(0, 0.5, 3.0, max_m=0.070)
    g.reset(0.0)
    out = np.concatenate([run(g, 0.0, 0.5), run(g, 0.090, 0.5), run(g, -0.01, 0.5)])
    assert out.min() >= 0.0 and out.max() <= 0.070


def test_reset_starts_from_where_the_gripper_is_and_off_switches():
    g = GripperLayer(150, 0.0, 3.0)
    g.reset(0.027)
    assert g.cmd == pytest.approx(0.027)
    assert g.update(0.027, DT) == pytest.approx(0.027)           # no sine, no target change: exactly the same value
    g2 = GripperLayer(0, 0.0, 0.0)
    g2.reset(0.0)
    assert g2.update(0.050, DT) == pytest.approx(0.050)          # no ramp, no sine: plain pass-through


def test_step_lines_the_sine_up_open_and_close():
    """How soon the gripper reacts to a step depends on the sine phase at the step: open -> 270 deg, close -> 90 deg."""
    import math
    for start_phase in (0.0, 1.0, 2.5, 4.0, 5.5):
        g = GripperLayer(slew_mm_s=0, dither_mm=0.5, dither_hz=6.0)
        g.reset(0.035)
        g._phase = start_phase
        g.update(0.035, 0.005)                          # hold: not a step
        assert g._last_target == pytest.approx(0.035)
        up = g.update(0.070, 0.005)                     # opening step
        assert up == pytest.approx(0.070 - 0.0005, abs=1e-9)           # sin(270 deg) = -1, clipped by the stroke
        assert math.sin(g._phase) == pytest.approx(-1.0, abs=1e-6)
        g.update(0.070, 0.005)
        down = g.update(0.010, 0.005)                   # closing step
        assert down == pytest.approx(0.010 + 0.0005, abs=1e-9)         # sin(90 deg) = +1
        assert math.sin(g._phase) == pytest.approx(1.0, abs=1e-6)


def test_smooth_targets_and_small_changes_keep_the_sine_running():
    g = GripperLayer(slew_mm_s=0, dither_mm=0.5, dither_hz=6.0)
    g.reset(0.035)
    phases = []
    for i in range(50):                                 # a trigger sweep: 1 mm per tick (200 Hz = 200 mm/s) is not a step
        g.update(0.035 + 0.001 * i, 0.005)
        phases.append(g._phase)
    d = [(b - a) % (2 * 3.141592653589793) for a, b in zip(phases, phases[1:])]
    assert max(d) == pytest.approx(min(d), abs=1e-9)    # the phase advanced at a constant rate: never re-aligned


def test_alignment_can_be_switched_off():
    g = GripperLayer(slew_mm_s=0, dither_mm=0.5, dither_hz=6.0, align=False)
    g.reset(0.035)
    g._phase = 1.0
    g.update(0.035, 0.005)
    p = g._phase
    g.update(0.070, 0.005)
    assert g._phase == pytest.approx(p + 2 * 3.141592653589793 * 6.0 * 0.005)
