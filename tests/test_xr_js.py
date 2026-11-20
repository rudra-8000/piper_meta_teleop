"""Run the real web/quest.js against a fake WebXR (tests/xr_harness.js) and feed its packets to the real decoder."""
import json
import pathlib
import shutil
import subprocess

import pytest

from piper_teleop.protocol import decode_pose

HERE = pathlib.Path(__file__).parent
QUEST_JS = HERE.parent / "web" / "quest.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run(scenario):
    r = subprocess.run(["node", str(HERE / "xr_harness.js"), str(QUEST_JS), scenario], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr + r.stdout
    return json.loads(r.stdout.strip().splitlines()[-1])


def states(out):
    return [x for x in out["reports"] if x.get("t") == "xr"]


def test_frames_start_even_if_update_target_frame_rate_never_resolves():
    """Regression for the black-screen bug: the frame loop must not wait for updateTargetFrameRate."""
    out = run("hang_rate")
    assert out["raf"] >= 4, "frame loop never started / never continued"
    assert out["binary"], "no pose packets sent"
    assert any(x["state"] == "running" for x in states(out))
    assert not [x for x in states(out) if x["state"] == "error"]


def test_packet_from_js_decodes_in_python_with_the_right_buttons():
    out = run("ar")
    hands = decode_pose(bytes.fromhex(out["binary"][-1]), 0.0)
    left, right = hands["left"], hands["right"]
    assert left.squeeze == pytest.approx(1.0) and left.trigger == pytest.approx(0.25)     # GRIP / TRIGGER mapping
    assert right.squeeze == pytest.approx(0.0) and right.trigger == pytest.approx(1.0)
    assert left.pos[0] == pytest.approx(-0.2, abs=1e-6) and right.pos[0] == pytest.approx(0.2, abs=1e-6)
    assert left.tracked and not left.emulated


def test_passthrough_is_fully_transparent_and_status_squares_are_drawn():
    out = run("ar")
    clears = [c for c in out["gl"] if c[0] == "clearColor"]
    assert clears[0][1:] == [0, 0, 0, 0], "AR background must be fully transparent (passthrough visible)"
    assert any(c[0] == "scissor" for c in out["gl"])                          # head-locked status squares drawn
    # left GRIP is held in the fake: a white dot (clearColor 1,1,1,1) must be drawn for it
    assert [c for c in clears if c[1:] == [1, 1, 1, 1]]


def test_vr_background_is_opaque_dark():
    out = run("vr")
    first = [c for c in out["gl"] if c[0] == "clearColor"][0]
    assert first[4] == 1 and first[1] < 0.1


def test_watchdog_reports_when_the_browser_never_calls_back():
    out = run("no_frames")
    errs = [x for x in states(out) if x["state"] == "error"]
    assert errs and "has not called the frame callback" in errs[0]["msg"]


def test_frame_errors_are_reported_not_swallowed():
    out = run("frame_error")
    errs = [x for x in states(out) if x["state"] == "error"]
    assert errs and "frame loop: boom" in errs[0]["msg"]
    assert out["raf"] >= 3                                                    # and the loop keeps running
