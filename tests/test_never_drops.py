"""Nothing in this project may cut motor power unless the user explicitly asks (a disabled Piper drops its wrist)."""
import ast
import pathlib

import numpy as np

from piper_teleop.arm import SimArm
from piper_teleop.controller import ArmConfig, TeleopArm
from piper_teleop.system import TeleopSystem

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _system():
    be = SimArm(q0=np.radians(ArmConfig().home_deg), lag_s=0.0, max_speed=100)
    be.enable()
    return TeleopSystem([TeleopArm(ArmConfig(name="t"), be)], rate_hz=200), be


def test_shutdown_parks_but_leaves_motors_enabled_by_default():
    sys_, be = _system()
    sys_.start()
    sys_.shutdown(park=True, timeout=15)
    assert be._enabled is True                      # parked at rest and STILL HOLDING
    assert np.max(np.abs(be._q)) < np.radians(1.0)


def test_shutdown_releases_only_when_asked():
    sys_, be = _system()
    sys_.start()
    sys_.shutdown(park=True, timeout=15, release=True)
    assert be._enabled is False


def test_no_call_site_disables_without_a_release_flag():
    """Static guard: every .disable() call outside the backends must sit under a `release` or `consent` condition."""
    offenders = []
    for path in list((ROOT / "piper_teleop").glob("*.py")) + list((ROOT / "tools").glob("*.py")):
        if path.name in ("arm.py", "agx_arm.py"):   # the backends implement disable()/reset() themselves
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ("release" in ast.unparse(node.test) or "consent" in ast.unparse(node.test)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        sub._guarded = True
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (node.func.attr in ("disable", "DisablePiper") or (node.func.attr == "reset" and ast.unparse(node.func.value) in ("robot", "self._r", "r", "p", "self._p"))):
                if not getattr(node, "_guarded", False):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"unguarded disable() calls: {offenders}"
