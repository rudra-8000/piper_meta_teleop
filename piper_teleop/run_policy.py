"""Run a policy / planner on the robot at a fixed rate.

    python -m piper_teleop.run_policy --policy examples/policies.py:sine --sim
    python -m piper_teleop.run_policy --policy my_pkg.my_module:make_policy --prompt "clear the table" --cameras xtion_top
    python -m piper_teleop.run_policy --policy examples/policies.py:replay --arg root=datasets/clear_table --arg episode=3
    python -m piper_teleop.run_policy --policy my_pkg.my_module:make_policy --remote 192.168.1.20:8800 --token SECRET   # via robot_server

``--policy MODULE:FACTORY`` (a dotted module or a path to a .py file).  The factory is called with the ``--arg key=value`` pairs and
returns the policy: a callable ``policy(obs) -> action``, where ``obs`` is ``robot.get_observation()`` plus ``obs["prompt"]``, and
``action`` is one (7,) action or a (N, 7) chunk.  A policy may also have ``reset()`` (called at the start of a run).  Chunks are
executed open-loop, ``--execute N`` actions of each before the policy is asked again (default: all of them).

What it does: confirms, folds the arm to rest, then every 1/hz s sends the next action.  Ctrl-C (or --steps / --duration-s) stops
the run: the arm holds, then folds to rest, and stays ENABLED - a second Ctrl-C skips the folding and just leaves it holding.
Importable too: ``run(robot, policy, hz=30, ...)``.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import signal
import sys
import time
from pathlib import Path

import numpy as np

from .robot import PiperRobot, Rate


def load_policy(spec: str, kwargs: dict):
    """``module:factory`` or ``path/to/file.py:factory`` -> ``factory(**kwargs)``."""
    if ":" not in spec:
        raise SystemExit("--policy needs MODULE:FACTORY (e.g. examples/policies.py:sine)")
    mod_name, _, attr = spec.rpartition(":")
    if mod_name.endswith(".py"):
        path = Path(mod_name).resolve()
        if not path.exists():
            raise SystemExit(f"policy file not found: {path}")
        sys.path.insert(0, str(path.parent))
        s = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    else:
        sys.path.insert(0, str(Path.cwd()))
        mod = importlib.import_module(mod_name)
    if not hasattr(mod, attr):
        raise SystemExit(f"{mod_name} has no '{attr}'")
    return getattr(mod, attr)(**kwargs)


def run(robot: PiperRobot, policy, *, hz: float = 30.0, prompt: str = "", steps: int = 0, duration_s: float = 0.0,
        execute: int = 0, reset: bool = True, stop=lambda: False, log=print) -> dict:
    """The control loop.  Returns {"steps", "policy_calls", "seconds", "late"} (``late`` = ticks that overran their slot).

    A policy exception, or an action the robot refuses, stops the run: the arm is told to hold first, then the error is raised."""
    if hasattr(policy, "reset"):
        policy.reset()
    if reset:
        robot.go_rest()
    queue: list[np.ndarray] = []
    rate = Rate(hz)
    t0 = time.perf_counter()
    n = calls = late = 0
    period = 1.0 / hz
    try:
        while not stop():
            tick = time.perf_counter()
            if (steps and n >= steps) or (duration_s and tick - t0 >= duration_s):
                break
            if not queue:
                obs = robot.get_observation()
                obs["prompt"] = prompt
                out = np.asarray(policy(obs), dtype=float)
                calls += 1
                if out.ndim == 1:
                    out = out[None]
                if out.ndim != 2 or out.shape[1] != 7 or len(out) == 0:
                    raise ValueError(f"the policy must return a (7,) action or a (N, 7) chunk, got shape {out.shape}")
                queue = list(out[: execute] if execute else out)
            robot.send_action(queue.pop(0))
            n += 1
            rate.sleep()
            if time.perf_counter() - tick > 1.5 * period:
                late += 1
    except BaseException:
        robot.hold()
        raise
    robot.hold()
    sec = time.perf_counter() - t0
    log(f"[run] {n} steps in {sec:.1f} s ({n / max(sec, 1e-9):.1f} Hz), {calls} policy calls, {late} late ticks")
    return {"steps": n, "policy_calls": calls, "seconds": sec, "late": late}


def _kv(items) -> dict:
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"--arg needs key=value, got {it!r}")
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="MODULE:FACTORY or FILE.py:FACTORY")
    ap.add_argument("--arg", action="append", metavar="KEY=VALUE", help="keyword argument for the factory (repeatable; numbers/lists are parsed)")
    ap.add_argument("--prompt", default="", help="task prompt, passed to the policy as obs['prompt']")
    ap.add_argument("--arm", help="physical arm to drive (default robot.arm in the config)")
    ap.add_argument("--hz", type=float, default=30.0, help="action rate (should match the rate the policy was trained at; default 30)")
    ap.add_argument("--steps", type=int, default=0, help="stop after this many actions (0 = until Ctrl-C)")
    ap.add_argument("--duration-s", type=float, default=0.0, help="stop after this many seconds")
    ap.add_argument("--execute", type=int, default=0, help="actions to run from each chunk before asking the policy again (0 = whole chunk)")
    ap.add_argument("--cameras", metavar="SET", help="record.camera_sets.SET instead of record.cameras (e.g. xtion_top)")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--max-joint-speed", type=float, help="rad/s cap (default robot.max_joint_speed)")
    ap.add_argument("--no-reset", action="store_true", help="do not fold to rest before starting")
    ap.add_argument("--no-park", action="store_true", help="do not fold to rest at the end (the arm just holds)")
    ap.add_argument("--remote", metavar="HOST:PORT", help="drive the arm through a running `python -m piper_teleop.robot_server` instead "
                    "of directly (cameras / sim / speed cap are then the server's settings)")
    ap.add_argument("--token", help="the server's token (with --remote)")
    ap.add_argument("--sim", action="store_true", help="simulated arm, no hardware")
    ap.add_argument("--config")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation before the arm moves")
    a = ap.parse_args(argv)

    policy = load_policy(a.policy, _kv(a.arg))
    stop = {"n": 0}

    def on_sigint(*_):
        stop["n"] += 1
        if stop["n"] >= 2:
            robot.park_on_exit = False
            robot.cancel_moves()
            print("\n[run] second Ctrl-C: not folding to rest - the arm holds its pose.", flush=True)

    if a.remote:
        if a.sim or a.cameras or a.no_cameras or a.max_joint_speed or a.arm:
            raise SystemExit("--sim / --arm / --cameras / --no-cameras / --max-joint-speed are the SERVER's settings with --remote")
        from .robot_client import PiperClient
        host, _, port = a.remote.rpartition(":")
        robot = PiperClient(host or "127.0.0.1", int(port), token=a.token)
    else:
        cams = False if a.no_cameras else (a.cameras if a.cameras else None)
        robot = PiperRobot(a.arm, sim=a.sim, cameras=cams, config=a.config, max_joint_speed=a.max_joint_speed,
                           park_on_exit=False if a.no_park else None)
    if not a.sim and not a.yes:
        input(f">>> The arm will fold to the rest pose, then {a.policy.rpartition(':')[2]} drives it "
              f"({'via ' + a.remote if a.remote else 'directly'}). Clear the workspace, then press Enter (Ctrl-C aborts)... ")
    signal.signal(signal.SIGINT, on_sigint)
    try:
        robot.connect()
        run(robot, policy, hz=a.hz, prompt=a.prompt, steps=a.steps, duration_s=a.duration_s, execute=a.execute,
            reset=not a.no_reset, stop=lambda: stop["n"] > 0)
    finally:
        if a.remote and not a.no_park and stop["n"] < 2:            # a local robot parks itself on close(); a client must ask the server
            try:
                robot.go_rest()
            except Exception as e:                                  # noqa: BLE001
                print(f"[run] could not fold to rest ({type(e).__name__}: {e}) - the arm holds its pose", flush=True)
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
