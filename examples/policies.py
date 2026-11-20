"""Example policies for  python -m piper_teleop.run_policy --policy examples/policies.py:NAME [--arg k=v] ...

A policy factory takes keyword arguments and returns ``policy(obs) -> action``: obs is ``robot.get_observation()`` (+ ``prompt``),
the action one (7,) vector [6 joints rad, gripper 0..1] or an (N, 7) chunk.  Replace these with your own model.
"""
from pathlib import Path

import numpy as np


def hold():
    """Do nothing: always target the pose the arm is at now.  A good first run to see the plumbing works."""
    return lambda obs: obs["observation.state"].astype(float)


def sine(amplitude_deg: float = 10.0, period_s: float = 6.0, hz: float = 30.0, gripper: bool = True):
    """Gently swing joints 2 and 3 around the pose the run started in (and open/close the gripper).  Moves the arm - clear the area."""
    t = {"i": 0, "base": None}

    def policy(obs):
        s = obs["observation.state"].astype(float)
        if t["base"] is None:
            t["base"] = s.copy()
        ph = 2 * np.pi * t["i"] / (period_s * hz)
        t["i"] += 1
        a = t["base"].copy()
        a[1] += np.radians(amplitude_deg) * np.sin(ph)
        a[2] -= np.radians(amplitude_deg) * np.sin(ph)
        if gripper:
            a[6] = 0.5 + 0.5 * np.sin(ph)
        return a

    return policy


def replay(root: str, episode: int = 0):
    """Replay a recorded episode's ACTIONS (LeRobot v3.0 or v2.1 dataset made by piper_teleop.record) as one chunk.  A good check
    that the whole action path reproduces the demonstration; the arm starts from rest, as the recording did."""
    import pyarrow.parquet as pq
    files = sorted(Path(root).glob("data/chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"no data files under {root}/data")
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "frame_index", "action"]).to_pydict()
        rows += [(fi, a) for e, fi, a in zip(t["episode_index"], t["frame_index"], t["action"]) if e == episode]
    if not rows:
        raise SystemExit(f"episode {episode} not found in {root}")
    actions = np.array([a for _, a in sorted(rows, key=lambda r: r[0])], dtype=float)
    done = {"v": False}

    def policy(obs):
        if done["v"]:
            return obs["observation.state"].astype(float)          # finished: hold
        done["v"] = True
        return actions

    policy.reset = lambda: done.update(v=False)
    return policy
