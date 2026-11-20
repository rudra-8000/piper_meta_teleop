"""Record a LeRobot dataset (v3.0 or v2.1, 30 Hz) while teleoperating arm-to-arm.

    python -m piper_teleop.record --root datasets/pick_cube --task "pick up the cube"
    python -m piper_teleop.record --root datasets/pick_cube --format v2.1          # for OpenPI
    python -m piper_teleop.record --root datasets/pick_cube --sim --fake-camera    # no hardware at all

The arms behave exactly as in ``python -m piper_teleop.leader_follower`` (same options: --mode, --j1-offset, --leader ...).
Keys (see piper_teleop/recording/session.py): Enter start / finish, p pause / resume, r discard & retake, k keep,
d discard, q quit, h help.  Pointing --root at an existing dataset of the same format APPENDS to it.

Per frame (30 Hz):
    observation.state   (7,) float32   follower joint angles 1..6 [rad], gripper opening 0..1 (1 = fully open, 70 mm)
    action              (7,) float32   the leader's joints mapped onto the follower [rad], gripper target 0..1
Both are named after the physical arm the policy drives (the follower): left_joint_1 ... left_joint_6, left_gripper.
    observation.images.<camera>        (H, W, 3) video, one per camera in config/teleop.yaml `record.cameras`
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import leader_follower as lf
from .arm import GRIPPER_MAX_M
from .recording.cameras import make_camera
from .recording.session import KeyReader, RecordingSession
from .recording.writer import LeRobotWriter



def feature_names(side: str) -> list[str]:
    """`left_joint_1` ... `left_joint_6`, `left_gripper`: named after the physical arm the policy controls (the follower)."""
    return [f"{side}_joint_{i}" for i in range(1, 7)] + [f"{side}_gripper"]


class ArmToArmSource:
    """Turns the rig's last control tick into one (observation.state, action) pair."""

    def __init__(self, rig: lf.LeaderFollower):
        self.rig = rig

    def sample(self):
        r = self.rig
        st = r.last_follower
        if not r.synced or not r.last_tick_ok or st is None or not st.fresh:     # still aligning / leader silent: not a demonstration
            return None
        state = np.append(st.q, np.clip(st.gripper / GRIPPER_MAX_M, 0.0, 1.0))
        action = np.append(r.last_goal, np.clip(r.last_g_target / GRIPPER_MAX_M, 0.0, 1.0))
        return state.astype(np.float32), action.astype(np.float32)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    lf.add_args(ap)
    ap.add_argument("--root", required=True, help="dataset directory (created, or appended to if it already holds a dataset)")
    ap.add_argument("--format", choices=["v3.0", "v2.1"], help="LeRobot codebase version (default from config; v2.1 for OpenPI)")
    ap.add_argument("--task", help="task description stored with every episode")
    ap.add_argument("--fps", type=int, help="recording rate (default 30)")
    ap.add_argument("--num-episodes", type=int, default=0, help="stop after this many KEPT episodes (0 = until you quit)")
    ap.add_argument("--episode-time-s", type=float, help="end each episode by itself after this long (0 = only when you press a key)")
    ap.add_argument("--reset-time-s", type=float, help="start the next episode by itself this long after the last (0 = wait for Enter)")
    ap.add_argument("--no-review", action="store_true", help="save each episode as soon as it ends (no keep / retake step)")
    ap.add_argument("--cameras", metavar="SET", help="use the camera set `record.camera_sets.SET` from the config instead of `record.cameras` "
                                                      "(e.g. xtion_top: the ASUS Xtion as the top camera)")
    ap.add_argument("--no-home", action="store_true", help="do not send the robots home before each episode (see `record.home` in the config)")
    ap.add_argument("--no-cameras", action="store_true", help="record joints only, whatever the config lists")
    ap.add_argument("--fake-camera", action="store_true", help="use a synthetic camera (testing)")
    return ap.parse_args(argv)


def episode_home_deg(raw: dict) -> list:
    """The pose the robots start every episode from (follower joint space, deg): ``record.home.deg``, by default the folded REST
    pose ``defaults.park_deg`` - easy to match by hand, because the leader just rests there (``defaults.home_deg`` is the raised
    IK-teleop start pose, which is hard to hit)."""
    deg = (raw.get("record", {}).get("home") or {}).get("deg") or raw.get("defaults", {}).get("park_deg")
    if deg is None or len(deg) != 6:
        raise SystemExit("record.home needs 6 joint angles (deg): set record.home.deg or defaults.park_deg in the config")
    return list(deg)


def camera_specs(cfg: dict, a) -> dict:
    """The camera config this run uses: --fake-camera, a named set (--cameras), or the default ``record.cameras``."""
    if a.no_cameras:
        return {}
    if a.fake_camera:
        return {"fake": {"type": "fake", "width": 64, "height": 48}}
    if a.cameras:
        sets = cfg.get("camera_sets") or {}
        if a.cameras not in sets:
            raise SystemExit(f"no camera set '{a.cameras}' in record.camera_sets (have: {', '.join(sets) or 'none'})")
        return sets[a.cameras] or {}
    return cfg.get("cameras") or {}


def build_cameras(cfg: dict, a) -> dict:
    return {name: make_camera(name, spec) for name, spec in camera_specs(cfg, a).items()}


def main(argv=None):
    a = parse(argv)
    raw = cfgmod.load(a.config)
    rc = raw.get("record", {})
    fps = a.fps or rc.get("fps", 30)
    fmt = a.format or rc.get("format", "v3.0")
    task = a.task or rc.get("task", "teleoperated demonstration")
    side = "follower" if a.follower else (a.follower_arm or raw.get("leader_follower", {}).get("follower_arm", "left"))
    names = feature_names(side)
    video = rc.get("video", {})

    specs = camera_specs(rc, a)
    cams = build_cameras(rc, a)
    for name, cam in cams.items():
        print(f"[rec] starting camera '{name}' ...", flush=True)
        cam.connect()
    writer = LeRobotWriter(Path(a.root), fps=fps, state_names=names, action_names=names,
                           cameras={n: (c.height, c.width) for n, c in cams.items()}, robot_type=rc.get("robot_type", "piper"),
                           fmt=fmt, vcodec=video.get("codec", "libx264"), crf=video.get("crf", 23), gop=video.get("gop", 2))
    rig = leader = follower = session = None
    keys = KeyReader()
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    try:
        rig, leader, follower = lf.setup(a, raw)
        hc = rc.get("home") or {}
        home_on = bool(hc.get("enabled", True)) and not a.no_home
        if home_on:
            deg = episode_home_deg(raw)
            rig.home_q = np.clip(np.radians(np.asarray(deg, float)), lf.JOINT_LO, lf.JOINT_HI)
            rig.home_speed = float(hc.get("speed", raw.get("defaults", {}).get("home_speed", 0.5)))
            rig.home_tol = math.radians(float(hc.get("tol_deg", 10.0)))
            if a.sim:                                                  # the simulated "person" puts the leader at home by itself
                rig.on_home_reached = lambda: leader.set_pose(rig.jmap.inverse(rig.home_q))
            print(f"[rec] robots go home before every episode: follower to {np.round(deg, 0).tolist()} deg (rest pose by default), then you put the leader there "
                  f"(within {hc.get('tol_deg', 10.0):g} deg)", flush=True)
        session = RecordingSession(
            writer, cams, task=task, fps=fps,
            episode_time_s=a.episode_time_s if a.episode_time_s is not None else rc.get("episode_time_s", 0.0),
            reset_time_s=a.reset_time_s if a.reset_time_s is not None else rc.get("reset_time_s", 0.0),
            num_episodes=a.num_episodes, review=not a.no_review,
            on_wait=rig.go_home if home_on else None, ready=lambda: rig.synced, info=lambda: rig.msg if home_on else "")
        if home_on:
            rig.go_home()
        source = ArmToArmSource(rig)
        if not keys.start():
            print("[rec] WARNING: no terminal - keys do not work; run this from a real terminal.", flush=True)

        def on_tick(now):
            for k in keys.poll():
                if session.state == "WAIT" and k in ("\n", "\r", " ", "s") and not rig.synced:
                    session.say(rig.msg if rig.homing else "the follower is still moving to the leader's pose - wait a moment, then press Enter again")
                    continue
                session.key(k)
            session.tick(now, source.sample() if session.state == "REC" else None)
            session.maybe_status(now)
            if session.done:
                stop["flag"] = True

        rig.run(lambda: stop["flag"], status_every=None, on_tick=on_tick)
        if not session.done:
            session.stop("interrupted")
    finally:
        keys.stop()
        print()
        if session is not None and session.kept:
            lfc = raw.get("leader_follower", {})
            writer.log_session({
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "episodes_kept": session.kept, "task": task, "fps": fps, "format": fmt,
                "follower_arm": side, "leader_arm": a.leader_arm or lfc.get("leader_arm", "right"),
                "joint_mapping": a.mode or lfc.get("mode", "copy"), "j1_offset_deg": a.j1_offset if a.j1_offset is not None else lfc.get("j1_offset_deg", 0.0),
                "leader_gripper_mm": lfc.get("leader_gripper_mm"), "video": video,
                "cameras": {n: specs.get(n, {"type": "fake"}) for n in cams}})
        writer.close()
        for cam in cams.values():
            cam.disconnect()
        if rig is not None:
            lf.shutdown(a, rig, follower)
    print(f"[rec] dataset at {a.root}: {writer.num_episodes} episodes, {writer.total_frames} frames ({fmt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
