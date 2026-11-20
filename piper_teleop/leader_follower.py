"""Arm-to-arm teleop: one Piper (leader, moved by hand) drives the other (follower).

    python -m piper_teleop.leader_follower --sim
    python -m piper_teleop.leader_follower --leader can_leader --follower can_follower

Two ways to run leader/follower with Piper - this module is the software one:

* HARDWARE LINKAGE (AgileX native, no PC in the loop): wire both arms onto one CAN bus,
  set one to 0xFA (leader) and one to 0xFC (follower).  Lowest possible latency.  See
  the user guide, section "Mode A".
* SOFTWARE (this file): each arm is on its *own* CAN bus; the PC reads the leader and
  commands the follower.  Costs ~1-3 ms but gives speed limits, startup alignment,
  gripper scaling and safe stop - and the same code can later record data.

The leader is only ever *read*: we never enable its motors, so it stays free to drag.
The leader must be in master mode (0xFA) so it broadcasts its pose as command frames;
use ``tools/set_role.py leader``.
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time

import numpy as np

from . import config as cfgmod
from .arm import GRIPPER_MAX_M, RAD_TO_MDEG, ArmBackend, ArmState, PiperArm, SimArm, _make_interface
from .gripper_layer import GripperLayer
from .kinematics import JOINT_HI, JOINT_LO


class PiperLeader:
    """Read-only view of a leader arm (never enabled, never commanded)."""

    def __init__(self, can_name: str, source: str = "ctrl", dh_is_offset: int = 1, judge_flag=True, iface=None):
        self.source = source
        self._p = iface or _make_interface(can_name, dh_is_offset, judge_flag)

    def connect(self):
        self._p.ConnectPort()
        time.sleep(0.1)

    def read(self) -> ArmState:
        p = self._p
        if self.source == "ctrl":          # command frames 0x155-0x157 / 0x159 broadcast by a 0xFA arm
            m = p.GetArmJointCtrl()
            j = m.joint_ctrl
            g = p.GetArmGripperCtrl()
            grip = g.gripper_ctrl.grippers_angle
        else:                              # joint feedback frames 0x2A5-0x2A7
            m = p.GetArmJointMsgs()
            j = m.joint_state
            grip = p.GetArmGripperMsgs().gripper_state.grippers_angle
        q = np.array([j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6], float) / RAD_TO_MDEG
        return ArmState(q, max(0.0, grip / 1e6), getattr(m, "Hz", 0.0) > 10, True, self.source)


class SimLeader:
    """Pretend leader: a person slowly waving the arm around."""

    def __init__(self, home):
        self.home = np.array(home, float)
        self.t0 = time.perf_counter()
        self.held = None                     # set_pose(): the "person" holds the arm still at this pose

    def connect(self): pass

    def set_pose(self, q):
        self.held = None if q is None else np.array(q, float)

    def read(self) -> ArmState:
        if self.held is not None:
            return ArmState(self.held.copy(), 0.035, True, True, "sim")
        t = time.perf_counter() - self.t0
        q = self.home + np.radians([20 * math.sin(0.7 * t), 12 * math.sin(0.5 * t), 10 * math.sin(0.9 * t),
                                    15 * math.sin(0.6 * t), 12 * math.sin(0.8 * t), 25 * math.sin(0.4 * t)])
        return ArmState(q, 0.035 + 0.03 * math.sin(t), True, True, "sim")


# How the leader's joints become the follower's.  q_follower = sign * q_leader + offset, then clipped to the joint limits.
#   copy   - same joint angles.  Use when both arms face the SAME way (e.g. the leader's base turned round by hand: with
#            j1_offset_deg 180 the leader's J1 -150 becomes the follower's +30 - the same heading in the room).
#   mirror - the arms FACE EACH OTHER: the follower is the leader's mirror image across the plane between them.  The
#            swing/roll joints (J1 base yaw, J4 and J6 forearm/wrist roll) flip sign, the pitch joints (J2, J3, J5) do not:
#            the leader sweeping to ITS left makes the follower sweep to ITS right, i.e. the same side of the room, like a
#            reflection.  J1 is the "base flip"; its offset is the angle the follower's base zero sits from the leader's.
MAPPINGS = {
    "copy": np.ones(6),
    "mirror": np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0]),
}


class JointMap:
    def __init__(self, mode: str = "copy", offset_deg=None, sign=None):
        if mode not in MAPPINGS:
            raise ValueError(f"mapping must be one of {sorted(MAPPINGS)}, not {mode!r}")
        self.mode = mode
        self.sign = np.asarray(MAPPINGS[mode] if sign is None else sign, float)
        off = np.zeros(6) if offset_deg is None else np.atleast_1d(np.asarray(offset_deg, float))
        if off.size == 1:                                   # a bare number = the base (J1) offset
            off = np.concatenate([off, np.zeros(5)])
        self.offset = np.radians(off)

    def inverse(self, q_follower) -> np.ndarray:
        """The leader pose that maps onto ``q_follower`` (the sign is +-1, so it is its own inverse)."""
        return self.sign * (np.asarray(q_follower, float) - self.offset)

    def __call__(self, q, ref=None) -> np.ndarray:
        """Map leader joints to follower joints.  ``ref`` = where the follower is now: each joint takes the equivalent angle
        (+-360 deg) NEAREST to it, so a +180 deg base flip never makes the follower jump from +150 to -150 when the leader's
        base sweeps through the unreachable sector - it just waits at the limit."""
        x = self.sign * np.asarray(q, float) + self.offset
        r = np.zeros(6) if ref is None else np.asarray(ref, float)
        cands = np.stack([x - 2.0 * math.pi, x, x + 2.0 * math.pi])
        x = cands[np.argmin(np.abs(cands - r), axis=0), np.arange(6)]
        return np.clip(x, JOINT_LO, JOINT_HI)


class LeaderFollower:
    def __init__(self, leader, follower: ArmBackend, *, max_joint_speed=3.0, sync_speed=0.5, joint_map: JointMap | None = None,
                 gripper_scale=1.0, leader_gripper_mm=None, unstick=False, stale_s=0.15, rate_hz=200.0, gripper_slew_mm_s=0.0, gripper_dither_mm=0.5,
                 gripper_dither_hz=6.0):
        self.leader, self.follower = leader, follower
        self.jmap = joint_map or JointMap("copy")
        self.vmax, self.vsync = max_joint_speed, sync_speed
        self.gscale = gripper_scale
        # (closed, open) opening of the LEADER's gripper/pendant in mm, mapped onto the follower's 0..70 mm stroke; None = 1:1 x scale
        self.lrange = None if leader_gripper_mm is None else (float(leader_gripper_mm[0]), float(leader_gripper_mm[1]))
        # OFF by default - a gripper pushing at its effort limit is normally GRIPPING an object (the pendant gives no feedback, so
        # the operator over-closes) and relaxing would drop it.  Release-and-retry: a gripper that pushes at its effort limit without moving (seen live: fast closes ended blocked at
        # 47-54 mm, effort -1040 of 1000) is told to relax for a moment, then the target is re-issued.  Gripper frames only.
        self.unstick = unstick
        self._blocked_since: float | None = None
        self._relax_until = 0.0
        self._last_relax = -1e9
        self._gm_hist: list[tuple[float, float]] = []
        self.relaxes = 0
        self.lg_seen = [1e9, -1e9]           # smallest / largest leader opening seen (mm): tune leader_gripper_mm from it
        self.stale_s = stale_s
        self.period = 1.0 / rate_hz
        self.q_cmd = np.zeros(6)
        self.g_cmd = 0.0
        self.glayer = GripperLayer(gripper_slew_mm_s, gripper_dither_mm, gripper_dither_hz, GRIPPER_MAX_M)
        self.synced = False
        # "home before each episode" (see go_home): the follower drives to ``home_q`` (follower joint space) at ``home_speed``
        # and holds there, ignoring the leader, until the LEADER - which a person moves by hand - is within ``home_tol`` of it.
        self.home_q: np.ndarray | None = None
        self.home_speed = 0.5                # rad/s
        self.home_tol = math.radians(10.0)
        self.homing = False
        self.on_home_reached = None          # optional callback, called once per homing when the follower arrives (the simulator's "person")
        self._home_arrived = False
        self.log = None                      # optional csv.writer: one row per tick (leader/follower gripper + base joint)
        self.msg = ""
        self.last_fresh = 0.0
        self.loop_hz = 0.0
        self.lag = 0.0                       # |leader - follower feedback| (rad), max over joints
        # the last tick, for a dataset recorder: mapped leader joints (rad), the follower's gripper target (m), its measured state
        self.last_goal = np.zeros(6)
        self.last_g_target = 0.0
        self.last_follower: ArmState | None = None
        self.last_tick_ok = False            # False while the leader is silent / before the first tick

    def start(self):
        st = self.follower.read()
        self.q_cmd, self.g_cmd = st.q.copy(), st.gripper
        self.glayer.reset(st.gripper)

    def go_home(self):
        """Start a homing: the follower drives to ``home_q`` and waits there; following resumes (through the usual slow alignment)
        once the leader has been moved to within ``home_tol`` of that pose.  The leader is a hand-driven arm in drag-teach, so
        it can only be asked, never moved - that is what the status message says.  No-op without a ``home_q``."""
        if self.home_q is None:
            return
        self.homing, self.synced, self._home_arrived = True, False, False

    def _home_step(self, ld, dt: float):
        self.last_tick_ok = False                        # nothing recorded / no demonstration while homing
        d = self.home_q - self.q_cmd
        m = float(np.max(np.abs(d)))
        if m > 1e-3:
            self.q_cmd = np.clip(self.q_cmd + d * min(1.0, self.home_speed * dt / m), JOINT_LO, JOINT_HI)
            self.msg = f"homing the follower ({math.degrees(m):.0f} deg to go)"
        else:
            self.q_cmd = self.home_q.copy()
            if not self._home_arrived:
                self._home_arrived = True
                if self.on_home_reached is not None:
                    self.on_home_reached()
            if ld.fresh and np.all(np.isfinite(ld.q)):
                dev = float(np.max(np.abs(self.jmap(ld.q, self.q_cmd) - self.home_q)))
                if dev <= self.home_tol:
                    self.homing = False                  # leader is home: follow it (aligning from here)
                    self.msg = "following"
                    return
                self.msg = f"follower is home - move the leader to its home pose ({math.degrees(dev):.0f} deg off, need <= {math.degrees(self.home_tol):.0f})"
            else:
                self.msg = "follower is home - leader silent"
        self.follower.command(self.q_cmd, self.glayer.update(self.g_cmd, dt))    # gripper: hold whatever it is doing

    def align_error(self) -> float:
        ld = self.leader.read()
        return float(np.max(np.abs(self.jmap(ld.q, self.q_cmd) - self.q_cmd)))

    def step(self, dt: float):
        ld = self.leader.read()
        fst = self.follower.read()
        now = time.monotonic()
        if ld.fresh and np.all(np.isfinite(ld.q)):
            self.last_fresh = now
        if self.homing:
            self._home_step(ld, dt)
            return
        if now - self.last_fresh > self.stale_s:
            self.msg = "leader silent - holding"
            self.last_tick_ok = False
            self.follower.command(self.q_cmd, self.glayer.update(self.g_cmd, dt))   # a plain constant would freeze it
            return
        goal = self.jmap(ld.q, self.q_cmd)
        self.last_goal, self.last_follower, self.last_tick_ok = goal, fst, True
        err = goal - self.q_cmd
        m = float(np.max(np.abs(err)))
        v = self.vmax if self.synced else self.vsync
        lim = v * dt
        if m > lim:
            err *= lim / m
            if not self.synced:
                self.msg = f"aligning follower to leader ({math.degrees(m):.0f} deg to go)"
        elif not self.synced:
            self.synced = True
        if self.synced:
            self.msg = "following"
        self.q_cmd = np.clip(self.q_cmd + err, JOINT_LO, JOINT_HI)
        lg = ld.gripper * 1e3
        self.lg_seen = [min(self.lg_seen[0], lg), max(self.lg_seen[1], lg)]
        if self.lrange is not None:
            lo, hi = self.lrange
            g_target = float(np.clip((lg - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * GRIPPER_MAX_M)
        else:
            g_target = float(np.clip(ld.gripper * self.gscale, 0.0, GRIPPER_MAX_M))
        if self.unstick:
            g_target = self._unstick(g_target, fst, now)
        self.last_g_target = g_target
        g_raw = self.glayer.update(g_target, dt)       # ramp + slow sine (see gripper_layer.py)
        self.g_cmd = self.glayer.ramp
        self.follower.command(self.q_cmd, g_raw)
        if fst.fresh:
            self.lag = float(np.max(np.abs(goal - fst.q)))
        if self.log is not None:
            self.log.writerow([f"{time.time():.4f}", f"{ld.gripper * 1e3:.2f}", f"{g_target * 1e3:.2f}", f"{g_raw * 1e3:.2f}",
                               f"{fst.gripper * 1e3:.2f}", f"{math.degrees(ld.q[0]):.2f}", f"{math.degrees(fst.q[0]):.2f}", int(ld.fresh), *self._grip_diag(), int(time.monotonic() < self._relax_until)])

    def _unstick(self, g_target: float, fst, now: float) -> float:
        gs = getattr(self.follower, "gripper_status", None)
        d = gs() if callable(gs) else None
        if not d or not fst.fresh:
            return g_target
        if now < self._relax_until:
            return fst.gripper                                  # relax: ask for where it already is
        self._gm_hist = [(t, g) for t, g in self._gm_hist if now - t < 0.3] + [(now, fst.gripper)]
        still = max(g for _, g in self._gm_hist) - min(g for _, g in self._gm_hist) < 0.0005     # moved < 0.5 mm in 0.3 s
        off = abs(g_target - fst.gripper) > 0.004
        pushing = abs(d["effort"]) > 800
        if still and off and pushing and now - self._gm_hist[0][0] > 0.2:
            if self._blocked_since is None:
                self._blocked_since = now
            if now - self._blocked_since > 0.05 and now - self._last_relax > 1.5:
                self._relax_until, self._last_relax, self.relaxes = now + 0.2, now, self.relaxes + 1
                self._blocked_since = None
                return fst.gripper
        else:
            self._blocked_since = None
        return g_target

    def _grip_diag(self):
        gs = getattr(self.follower, "gripper_status", None)
        d = gs() if callable(gs) else None
        return ("", "", "", "") if not d else (d["effort"], int(d["enabled"]), int(d["zeroed"]), int(d["error"]))

    def run(self, stop, status_every=1.0, on_tick=None):
        nxt = last = time.perf_counter()
        report, n = last, 0
        while not stop():
            t0 = time.perf_counter()
            self.step(min(max(t0 - last, 1e-3), 0.02))
            if on_tick is not None:
                on_tick(t0)
            last = t0
            n += 1
            if status_every and t0 - report >= status_every:
                self.loop_hz = n / (t0 - report)
                n, report = 0, t0
                print(f"[lf] {self.loop_hz:5.1f} Hz | {self.msg} | leader-vs-follower {math.degrees(self.lag):5.1f} deg | "
                      f"gripper {self.g_cmd * 1000:4.0f} mm (leader seen {self.lg_seen[0]:.0f}-{self.lg_seen[1]:.0f}, unstick {self.relaxes}x)", flush=True)
            nxt += self.period
            slack = nxt - time.perf_counter()
            if slack < -self.period:
                nxt = time.perf_counter()
            elif slack > 0.0007:
                time.sleep(slack - 0.0005)
            while time.perf_counter() < nxt:
                pass

    def park(self, speed=0.5, timeout=25.0):
        """Slowly fold the follower to the zero pose so it does not drop when disabled."""
        t0 = time.monotonic()
        dt = 0.005
        while time.monotonic() - t0 < timeout:
            d = -self.q_cmd
            m = float(np.max(np.abs(d)))
            if m < 1e-3:
                break
            self.q_cmd = self.q_cmd + d * min(1.0, speed * dt / m)
            self.follower.command(self.q_cmd, self.glayer.update(self.g_cmd, dt))
            time.sleep(dt)


def add_args(ap):
    ap.add_argument("--config")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--leader-arm", choices=["left", "right"], help="the physical arm you move by hand (default from config)")
    ap.add_argument("--follower-arm", choices=["left", "right"], help="the physical arm that copies it (default from config)")
    ap.add_argument("--leader", help="CAN name of the leader (overrides --leader-arm)")
    ap.add_argument("--follower", help="CAN name of the follower (overrides --follower-arm)")
    ap.add_argument("--source", choices=["ctrl", "msgs"], help="how to read the leader (see config)")
    ap.add_argument("--mode", choices=sorted(MAPPINGS), help="joint mapping: mirror = arms face each other, copy = same way")
    ap.add_argument("--j1-offset", type=float, help="deg added to the follower's base joint after mapping (default from config)")
    ap.add_argument("--no-park", action="store_true")
    ap.add_argument("--release", action="store_true", help="after parking, DISABLE the follower (wrist droops)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation before the follower moves")


def setup(a, raw=None):
    """Connect both arms, enable the follower, check the leader talks.  Returns (rig, leader, follower)."""
    raw = raw if raw is not None else cfgmod.load(a.config)
    lf = raw.get("leader_follower", {})
    p = raw.get("piper", {})
    home = np.radians(raw.get("defaults", {}).get("park_deg", [0] * 6))
    if a.sim:
        leader, follower = SimLeader(home + np.radians([0, 40, -40, 0, 20, 0])), SimArm("follower", q0=home)
    else:
        lcan = a.leader or lf.get("leader_can") or cfgmod.can_for(raw, a.leader_arm or lf.get("leader_arm", "right"))
        fcan = a.follower or lf.get("follower_can") or cfgmod.can_for(raw, a.follower_arm or lf.get("follower_arm", "left"))
        print(f"[lf] leader = {a.leader_arm or lf.get('leader_arm', 'right')} arm on {lcan}   "
              f"follower = {a.follower_arm or lf.get('follower_arm', 'left')} arm on {fcan}")
        if p.get("driver", "pyagxarm") == "pyagxarm":
            from .agx_arm import AgxLeader
            leader = AgxLeader(lcan, p.get("firmware", "auto"), source=a.source or lf.get("leader_source", "ctrl"))
        else:
            leader = PiperLeader(lcan, a.source or lf.get("leader_source", "ctrl"), p.get("dh_is_offset", 1),
                                 p.get("judge_flag", True))
        follower = cfgmod.make_backend({"can": fcan}, raw, False)
    leader.connect()
    follower.connect()
    time.sleep(0.3)
    if not follower.enable():
        raise SystemExit(f"follower could not be enabled: {getattr(follower, 'error', '') or 'enable failed'}\n"
                         "(check CAN + power; is it accidentally still set as a leader, 0xFA?)")
    jmap = JointMap(a.mode or lf.get("mode", "copy"),
                    a.j1_offset if a.j1_offset is not None else lf.get("j1_offset_deg", lf.get("offset_deg", 0.0)))
    print(f"[lf] joint mapping: {jmap.mode}  sign {jmap.sign.astype(int).tolist()}  offset {np.degrees(jmap.offset).round(1).tolist()} deg")
    rig = LeaderFollower(leader, follower, joint_map=jmap, max_joint_speed=lf.get("max_joint_speed", 3.0),
                         gripper_scale=lf.get("gripper_scale", 1.0), leader_gripper_mm=lf.get("leader_gripper_mm"),
                         unstick=lf.get("gripper_unstick", False), rate_hz=lf.get("rate_hz", 200),
                         gripper_slew_mm_s=lf.get("gripper_slew_mm_s", 0.0),
                         gripper_dither_mm=lf.get("gripper_dither_mm", 0.5), gripper_dither_hz=lf.get("gripper_dither_hz", 6.0))
    rig.start()
    t0 = time.monotonic()
    while not leader.read().fresh and time.monotonic() - t0 < 3.0:
        time.sleep(0.05)
    if not leader.read().fresh:
        raise SystemExit("no command frames from the leader. Is it powered, on its own CAN bus, and set to "
                         "leader mode (python tools/set_role.py leader)?  Or try --source msgs.")
    gap = math.degrees(rig.align_error())
    print(f"[lf] leader/follower pose difference: {gap:.0f} deg (max joint)")
    if not a.sim and not a.yes and gap > 10:
        input(">>> The follower will slowly move (<=0.5 rad/s) to match the leader. Clear the workspace, "
              "then press Enter (Ctrl-C aborts)... ")
    return rig, leader, follower


def shutdown(a, rig, follower):
    if not a.no_park:
        print("[lf] parking follower to zero pose (do not touch)...", flush=True)
        rig.park()
        if a.release and float(np.max(np.abs(rig.q_cmd))) < 0.02:
            follower.disable()
            print("[lf] follower motors RELEASED on request (the wrist will droop).", flush=True)
        else:
            print("[lf] follower motors left ENABLED and holding - nothing was powered down.", flush=True)
    follower.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    a = ap.parse_args(argv)
    rig, leader, follower = setup(a)
    import csv
    from .paths import log_dir
    log_path = log_dir() / time.strftime("leader_follower_%Y%m%d_%H%M%S.csv")
    log_fh = open(log_path, "w", newline="")
    rig.log = csv.writer(log_fh)
    rig.log.writerow(["t", "leader_grip_mm", "target_mm", "cmd_mm", "follower_grip_mm", "leader_j1_deg", "follower_j1_deg", "fresh",
                      "fgrip_effort", "fgrip_enabled", "fgrip_zeroed", "fgrip_error", "relax"])
    print(f"[lf] logging to {log_path}", flush=True)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    try:
        rig.run(lambda: stop["flag"])
    finally:
        log_fh.close()
        shutdown(a, rig, follower)


if __name__ == "__main__":
    main()
