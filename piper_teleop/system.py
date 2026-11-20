"""Real-time control loop shared by all teleop front-ends (Quest, ...).

The asyncio network thread only ever calls ``submit`` (store newest sample) and
``request`` (queue a command); everything that touches the arms happens in one
dedicated thread so the control period is not at the mercy of the network loop.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from typing import Callable

import numpy as np

from .calibration import GuidedCalibration
from .controller import ENGAGED, ESTOP, HOMING, IDLE, PARKING, HandSample, TeleopArm


class TeleopSystem:
    def __init__(self, arms: list[TeleopArm], rate_hz: float = 200.0,
                 on_event: Callable[..., None] | None = None, hard_estop: bool = False,
                 auto_calibrate: bool = False, pose_log: bool = True):
        self.arms = arms
        self.calib = GuidedCalibration()
        self.calibrated = False
        self._auto_calibrate = auto_calibrate     # start the guided calibration by itself once both hands are seen
        self._auto_done = False
        self._pose_log_on = pose_log
        self._pose_file = None
        self._pose_csv = None
        self._pose_tick = 0
        self._skipped = False                     # the operator deliberately skipped / cancelled the calibration
        self._both_since: float | None = None
        self.hard_estop = hard_estop      # also send the firmware fast-stop on E-STOP (default: freeze only)
        self.period = 1.0 / rate_hz
        self.on_event = on_event or (lambda *a: None)
        self.samples: dict[str, HandSample] = {}
        self._cmds: collections.deque = collections.deque()
        self._thread: threading.Thread | None = None
        self._run = False
        self._stick_prev = False
        # loop statistics
        self.loop_hz = 0.0
        self.step_ms_max = 0.0
        self.overruns = 0
        self.frame_yaw_deg = arms[0].frame_yaw_deg if arms else 0.0
        self.error: str | None = None

    # -- called from the network thread ----------------------------------
    def submit(self, hand: str, sample: HandSample):
        self.samples[hand] = sample

    def request(self, cmd: str, arm: str | None = None):
        self._cmds.append((cmd, arm))

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        for a in self.arms:
            a.start()
        self._run = True
        self._thread = threading.Thread(target=self._loop, name="control", daemon=True)
        self._thread.start()

    def shutdown(self, park: bool = True, timeout: float = 25.0, release: bool = False):
        """Optionally park the arms (slowly, to the folded zero pose), then stop.

        Motors are NOT disabled unless ``release`` is set: a disabled arm lets its wrist fall."""
        if park:
            for a in self.arms:
                a.go_park()
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout and not all(
                    a.reached_park or a.state == ESTOP for a in self.arms):
                time.sleep(0.05)
        self._run = False
        if self._thread:
            self._thread.join(2.0)
        if self._pose_file:
            self._pose_file.close()
        for a in self.arms:
            try:
                if park and a.reached_park and release:
                    a.backend.disable()
                elif park and not a.reached_park:
                    print(f"[system] {a.cfg.name}: did not finish parking - motors LEFT ENABLED so the arm "
                          f"does not drop. Lower it by hand while supporting it.", flush=True)
                else:
                    print(f"[system] {a.cfg.name}: motors left ENABLED and holding (use --release-on-exit to power down).",
                          flush=True)
                a.backend.close()
            except Exception:
                pass

    # -- control thread ----------------------------------------------------
    def _loop(self):
        nxt = time.perf_counter()
        last = nxt
        win_t, win_n = nxt, 0
        while self._run:
            t0 = time.perf_counter()
            dt = min(max(t0 - last, 1e-3), 0.02)
            last = t0
            try:
                self._tick(t0, dt)
            except Exception as e:           # never die silently: freeze + report
                self.error = f"{type(e).__name__}: {e}"
                self.on_event("error", self.error)
                for a in self.arms:
                    try:
                        a.estop()                       # freeze only: never a surprise motion or power cut
                    except Exception:
                        a.state = ESTOP
                self._run = False
                return
            t1 = time.perf_counter()
            self.step_ms_max = max(self.step_ms_max * 0.999, (t1 - t0) * 1e3)
            win_n += 1
            if t1 - win_t >= 1.0:
                self.loop_hz = win_n / (t1 - win_t)
                win_t, win_n = t1, 0
            nxt += self.period
            slack = nxt - time.perf_counter()
            if slack < -self.period:         # fell behind: resync instead of bursting
                self.overruns += 1
                nxt = time.perf_counter()
            elif slack > 0.0007:
                time.sleep(slack - 0.0005)
            while time.perf_counter() < nxt:   # last 0.5 ms spin for low jitter
                pass

    def _tick(self, now: float, dt: float):
        while self._cmds:
            cmd, which = self._cmds.popleft()
            targets = [a for a in self.arms if which in (None, a.cfg.name)]
            if cmd == "estop":
                for a in self.arms:          # an e-stop is always global
                    a.estop(hard=self.hard_estop)
                self.on_event("estop")
            elif cmd == "resume":
                for a in targets: a.resume()
            elif cmd == "home":
                for a in targets: a.go_home()
            elif cmd == "park":
                for a in targets: a.go_park()
            elif cmd == "calibrate":                       # guided: forward / right / up
                self._start_guided(now)
            elif cmd == "calibrate_cancel":                # cancel or skip: arms are released to the operator
                self._handle_calib_events(self.calib.cancel())
                self._skipped = True
            elif cmd == "calibrate_quick":                 # old method: "where the controllers point"
                self._calibrate()
        self._check_calibration_gesture(now)
        self._guided_tick(now)
        for a in self.arms:
            a.step(self.samples.get(a.cfg.hand), now, dt)
            if a.events:
                evs, a.events = a.events, []
                for ev in evs:
                    self.on_event(*ev)
        if self._pose_log_on:
            try:
                self._record_poses(now)
            except OSError:
                self._pose_log_on = False

    # -- pose recorder: one CSV row per engaged arm at ~40 Hz (logs/teleop_*.csv) ---------------------------------
    def _record_poses(self, now: float):
        self._pose_tick += 1
        if self._pose_tick % 5:
            return
        rows = []
        for a in self.arms:
            if a.state != ENGAGED:
                continue
            s = self.samples.get(a.cfg.hand)
            tcp = a.kin.fk_tcp(a.q_cmd)[1]
            rows.append([f"{now:.3f}", a.cfg.name, a.msg[:40], *(np.round(s.pos, 4) if s is not None else ["", "", ""]),
                         "" if s is None else round(s.squeeze, 2), "" if s is None else round(s.trigger, 2),
                         *np.round(tcp, 4), *np.round(a._target_p, 4), round(a.pos_err * 1000, 1),
                         *np.round(np.degrees(a.q_cmd), 1), round(a.grip_cmd * 1000, 1), round(a.grip_meas * 1000, 1)])
        if not rows:
            return
        if self._pose_csv is None:
            import csv, time as _t
            from .paths import log_dir
            self._pose_file = open(log_dir() / f"teleop_{_t.strftime('%Y%m%d_%H%M%S')}.csv", "w", newline="", buffering=1)
            self._pose_csv = csv.writer(self._pose_file)
            self._pose_csv.writerow(["t", "arm", "msg", "hand_x", "hand_y", "hand_z", "squeeze", "trigger", "tcp_x", "tcp_y", "tcp_z",
                                     "tgt_x", "tgt_y", "tgt_z", "ik_err_mm", "J1", "J2", "J3", "J4", "J5", "J6", "grip_cmd_mm", "grip_meas_mm"])
        self._pose_csv.writerows(rows)

    # -- heading calibration ---------------------------------------------
    def _check_calibration_gesture(self, now: float):
        l, r = self.samples.get("left"), self.samples.get("right")
        both = bool(l and r and l.stick_click and r.stick_click
                    and now - l.t_recv < 0.2 and now - r.t_recv < 0.2)
        if both and not self._stick_prev:              # both thumbsticks: "I am facing forward, controllers pointing forward"
            if self.calib.active:                      # (this replaces a running guided calibration instead of skipping it)
                self._handle_calib_events(self.calib.cancel())
            self._auto_done = True
            self._calibrate()
        self._stick_prev = both

    # -- guided axis calibration -------------------------------------------
    def _start_guided(self, now: float):
        self._auto_done = True
        self._skipped = False
        self._handle_calib_events(self.calib.start(now))

    def _guided_tick(self, now: float):
        active = self.calib.active
        # With auto-calibration on, the arms stay frozen until calibrated or deliberately skipped, so nobody drives them
        # with an unknown heading by accident.
        gated = self._auto_calibrate and not self.calibrated and not self._skipped
        for a in self.arms:
            a.suspended = active or gated
            a.suspend_msg = ("axis calibration in progress - holding" if active else
                             "waiting for axis calibration: hold both controllers in view (both thumbsticks = skip)")
        if active:
            self._handle_calib_events(self.calib.feed(self.samples, now))
            return
        if self._auto_calibrate and not self.calibrated and not self._auto_done:
            l, r = self.samples.get("left"), self.samples.get("right")
            ok = all(s is not None and now - s.t_recv < 0.25 and s.tracked and not s.emulated for s in (l, r))
            if not ok:
                self._both_since = None
            elif self._both_since is None:
                self._both_since = now
            elif now - self._both_since > 1.5 and not any(a.state == ENGAGED for a in self.arms):
                self._start_guided(now)

    def _handle_calib_events(self, events):
        for ev in events:
            if ev[0] == "done":
                self._apply_yaw(ev[1]["yaw_deg"])
                self.calibrated = True
                self.on_event("calibrated", ev[1]["yaw_deg"])
                try:
                    import json, time as _t
                    from .paths import log_dir
                    (log_dir() / "calibration.json").write_text(json.dumps({"time": _t.strftime("%Y-%m-%d %H:%M:%S"), **ev[1]}, indent=1))
                except OSError:
                    pass
            else:
                self.on_event(*ev)

    def _apply_yaw(self, yaw: float):
        self.frame_yaw_deg = yaw
        for a in self.arms:
            a.set_frame_yaw(yaw)
            if a.state == ENGAGED:
                a.state = IDLE
                a._need_release = True

    def _calibrate(self):
        """Make "where the controllers point right now" the robot's +x (forward)."""
        vs = []
        for a in self.arms:
            s = self.samples.get(a.cfg.hand)
            if s is not None:
                vs.append(a.ray_forward_robot(s))
        if not vs:
            return
        v = np.mean(vs, axis=0)
        if math.hypot(v[0], v[1]) < 0.35:   # pointing mostly up/down: refuse
            self.on_event("calib_fail")
            return
        # frame_yaw rotates robot-frame vectors; we need R(yaw) @ v ~ +x
        yaw = -math.degrees(math.atan2(v[1], v[0]))
        self._apply_yaw(yaw)
        self.calibrated = True                          # un-gates the arms (auto-calibration waits for this)
        for a in self.arms:
            self.on_event("haptic", a.cfg.hand, 0.6, 80)
        self.on_event("calibrated", yaw)

    # -- reporting ---------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "loop_hz": round(self.loop_hz, 1), "step_ms_max": round(self.step_ms_max, 2),
            "overruns": self.overruns, "frame_yaw_deg": round(self.frame_yaw_deg, 1),
            "calibration": self.calib.snapshot(), "calibrated": self.calibrated,
            "calibration_required": self._auto_calibrate and not self.calibrated and not self._skipped,
            "error": self.error,
            "arms": {a.cfg.name: a.snapshot() for a in self.arms},
        }
