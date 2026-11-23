"""Drive a Piper arm from Python: a trained policy, a high-level controller or a planner.  No ROS, no teleop needed.

    from piper_teleop.robot import PiperRobot, Rate

    with PiperRobot(arm="left", cameras="xtion_top") as robot:        # sim=True: no hardware at all
        robot.go_rest()                                                  # slowly to the folded rest pose
        rate = Rate(30)
        for _ in range(300):
            obs = robot.get_observation()                              # the same keys / units the recorder wrote
            action = my_policy(obs)                                    # (7,): 6 joints [rad] + gripper 0..1
            robot.send_action(action)
            rate.sleep()

Observation and action use EXACTLY the dataset's format (piper_teleop/record.py), so a policy trained on a recorded dataset
plugs in without conversion:

    observation.state                  (7,)  float32   follower joints 1..6 [rad], gripper opening 0..1 (1 = open, 70 mm)
    observation.images.<camera>        (H, W, 3) uint8 RGB, the cameras of config `record.cameras` (or a `record.camera_sets` entry)
    action                             (7,)            absolute joint targets [rad] + gripper target 0..1

How it moves the arm: a background thread streams commands at `robot.control_hz` (200 Hz): every tick it steps the commanded
joints toward the latest target, capped at `max_joint_speed` (all joints arrive together), through the same gripper input layer
as the teleop (the Piper gripper only follows a slowly varying command - see gripper_layer.py).  `send_action` just sets the
target, so call it at whatever rate your policy runs; the arm keeps moving smoothly in between and holds still if you stop.
The very first target after connecting is approached at the slow `sync_speed`, like the teleop's start-up alignment.

Safety, all inherited from the teleop: joints clipped to the arm's limits, speed capped, Cartesian targets checked against the
arm's workspace box, the joint lock (config/joint_lock.yaml) enforced at the CAN layer.  This class NEVER disables the motors:
on exit the arm is folded to the rest pose (slowly) and left enabled and holding.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from . import config as cfgmod
from .arm import GRIPPER_MAX_M, ArmBackend, ArmState
from .errors import IKError
from .gripper_layer import GripperLayer
from .kinematics import JOINT_HI, JOINT_LO, PiperKinematics, quat_to_mat
from .recording.cameras import Camera, make_camera


class Rate:
    """Hold a loop at a fixed rate: ``rate = Rate(30)`` ... ``rate.sleep()`` at the end of each iteration.  Never bursts to catch
    up: if an iteration overran, the next one just starts late."""

    def __init__(self, hz: float):
        self.period = 1.0 / float(hz)
        self._next = time.perf_counter() + self.period

    def sleep(self):
        slack = self._next - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
            self._next += self.period
        else:
            self._next = time.perf_counter() + self.period


def feature_names(arm: str) -> list[str]:
    """Names of the 7 state / action entries, e.g. ``left_joint_1`` ... ``left_joint_6``, ``left_gripper``."""
    return [f"{arm}_joint_{i}" for i in range(1, 7)] + [f"{arm}_gripper"]


class PiperRobot:
    """One Piper arm, driven by joint-space (or tool-pose) targets.

    ``arm``      'left' / 'right' (the PHYSICAL arm; resolved to a CAN interface by ``hardware.can`` in the config) or a CAN name.
    ``sim``      simulated arm, nothing is sent anywhere.
    ``cameras``  None = the config's ``record.cameras`` (no cameras in ``sim``); a string = ``record.camera_sets.<name>``
                 (e.g. 'xtion_top'); a dict of camera specs; ``False`` / ``{}`` = none; 'fake' = a synthetic test camera.
    Everything else defaults from the ``robot:`` section of the config.
    """

    def __init__(self, arm: str | None = None, *, sim: bool = False, cameras=None, config=None,
                 max_joint_speed: float | None = None, sync_speed: float | None = None, rest_speed: float | None = None,
                 control_hz: float | None = None, park_on_exit: bool | None = None,
                 command_ttl_s: float | None = None, allow_speed_override: bool | None = None):
        self.raw = config if isinstance(config, dict) else cfgmod.load(config)
        rc = self.raw.get("robot", {})
        lfc = self.raw.get("leader_follower", {})
        self.arm = arm or rc.get("arm", "left")
        self.sim = bool(sim)
        pick = lambda v, key, default: float(v if v is not None else rc.get(key, default))      # noqa: E731
        self.max_joint_speed = pick(max_joint_speed, "max_joint_speed", 2.0)
        self.sync_speed = pick(sync_speed, "sync_speed", 0.5)
        self.rest_speed = pick(rest_speed, "rest_speed", 0.5)
        self.control_hz = pick(control_hz, "control_hz", 200.0)
        # Command freshness: a target older than command_ttl_s is replaced by "hold where commanded" in the control tick (so a client that
        # stops sending commands stops the arm, even mid-approach: send targets at least every ttl seconds);
        #     allow_speed_override=False refuses per-command speeds, and speeds above max_joint_speed are always refused
        ttl = command_ttl_s if command_ttl_s is not None else rc.get("command_ttl_s")
        if ttl is not None and not 0.0 < float(ttl) <= 60.0:
            raise ValueError(f"command_ttl_s must be in (0, 60] s or None, got {ttl}")
        self.command_ttl_s = None if ttl is None else float(ttl)
        self.allow_speed_override = bool(rc.get("allow_speed_override", True) if allow_speed_override is None else allow_speed_override)
        self.park_on_exit = bool(rc.get("park_on_exit", True) if park_on_exit is None else park_on_exit)
        for name, v in (("max_joint_speed", self.max_joint_speed), ("sync_speed", self.sync_speed), ("rest_speed", self.rest_speed)):
            if not 0.0 < v <= 6.0:
                raise ValueError(f"{name} must be in (0, 6] rad/s, got {v}")
        self.acfg = cfgmod.arm_config(self.arm, self.raw, self.raw.get("arms", {}).get(self.arm, {}))
        self.kin = PiperKinematics(self.raw.get("piper", {}).get("dh_is_offset", 1), self.acfg.tcp_offset)
        self.rest_q = np.radians(np.asarray(rc.get("rest_deg") or self.raw.get("defaults", {}).get("park_deg", [0] * 6), float))
        self.state_names = feature_names(self.arm)
        self._glayer = GripperLayer(lfc.get("gripper_slew_mm_s", 0.0), lfc.get("gripper_dither_mm", 0.5),
                                    lfc.get("gripper_dither_hz", 6.0), GRIPPER_MAX_M)
        self._cam_specs = self._camera_specs(cameras)
        self.cameras: dict[str, Camera] = {}
        self.backend: ArmBackend | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cancel = threading.Event()       # cancel_moves(): abort a blocking go_to / go_rest that is in progress
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._state: ArmState | None = None
        self._q_cmd = np.zeros(6)
        self._tgt_q = np.zeros(6)
        self._tgt_g = 0.0                      # normalised 0..1
        self._speed: float | None = None       # per-move speed override (go_to), None = max_joint_speed
        self.synced = False                    # False until the first target has been reached (that approach is slow)
        self._have_target = False              # set by the first send_action: until then the 'target' is just where the arm is
        self.connected = False
        self.stale_frames = {}                 # camera name -> number of observations that used a frame older than 0.15 s
        self.loop_hz = 0.0
        self._tgt_t = time.monotonic()         # when the current target was set (command freshness)
        self._moving = False                   # a blocking go_to / go_rest is in progress: exempt from the command expiry
        self.expired = 0                       # number of targets replaced by "hold" because they were older than command_ttl_s
        self._state_t = 0.0                    # wall time of the latest feedback read in the control loop

    # ------------------------------------------------------------------ setup
    def _camera_specs(self, cameras) -> dict:
        rc = self.raw.get("record", {})
        if cameras is False or cameras == {}:
            return {}
        if cameras is None:
            return {} if self.sim else dict(rc.get("cameras") or {})
        if cameras == "fake":
            return {"cam_high": {"type": "fake", "width": 64, "height": 48}}
        if isinstance(cameras, str):
            sets = rc.get("camera_sets") or {}
            if cameras not in sets:
                raise ValueError(f"no camera set '{cameras}' in record.camera_sets (have: {', '.join(sets) or 'none'})")
            return dict(sets[cameras] or {})
        return dict(cameras)

    def connect(self) -> "PiperRobot":
        if self.connected:
            return self
        lfc = self.raw.get("leader_follower", {})
        can = cfgmod.can_for(self.raw, self.arm)
        self.backend = cfgmod.make_backend({"can": can}, self.raw, self.sim)
        self.backend.connect()
        time.sleep(0.3 if not self.sim else 0.0)
        if not self.backend.enable(gripper=self.acfg.gripper_enabled):
            raise RuntimeError(f"the {self.arm} arm ({can}) could not be enabled: {getattr(self.backend, 'error', '') or 'enable failed'}")
        t0 = time.monotonic()
        st = self.backend.read()
        while not st.fresh and time.monotonic() - t0 < 3.0:
            time.sleep(0.02)
            st = self.backend.read()
        if not st.fresh:
            raise RuntimeError(f"no feedback from the {self.arm} arm ({can}): powered? on CAN?")
        self._state = st
        self._q_cmd, self._tgt_q = st.q.copy(), st.q.copy()
        self._tgt_g = float(np.clip(st.gripper / GRIPPER_MAX_M, 0.0, 1.0))
        self._glayer.reset(st.gripper)
        for name, spec in self._cam_specs.items():
            print(f"[robot] starting camera '{name}' ...", flush=True)
            cam = make_camera(name, spec)
            cam.connect()
            self.cameras[name] = cam
            self.stale_frames[name] = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="piper-control", daemon=True)
        self._thread.start()
        self.connected = True
        return self

    def __enter__(self) -> "PiperRobot":
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ---------------------------------------------------------- control thread
    def _loop(self):
        period = 1.0 / self.control_hz
        last = nxt = time.perf_counter()
        n, report = 0, last
        while not self._stop.is_set():
            t0 = time.perf_counter()
            dt = min(max(t0 - last, 1e-3), 0.02)
            last = t0
            try:
                self._tick(dt)
            except BaseException as e:                         # noqa: BLE001  (JointLockError, CAN failure ...): surface it
                self._error = e
                return
            n += 1
            if t0 - report >= 1.0:
                self.loop_hz, n, report = n / (t0 - report), 0, t0
            nxt += period
            slack = nxt - time.perf_counter()
            if slack < -period:
                nxt = time.perf_counter()
            elif slack > 0:
                time.sleep(slack)

    def _tick(self, dt: float):
        st = self.backend.read()
        self._state_t = time.time()
        with self._lock:
            if (self.command_ttl_s is not None and self._have_target and not self._moving
                    and time.monotonic() - self._tgt_t > self.command_ttl_s and float(np.max(np.abs(self._tgt_q - self._q_cmd))) > 1e-9):
                self._tgt_q = self._q_cmd.copy()               # stale target: stop where commanded now (the gripper target is kept)
                self._speed = None
                self.expired += 1
            tgt_q, tgt_g, speed = self._tgt_q.copy(), self._tgt_g, self._speed
        err = tgt_q - self._q_cmd
        m = float(np.max(np.abs(err)))
        v = speed if speed is not None else (self.max_joint_speed if self.synced else self.sync_speed)
        lim = v * dt
        if m > lim:
            err *= lim / m                                     # all joints arrive together
        elif not self.synced and self._have_target:
            self.synced = True
        self._q_cmd = np.clip(self._q_cmd + err, JOINT_LO, JOINT_HI)
        g_raw = self._glayer.update(tgt_g * GRIPPER_MAX_M, dt)
        self.backend.command(self._q_cmd, g_raw if self.acfg.gripper_enabled else None)
        self._state = st

    def _check(self):
        if self._error is not None:
            raise RuntimeError(f"the control thread stopped: {type(self._error).__name__}: {self._error}") from self._error
        if not self.connected:
            raise RuntimeError("not connected: use `with PiperRobot(...) as robot:` or call robot.connect()")

    # ------------------------------------------------------------ observation
    def get_observation(self) -> dict:
        """The latest state and camera frames, keyed like the recorded dataset."""
        self._check()
        st = self._state
        if st is None or not st.fresh:
            raise RuntimeError(f"no fresh feedback from the {self.arm} arm")
        obs = {"observation.state": np.append(st.q, np.clip(st.gripper / GRIPPER_MAX_M, 0.0, 1.0)).astype(np.float32)}
        for name, cam in self.cameras.items():
            obs[f"observation.images.{name}"] = cam.read()
            if cam.frame_age() > 0.15:
                self.stale_frames[name] += 1
        obs["timestamp"] = time.time()
        obs["frame_age_s"] = {name: float(cam.frame_age()) for name, cam in self.cameras.items()}      # age of each camera frame at read time [s]
        obs["state_age_s"] = (time.time() - self._state_t) if self._state_t else float("inf")
        return obs

    @property
    def state(self) -> np.ndarray:
        """(7,) measured joints [rad] + gripper 0..1 (the policy's ``observation.state``)."""
        self._check()
        st = self._state
        return np.append(st.q, np.clip(st.gripper / GRIPPER_MAX_M, 0.0, 1.0)).astype(np.float32)

    @property
    def commanded_q(self) -> np.ndarray:
        """The joints currently being commanded (follows the target at the speed cap), rad."""
        return self._q_cmd.copy()

    def tcp_pose(self, measured: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(R 3x3, p [m]) of the tool tip in the arm's base frame (x forward, y left, z up), from the measured joints."""
        self._check()
        return self.kin.fk_tcp(self._state.q if measured else self._q_cmd)

    @property
    def camera_names(self) -> list[str]:
        return list(self.cameras)

    # ------------------------------------------------------------------ action
    def send_action(self, action, *, speed: float | None = None):
        """Set the target: ``action`` = (7,) absolute joint targets [rad] + gripper 0..1 (the dataset's ``action``).  Returns at
        once; the control thread moves the arm there at the speed cap.  ``speed`` (rad/s) may only LOWER the speed for this target:
        anything above ``max_joint_speed`` is refused (and every per-command speed is, if ``allow_speed_override`` is false)."""
        self._check_speed(speed)
        self._set_target(action, speed)

    def _check_speed(self, speed):
        if speed is None:
            return
        if not self.allow_speed_override:
            raise ValueError("a per-command speed is refused (robot.allow_speed_override is false): the configured cap applies")
        if speed > self.max_joint_speed:
            raise ValueError(f"a per-command speed above the configured cap max_joint_speed={self.max_joint_speed:g} rad/s is refused, got {speed}")

    def _set_target(self, action, speed):
        self._check()
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape != (7,) or not np.all(np.isfinite(a)):
            raise ValueError(f"action must be 7 finite numbers (6 joints [rad], gripper 0..1), got shape {a.shape}: {a}")
        if speed is not None and not 0.0 < speed <= 6.0:
            raise ValueError(f"speed must be in (0, 6] rad/s, got {speed}")
        with self._lock:
            self._tgt_q = np.clip(a[:6], JOINT_LO, JOINT_HI)
            self._tgt_g = float(np.clip(a[6], 0.0, 1.0))
            self._speed = speed
            self._have_target = True
            self._tgt_t = time.monotonic()

    def run_chunk(self, actions, hz: float = 30.0, on_step=None):
        """Execute a (N, 7) action chunk open-loop at ``hz``; blocks for N / hz seconds.  ``on_step(i)`` is called before each."""
        actions = np.asarray(actions, float)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"an action chunk is (N, 7), got {actions.shape}")
        rate = Rate(hz)
        for i, a in enumerate(actions):
            if on_step is not None:
                on_step(i)
            self.send_action(a)
            rate.sleep()

    def hold(self):
        """Stop moving: the target becomes where the arm is commanded to be now (the gripper target is kept)."""
        with self._lock:
            self._tgt_q = self._q_cmd.copy()
            self._speed = None
            self._tgt_t = time.monotonic()

    def go_to(self, q, gripper: float | None = None, *, speed: float | None = None, timeout: float = 30.0, tol_deg: float = 1.5):
        """Blocking joint-space move at ``speed`` (default ``rest_speed``): all joints arrive together.  Raises TimeoutError if the
        arm is not there after ``timeout`` s - it stays enabled and holding."""
        self._check()
        self._check_speed(speed)                                   # a blocking move obeys the same cap as send_action
        q = np.clip(np.asarray(q, float).reshape(6), JOINT_LO, JOINT_HI)
        g = self._tgt_g if gripper is None else float(gripper)
        self._cancel.clear()
        tol = math.radians(tol_deg)
        t0 = time.monotonic()
        self._moving = True                    # set INSIDE the try so a refused target can never leave the expiry disabled
        try:
            self._set_target(np.append(q, g), speed or self.rest_speed)
            while True:
                self._check()
                if self._cancel.is_set():
                    self.hold()
                    raise InterruptedError("move cancelled - the arm holds its pose")
                if float(np.max(np.abs(self._q_cmd - q))) < 1e-3 and float(np.max(np.abs(self._state.q - q))) < tol:
                    return
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError(f"the arm did not reach {np.degrees(q).round(1).tolist()} deg within {timeout:g} s "
                                       f"(now {np.degrees(self._state.q).round(1).tolist()}); it holds its pose")
                time.sleep(0.01)
        finally:
            self._moving = False
            with self._lock:
                self._speed = None
                self._tgt_t = time.monotonic()
            self.synced = True

    def cancel_moves(self):
        """Abort a blocking ``go_to`` / ``go_rest`` in progress (callable from a signal handler or another thread): the arm holds."""
        self._cancel.set()

    def go_rest(self, gripper: float | None = None, **kw):
        """(Named go_rest, not reset: the Piper SDK's reset() drops the arm, and tests/test_never_drops.py flags any `.reset()` on a robot.)
        Fold to the rest pose (the pose every recorded episode starts from) at ``rest_speed``."""
        self.go_to(self.rest_q, gripper, **kw)

    # ------------------------------------------------------- tool-pose targets
    def send_tcp_pose(self, position, rotation=None, gripper: float | None = None, *, speed: float | None = None,
                      tol_pos: float = 0.004, tol_rot: float = 0.08, max_joint_change: float = 1.0):
        """Move the tool tip to ``position`` (x, y, z [m], arm base frame: x forward, y left, z up), solved with the teleop's
        task-priority IK (position first, orientation second).

        ``rotation``  3x3 matrix or quaternion (x, y, z, w) = the wanted tool orientation; None = keep the current one;
                      ``"free"`` = position only, the orientation is whatever the solution needs (the arm moves least).
        Refuses instead of sending something different: ``ValueError`` if the target is outside the workspace box,
        ``IKError`` if it is not reached within ``tol_pos`` / ``tol_rot``, or if the solution needs a joint to swing more than
        ``max_joint_change`` rad (57 deg) from where the arm is - that is a different arm configuration, and the straight joint-space
        move to it would not be the path you want: go via intermediate waypoints, or use ``rotation="free"``.
        Returns the joint solution (6,).  The arm moves there at the speed cap (or ``speed``); poll ``tcp_pose()`` to see it arrive."""
        self._check()
        p = np.asarray(position, float).reshape(3)
        ws = self.acfg.workspace
        lo, hi = np.array([ws[k][0] for k in "xyz"]), np.array([ws[k][1] for k in "xyz"])
        if np.any(p < lo) or np.any(p > hi):
            raise ValueError(f"target {p.round(3).tolist()} m is outside the workspace box x{ws['x']} y{ws['y']} z{ws['z']} "
                             "(config defaults.workspace / arms.<arm>.workspace)")
        free = isinstance(rotation, str) and rotation == "free"
        if rotation is None or free:
            R = self.kin.fk_tcp(self._q_cmd)[0]
        else:
            R = np.asarray(rotation, float)
            R = quat_to_mat(R.reshape(4)) if R.size == 4 else R.reshape(3, 3)
        q0 = self._q_cmd.copy()
        q, ep, er, pt = q0.copy(), 1.0, 1.0, p
        if free:
            q = self._solve_position(q0, p)
            pt = self.kin.fk_tcp(q)[1]
            ep, er = float(np.linalg.norm(pt - p)), 0.0
        else:
            for _ in range(60):
                q, _, _ = self.kin.ik_tcp(q, R, p, iters=3)
                Rt, pt = self.kin.fk_tcp(q)
                ep, er = float(np.linalg.norm(pt - p)), float(np.linalg.norm(_rot_err(Rt, R)))
                if ep < tol_pos * 0.25 and er < tol_rot * 0.25:
                    break
        if ep > tol_pos or er > tol_rot:
            raise IKError(f"cannot reach {p.round(3).tolist()} m{'' if free else ' with that orientation'} from here: best is "
                          f"{ep * 1000:.1f} mm / {math.degrees(er):.1f} deg off (limits {tol_pos * 1000:.0f} mm / "
                          f"{math.degrees(tol_rot):.0f} deg)" + ("" if free else "; try rotation='free' or intermediate waypoints"))
        if float(pt[2]) < lo[2] - 1e-4:
            raise IKError(f"the solved pose puts the tool tip at z = {pt[2]:.3f} m, below the table-side limit {lo[2]:.3f} m")
        swing = np.abs(q - q0)
        if float(swing.max()) > max_joint_change:
            j = int(swing.argmax())
            raise IKError(f"reaching {p.round(3).tolist()} m needs joint {j + 1} to swing {math.degrees(swing[j]):.0f} deg "
                          f"(limit {math.degrees(max_joint_change):.0f}): a different arm configuration. Go via intermediate "
                          "waypoints" + ("" if free else ", or use rotation='free'"))
        g = self._tgt_g if gripper is None else float(gripper)
        self.send_action(np.append(q, g), speed=speed)
        return q

    def _solve_position(self, q_seed: np.ndarray, p: np.ndarray, iters: int = 150) -> np.ndarray:
        """Tool-tip POSITION only (orientation free): damped least squares on the tip Jacobian, from ``q_seed``, so the arm moves
        as little as it can.  (The teleop IK always holds an orientation - it aims the flange at ``p - R_des * tcp_offset`` - which
        near the folded wrist singularity makes J4 / J6 counter-rotate wildly; with the orientation free there is nothing to hold.)"""
        q = np.asarray(q_seed, float).copy()
        for _ in range(iters):
            R, pf, J = self.kin.fk_jac(q)
            off = R @ self.kin.tcp_offset
            e = p - (pf + off)
            if float(np.linalg.norm(e)) < 2e-4:
                break
            skew = np.array([[0.0, -off[2], off[1]], [off[2], 0.0, -off[0]], [-off[1], off[0], 0.0]])
            Jt = J[:3] - skew @ J[3:]                         # tip velocity = v_flange + w x (R * offset)
            dq = Jt.T @ np.linalg.solve(Jt @ Jt.T + (0.02 ** 2) * np.eye(3), e)
            m = float(np.max(np.abs(dq)))
            if m > 0.1:
                dq *= 0.1 / m
            q = np.clip(q + dq, JOINT_LO, JOINT_HI)
        return q

    # ----------------------------------------------------------------- shutdown
    def close(self, park: bool | None = None):
        """Stop.  By default folds the arm to the rest pose first (slowly); the motors stay ENABLED and holding - this class never
        disables them.  Safe to call twice."""
        if not self.connected:
            return
        park = self.park_on_exit if park is None else park
        try:
            if park and self._error is None:
                try:
                    self.go_to(self.rest_q, speed=self.rest_speed, timeout=40.0)
                except Exception as e:                         # noqa: BLE001  (could not park: leave it holding, say why)
                    print(f"[robot] could not fold to rest ({type(e).__name__}: {e}) - the arm holds its pose, motors stay enabled", flush=True)
            else:
                self.hold()
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except Exception:                              # noqa: BLE001
                    pass
            self.cameras = {}
            try:
                self.backend.close()
            finally:
                self.connected = False
                print("[robot] closed; the motors were left enabled and holding.", flush=True)


def _rot_err(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Rotation vector taking R_a to R_b (rad)."""
    from .kinematics import so3_log
    return so3_log(R_b @ R_a.T)
