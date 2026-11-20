"""Per-arm teleop logic: VR controller sample -> joint command.

One ``TeleopArm`` drives one robot arm from one hand.  It is deliberately a pure
state machine (no threads, no I/O except the backend), stepped by the control loop
at ~200 Hz.  Safety properties, all enforced here and not in the UI:

* relative (clutch) mapping - the arm never jumps when you grab; hold GRIP to move
* every engage needs a *fresh rising edge* of GRIP (also after any tracking loss)
* stale network data / lost or emulated tracking -> arm holds its last command
* TCP clamped to a workspace box, joints clamped to limits, joint speed limited
* IK failures never move the arm (command simply holds)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .arm import GRIPPER_MAX_M, ArmBackend
from .filters import OneEuro, OneEuroQuat
from .gripper_layer import GripperLayer
from .kinematics import JOINT_HI, JOINT_LO, PiperKinematics, mat_to_quat, quat_to_mat, so3_exp, so3_log

# Quest/WebXR reference space: x right, y up, -z forward.  Robot base: x fwd, y left, z up.
_C_QUEST_TO_ROBOT = np.array([[0.0, 0.0, -1.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]])


def yaw_matrix(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class ArmConfig:
    name: str = "arm"
    hand: str = "right"                    # which Quest controller drives this arm
    pos_scale: float = 1.0
    rot_scale: float = 1.0
    fine_factor: float = 0.4               # multiplier while fine mode (A/X) is on
    max_joint_speed: float = 2.0           # rad/s, engaged
    # Tool-tip speed caps (m/s).  The joint-speed cap alone lets a stretched-out arm move its tip at > 1 m/s (115 deg/s on a
    # 0.6 m lever), which is how a fast approach slammed into a peg.  0 = off.  Lower near the desk, where contact happens.
    max_tcp_speed: float = 0.5
    slow_zone_z: float = 0.12              # m: below this tip height ...
    slow_zone_speed: float = 0.25          # ... the cap is this (insertion / peg work)
    home_speed: float = 0.5                # rad/s, homing / parking
    smoothing_ms: float = 3.0              # joint-space first-order lag (0 = off)
    stale_ms: float = 150.0                # no fresh controller data for this long -> hold
    tcp_offset: tuple = (0.0, 0.0, 0.12)   # TCP in flange frame [m]; measure yours!
    workspace: dict = field(default_factory=lambda: {
        "x": [0.05, 1.0], "y": [-1.0, 1.0], "z": [0.035, 1.0]})     # open: rear plane + floor only (same as the YAML)
    # Start pose: the folded rest pose with the TCP raised 10 cm and 5 cm further out, tool tilted 15 deg
    # further down than at rest (a more horizontal tool is near a wrist singularity: no sideways motion).
    home_deg: tuple = (0.0, 35.0, -34.0, -4.0, 37.0, 10.0)
    # "world": gripper orientation = hand orientation in the world (a sideways sweep must be cancelled by the wrist:
    #          J1 swings AND J4/J6 counter-rotate - badly conditioned near the folded start pose).
    # "follow_base": the gripper heading also follows the swing of the arm around its base, like a pointing arm; hand
    #          rotation is added on top.  Sideways motion then needs only J1.
    # Hand rotation -> gripper rotation gains about the ARM's x (roll), y (pitch) and z (yaw) axes.  A Piper wrist turns
    # a small hand yaw/roll into big J4/J6 swings, so by default pitch follows 1:1 and roll/yaw are tamed.
    rot_axis_scale: tuple = (0.5, 1.0, 0.3)
    # Keep J5 above this: near J5 = 0 the wrist axes J4/J6 line up and the IK spins them against each other.
    wrist_pitch_min_deg: float = 20.0
    # Gentle pull of (J4, J5, J6) toward the home values inside the IK.  Stops the wrist from wandering (J4 and J6
    # swinging against each other) where orientation is nearly redundant.  0 disables.
    wrist_posture_weight: tuple = (0.3, 0.05, 0.3)
    orientation_mode: str = "follow_base"
    base_yaw_deg: float = 0.0              # arm's x axis vs the operator frame, ccw (arms facing each other: -90 / +90)
    park_deg: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # folded zero pose (manual)
    filter_min_cutoff: float = 5.0         # Hz: smoothing of a slow/still hand (kills tracking jitter)
    filter_beta: float = 60.0              # how fast the cutoff rises with hand speed (higher = less lag)
    filter_rot_beta: float = 3.0           # same for hand rotation
    ik_iters: int = 3
    ik_max_pos_err: float = 0.15           # reject (hold) IK solutions worse than this [m]; was 5 cm = froze the arm too often
    # When the hand goes past what the arm can reach, slide the hand<->arm offset so the overshoot does not pile up as a
    # dead zone you must travel back through before the arm responds (0 = off, 1 = absorb all of it).
    anti_windup: float = 0.85
    dh_is_offset: int = 1
    squeeze_on: float = 0.6                # GRIP threshold for clutch
    stop_snap_rad: float = 0.01            # on release/freeze, stop at the measured pose if the arm trails by more
    trigger_deadzone: float = 0.03
    gripper_max: float = GRIPPER_MAX_M
    gripper_closed_on_trigger: bool = True  # squeeze trigger = close gripper
    gripper_enabled: bool = True            # False: send NO gripper commands at all (e.g. while a gripper is out of order)
    # Gripper input layer (piper_teleop/gripper_layer.py), measured on the real arm (firmware S-V1.8-9): the gripper
    # freezes on held targets unless a slow sine rides on the command.  With +-0.5 mm @ 6 Hz, a plain (unramped) target
    # was the fastest AND most reliable: 27/27 held targets over 3 runs, ~0.25-0.3 s to arrive; 150 mm/s ramp 0.52 s;
    # 300/600 mm/s ramps occasionally failed.  10-15 Hz sines were unreliable, 25 Hz+ fail.  0 = off for each.
    gripper_slew_mm_s: float = 0.0
    gripper_dither_mm: float = 0.5
    gripper_dither_hz: float = 6.0


@dataclass
class HandSample:
    t_recv: float                # server monotonic time of arrival [s]
    t_client: float = 0.0        # client performance.now() [ms]
    seq: int = 0
    tracked: bool = True
    emulated: bool = False       # position is IMU-predicted only (controller out of view)
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    ray_quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    trigger: float = 0.0
    squeeze: float = 0.0
    stick: tuple = (0.0, 0.0)
    btn_a: bool = False
    btn_b: bool = False
    stick_click: bool = False


IDLE, ENGAGED, HOMING, PARKING, ESTOP = "idle", "engaged", "homing", "parking", "estop"


def _euler_zyx(R):
    """R = Rz(yaw) Ry(pitch) Rx(roll)  ->  (roll, pitch, yaw)"""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(pitch)) > 1e-6:
        return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])
    return 0.0, pitch, math.atan2(-R[0, 1], R[1, 1])


def _from_euler_zyx(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def scale_rotation_axes(dR, k):
    """Scale a rotation about the x/y/z axes independently (k = roll, pitch, yaw gains).

    Done on the rotation VECTOR (axis * angle), not on Euler angles: scaling Euler angles turns a combined twist into a
    different rotation and has a singularity at pitch = +-90 deg.  A rotation about a single axis scales exactly the same
    either way."""
    if k[0] == k[1] == k[2] == 1.0:
        return dR
    return so3_exp(np.asarray(k, dtype=float) * so3_log(dR))


class TeleopArm:
    def __init__(self, cfg: ArmConfig, backend: ArmBackend, frame_yaw_deg: float = 0.0):
        self.cfg = cfg
        self.backend = backend
        self.kin = PiperKinematics(cfg.dh_is_offset, cfg.tcp_offset)
        self.kin.lo = self.kin.lo.copy()
        self.kin.lo[4] = max(self.kin.lo[4], math.radians(cfg.wrist_pitch_min_deg))   # IK never enters the wrist singularity
        self.frame_yaw_deg = frame_yaw_deg
        self._set_frame()
        self.state = IDLE
        self.q_cmd = np.zeros(6)
        self.q_meas = np.zeros(6)
        self.grip_cmd = 0.0            # what is sent (rate-limited)
        self.grip_target = 0.0         # what the trigger asks for
        self.grip_meas = 0.0
        self._dt = 0.005
        self.fine = False
        self.msg = ""
        self.pos_err = 0.0
        self.rot_err = 0.0
        self.last_seq = -1
        self._sq_prev = False
        self._a_prev = False
        self._b_prev = False
        self._need_release = True
        self._f_pos = OneEuro(cfg.filter_min_cutoff, cfg.filter_beta)
        self._f_rot = OneEuroQuat(cfg.filter_min_cutoff, cfg.filter_rot_beta)
        self._last_t = None
        self._target_p = np.zeros(3)
        self._target_R = np.eye(3)
        self._q_des = np.zeros(6)
        self._anchor = None
        self._theta0 = 0.0
        self._q_ref = np.radians(np.asarray(cfg.home_deg, float))
        pw = np.zeros(6)
        pw[3:6] = cfg.wrist_posture_weight
        self._posture = pw if np.any(pw > 0) else None
        self._ws_lo = np.array([cfg.workspace[k][0] for k in "xyz"])
        self._ws_hi = np.array([cfg.workspace[k][1] for k in "xyz"])
        self.events: list[tuple] = []     # (kind, payload) drained by the system
        self.reached_park = False
        self._hard_stop = False
        self.suspended = False             # True while calibrating / waiting for calibration: the arm only holds still
        self.suspend_msg = "axis calibration in progress - holding"
        self._last_limit_buzz = -1e9
        self._limited = False                      # currently pushing against a limit (for the haptic edge)
        self._short_n = 0
        self._lost_at: float | None = None         # when the controller data was last good -> "tracked again" cue
        self.grip_warn = ""
        self._gm_since: float | None = None
        self._gm_meas = 0.0

    # -- setup ----------------------------------------------------------
    def _set_frame(self):
        # quest -> operator frame (heading calibration) -> this arm's own base frame
        self.C = yaw_matrix(-self.cfg.base_yaw_deg) @ yaw_matrix(self.frame_yaw_deg) @ _C_QUEST_TO_ROBOT

    def set_frame_yaw(self, deg: float):
        self.frame_yaw_deg = deg
        self._set_frame()

    def start(self):
        """Read the arm once and initialise the command to where it really is."""
        st = self.backend.read()
        self.q_meas = st.q.copy()
        self.q_cmd = st.q.copy()
        self._q_des = st.q.copy()
        self.grip_meas = st.gripper
        # grippers start OPEN: ramp there from where the gripper really is (a jump would freeze it)
        self.gripper_layer = GripperLayer(self.cfg.gripper_slew_mm_s, self.cfg.gripper_dither_mm,
                                          self.cfg.gripper_dither_hz, self.cfg.gripper_max)
        self.gripper_layer.reset(st.gripper)
        self.grip_cmd = self.gripper_layer.ramp
        self.grip_target = self.cfg.gripper_max

    # -- helpers --------------------------------------------------------
    def _to_robot(self, s: HandSample):
        p = self.C @ s.pos
        R = self.C @ quat_to_mat(s.quat) @ self.C.T
        return p, R

    def ray_forward_robot(self, s: HandSample) -> np.ndarray:
        """Pointing direction of the controller ray in the *unrotated* robot frame."""
        d = quat_to_mat(s.ray_quat) @ np.array([0.0, 0.0, -1.0])
        return _C_QUEST_TO_ROBOT @ d

    def _anchor_now(self, s: HandSample):
        p, R = self._to_robot(s)
        Rt, pt = self.kin.fk_tcp(self.q_cmd)
        self._anchor = (p, R, pt, Rt)
        self._q_ref = np.asarray(self.q_cmd, float).copy()   # wrist tidy-up pulls toward the pose at GRIP press, not a fixed home
        self._limited, self._short_n = False, 0
        self._p0_hand = p.copy()                       # the hand reference as pressed: the anti-windup slide never moves this
        self._theta0 = math.atan2(pt[1], pt[0])        # azimuth of the TCP around the arm's base axis
        self._f_pos.reset()
        self._f_rot.reset()
        self._last_t = None

    def _scale(self):
        f = self.cfg.fine_factor if self.fine else 1.0
        return self.cfg.pos_scale * f, self.cfg.rot_scale * f

    def _step_toward(self, q_goal, speed, dt) -> bool:
        d = q_goal - self.q_cmd
        m = float(np.max(np.abs(d)))
        lim = speed * dt
        if m <= lim:
            self.q_cmd = np.array(q_goal, float)
            return True
        self.q_cmd = self.q_cmd + d * (lim / m)
        return False

    def _emit(self, kind, *a):
        self.events.append((kind, *a))

    def _send(self):
        """Joint command every tick; gripper command only when the gripper is switched on in the config."""
        if self.state == ESTOP:
            self.grip_target = self.grip_cmd           # a freeze freezes the gripper ramp too
        if not hasattr(self, "gripper_layer"):         # step() before start(): begin from the current command
            self.gripper_layer = GripperLayer(self.cfg.gripper_slew_mm_s, self.cfg.gripper_dither_mm,
                                              self.cfg.gripper_dither_hz, self.cfg.gripper_max)
            self.gripper_layer.reset(self.grip_cmd)
        grip_raw = self.gripper_layer.update(self.grip_target, self._dt)   # ramp + slow sine -> raw gripper command
        self.grip_cmd = self.gripper_layer.ramp
        if getattr(self.backend, "joint_locked", None):
            # Gripper-only session (config/joint_lock.yaml): the backend sends no joint frame, and the joint target is
            # pinned to where the arm really is, so nothing can be "owed" when a lock expires.  The lock state is fixed
            # for the whole session (decided at enable), so a lock expiring mid-session never re-enables the joints.
            self.q_cmd = self.q_meas.copy()
            self._q_des = self.q_cmd.copy()
        self.backend.command(self.q_cmd, grip_raw if self.cfg.gripper_enabled else None)

    @staticmethod
    def _limit_words(e) -> str:
        """Which way the hand asked for more than the arm can do (arm frame: x out from its base, y sideways, z up)."""
        i = int(np.argmax(np.abs(e)))
        return (("too far out" if e[0] > 0 else "too close to its own base"), ("too far sideways"),
                ("too high" if e[2] > 0 else "too low"))[i] if i != 1 else "too far sideways"

    def _monitor_gripper(self, now: float):
        """Warn when the gripper sits still far from its commanded opening (not zeroed / not enabled / blocked)."""
        if not self.cfg.gripper_enabled:
            self._gm_since, self.grip_warn = None, ""
            return
        if abs(self.grip_cmd - self.grip_meas) < 0.008:
            self._gm_since, self.grip_warn = None, ""
            return
        if self._gm_since is None or abs(self.grip_meas - self._gm_meas) > 0.0015:
            self._gm_since, self._gm_meas = now, self.grip_meas      # it is moving (or this is new): restart the window
        elif now - self._gm_since > 3.0 and not self.grip_warn:
            self.grip_warn = (f"gripper is not reaching its commanded opening (commanded {self.grip_cmd * 1000:.0f} mm, "
                              f"measured {self.grip_meas * 1000:.0f} mm, not moving): not zeroed / not enabled / blocked")
            self._emit("gripper_warn", self.cfg.name, self.grip_warn)

    # -- state transitions ---------------------------------------------
    def go_home(self):
        if self.state in (IDLE, HOMING):
            self.state = HOMING
            self.msg = "homing"

    def go_park(self):
        if self.state != ESTOP:
            self.state = PARKING
            self.reached_park = False
            self.msg = "parking"

    def _held(self, s: HandSample) -> bool:
        """Deadman: the arm may only move while the GRIP button is held.  (The trigger only works the gripper.)"""
        return s.squeeze > self.cfg.squeeze_on

    def _stop_here(self):
        """Command the joints to where the arm physically IS, so it stops now instead of coasting on to the
        last command.  Joints only: the gripper is never touched (it may be holding something)."""
        d = float(np.max(np.abs(self.q_meas - self.q_cmd)))
        # q_meas only ever comes from fresh feedback; the 1 rad bound is purely a guard against garbage values
        if np.all(np.isfinite(self.q_meas)) and self.cfg.stop_snap_rad < d < 1.0:
            self.q_cmd = np.clip(self.q_meas, JOINT_LO, JOINT_HI).copy()
            self._q_des = self.q_cmd.copy()

    def estop(self, hard: bool = False):
        """FREEZE: hold every joint where it is.  Default is software-only - it keeps commanding the held pose and
        never disables motors.  ``hard=True`` additionally sends the firmware fast-stop (behaviour not verified)."""
        if self.state != ESTOP:
            self._stop_here()
        self.state = ESTOP
        self._hard_stop = hard
        self.msg = "FROZEN" + (" (firmware stop)" if hard else "") + " - press Resume to continue"
        if hard:
            self.backend.emergency_stop()

    def resume(self):
        if self.state == ESTOP:
            if self._hard_stop:
                self.backend.resume()
            self._hard_stop = False
            self.start()
            self.state = IDLE
            self._need_release = True
            self.msg = "resumed - release, then press the deadman to move"

    # -- main step ------------------------------------------------------
    def step(self, s: HandSample | None, now: float, dt: float):
        cfg = self.cfg
        self._dt = dt
        st = self.backend.read()
        if st.fresh:
            self.q_meas = st.q
            self.grip_meas = st.gripper
        self._monitor_gripper(now)

        if self.state == ESTOP:
            if not self._hard_stop:                  # soft freeze: keep actively holding the frozen pose
                self._send()
            return

        fresh = (s is not None and (now - s.t_recv) * 1000.0 < cfg.stale_ms
                 and s.tracked and not s.emulated)

        # ---- non-teleop motions first (do not depend on controller data) ----
        if self.state in (HOMING, PARKING):
            goal = np.radians(cfg.home_deg if self.state == HOMING else cfg.park_deg)
            if self._step_toward(goal, cfg.home_speed, dt):
                if self.state == PARKING:
                    self.reached_park = True
                else:
                    self.state = IDLE
                    self.msg = "home reached"
                    self._emit("haptic", cfg.hand, 0.4, 60)
            if self.state == HOMING and fresh and self._held(s):
                self.state = IDLE            # grabbing cancels homing, holds position
                self.msg = "homing cancelled"
            self._send()
            return

        if self.suspended:
            if self.state == ENGAGED:
                self.state = IDLE
                self._stop_here()
            self._need_release = True        # after calibrating: release, then press GRIP again to move
            self._sq_prev = False
            self.msg = self.suspend_msg
            self._send()
            return

        if not fresh:
            if self._lost_at is None:
                self._lost_at = now
            if self.state == ENGAGED:
                self.state = IDLE
                self._stop_here()
                self.msg = "tracking/stream lost - holding"
            elif s is None or (now - s.t_recv) * 1000.0 >= cfg.stale_ms:
                self.msg = "no controller data" if s is None else "stale data - holding"
            else:
                self.msg = "controller not tracked - holding"
            self._need_release = True
            self._sq_prev = False
            self._send()
            return

        # ---- fresh sample ----
        if self._lost_at is not None:
            if now - self._lost_at > 1.0:                   # tracked again after a real gap: a short tick tells the operator
                self._emit("haptic", cfg.hand, 0.35, 45)
            self._lost_at = None
        squeeze = self._held(s)
        sq_edge = squeeze and not self._sq_prev
        a_edge = s.btn_a and not self._a_prev
        b_edge = s.btn_b and not self._b_prev
        self._sq_prev, self._a_prev, self._b_prev = squeeze, s.btn_a, s.btn_b
        if not squeeze:
            self._need_release = False

        # gripper follows the trigger whenever the hand is tracked
        trig = 0.0 if s.trigger < cfg.trigger_deadzone else s.trigger
        open_frac = (1.0 - trig) if cfg.gripper_closed_on_trigger else trig
        self.grip_target = float(np.clip(open_frac, 0.0, 1.0) * cfg.gripper_max)

        if a_edge:
            self.fine = not self.fine
            if self.state == ENGAGED:
                self._anchor_now(s)
            self._emit("haptic", cfg.hand, 0.25, 30)
            self.msg = "fine mode ON" if self.fine else "fine mode off"

        if self.state == IDLE:
            if b_edge and not squeeze:
                self.go_home()
            elif sq_edge and not self._need_release:
                self.state = ENGAGED
                self._anchor_now(s)
                self._q_des = self.q_cmd.copy()
                self.msg = "engaged"
                self._emit("haptic", cfg.hand, 0.5, 40)
            else:
                self._send()
                return

        if self.state == ENGAGED:
            if not squeeze:
                self.state = IDLE
                self._stop_here()                    # deadman released: stop where the arm actually is
                self.msg = "released"
                self._emit("haptic", cfg.hand, 0.2, 25)
                self._send()
                return
            self._engaged_step(s, dt)
        self._send()

    def _engaged_step(self, s: HandSample, dt: float):
        cfg = self.cfg
        p0, R0, pt0, Rt0 = self._anchor
        p, R = self._to_robot(s)
        sp, sr = self._scale()

        # new network sample -> advance the adaptive filters at the sample's own period
        if s.seq != self.last_seq:
            self.last_seq = s.seq
            t = s.t_recv
            fdt = (t - self._last_t) if self._last_t is not None else 0.0
            self._last_t = t
            dp = sp * (p - p0)
            dR = scale_rotation_axes(R @ R0.T, cfg.rot_axis_scale)     # rotation delta, per-axis gains
            dR = so3_exp(sr * so3_log(dR)) if sr != 1.0 else dR
            # the box may never pull the target away from where the TCP already is
            raw_p = pt0 + dp
            tgt_p = np.clip(raw_p, np.minimum(self._ws_lo, pt0), np.maximum(self._ws_hi, pt0))
            tgt_R = dR @ Rt0
            # follow_base: the gripper heading follows the swing of the HAND-DRIVEN target around the base.  It must not use
            # the slid (anti-windup) target: then a wrist-limited twist -> sideways shortfall -> slide -> more heading ->
            # more shortfall ... ran away (J1 swung to ~85 deg for a 15-20 deg hand twist; tests/test_core.py).
            hand_p = np.clip(pt0 + sp * (p - self._p0_hand), np.minimum(self._ws_lo, pt0), np.maximum(self._ws_hi, pt0))
            if cfg.orientation_mode == "follow_base" and math.hypot(hand_p[0], hand_p[1]) > 0.06:
                dth = math.atan2(hand_p[1], hand_p[0]) - self._theta0
                dth = (dth + math.pi) % (2.0 * math.pi) - math.pi
                c, sn = math.cos(dth), math.sin(dth)
                tgt_R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]) @ tgt_R
            tgt_q = mat_to_quat(tgt_R)
            self._target_p = self._f_pos(tgt_p, fdt)
            self._target_R = quat_to_mat(self._f_rot(tgt_q, fdt))
            q_ik, ep, er = self.kin.ik_tcp(self._q_des, self._target_R, self._target_p, iters=cfg.ik_iters,
                                           q_ref=self._q_ref, posture=self._posture)
            self.pos_err, self.rot_err = ep, er
            ok = ep <= cfg.ik_max_pos_err and bool(np.all(np.isfinite(q_ik)))
            below_floor = False
            if ok:
                # Floor on where the tool tip really ends up (not just on the target): an IK compromise (wrist limits, a
                # twist) can sink the tip below the box.  Never accept a solution that is below the floor AND lower than now.
                z_new = float(self.kin.fk_tcp(q_ik)[1][2])
                z_now = float(self.kin.fk_tcp(self._q_des)[1][2])
                below_floor = z_new < self._ws_lo[2] - 0.01 and z_new < z_now - 1e-4
                ok = not below_floor
            q_use = q_ik if ok else self._q_des
            # how much of what the HAND asked for is unreachable: box clamp excess + kinematic shortfall (arm frame)
            # Measured against the UNFILTERED target: the filtered one lags behind an anchor slide, so the same shortfall was
            # counted again on every hand sample and the target retreated 10-17 cm while the hand sat still.
            e_kin = tgt_p - self.kin.fk_tcp(q_use)[1]
            e = (raw_p - tgt_p) + e_kin
            gap = float(np.linalg.norm(e))
            if ok:
                self._q_des = q_ik
                if gap > 0.012:
                    self.msg = "at reach limit: " + self._limit_words(e)
                else:
                    self.msg = "tracking"
            elif below_floor:
                self.msg = "at reach limit: too low (floor)"
            else:
                self.msg = "target far out of reach - holding: " + self._limit_words(e)
            # Vibration on a limit: IMMEDIATELY on the first sample (box clamp / floor / out of reach are certain), the
            # kinematic shortfall must persist for two samples so a fast hand's transient IK lag does not buzz.
            hard = below_floor or (not ok) or float(np.linalg.norm(raw_p - tgt_p)) > 0.003
            self._short_n = self._short_n + 1 if gap > 0.012 else 0
            limited = hard or self._short_n >= 2
            if limited:
                edge = not self._limited
                if edge or s.t_recv - self._last_limit_buzz > 0.12:      # first hit: strong; while pushing: steady pulses
                    self._emit("haptic", cfg.hand, 0.8 if edge else min(0.6, 0.25 + 4.0 * gap), 70 if edge else 40)
                    self._last_limit_buzz = s.t_recv
            self._limited = limited
            if cfg.anti_windup > 0 and sp > 0:
                # slide the offset so there is no dead zone on the way back: the box overshoot at once (certain), a
                # kinematic shortfall only once it has persisted (a few samples of IK still converging are not a limit)
                # and only the part beyond a 3 cm band: a smaller lag is IK convergence/damping, and sliding on it chases
                # its own tail (the arm trails the retreating target, the lag persists, the target keeps retreating)
                slide = raw_p - tgt_p
                if self._short_n >= 4 and er < 0.12:         # not while the WRIST is the limit: orientation gives, position holds
                    n_k = float(np.linalg.norm(e_kin))
                    if n_k > 0.03:
                        slide = slide + e_kin * (1.0 - 0.03 / n_k)
                if float(np.linalg.norm(slide)) > 0.004:
                    self._anchor = (p0 + slide / sp * cfg.anti_windup, R0, pt0, Rt0)

        # control-rate smoothing + speed limit toward the latest IK solution
        if cfg.smoothing_ms > 0:
            a = 1.0 - math.exp(-dt / (cfg.smoothing_ms * 1e-3))
            goal = self.q_cmd + (self._q_des - self.q_cmd) * a
        else:
            goal = self._q_des
        d = goal - self.q_cmd
        m = float(np.max(np.abs(d)))
        lim = cfg.max_joint_speed * dt
        if m > lim:
            d *= lim / m
        if cfg.max_tcp_speed > 0 and dt > 0:                 # cap the TIP speed too (scales the whole joint step)
            p_now = self.kin.fk_tcp(self.q_cmd)[1]
            v = float(np.linalg.norm(self.kin.fk_tcp(self.q_cmd + d)[1] - p_now)) / dt
            vmax = cfg.max_tcp_speed
            if cfg.slow_zone_speed > 0 and p_now[2] < cfg.slow_zone_z:
                vmax = min(vmax, cfg.slow_zone_speed)
            if v > vmax:
                d *= vmax / v
        self.q_cmd = np.clip(self.q_cmd + d, JOINT_LO, JOINT_HI)

    # -- reporting ------------------------------------------------------
    def snapshot(self) -> dict:
        R, p = self.kin.fk_tcp(self.q_cmd)
        return {
            "name": self.cfg.name, "hand": self.cfg.hand, "state": self.state, "msg": self.msg,
            "fine": self.fine, "q": [round(float(x), 4) for x in self.q_cmd],
            "q_meas": [round(float(x), 4) for x in self.q_meas],
            "grip": round(self.grip_cmd, 4), "grip_target": round(self.grip_target, 4), "grip_meas": round(self.grip_meas, 4), "grip_warn": self.grip_warn, "tcp": [round(float(x), 4) for x in p],
            "pos_err_mm": round(self.pos_err * 1e3, 2),
            "pts": [[round(float(v), 4) for v in row] for row in self.kin.points(self.q_cmd)],
        }
