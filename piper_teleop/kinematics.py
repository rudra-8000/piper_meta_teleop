"""Piper forward kinematics, Jacobian and damped-least-squares IK in plain numpy.

DH parameters are the modified (Craig) DH table published in AgileX's
``piper_sdk`` (MIT licensed, ``piper_sdk/kinematics/piper_fk.py``).  Lengths are
converted to metres.  ``dh_is_offset=1`` is the table for firmware >= S-V1.6-3.

Everything here is pure maths (no CAN, no hardware) so it is unit-testable and
fast enough to run every 5 ms control tick (~100 us per IK iteration).
"""
from __future__ import annotations

import math

import numpy as np

# Joint limits [rad].  Intersection of the SDK JointCtrl table and the quick
# start manual (manual J6 is +-100 deg, SDK says +-120 deg -> use the smaller).
JOINT_LO = np.radians([-150.0, 0.0, -170.0, -100.0, -70.0, -100.0])
JOINT_HI = np.radians([150.0, 180.0, 0.0, 100.0, 70.0, 100.0])

_PI = math.pi


def _dh_table(dh_is_offset: int):
    a = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0]) * 1e-3
    alpha = np.array([0.0, -_PI / 2, 0.0, _PI / 2, -_PI / 2, _PI / 2])
    d = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0]) * 1e-3
    if dh_is_offset:
        theta0 = np.array([0.0, -_PI * 172.22 / 180, -102.78 / 180 * _PI, 0.0, 0.0, 0.0])
    else:
        theta0 = np.array([0.0, -_PI * 174.22 / 180, -100.78 / 180 * _PI, 0.0, 0.0, 0.0])
    return a, alpha, theta0, d


# ---------------------------------------------------------------- SO(3) helpers
def quat_to_mat(q) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1 - xx - yy],
    ])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (x, y, z, w), w >= 0."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s,
                      (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
                      0.25 * s, (R[1, 0] - R[0, 1]) / s])
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis*angle vector."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * v
    if ang > _PI - 1e-4:  # near 180 deg the antisymmetric part vanishes
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(B), 0.0))
        k = int(np.argmax(axis))
        axis[k] = max(axis[k], 1e-9)
        for i in range(3):
            if i != k:
                axis[i] = math.copysign(axis[i], B[k, i])
        axis /= np.linalg.norm(axis)
        return axis * ang
    return v * (ang / (2.0 * math.sin(ang)))


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Axis*angle vector -> rotation matrix (Rodrigues)."""
    ang = float(np.linalg.norm(w))
    if ang < 1e-9:
        return np.eye(3) + _skew(w)
    k = w / ang
    K = _skew(k)
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def _skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


# ------------------------------------------------------------------ kinematics
class PiperKinematics:
    """FK / Jacobian / IK for one Piper arm (6 revolute joints).

    ``tcp_offset``: tool-centre-point offset in the flange frame [m].  FK/IK
    helpers named ``*_tcp`` use it; the plain ones refer to the flange (frame 6).
    """

    def __init__(self, dh_is_offset: int = 1, tcp_offset=(0.0, 0.0, 0.0)):
        self.a, self.alpha, self.theta0, self.d = _dh_table(dh_is_offset)
        self._ca = np.cos(self.alpha)
        self._sa = np.sin(self.alpha)
        self.tcp_offset = np.asarray(tcp_offset, dtype=float)
        # constant parts of the per-link transforms; frames() fills the rest
        L = np.zeros((6, 4, 4))
        L[:, 0, 3] = self.a
        L[:, 1, 2] = -self._sa
        L[:, 1, 3] = -self._sa * self.d
        L[:, 2, 2] = self._ca
        L[:, 2, 3] = self._ca * self.d
        L[:, 3, 3] = 1.0
        self._L = L
        self.lo = JOINT_LO.copy()
        self.hi = JOINT_HI.copy()

    # -- forward ---------------------------------------------------------
    def _chain(self, q) -> np.ndarray:
        """(6,4,4) array of base->frame_i transforms (fresh each call)."""
        th = np.asarray(q, dtype=float) + self.theta0
        ct, st = np.cos(th), np.sin(th)
        L = self._L
        L[:, 0, 0] = ct
        L[:, 0, 1] = -st
        L[:, 1, 0] = st * self._ca
        L[:, 1, 1] = ct * self._ca
        L[:, 2, 0] = st * self._sa
        L[:, 2, 1] = ct * self._sa
        F = np.empty((6, 4, 4))
        F[0] = L[0]
        for i in range(1, 6):
            np.matmul(F[i - 1], L[i], out=F[i])
        return F

    def frames(self, q) -> list[np.ndarray]:
        """4x4 base->frame_i transforms for i = 1..6."""
        return list(self._chain(q))

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Flange pose: (R 3x3, p 3) in the base frame, metres."""
        T = self._chain(q)[-1]
        return T[:3, :3], T[:3, 3]

    def fk_tcp(self, q) -> tuple[np.ndarray, np.ndarray]:
        R, p = self.fk(q)
        return R, p + R @ self.tcp_offset

    def points(self, q) -> np.ndarray:
        """Origins of base, frames 1..6 and TCP (8x3) - for visualisation."""
        F = self._chain(q)
        R, p = F[-1, :3, :3], F[-1, :3, 3]
        return np.vstack([np.zeros(3), F[:, :3, 3], p + R @ self.tcp_offset])

    def fk_jac(self, q):
        """Flange pose and 6x6 geometric Jacobian [Jv; Jw] (base frame)."""
        F = self._chain(q)
        pe = F[-1, :3, 3]
        z = F[:, :3, 2]
        J = np.empty((6, 6))
        J[:3] = np.cross(z, pe - F[:, :3, 3]).T
        J[3:] = z.T
        return F[-1, :3, :3], pe, J

    # -- inverse ---------------------------------------------------------
    def ik_tcp(self, q_seed, R_des, p_tcp_des, *, iters=3, rot_weight=1.0,
               damping=0.04, rot_damping=0.0, rot_clip=0.2, max_step=0.12, q_ref=None, posture=None):
        """Task-priority damped-least-squares IK for a TCP pose, warm-started at ``q_seed``.

        POSITION IS PRIORITY 1, orientation is solved in the null space of the position task (priority 2), and the
        optional ``posture`` pull toward ``q_ref`` is the lowest.  A single weighted least squares mixes the two: a
        45 deg orientation error (0.8 rad) then outweighs a 10 cm position error, so as soon as the wrist hit a joint limit
        the solver dragged the tool tip 10-25 cm away to chase an orientation it could not reach.  Here an unreachable
        orientation only costs orientation.

        Joint limits are handled by an active set: a joint whose step would cross its limit is pinned at the limit and the
        step is re-solved with that joint frozen (its column removed), so the others take over instead of the result being
        clipped into a different pose.  ``rot_weight`` is the gain on the orientation error (1 = full).

        Returns ``(q, pos_err_m, rot_err_rad)`` (residual before the final step).
        """
        R_des = np.asarray(R_des)
        p_fl_des = np.asarray(p_tcp_des) - R_des @ self.tcp_offset
        q = np.array(q_seed, dtype=float)
        M = None if posture is None else np.diag(np.asarray(posture, float) ** 2)
        qr = None if q_ref is None else np.asarray(q_ref, float)
        eye = np.eye(6)
        n_p = n_r = 0.0
        for _ in range(iters):
            R, p, J = self.fk_jac(q)
            e_p = p_fl_des - p
            e_r = rot_weight * so3_log(R_des @ R.T)
            n_p = float(np.linalg.norm(e_p))
            n_r = float(np.linalg.norm(e_r)) / max(rot_weight, 1e-9)
            if np.linalg.norm(e_r) > rot_clip:                # a far-off (unreachable) orientation must not shove the position task
                e_r = e_r * (rot_clip / np.linalg.norm(e_r))
            if n_p < 2e-4 and n_r < 2e-3:
                break
            free = np.ones(6)
            dq = np.zeros(6)
            for _pass in range(4):                         # active set: pin joints that run into a limit
                Jf = J * free                              # frozen joints = zero columns (they cannot move)
                Jp, Jr = Jf[:3], Jf[3:]
                Ap = Jp @ Jp.T
                manip = math.sqrt(max(np.linalg.det(Ap), 0.0))
                lam = damping + (0.12 * (1.0 - manip / 0.004) if manip < 0.004 else 0.0)
                Jp_pinv = Jp.T @ np.linalg.inv(Ap + (lam * lam) * np.eye(3))
                dq1 = Jp_pinv @ e_p
                N = (eye - Jp_pinv @ Jp) * free[None, :] * free[:, None]
                A = Jr @ N
                H = A.T @ A + (max(lam, rot_damping) ** 2) * eye
                g = A.T @ (e_r - Jr @ dq1)
                if M is not None:
                    Mf = M * free[None, :] * free[:, None]
                    H = H + N.T @ Mf @ N
                    g = g + N.T @ (Mf @ (qr - q - dq1))
                dq = dq1 + N @ np.linalg.solve(H, g)
                dq = dq * free
                m = np.max(np.abs(dq))
                if m > max_step:
                    dq *= max_step / m
                qn = q + dq
                hit = ((qn < self.lo) | (qn > self.hi)) & (free > 0)
                if not hit.any():
                    break
                pin = np.clip(qn, self.lo, self.hi)        # freeze them AT the limit, re-solve with the rest
                q = np.where(hit, pin, q)
                free = np.where(hit, 0.0, free)
                R, p, J = self.fk_jac(q)
                e_p = p_fl_des - p
                e_r = rot_weight * so3_log(R_des @ R.T)
                if np.linalg.norm(e_r) > rot_clip:
                    e_r = e_r * (rot_clip / np.linalg.norm(e_r))
            q = np.clip(q + dq, self.lo, self.hi)
        # residuals are from the last evaluated iterate (one step stale at most)
        return q, n_p, n_r
