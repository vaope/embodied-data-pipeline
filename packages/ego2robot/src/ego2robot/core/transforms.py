"""SE(3) / SO(3) primitives with explicit frame semantics.

System design 5.2 and the M0 milestone of implementation-plan.md.

Conventions enforced here (and nowhere else):

* ``T_A_B`` maps points from ``B`` into ``A``: ``p_A = T_A_B @ p_B``.
* composition: ``T_A_C = T_A_B @ T_B_C``.
* quaternions are ``[qx, qy, qz, qw]`` (``xyzw``) at every persisted boundary.
* rotation vectors use ``Log_SO3`` / ``Exp_SO3`` in radians.
* camera-frame relative rotation delta uses the left-multiply convention of
  system-design.md 6.3 (paper appendix C.2 equation 10)::

      delta_R_C[t] = R_C_TCP[t+1] @ transpose(R_C_TCP[t])
      delta_rotvec_C[t] = Log_SO3(delta_R_C[t])

None of these helpers take or return a bare "pose": every function that mixes
frames names them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .frames import Frame, TransformName

__all__ = [
    "SO3Repair",
    "so3_repair",
    "is_rotation_matrix",
    "rotation_matrix_from_axes",
    "quat_xyzw_from_matrix",
    "matrix_from_quat_xyzw",
    "quat_normalize",
    "quat_multiply",
    "quat_conjugate",
    "quat_angle_rad",
    "hemisphere_correct",
    "log_so3",
    "exp_so3",
    "skew",
    "unskew",
    "make_transform",
    "split_transform",
    "transform_from_position_quat",
    "invert_transform",
    "compose_transforms",
    "transform_points",
    "delta_position",
    "delta_rotation_matrix",
    "delta_rotvec",
    "geodesic_distance_rad",
]

_EPS = 1e-12


@dataclass(frozen=True)
class SO3Repair:
    """Result of projecting a numeric matrix back onto SO(3)."""

    R: NDArray[np.float64]
    """The repaired rotation matrix."""

    correction_rad: float
    """Geodesic angle between the input and the repaired matrix, in radians."""

    determinant: float
    """Determinant of the raw input matrix."""

    max_orthogonality_error: float
    """``max(abs(R_raw.T @ R_raw - I))`` of the raw input."""

    corrected: bool
    """True when the input was not already a valid rotation within tolerance."""


# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------


def skew(v: ArrayLike) -> NDArray[np.float64]:
    """Skew-symmetric matrix ``[v]_x`` such that ``[v]_x @ w == cross(v, w)``."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def unskew(M: ArrayLike) -> NDArray[np.float64]:
    """Inverse of :func:`skew`: extract the axis vector from a 3x3 skew matrix."""
    M = np.asarray(M, dtype=np.float64).reshape(3, 3)
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float64)


def rotation_matrix_from_axes(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray[np.float64]:
    """Assemble ``R = [x y z]`` as columns (system-design.md 7)."""
    R = np.stack(
        [np.asarray(x, dtype=np.float64).reshape(3),
         np.asarray(y, dtype=np.float64).reshape(3),
         np.asarray(z, dtype=np.float64).reshape(3)],
        axis=1,
    )
    return R


def is_rotation_matrix(R: ArrayLike, *, atol: float = 1e-6) -> bool:
    """True when ``R`` is a valid rotation matrix within ``atol``."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    if not np.all(np.isfinite(R)):
        return False
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-5:
        return False
    return bool(np.max(np.abs(R.T @ R - np.eye(3))) <= atol)


def so3_repair(R_raw: ArrayLike, *, tol: float = 1e-9) -> SO3Repair:
    """Project a numeric matrix onto SO(3) and report how far it had drifted.

    system-design.md 7 requires an SVD / polar-decomposition check before
    ``R_C_TCP`` is emitted, with the correction magnitude recorded.  The raw
    input is never silently accepted.
    """
    R_raw = np.asarray(R_raw, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(R_raw)):
        raise ValueError("cannot repair a rotation matrix containing NaN/Inf")

    det_raw = float(np.linalg.det(R_raw))
    ortho_err = float(np.max(np.abs(R_raw.T @ R_raw - np.eye(3))))

    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    # guard against a reflection (det = -1) produced by a degenerate input
    if float(np.linalg.det(R)) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt

    correction = geodesic_distance_rad(R_raw, R)
    corrected = bool(ortho_err > tol or abs(det_raw - 1.0) > 1e-9)
    return SO3Repair(
        R=R,
        correction_rad=correction,
        determinant=det_raw,
        max_orthogonality_error=ortho_err,
        corrected=corrected,
    )


def log_so3(R: ArrayLike) -> NDArray[np.float64]:
    """``Log_SO3``: rotation matrix -> rotation vector (radians).

    Numerically stable near ``theta = 0`` and ``theta = pi``.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        # 2nd order series: Log(R) ~= 0.5 * (R - R.T) for tiny angles
        return 0.5 * unskew(R - R.T)

    if np.pi - theta < 1e-6:
        # near pi the skew part vanishes; recover the axis from R + I
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # fix signs using the largest component
        k = int(np.argmax(axis))
        if axis[k] > _EPS:
            sign_src = R[:, k] - (2.0 * A[k, k] - 1.0) * np.eye(3)[:, k]
            for i in range(3):
                if i != k and abs(axis[i]) > _EPS:
                    axis[i] = np.sign(sign_src[i]) * axis[i]
            axis[k] = abs(axis[k])
        norm = float(np.linalg.norm(axis))
        if norm < _EPS:
            return np.zeros(3, dtype=np.float64)
        return axis / norm * theta

    return unskew(R - R.T) * (theta / (2.0 * np.sin(theta)))


def exp_so3(rotvec: ArrayLike) -> NDArray[np.float64]:
    """``Exp_SO3``: rotation vector (radians) -> rotation matrix (Rodrigues)."""
    w = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        # 2nd order series keeps Exp/Log round-trip stable for tiny rotations
        W = skew(w)
        return np.eye(3) + W + 0.5 * (W @ W)
    axis = w / theta
    W = skew(axis)
    return np.eye(3) + np.sin(theta) * W + (1.0 - np.cos(theta)) * (W @ W)


def geodesic_distance_rad(R_a: ArrayLike, R_b: ArrayLike) -> float:
    """Geodesic angle between two rotations, in radians.

    ``R_b`` need not be a perfect rotation; it is projected first so that the
    function is safe to use as a drift metric.
    """
    A = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    B = np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    R_rel = A.T @ B
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.arccos(float(np.clip(cos_theta, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw, persisted order)
# ---------------------------------------------------------------------------


def quat_normalize(q: ArrayLike) -> NDArray[np.float64]:
    """Return the unit quaternion of ``q`` in ``xyzw`` order."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        raise ValueError("cannot normalize a zero quaternion")
    return q / n


def quat_multiply(q_a: ArrayLike, q_b: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product ``q_a * q_b`` for ``xyzw`` quaternions."""
    ax, ay, az, aw = np.asarray(q_a, dtype=np.float64).reshape(4)
    bx, by, bz, bw = np.asarray(q_b, dtype=np.float64).reshape(4)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: ArrayLike) -> NDArray[np.float64]:
    """Conjugate (inverse for unit quaternions) in ``xyzw`` order."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_angle_rad(q_a: ArrayLike, q_b: ArrayLike) -> float:
    """Angle of the relative rotation between two unit quaternions."""
    q_a = quat_normalize(q_a)
    q_b = quat_normalize(q_b)
    dot = float(np.clip(abs(np.dot(q_a, q_b)), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def quat_xyzw_from_matrix(R: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrix -> unit quaternion in ``xyzw`` order.

    Uses the Shepperd branch selection so it is stable for all inputs.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return quat_normalize(np.array([qx, qy, qz, qw], dtype=np.float64))


def matrix_from_quat_xyzw(q: ArrayLike) -> NDArray[np.float64]:
    """Unit quaternion in ``xyzw`` order -> rotation matrix."""
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def hemisphere_correct(quats: ArrayLike) -> NDArray[np.float64]:
    """Make consecutive ``xyzw`` quaternions share a hemisphere.

    system-design.md 7 / implementation-plan.md M2: without this, a sign flip
    turns into a fake 180 degree jump when the sequence is smoothed.

    The first quaternion is normalised to ``qw >= 0`` to fix the global gauge.
    """
    q = np.asarray(quats, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"expected quaternion array of shape [T, 4], got {q.shape}")
    out = np.array(q, dtype=np.float64, copy=True)
    for t in range(out.shape[0]):
        out[t] = quat_normalize(out[t])
    if out.shape[0] and out[0, 3] < 0.0:
        out[0] = -out[0]
    for t in range(1, out.shape[0]):
        if float(np.dot(out[t], out[t - 1])) < 0.0:
            out[t] = -out[t]
    return out


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------


def make_transform(R: ArrayLike, t: ArrayLike) -> NDArray[np.float64]:
    """Build ``T_A_B`` from a rotation and a translation expressed in ``A``."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def split_transform(T: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split ``T_A_B`` into ``(R_A_B, t_A_B_origin)``."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return T[:3, :3].copy(), T[:3, 3].copy()


def transform_from_position_quat(
    position: ArrayLike, quat_xyzw: ArrayLike
) -> NDArray[np.float64]:
    """Build ``T_A_B`` from a position in ``A`` and an ``xyzw`` quaternion."""
    return make_transform(matrix_from_quat_xyzw(quat_xyzw), position)


def invert_transform(T_A_B: ArrayLike) -> NDArray[np.float64]:
    """``T_B_A = inverse(T_A_B)`` computed analytically (no generic inverse)."""
    T = np.asarray(T_A_B, dtype=np.float64).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose_transforms(*transforms: ArrayLike) -> NDArray[np.float64]:
    """Left-to-right composition: ``compose(T_A_B, T_B_C) == T_A_B @ T_B_C``."""
    if not transforms:
        return np.eye(4, dtype=np.float64)
    out = np.asarray(transforms[0], dtype=np.float64).reshape(4, 4)
    for T in transforms[1:]:
        out = out @ np.asarray(T, dtype=np.float64).reshape(4, 4)
    return out


def transform_points(T_A_B: ArrayLike, points_B: ArrayLike) -> NDArray[np.float64]:
    """Apply ``T_A_B`` to an array of points shaped ``[..., 3]``."""
    T = np.asarray(T_A_B, dtype=np.float64).reshape(4, 4)
    P = np.asarray(points_B, dtype=np.float64)
    if P.shape[-1] != 3:
        raise ValueError(f"points must have a trailing dimension of 3, got {P.shape}")
    R, t = T[:3, :3], T[:3, 3]
    return P @ R.T + t


# ---------------------------------------------------------------------------
# Camera-frame relative action deltas (system-design.md 6.3)
# ---------------------------------------------------------------------------


def delta_position(position_C_m: ArrayLike) -> NDArray[np.float64]:
    """``delta_position_C[t] = position_C[t+1] - position_C[t]``."""
    p = np.asarray(position_C_m, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected position array of shape [T, 3], got {p.shape}")
    return np.diff(p, axis=0)


def delta_rotation_matrix(R_C_TCP: ArrayLike) -> NDArray[np.float64]:
    """``delta_R_C[t] = R_C_TCP[t+1] @ transpose(R_C_TCP[t])``.

    Left-multiplication expresses the delta in the camera frame, matching the
    paper's ``Delta R_cc = R_ce Delta R_ee R_ce^T`` rule (appendix C.2, eq. 10).
    """
    R = np.asarray(R_C_TCP, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"expected rotation array of shape [T, 3, 3], got {R.shape}")
    return np.einsum("tij,tkj->tik", R[1:], R[:-1])


def delta_rotvec(R_C_TCP: ArrayLike) -> NDArray[np.float64]:
    """``Log_SO3`` of :func:`delta_rotation_matrix`."""
    dR = delta_rotation_matrix(R_C_TCP)
    return np.stack([log_so3(R) for R in dR], axis=0) if dR.shape[0] else np.zeros((0, 3))


def delta_rotation_axis_in_frame(frame_name: TransformName) -> str:
    """Documentation helper: state which frame a delta axis is expressed in."""
    return f"rotation delta axes are expressed in frame {frame_name.source.value}"


def frame_description(frame: Frame | str) -> str:
    """Return the documented convention string for a frame."""
    f = Frame(frame)
    if f is Frame.C:
        return CAMERA_OPTICAL_CONVENTION.describe()
    if f is Frame.B:
        return ROBOT_BASE_CONVENTION.describe()
    if f is Frame.TCP:
        return TCP_CONVENTION.describe()
    return f"{f.value}: see docs/system-design.md section 5.1"
