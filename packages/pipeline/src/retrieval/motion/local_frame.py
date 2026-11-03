"""Local (self-frame) effector motion representation.

Feature version: ``local-directed-v1``.

This module is the *new, independent* representation requested after the 8785
prototype was found to drop direction: the prototype transformed to the first
effector frame but then took ``norm`` of every local delta, so +Z and -Z motion
of equal amplitude produced identical features.

Conventions (column vectors, as in the handoff document):

    p_t : reference point position in the source world frame (metres)
    R_t : rotation taking a local vector to the source world frame

A. Fixed initial-effector frame trajectory

    x_t = R_0^T (p_t - p_0)

B. Per-step displacement in each step's own effector frame

    dx_t = R_t^T (p_{t+1} - p_t)

Rotation increment (diagnostic only here):

    w_t = Log(R_t^T R_{t+1})

A and B are different objects.  ``R_0^T R_t dx_t`` is the step in the fixed
initial frame, so B is accumulated into A only after that rotation:

    x_t = sum_{k<t} R_0^T R_k dx_k

Naive ``cumsum(dx_t)`` is wrong whenever the effector rotates; the test suite
keeps that counterexample.

NumPy arrays below hold row vectors, so the row forms are used::

    x    = (p - p_0) @ R_0                   (= R_0^T (p - p_0))
    dx   = diff(p) @ R[:-1]                  (= R_t^T (p_{t+1} - p_t))
    w    = Rotation(R[:-1]^T R[1:]).as_rotvec()

Nothing here needs a shared world frame: x_t and dx_t are invariant to a rigid
transform of the whole sequence.  Nothing here aligns the human wrist axes to
the robot end-effector axes; that mapping is a declared input, not an implicit
identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FEATURE_VERSION = 'local-directed-v1'

# Proper signed permutations of the 3 coordinate axes (24 rotations).
def signed_permutation_group():
    from itertools import permutations, product
    group = []
    for perm in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if np.linalg.det(m) > 0.5:  # keep det=+1 only, i.e. proper rotations
                group.append(m)
    assert len(group) == 24
    return group


AXIS_GROUP = signed_permutation_group()
IDENTITY = np.eye(3)


def mapping_name(matrix):
    """Human-readable descriptor of a signed axis mapping (rows = new axes)."""
    letters = 'xyz'
    names = []
    for row in np.asarray(matrix):
        col = int(np.argmax(np.abs(row)))
        if abs(abs(row[col]) - 1) > 1e-9 or np.count_nonzero(np.abs(row) > 1e-9) != 1:
            raise ValueError('mapping is not a signed axis permutation')
        names.append(('+' if row[col] > 0 else '-') + letters[col])
    return ','.join(names)


def describe_mapping(matrix):
    """Name a mapping: signed-permutation name, or ``general_rotation(...)``."""
    try:
        return mapping_name(matrix)
    except ValueError:
        angle = float(np.degrees(Rotation.from_matrix(np.asarray(matrix, float)).magnitude()))
        return f'general_rotation({angle:.3f} deg)'


def rotation_matrices(sequence):
    r = np.full((len(sequence.timestamps_s), 3, 3), np.nan)
    r[sequence.valid_mask] = Rotation.from_quat(sequence.rotation_xyzw[sequence.valid_mask]).as_matrix()
    return r


def resample_poses(sequence, n_points, min_valid_ratio=0.8, max_gap_s=0.15):
    """Resample one event onto ``n_points`` equal-progress poses.

    The event window is exactly the sequence's own extent (callers crop the
    event first), so no extrapolation is allowed: both endpoints must be valid.
    Invalid runs are interpolated through, but the valid ratio and largest time
    gap are gated and recorded.  No value is silently filled with zero.
    """
    n_points = int(n_points)
    if n_points < 3:
        raise ValueError('n_points must be >= 3')
    t, valid = sequence.timestamps_s, sequence.valid_mask
    if valid.sum() < 3:
        raise ValueError(f'only {int(valid.sum())} valid poses in event')
    if not (valid[0] and valid[-1]):
        raise ValueError('event endpoint pose invalid; refusing to extrapolate')
    ratio = float(valid.mean())
    if ratio < min_valid_ratio:
        raise ValueError(f'valid ratio {ratio:.3f} below {min_valid_ratio}')
    vt = t[valid]
    gap = float(np.diff(vt).max()) if len(vt) > 1 else 0.0
    if gap > max_gap_s + 1e-9:
        raise ValueError(f'tracking/time gap {gap:.3f}s exceeds {max_gap_s}s')
    grid = np.linspace(vt[0], vt[-1], n_points)
    p = np.stack([np.interp(grid, vt, sequence.position_m[valid, j]) for j in range(3)], axis=1)
    rot = Slerp(vt, Rotation.from_quat(sequence.rotation_xyzw[valid]))(grid).as_matrix()
    return grid, p, rot, {
        'source_samples': int(len(t)),
        'source_valid': int(valid.sum()),
        'source_valid_ratio': ratio,
        'max_valid_gap_s': gap,
        'resample_points': n_points,
        'span_s': [float(vt[0]), float(vt[-1])],
        'duration_s': float(vt[-1] - vt[0]),
        'interpolated': True,
    }


@dataclass
class LocalFrameTrajectory:
    """Local directed representation of one event for one effector/hand."""

    source: str
    sequence_id: str
    side: str
    progress: np.ndarray            # [N] normalised progress in [0, 1]
    positions_fixed: np.ndarray     # [N,3]  A  metres, initial effector frame
    deltas_self: np.ndarray         # [N-1,3] B  metres per progress step, own frame
    deltas_fixed: np.ndarray        # [N-1,3] R_0^T (p_{t+1}-p_t)
    rotation_increments: np.ndarray  # [N-1,3] rotvec, own frame
    frame_mapping: np.ndarray       # [3,3] declared local-frame mapping C
    mapping_declared: bool
    reference_point: str
    axis_convention: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ('progress', 'positions_fixed', 'deltas_self', 'deltas_fixed',
                     'rotation_increments', 'frame_mapping'):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        n = len(self.progress)
        if self.positions_fixed.shape != (n, 3) or self.deltas_self.shape != (n - 1, 3) \
                or self.deltas_fixed.shape != (n - 1, 3) or self.rotation_increments.shape != (n - 1, 3):
            raise ValueError('local trajectory shape mismatch')
        if self.frame_mapping.shape != (3, 3):
            raise ValueError('frame mapping must be 3x3')
        if abs(np.linalg.det(self.frame_mapping) - 1) > 1e-6 or \
                np.max(abs(self.frame_mapping.T @ self.frame_mapping - np.eye(3))) > 1e-6:
            raise ValueError('frame mapping must be a proper rotation')
        for name in ('positions_fixed', 'deltas_self', 'deltas_fixed', 'rotation_increments'):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f'non-finite values in {name}')

    @property
    def endpoint_displacement(self):
        return self.positions_fixed[-1]

    @property
    def path_length_m(self):
        return float(np.linalg.norm(self.deltas_fixed, axis=1).sum())

    def mapped(self, matrix):
        """Express the same physical motion in another local frame.

        For a frame change ``T'_t = T_t C`` (same reference point), positions and
        per-step displacements rotate as ``x' = x C`` (row convention) and the
        rotation increment rotates as ``w' = w C``.  This is a *coordinate*
        change only; it does not fit, mirror or otherwise reshape the motion.
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError('mapping must be 3x3')
        return LocalFrameTrajectory(
            source=self.source, sequence_id=self.sequence_id, side=self.side,
            progress=self.progress.copy(), positions_fixed=self.positions_fixed @ matrix,
            deltas_self=self.deltas_self @ matrix, deltas_fixed=self.deltas_fixed @ matrix,
            rotation_increments=self.rotation_increments @ matrix,
            frame_mapping=self.frame_mapping @ matrix, mapping_declared=self.mapping_declared,
            reference_point=self.reference_point, axis_convention=self.axis_convention,
            metadata=dict(self.metadata, mapped_with=describe_mapping(matrix)),
        )

    def as_dict(self, ndigits=9):
        def rows(a):
            return np.round(a, ndigits).tolist()
        return {
            'source': self.source, 'sequence_id': self.sequence_id, 'side': self.side,
            'feature_version': FEATURE_VERSION,
            'progress': np.round(self.progress, 9).tolist(),
            'positions_fixed': rows(self.positions_fixed),
            'deltas_self': rows(self.deltas_self),
            'deltas_fixed': rows(self.deltas_fixed),
            'rotation_increments': rows(self.rotation_increments),
            'frame_mapping': rows(self.frame_mapping),
            'mapping_name': describe_mapping(self.frame_mapping),
            'mapping_declared': self.mapping_declared,
            'reference_point': self.reference_point,
            'axis_convention': self.axis_convention,
            'metadata': self.metadata,
        }


def build_local_trajectory(sequence, n_points=32, min_valid_ratio=0.8, max_gap_s=0.15,
                           frame_mapping=None, mapping_declared=False,
                           reference_point=None, axis_convention=None, extra=None):
    """Build the local directed representation for one cropped event."""
    grid, p, r, info = resample_poses(sequence, n_points, min_valid_ratio, max_gap_s)
    r0 = r[0]
    positions = (p - p[0]) @ r0
    step = np.diff(p, axis=0)
    deltas_self = np.einsum('ti,tij->tj', step, r[:-1])
    deltas_fixed = step @ r0
    rotation_increments = Rotation.from_matrix(
        np.einsum('tji,tjk->tik', r[:-1], r[1:])).as_rotvec()
    mapping = IDENTITY if frame_mapping is None else np.asarray(frame_mapping, float)
    traj = LocalFrameTrajectory(
        source=sequence.source, sequence_id=sequence.sequence_id,
        side=sequence.metadata.get('effector_side') or sequence.effector_frame,
        progress=(grid - grid[0]) / max(grid[-1] - grid[0], 1e-12),
        positions_fixed=positions @ mapping,
        deltas_self=deltas_self @ mapping,
        deltas_fixed=deltas_fixed @ mapping,
        rotation_increments=rotation_increments @ mapping,
        frame_mapping=mapping.copy(),
        mapping_declared=bool(mapping_declared),
        reference_point=reference_point or sequence.reference_point,
        axis_convention=axis_convention or sequence.effector_frame,
        metadata=dict(info, **{
            'world_frame': sequence.source_frame,
            'effector_frame': sequence.effector_frame,
            'time_source': sequence.metadata.get('timestamp_source'),
            'event_frames_half_open': sequence.metadata.get('original_frames_half_open'),
            'event_span_s': sequence.metadata.get('original_span_s'),
            **(extra or {}),
        }),
    )
    return traj


def _step_rotations(traj):
    """Cumulative rotations R_t relative to R_0, from self-frame increments."""
    steps = Rotation.from_rotvec(traj.rotation_increments).as_matrix()
    out = np.empty((len(steps) + 1, 3, 3))
    out[0] = np.eye(3)
    for i, s in enumerate(steps):
        out[i + 1] = out[i] @ s
    return out


def reconstruct_fixed_from_self(traj):
    """x_t rebuilt from B and the relative rotations, in R_0 coordinates.

    Returns [N,3] with ``out[0] = 0``.  This must equal ``positions_fixed`` to
    numerical tolerance (tested).  It exists so that the difference between A
    and B is explicit and auditable rather than conflated.
    """
    r = _step_rotations(traj)                       # R0^T R_t, shape [N,3,3]
    fixed_steps = np.einsum('tij,tj->ti', r[:-1], traj.deltas_self)
    out = np.zeros((len(traj.progress), 3))
    out[1:] = np.cumsum(fixed_steps, axis=0)
    return out


def naive_cumsum(traj):
    """Deliberately wrong accumulation kept as an executable counterexample."""
    out = np.zeros((len(traj.progress), 3))
    out[1:] = np.cumsum(traj.deltas_self, axis=0)
    return out


def apply_reference_offset(sequence, offset_local):
    """Attach a fixed body-frame offset to the reference point: p' = p + R c.

    Used to test what a different (rigidly attached) reference point does to A
    and B.  A constant offset does *not* cancel once the body rotates.
    """
    from dataclasses import replace
    c = np.asarray(offset_local, float)
    if c.shape != (3,):
        raise ValueError('offset must be a 3-vector')
    r = rotation_matrices(sequence)
    p = sequence.position_m + np.einsum('tij,j->ti', r, c)
    return replace(sequence, position_m=p)


# --------------------------------------------------------------------------- distance
def _check_scale(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite scale')


def _duration_ratio(query, ego):
    a, b = query.metadata.get('duration_s'), ego.metadata.get('duration_s')
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return None
    return float(b / a)


def _weighted_mean(parts, weights):
    used = {k: float(v) for k, v in parts.items() if v is not None}
    total = sum(weights[k] for k in used)
    if total <= 0:
        raise ValueError('no comparable local component')
    return sum(weights[k] * v for k, v in used.items()) / total, used, total


def local_motion_distance(query, ego, config):
    """Directed local-frame distance between two event trajectories.

    Components (all signed / per-axis, never a bare norm of the displacement):

    ``traj_axis``  mean over progress x axes of |x_q - x_e| / position_scale_m
    ``direction``  1 - cos(angle) between the two endpoint displacements; None
                   (missing, not zero) when either displacement is below
                   ``min_displacement_m``
    ``delta_axis`` mean over steps x axes of |dx_q - dx_e| / delta_scale_m, i.e.
                   the per-step motion in the effector's own frame

    Diagnostics (not scored): endpoint magnitude ratio, path-length ratio,
    rotation-increment distance and the raw endpoint vectors.  Weights are
    exploratory and explicitly not calibrated.
    """
    if not isinstance(query, LocalFrameTrajectory) or not isinstance(ego, LocalFrameTrajectory):
        raise TypeError('expected LocalFrameTrajectory')
    if len(query.progress) != len(ego.progress):
        raise ValueError('both trajectories must use the same number of progress points')
    pos_scale = float(config['position_scale_m'])
    delta_scale = float(config['delta_scale_m'])
    min_disp = float(config['min_displacement_m'])
    _check_scale(pos_scale, 'position_scale_m')
    _check_scale(delta_scale, 'delta_scale_m')
    weights = {'traj_axis': float(config['weight_traj_axis']),
               'direction': float(config['weight_direction']),
               'delta_axis': float(config['weight_delta_axis'])}
    if any(v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('local distance weights must be non-negative and not all zero')

    traj_axis = float(np.mean(np.abs(query.positions_fixed - ego.positions_fixed)) / pos_scale)
    delta_axis = float(np.mean(np.abs(query.deltas_self - ego.deltas_self)) / delta_scale)
    dq, de = query.endpoint_displacement, ego.endpoint_displacement
    nq, ne = float(np.linalg.norm(dq)), float(np.linalg.norm(de))
    direction = None
    angle = None
    if nq >= min_disp and ne >= min_disp:
        cos = float(np.clip(np.dot(dq, de) / (nq * ne), -1.0, 1.0))
        direction = 1.0 - cos
        angle = float(np.degrees(np.arccos(cos)))
    parts = {'traj_axis': traj_axis, 'direction': direction, 'delta_axis': delta_axis}
    distance, used, total = _weighted_mean(parts, weights)

    rot_scale = float(config['rotation_scale_rad'])
    _check_scale(rot_scale, 'rotation_scale_rad')
    floor = float(config['path_floor_m'])
    _check_scale(floor, 'path_floor_m')
    diag = {
        'query_endpoint': dq.tolist(), 'ego_endpoint': de.tolist(),
        'query_endpoint_norm_m': nq, 'ego_endpoint_norm_m': ne,
        'direction_angle_deg': angle,
        'endpoint_magnitude_ratio': float(abs(np.log((nq + floor) / (ne + floor)))),
        'path_length_ratio': float(abs(np.log((query.path_length_m + floor) / (ego.path_length_m + floor)))),
        'query_path_m': query.path_length_m, 'ego_path_m': ego.path_length_m,
        'query_duration_s': query.metadata.get('duration_s'),
        'ego_duration_s': ego.metadata.get('duration_s'),
        'duration_ratio': _duration_ratio(query, ego),
        'rotation_increment_distance': float(
            np.mean(np.abs(query.rotation_increments - ego.rotation_increments)) / rot_scale),
        'weighted_parts': {k: weights[k] * v for k, v in used.items()},
        'weights': weights, 'used_weight_total': total,
        'missing_components': [k for k, v in parts.items() if v is None],
        'query_mapping': describe_mapping(query.frame_mapping),
        'ego_mapping': describe_mapping(ego.frame_mapping),
        'query_mapping_declared': query.mapping_declared,
        'ego_mapping_declared': ego.mapping_declared,
    }
    return {'distance': float(distance), 'score': float(1.0 / (1.0 + distance)),
            'parts': parts, 'diagnostics': diag, 'feature_version': FEATURE_VERSION}


def axis_mapping_sensitivity(query, ego, config, group=None):
    """Distance under every proper signed axis permutation of the Ego frame.

    The human wrist frame and the robot end-effector frame are not proven to be
    axis-aligned.  Instead of silently assuming identity, the distance is
    evaluated for all 24 proper signed permutations and the spread is reported.
    """
    group = AXIS_GROUP if group is None else group
    results = {}
    for m in group:
        name = mapping_name(m)
        results[name] = local_motion_distance(query, ego.mapped(m), config)['distance']
    names = list(results)
    values = np.array([results[n] for n in names])
    identity = results.get(mapping_name(IDENTITY))
    order = np.argsort(values, kind='stable')
    return {
        'mapping_count': len(names),
        'identity_mapping': mapping_name(IDENTITY),
        'identity_distance': identity,
        'min_distance': float(values[order[0]]),
        'max_distance': float(values[order[-1]]),
        'best_mapping': names[int(order[0])],
        'worst_mapping': names[int(order[-1])],
        'identity_rank': int(np.flatnonzero(order == names.index(mapping_name(IDENTITY)))[0]) + 1,
        'spread': float(values[order[-1]] - values[order[0]]),
        'distances': results,
    }


def procrustes_shape_only(query, ego, config):
    """DIAGNOSTIC ONLY: distance after one free rotation fitted per pair.

    This is deliberately *not* used for ranking.  It shows how much of the
    apparent mismatch a per-pair rotation fit can remove; such a fit throws away
    the absolute local direction, which is exactly what the user asked to keep.

    The fitted ``m`` satisfies ``query_positions @ m ~= ego_positions``; the ego
    trajectory is then re-expressed as ``ego.mapped(m.T)`` so both sides live in
    comparable coordinates before the same components are scored.
    """
    a, b = query.positions_fixed, ego.positions_fixed
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    m = u @ vt
    if np.linalg.det(m) < 0:
        u[:, -1] *= -1
        m = u @ vt
    result = local_motion_distance(query, ego.mapped(m.T), config)
    angle = float(np.degrees(Rotation.from_matrix(m).magnitude()))
    return {
        'distance': result['distance'], 'score': result['score'],
        'fitted_rotation': np.round(m, 9).tolist(),
        'fitted_rotation_deg': angle,
        'used_for_ranking': False,
        'disclosure': 'fits one rotation per pair; removes absolute local direction; diagnostic only',
    }
