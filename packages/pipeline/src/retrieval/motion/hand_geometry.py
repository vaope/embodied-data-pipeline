"""Human hand geometry helpers used by the local-frame axis audit.

The EgoDex HDF5 files store a wrist-root pose per hand plus a full ARKit hand
skeleton (metacarpals, phalanges, fingertips).  The metacarpal joints are rigid
in the wrist frame, so they define a *palm frame* independent of finger
articulation:

    u = normalise(middleFingerMetacarpal - wrist)      # distal / palm forward
    v = normalise(indexFingerMetacarpal - littleFingerMetacarpal)  # radial
    n = u x v                                          # palm normal
    R_geo = [u v n]                                    # columns

``C = R_wrist^T R_geo`` is the fixed rotation taking vectors from the raw wrist
frame to that palm frame.  Measuring C on real clips is how this project checks
whether a "self frame" is anatomically meaningful, and whether left and right
hands use the same convention.

None of this makes the human wrist frame equal to the DROID end-effector frame;
it only audits the human side and keeps the cross-embodiment mapping explicit.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _positions(transforms):
    a = np.asarray(transforms, float)
    if a.ndim != 3 or a.shape[1:] != (4, 4):
        raise ValueError('expected [T,4,4] transforms')
    return a[:, :3, 3], a[:, :3, :3]


def palm_frame(wrist, middle_metacarpal, index_metacarpal, little_metacarpal, epsilon=1e-6):
    """Palm frame per frame; returns (positions, rotations, valid, details)."""
    pw, rw = _positions(wrist)
    pm, _ = _positions(middle_metacarpal)
    pi, _ = _positions(index_metacarpal)
    pl, _ = _positions(little_metacarpal)
    u = pm - pw
    v = pi - pl
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    valid = np.isfinite(u).all(axis=1) & np.isfinite(v).all(axis=1) & (nu > epsilon) & (nv > epsilon)
    r = np.full((len(pw), 3, 3), np.nan)
    if valid.any():
        un = u[valid] / nu[valid, None]
        vn = v[valid] / nv[valid, None]
        nn = np.cross(un, vn)
        nn /= np.linalg.norm(nn, axis=1, keepdims=True)
        vn = np.cross(nn, un)  # re-orthogonalise
        r[valid] = np.stack([un, vn, nn], axis=2)
    return pw, r, valid, {'middle_metacarpal_m': nu, 'index_little_span_m': nv}


def wrist_to_palm_rotation(wrist_transforms, middle_metacarpal, index_metacarpal,
                           little_metacarpal):
    """C = R_wrist^T R_geo per frame, plus mean/consistency statistics."""
    _, rw = _positions(wrist_transforms)
    _, rg, valid, _ = palm_frame(wrist_transforms, middle_metacarpal, index_metacarpal,
                                 little_metacarpal)
    if not valid.any():
        raise ValueError('no valid palm frames')
    c = np.full((len(rw), 3, 3), np.nan)
    c[valid] = np.einsum('tji,tjk->tik', rw[valid], rg[valid])
    if not valid.any():
        raise ValueError('no valid palm frames')
    matrices = c[valid]
    mean = Rotation.from_matrix(matrices).mean().as_matrix()
    angle = Rotation.from_matrix(np.einsum('ji,tjk->tik', mean, matrices)).magnitude()
    return {
        'per_frame': c,
        'mean': mean,
        'valid_frames': int(valid.sum()),
        'total_frames': int(len(valid)),
        'max_angle_dev_deg': float(np.degrees(angle.max())),
        'mean_angle_dev_deg': float(np.degrees(angle.mean())),
    }


def palm_center_offset(wrist_transforms, middle_metacarpal, index_metacarpal,
                       little_metacarpal):
    """Offset of the palm centre (mean of wrist + 3 metacarpals) in the wrist frame."""
    pw, rw = _positions(wrist_transforms)
    pm, _ = _positions(middle_metacarpal)
    pi, _ = _positions(index_metacarpal)
    pl, _ = _positions(little_metacarpal)
    center = (pw + pm + pi + pl) / 4.0
    offset = np.einsum('ti,tij->tj', center - pw, rw)  # row form of R_wrist^T (center - wrist)
    finite = np.isfinite(offset).all(axis=1)
    if not finite.any():
        raise ValueError('no valid palm centre offset')
    return {
        'per_frame': offset,
        'mean': offset[finite].mean(axis=0),
        'max_spread_m': float(np.max(np.ptp(offset[finite], axis=0))),
        'valid_frames': int(finite.sum()),
    }


def rotation_between_sides(c_left, c_right, group):
    """Best signed-permutation relation C_right ~ C_left @ D, for documentation.

    Returns the best D, its name, the residual angle and whether the relation is
    proper.  A reflection (det=-1) is never an accepted "alignment".
    """
    c_left = np.asarray(c_left, float)
    c_right = np.asarray(c_right, float)
    best = None
    for d in group:
        # nearest group element to X = C_left^T C_right in the geodesic metric
        residual = Rotation.from_matrix(c_right.T @ c_left @ d).magnitude()
        if best is None or residual < best[0]:
            best = (float(residual), d)
    residual, d = best
    from .local_frame import mapping_name
    return {
        'delta': d.tolist(),
        'delta_name': mapping_name(d),
        'residual_deg': float(np.degrees(residual)),
        'det_delta': float(np.linalg.det(d)),
        'proper': bool(np.linalg.det(d) > 0.5),
    }
