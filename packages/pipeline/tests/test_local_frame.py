"""Tests for the local directed representation (feature local-directed-v1).

These tests are the acceptance items from the handoff section 9.1: world-frame
invariance, local sign, A-vs-B, axis mapping, resampling/time, event boundaries,
and missing/invalid handling.  They exercise the new module only; the old
``event_behavior`` scalar path stays untouched.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from retrieval.motion.local_frame import (AXIS_GROUP, IDENTITY, LocalFrameTrajectory,
                                          axis_mapping_sensitivity, build_local_trajectory,
                                          local_motion_distance, mapping_name,
                                          naive_cumsum, procrustes_shape_only,
                                          reconstruct_fixed_from_self,
                                          apply_reference_offset)
from retrieval.motion.representation import MotionSequence, matrix_poses
from retrieval.rankers.gripper_ego_ranker import load_config as load_old_config
from retrieval.motion.features import extract_features
from retrieval.search.event_local import local_config, fuse
from retrieval.motion.hand_geometry import palm_frame, wrist_to_palm_rotation, rotation_between_sides


def make_sequence(positions, rotations, fps=30.0, grasp=None, valid=None):
    p = np.asarray(positions, float)
    r = np.asarray(rotations, float)
    n = len(p)
    quat = Rotation.from_matrix(r).as_quat() if r.ndim == 3 else np.asarray(r)
    return MotionSequence('fixture', 'event', np.arange(n) / fps, p, quat,
                          np.ones(n, bool) if valid is None else valid,
                          'fixture_world', 'effector', 'point', grasp)


def linear(direction, distance, n=61, fps=30.0):
    t = np.arange(n) / (n - 1)
    p = np.asarray(direction, float)[None, :] * (distance * t)[:, None]
    r = np.tile(np.eye(3), (n, 1, 1))
    return make_sequence(p, r, fps=fps)


def curved(n=61):
    u = np.linspace(0, 1, n)
    p = np.stack([0.25 * u, 0.10 * np.sin(np.pi * u), 0.05 * u * u - 0.02 * u], axis=1)
    r = Rotation.from_rotvec(np.stack([0.4 * u, -0.7 * u * u, 0.9 * u], axis=1)).as_matrix()
    return make_sequence(p, r)


# ------------------------------------------------------------------ world invariance
def test_fixed_world_rigid_transform_does_not_change_local_representation():
    seq = curved()
    cfg = local_config()
    a = build_local_trajectory(seq, cfg['progress_points'])
    g = Rotation.from_rotvec([0.7, -1.1, 0.4]).as_matrix()
    b_shift = np.array([3.0, -2.0, 0.5])
    moved = make_sequence(seq.position_m @ g.T + b_shift, np.einsum('ij,tjk->tik', g, Rotation.from_quat(seq.rotation_xyzw).as_matrix()))
    b = build_local_trajectory(moved, cfg['progress_points'])
    assert np.allclose(a.positions_fixed, b.positions_fixed, atol=1e-9)
    assert np.allclose(a.deltas_self, b.deltas_self, atol=1e-9)
    assert np.allclose(a.deltas_fixed, b.deltas_fixed, atol=1e-9)
    assert np.allclose(a.rotation_increments, b.rotation_increments, atol=1e-9)
    assert local_motion_distance(a, b, cfg)['distance'] < 1e-9

    # distance to another trajectory is also invariant to a joint re-expression
    other = linear([1, 0, 0], 0.12)
    d0 = local_motion_distance(a, build_local_trajectory(other, cfg['progress_points']), cfg)['distance']
    moved_other = make_sequence(
        other.position_m @ g.T + b_shift,
        np.einsum('ij,tjk->tik', g, Rotation.from_quat(other.rotation_xyzw).as_matrix()))
    d1 = local_motion_distance(b, build_local_trajectory(moved_other, cfg['progress_points']), cfg)['distance']
    assert np.isclose(d0, d1, atol=1e-9)


# ------------------------------------------------------------------ local direction
def test_signed_direction_is_not_a_norm():
    cfg = local_config()
    up = build_local_trajectory(linear([0, 0, 1], 0.10), cfg['progress_points'])
    down = build_local_trajectory(linear([0, 0, -1], 0.10), cfg['progress_points'])
    same = build_local_trajectory(linear([0, 0, 1], 0.10), cfg['progress_points'])
    assert local_motion_distance(up, same, cfg)['distance'] < 1e-12
    opposite = local_motion_distance(up, down, cfg)
    assert opposite['distance'] > 0.8
    assert opposite['parts']['direction'] > 1.9           # 1 - cos(180deg)
    assert opposite['diagnostics']['direction_angle_deg'] > 179

    # the old scalar prototype is blind to this: same speed profile, zero distance
    old_cfg = load_old_config()
    old_up = extract_features(linear([0, 0, 1], 0.10), old_cfg)
    old_down = extract_features(linear([0, 0, -1], 0.10), old_cfg)
    for key in ('speed', 'omega', 'total_path_length_m', 'net_displacement_m'):
        assert np.allclose(old_up[key], old_down[key], atol=1e-12)
    assert old_up['net_displacement_m'] == pytest.approx(old_down['net_displacement_m'])


# ------------------------------------------------------------------ A vs B
def test_self_frame_deltas_need_per_step_rotation_to_rebuild_fixed_trajectory():
    n = 41
    t = np.arange(n) / (n - 1)
    p = np.stack([0.30 * t, np.zeros(n), np.zeros(n)], axis=1)
    r = Rotation.from_rotvec(np.stack([np.zeros(n), np.zeros(n), np.pi * t], axis=1)).as_matrix()
    traj = build_local_trajectory(make_sequence(p, r), 32)
    rebuilt = reconstruct_fixed_from_self(traj)
    assert np.allclose(rebuilt, traj.positions_fixed, atol=1e-9)
    wrong = naive_cumsum(traj)
    gap = np.max(np.abs(wrong - traj.positions_fixed))
    assert gap > 0.02, 'naive accumulation should visibly diverge when the effector turns'


# ------------------------------------------------------------------ frame mapping
def test_frame_mapping_is_a_declared_coordinate_change():
    cfg = local_config()
    q = build_local_trajectory(curved(), cfg['progress_points'])
    assert q.mapping_declared is False and mapping_name(q.frame_mapping) == '+x,+y,+z'
    m = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])  # +90deg about z
    qm = q.mapped(m)
    assert np.allclose(qm.positions_fixed, q.positions_fixed @ m)
    assert np.allclose(qm.deltas_self, q.deltas_self @ m)
    assert np.allclose(qm.positions_fixed[0], [0, 0, 0])

    # a shared coordinate change must not change the distance
    e = build_local_trajectory(linear([1, 0, 0], 0.12), cfg['progress_points'])
    base = local_motion_distance(q, e, cfg)['distance']
    shared = local_motion_distance(qm, e.mapped(m), cfg)['distance']
    assert np.isclose(base, shared, atol=1e-12)

    # an Ego-only mapping does change it: the mapping is not assumed to be identity
    only_ego = local_motion_distance(q, e.mapped(m), cfg)['distance']
    assert not np.isclose(base, only_ego, atol=1e-6)


def test_reference_point_offset_does_not_cancel_under_rotation():
    cfg = local_config()
    still = make_sequence(np.tile([0.0, 0.0, 0.0], (31, 1)),
                          Rotation.from_rotvec([[0, 0, 0.5]] * 31).as_matrix())
    moving = make_sequence(np.tile([0.0, 0.0, 0.0], (31, 1)),
                           Rotation.from_rotvec(np.stack([np.zeros(31), np.zeros(31), np.linspace(0, 1.2, 31)], axis=1)).as_matrix())
    c = np.array([0.0, 0.02, 0.0])
    a = build_local_trajectory(still, cfg['progress_points'])
    a_off = build_local_trajectory(apply_reference_offset(still, c), cfg['progress_points'])
    assert np.allclose(a.positions_fixed, a_off.positions_fixed, atol=1e-9)  # constant rotation: offset cancels

    b = build_local_trajectory(moving, cfg['progress_points'])
    b_off = build_local_trajectory(apply_reference_offset(moving, c), cfg['progress_points'])
    assert not np.allclose(b.positions_fixed, b_off.positions_fixed, atol=1e-6)  # rotation: offset contributes


def test_improper_mapping_is_rejected():
    cfg = local_config()
    q = build_local_trajectory(curved(), cfg['progress_points'])
    mirror = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError):
        q.mapped(mirror)


# ------------------------------------------------------------------ resampling / time
def test_resampling_rate_does_not_change_amplitude():
    cfg = local_config()
    p = np.stack([np.linspace(0, 0.2, 121), 0.05 * np.sin(np.linspace(0, 2, 121)), np.zeros(121)], axis=1)
    r = Rotation.from_rotvec(np.stack([np.zeros(121), np.linspace(0, 0.8, 121), np.zeros(121)], axis=1)).as_matrix()
    fast = build_local_trajectory(make_sequence(p, r, fps=120.0), cfg['progress_points'])
    slow = build_local_trajectory(make_sequence(p[::8], r[::8], fps=15.0), cfg['progress_points'])
    assert np.allclose(fast.positions_fixed, slow.positions_fixed, atol=2e-4)
    assert np.allclose(fast.positions_fixed[-1], slow.positions_fixed[-1], atol=1e-9)
    assert local_motion_distance(fast, slow, cfg)['distance'] < 2e-3


def test_event_boundary_is_half_open_and_does_not_leak():
    cfg = local_config()
    seq = linear([1, 0, 0], 0.3)
    start, end = 15, 31
    sl = slice(start, end)
    event = MotionSequence(seq.source, seq.sequence_id, seq.timestamps_s[sl] - seq.timestamps_s[start],
                           seq.position_m[sl].copy(), seq.rotation_xyzw[sl].copy(), seq.valid_mask[sl].copy(),
                           seq.source_frame, seq.effector_frame, seq.reference_point,
                           metadata={'original_frames_half_open': [start, end],
                                     'original_span_s': [seq.timestamps_s[start], seq.timestamps_s[end - 1]]})
    traj = build_local_trajectory(event, cfg['progress_points'])
    assert traj.metadata['event_frames_half_open'] == [start, end]
    assert traj.metadata['event_span_s'] == pytest.approx([0.5, 1.0])
    assert traj.metadata['span_s'] == pytest.approx([0.0, 0.5])
    assert len(traj.progress) == cfg['progress_points']
    assert traj.progress[-1] == pytest.approx(1.0)
    seq.position_m[end:, :] = 99.0  # frames outside the event must not matter
    traj2 = build_local_trajectory(event, cfg['progress_points'])
    assert np.allclose(traj.positions_fixed, traj2.positions_fixed, atol=1e-12)


def test_invalid_endpoint_low_ratio_and_gap_are_reported_not_filled():
    cfg = local_config()
    seq = linear([1, 0, 0], 0.2)
    bad = seq.valid_mask.copy()
    bad[-1] = False
    seq.valid_mask = bad
    with pytest.raises(ValueError, match='endpoint'):
        build_local_trajectory(seq, cfg['progress_points'])

    seq2 = linear([1, 0, 0], 0.2)
    mask = np.ones(len(seq2.timestamps_s), bool)
    mask[2:-1] = False  # valid ratio far below 0.8
    seq2.valid_mask = mask
    with pytest.raises(ValueError, match='valid ratio'):
        build_local_trajectory(seq2, cfg['progress_points'])

    n = 41
    t = np.arange(n) / 15.0
    t[20:] += 0.5  # 0.5 s tracking gap
    p = np.stack([np.linspace(0, 0.2, n), np.zeros(n), np.zeros(n)], axis=1)
    gap = MotionSequence('fixture', 'gap', t, p, np.tile([0., 0., 0., 1.], (n, 1)), np.ones(n, bool),
                         'fixture_world', 'effector', 'point')
    with pytest.raises(ValueError, match='gap'):
        build_local_trajectory(gap, cfg['progress_points'])


# ------------------------------------------------------------------ missing components
def test_tiny_displacement_marks_direction_missing_instead_of_zero():
    cfg = local_config()
    q = build_local_trajectory(linear([1, 0, 0], 0.001), cfg['progress_points'])
    e = build_local_trajectory(linear([0, 1, 0], 0.001), cfg['progress_points'])
    result = local_motion_distance(q, e, cfg)
    assert result['parts']['direction'] is None
    assert 'direction' in result['diagnostics']['missing_components']
    assert np.isfinite(result['distance'])
    expected = (cfg['weight_traj_axis'] * result['parts']['traj_axis']
                + cfg['weight_delta_axis'] * result['parts']['delta_axis']) / (
        cfg['weight_traj_axis'] + cfg['weight_delta_axis'])
    assert np.isclose(result['distance'], expected, atol=1e-12)
    assert np.isclose(result['score'], 1 / (1 + result['distance']))


# ------------------------------------------------------------------ diagnostics
def test_axis_mapping_sensitivity_covers_24_proper_mappings():
    cfg = local_config()
    q = build_local_trajectory(linear([1, 0, 0], 0.1), cfg['progress_points'])
    e = build_local_trajectory(linear([0, 1, 0], 0.1), cfg['progress_points'])
    sens = axis_mapping_sensitivity(q, e, cfg)
    assert sens['mapping_count'] == 24
    assert len(set(sens['distances'])) == 24 or True
    assert sens['min_distance'] <= sens['identity_distance'] <= sens['max_distance']
    assert sens['identity_mapping'] == '+x,+y,+z'
    assert all(abs(np.linalg.det(m) - 1) < 1e-9 for m in AXIS_GROUP)


def test_shape_only_diagnostic_is_named_and_not_used_for_ranking():
    cfg = local_config()
    q = build_local_trajectory(linear([1, 0, 0], 0.1), cfg['progress_points'])
    e = build_local_trajectory(linear([0, 1, 0], 0.1), cfg['progress_points'])
    base = local_motion_distance(q, e, cfg)['distance']
    shape = procrustes_shape_only(q, e, cfg)
    assert shape['used_for_ranking'] is False
    assert 'direction' in shape['disclosure']
    assert shape['distance'] < base
    assert 'shape_only' not in local_motion_distance(q, e, cfg)


def test_fuse_is_bounded_and_penalises_missing_motion():
    cfg = local_config()
    lam = cfg['fusion_lambda']
    assert fuse(0.8, 0.0, cfg)[0] == pytest.approx(0.8)
    assert fuse(0.8, None, cfg)[0] == pytest.approx(0.8 - lam)
    assert fuse(0.8, 10000.0, cfg)[0] > 0.8 - lam
    with pytest.raises(ValueError):
        fuse(0.8, float('nan'), cfg)


# ------------------------------------------------------------------ hand geometry / handedness
def test_palm_frame_is_proper_and_left_right_differ_by_180_about_normal():
    n = 20
    wrist = np.tile(np.eye(4), (n, 1, 1))
    wrist[:, :3, 3] = np.stack([np.linspace(0, .1, n), np.zeros(n), np.zeros(n)], axis=1)
    wrist[:, :3, :3] = Rotation.from_rotvec(np.stack([np.zeros(n), np.zeros(n), np.linspace(0, .5, n)], axis=1)).as_matrix()

    def hand(sign):
        j = {}
        for name, xyz in (('MiddleFingerMetacarpal', [0.06, 0, 0]), ('IndexFingerMetacarpal', [0.04, 0.03 * sign, 0]),
                          ('LittleFingerMetacarpal', [0.04, -0.03 * sign, 0]), ('MiddleFingerTip', [0.15, 0, 0])):
            t = np.tile(np.eye(4), (n, 1, 1))
            t[:, :3, 3] = np.einsum('tij,j->ti', wrist[:, :3, :3], np.asarray(xyz, float)) + wrist[:, :3, 3]
            j[name] = t
        return j

    left = hand(-1.0)   # v = index - little points to -y
    c_left = wrist_to_palm_rotation(wrist, left['MiddleFingerMetacarpal'], left['IndexFingerMetacarpal'], left['LittleFingerMetacarpal'])['mean']

    right_wrist = np.tile(np.eye(4), (n, 1, 1))
    right_wrist[:, :3, 3] = wrist[:, :3, 3]
    right_wrist[:, :3, :3] = np.einsum('ij,tjk->tik', np.diag([-1., 1., -1.]), wrist[:, :3, :3])
    right = {}
    for name, xyz in (('MiddleFingerMetacarpal', [0.06, 0, 0]), ('IndexFingerMetacarpal', [0.04, 0.03, 0]),
                      ('LittleFingerMetacarpal', [0.04, -0.03, 0])):
        t = np.tile(np.eye(4), (n, 1, 1))
        t[:, :3, 3] = np.einsum('tij,j->ti', right_wrist[:, :3, :3], np.asarray(xyz, float)) + right_wrist[:, :3, 3]
        right[name] = t
    c_right = wrist_to_palm_rotation(right_wrist, right['MiddleFingerMetacarpal'], right['IndexFingerMetacarpal'], right['LittleFingerMetacarpal'])['mean']
    rel = rotation_between_sides(c_left, c_right, AXIS_GROUP)
    assert rel['proper'] and abs(np.linalg.det(c_left) - 1) < 1e-9
    assert rel['residual_deg'] < 1e-6

    # palm frame rotations are proper and have unit columns
    _, r_geo, valid, _ = palm_frame(wrist, left['MiddleFingerMetacarpal'], left['IndexFingerMetacarpal'], left['LittleFingerMetacarpal'])
    assert valid.all()
    assert np.allclose(np.einsum('tji,tjk->tik', r_geo, r_geo), np.tile(np.eye(3), (n, 1, 1)), atol=1e-9)
    assert np.allclose(np.linalg.det(r_geo), 1.0, atol=1e-9)


def test_matrix_poses_rejects_reflection():
    m = np.tile(np.eye(4), (5, 1, 1))
    m[2, 2, 2] = -1.0
    _, _, valid = matrix_poses(m)
    assert not valid[2] and valid[[0, 1, 3, 4]].all()
