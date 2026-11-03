from dataclasses import replace
import csv
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from retrieval.motion.representation import MotionSequence, matrix_poses
from retrieval.motion.relative_pose import relative_pose
from retrieval.motion.temporal_align import resample, sequence_distance
from retrieval.motion.features import extract_features, feature_distances
from retrieval.adapters.droid_motion import DROIDAdapter
from retrieval.adapters.egodex_wrist_motion import EgoWristAdapter
from retrieval.rankers.gripper_ego_ranker import load_config, GripperEgoMotionRanker


def motion(p=None, r=None):
    t = np.arange(31)/30
    p = np.c_[t, t*t, np.zeros(31)] if p is None else p
    r = Rotation.from_euler('z', t[:, None]) if r is None else r
    return MotionSequence('egodex', 'test', t, p, r.as_quat(), np.ones(31, bool), 'world', 'right_hand', 'wrist_root')


def test_left_multiplication_invariance():
    a = motion()
    g = Rotation.random(random_state=19)
    b = replace(a, position_m=g.apply(a.position_m)+[4, -2, 1], rotation_xyzw=(g*Rotation.from_quat(a.rotation_xyzw)).as_quat())
    for x, y in zip(relative_pose(a), relative_pose(b)):
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_geodesic_wrap_slerp_translation_spin():
    cfg = load_config()
    a = motion(r=Rotation.from_euler('z', np.linspace(3.1, 3.2, 31)[:, None]))
    sample = resample(a, 0, 1, cfg)
    f = extract_features(sample, cfg)
    np.testing.assert_allclose(f['omega'], .1, atol=1e-12)
    r = Rotation.from_quat(sample.rotation_xyzw).as_matrix()
    np.testing.assert_allclose(r.transpose(0, 2, 1)@r, np.broadcast_to(np.eye(3), r.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(r), 1)
    pure_t = motion(r=Rotation.identity(31))
    np.testing.assert_allclose(extract_features(pure_t, cfg)['omega'], 0)
    pure_r = motion(p=np.zeros((31, 3)))
    assert extract_features(pure_r, cfg)['total_path_length_m'] == 0


@pytest.mark.parametrize('kind', ['duplicate', 'nan', 'short', 'low', 'gap'])
def test_bad_data_rejected(kind):
    cfg, a = load_config(), motion()
    with pytest.raises(ValueError):
        if kind == 'duplicate':
            replace(a, timestamps_s=np.zeros(31))
        elif kind == 'nan':
            replace(a, position_m=np.full((31, 3), np.nan))
        elif kind == 'short':
            resample(a, 0, .1, cfg)
        elif kind == 'low':
            valid = np.arange(31) % 2 == 0
            resample(replace(a, valid_mask=valid), 0, 1, cfg)
        else:
            valid = np.ones(31, bool); valid[10:15] = False
            resample(replace(a, valid_mask=valid), 0, 1, cfg)


def test_short_missing_frame_allowed_with_metadata():
    a, cfg = motion(), load_config()
    valid = a.valid_mask.copy(); valid[10] = False
    b = resample(replace(a, valid_mask=valid), 0, 1, cfg)
    assert b.metadata['source_valid_ratio'] < 1
    assert np.isfinite(b.position_m).all()


def test_reverse_ambiguity_and_frame_rotation_invariance():
    cfg = load_config()
    t = np.arange(31)/30
    a = motion(p=np.c_[t, t*0, t*0], r=Rotation.identity(31))
    b = replace(a, position_m=a.position_m[::-1].copy())
    assert feature_distances(extract_features(a, cfg), extract_features(b, cfg), cfg)[0] < 1e-12
    g = Rotation.from_euler('xyz', [.4, .2, 1.])
    c = replace(a, position_m=g.apply(a.position_m), rotation_xyzw=(g*Rotation.from_quat(a.rotation_xyzw)).as_quat())
    assert feature_distances(extract_features(a, cfg), extract_features(c, cfg), cfg)[0] < 1e-12
    assert np.isnan(extract_features(motion(p=np.zeros((31, 3))), cfg)['turn']).all()


def test_egodex_crop_missing_confidence_hand_and_bad_rotation(tmp_path):
    path = tmp_path/'pose.h5'
    matrices = np.tile(np.eye(4), (61, 1, 1)); matrices[:, 0, 3] = np.arange(61)*.01
    with h5py.File(path, 'w') as h:
        h['transforms/rightHand'] = matrices
    row = {'clip_id': 'x', 'hdf5_path': str(path), 'start_frame': 30, 'end_frame': 60, 'fps': 30, 'start_time': 1., 'end_time': 2.}
    hands, issues = EgoWristAdapter().read(row)
    assert hands['right'].position_m[0, 0] == .3 and 'left' in issues
    assert hands['right'].metadata['confidence_missing']
    matrices[:, 0, 0] = -1
    assert not matrix_poses(matrices)[2].any()
    with h5py.File(path, 'a') as h:
        h['confidences/rightHand'] = np.zeros(61)
    hands, issues = EgoWristAdapter().read(row)
    assert not hands and 'right' in issues


def test_observation_required_and_preferred(tmp_path):
    p = tmp_path/'steps.csv'
    fields = ['step']+[f'{kind}_cartesian_position_{i}' for kind in ('action', 'observation') for i in range(6)]
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for t in range(31):
            w.writerow(dict(step=t, **{f'{kind}_cartesian_position_{i}': (99 if kind == 'action' else t*.01 if i == 0 else 0) for kind in ('action', 'observation') for i in range(6)}))
    a = DROIDAdapter().read(p)
    assert a.position_m[-1, 0] == .3 and a.metadata['pose_origin'] == 'observation'
    assert not a.grasp_valid_mask.any()
    p.write_text('step,action_cartesian_position_0\n0,0\n')
    with pytest.raises(ValueError, match='commanded_only'):
        DROIDAdapter().read(p)


def test_dtw_band_and_reproducible_ranker():
    cfg = load_config({'use_dtw': True})
    a = np.r_[np.zeros(5), np.ones(5), np.zeros(5)]
    assert sequence_distance(a, a, cfg) == 0
    b = np.r_[np.zeros(6), np.ones(5), np.zeros(4)]
    assert sequence_distance(a, b, cfg) <= sequence_distance(a, b, load_config())
    ranker = GripperEgoMotionRanker()
    rows = [{'clip_id': 'ok', 'rank': 1, 'start_time': 0, 'end_time': 1}, {'clip_id': 'bad', 'rank': 2, 'start_time': 0, 'end_time': 1}]
    motions = {'ok': ({'right': motion()}, {}), 'bad': ({}, {'left': 'missing'})}
    x = ranker.rank(motion(), rows, motions)
    assert x == ranker.rank(motion(), rows, motions)
    assert x[0]['matched_ego_hand'] == 'right' and x[0]['matched_droid_span_s'] == [0., 1.]
    assert x[1]['fallback_reason'] and x[1]['translation_rank'] is None


def test_legacy_source_files_unchanged():
    root = Path(__file__).resolve().parents[2]
    frozen = root/'experiments/droid_ego_adapter/run_20261025/prechange_hashes.json'
    if not frozen.exists():
        pytest.skip('local prechange evidence not distributed')
    import hashlib
    hashes = json.loads(frozen.read_text(encoding='utf-8'))
    # Includes old four-route, pose-language, V2 baseline and evaluator sources.
    for path, digest in hashes.items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest, path


def test_lever_arm_effect_is_quantified():
    # Same rotating rigid body, different observed origins: not equivalent paths.
    a = motion(p=np.zeros((31, 3)), r=Rotation.from_euler('z', np.linspace(0, np.pi/2, 31)[:, None]))
    offset = Rotation.from_quat(a.rotation_xyzw).apply(np.tile([.1, 0, 0], (31, 1)))
    b = replace(a, position_m=offset)
    cfg = load_config()
    assert extract_features(a, cfg)['total_path_length_m'] == 0
    assert extract_features(b, cfg)['net_displacement_m'] == pytest.approx(np.sqrt(2)*.1)
    assert extract_features(b, cfg)['total_path_length_m'] == pytest.approx(np.pi*.1/2, rel=.001)


def test_real_local_droid_route_and_pool_repeatability():
    root = Path(__file__).resolve().parents[2]
    frozen = root/'experiments/droid_ego_adapter/run_20261025/candidates.json'
    if not frozen.is_file():
        pytest.skip('real local data not distributed')
    from retrieval.search.droid_to_ego import retrieve_droid_to_ego
    q = json.loads(frozen.read_text(encoding='utf-8'))['queries'][0]
    candidates = q['candidates'][:2]
    a = retrieve_droid_to_ego(q['query_id'], candidates=candidates)
    b = retrieve_droid_to_ego(q['query_id'], candidates=candidates)
    assert a['results'] == b['results']
    assert a['query_motion_metadata']['pose_origin'] == 'observation'
    assert a['stats']['fallback_count'] == 0
    assert all(r['matched_droid_span_s'] is not None and r['hand_scores'] for r in a['results'])
