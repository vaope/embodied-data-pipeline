"""Tests for the EgoDex hand -> gripper-like grasp proxy (ego-grasp-proxy-v1)."""
import h5py
import numpy as np
import pytest

from retrieval.motion.hand_grasp import (GRASP_VERSION, grasp_from_positions,
                                         grasp_series_at_progress, load_grasp_config)
from retrieval.adapters.egodex_wrist_motion import EgoWristAdapter


def joint_positions(ratio, n=31, palm_width=0.04, closed_axis=1):
    """Build [T,3] positions with a controlled thumb-index distance."""
    thumb = np.zeros((n, 3))
    index_tip = np.zeros((n, 3))
    index_tip[:, closed_axis] = ratio * palm_width
    meta = np.zeros((n, 3))
    meta[:, 0] = -palm_width / 2
    little = np.zeros((n, 3))
    little[:, 0] = palm_width / 2
    return {'ThumbTip': thumb, 'IndexFingerTip': index_tip,
            'IndexFingerMetacarpal': meta, 'LittleFingerMetacarpal': little}


def test_open_and_closed_map_to_opposite_ends_and_are_monotone():
    cfg = load_grasp_config()
    open_hand = grasp_from_positions(joint_positions(2.1), cfg)
    closed_hand = grasp_from_positions(joint_positions(0.3), cfg)
    assert open_hand['status'] == closed_hand['status'] == 'ok'
    assert np.nanmean(open_hand['signal']) < 0.1
    assert np.nanmean(closed_hand['signal']) > 0.9
    assert np.all((open_hand['signal'] >= 0) & (open_hand['signal'] <= 1))

    ratios = np.linspace(2.1, 0.3, 7)
    means = [np.nanmean(grasp_from_positions(joint_positions(r), cfg)['signal']) for r in ratios]
    assert all(b >= a - 1e-9 for a, b in zip(means, means[1:]))


def test_invalid_frames_are_marked_not_filled():
    cfg = load_grasp_config()
    joints = joint_positions(1.0)
    joints['ThumbTip'][5] = np.nan
    result = grasp_from_positions(joints, cfg)
    assert result['status'] == 'ok'
    assert not result['valid_mask'][5] and np.isnan(result['signal'][5])

    bad_width = joint_positions(1.0)
    bad_width['LittleFingerMetacarpal'][:, 0] = 1.0  # palm width 1.02 m
    result = grasp_from_positions(bad_width, cfg)
    assert result['status'] == 'no_valid_frames'

    outlier = joint_positions(1.0)
    outlier['IndexFingerTip'][:, 1] = 0.04 * 10.0  # ratio 10 > max
    assert grasp_from_positions(outlier, cfg)['status'] == 'no_valid_frames'

    conf = np.ones(31)
    conf[:10] = 0.0
    result = grasp_from_positions(joint_positions(1.0), cfg, confidence=conf)
    assert result['status'] == 'ok'
    assert not result['valid_mask'][:10].any() and result['valid_mask'][10:].all()


def test_median_smoothing_keeps_transition_and_suppresses_single_frame_spike():
    cfg = load_grasp_config()
    joints = joint_positions(0.4)
    joints['IndexFingerTip'][10, 1] = 0.04 * 2.2  # one-frame open spike
    result = grasp_from_positions(joints, cfg)
    assert result['signal'][10] > 0.8  # spike suppressed by the median filter
    assert result['signal'][10] > result['signal'][9] - 1e-9

    half = joint_positions(0.4)
    half['IndexFingerTip'][15:, 1] = 0.04 * 2.2
    result = grasp_from_positions(half, cfg)
    assert result['signal'][5] > 0.8 and result['signal'][-5] < 0.1


def test_series_sampling_marks_invalid_and_keeps_ends():
    cfg = load_grasp_config()
    joints = joint_positions(0.4)
    joints['ThumbTip'][0] = np.nan
    result = grasp_from_positions(joints, cfg)
    progress = np.linspace(0, 1, 17)
    values, valid = grasp_series_at_progress(result['signal'], result['valid_mask'], progress, 0, 31)
    assert len(values) == 17
    assert np.isnan(values[0]) and not valid[0]
    assert valid[-1]


def _write_h5(path, joints, n=31, with_confidence=True, include_fingers=True):
    with h5py.File(path, 'w') as h:
        for side in ('left', 'right'):
            m = np.tile(np.eye(4), (n, 1, 1))
            m[:, 0, 3] = np.linspace(0, 0.05, n)
            h[f'transforms/{side}Hand'] = m
            if include_fingers:
                for name, arr in joints.items():
                    m2 = np.tile(np.eye(4), (n, 1, 1))
                    m2[:, :3, 3] = arr
                    h[f'transforms/{side}{name}'] = m2
            if with_confidence:
                h[f'confidences/{side}Hand'] = np.ones(n)


def test_adapter_returns_grasp_signal_and_survives_missing_joints(tmp_path):
    row = {'clip_id': 'x', 'start_frame': 0, 'end_frame': 30, 'fps': 30,
           'start_time': 0.0, 'end_time': 1.0}
    path = tmp_path / 'with_fingers.h5'
    joints = joint_positions(0.35)
    _write_h5(path, joints)
    hands, issues = EgoWristAdapter().read({**row, 'hdf5_path': str(path)})
    assert not issues
    for side in ('left', 'right'):
        seq = hands[side]
        assert seq.metadata['grasp']['status'] == 'ok'
        assert seq.grasp_signal is not None and seq.grasp_signal.shape == (31,)
        assert seq.grasp_valid_mask.all()
        assert np.nanmean(seq.grasp_signal) > 0.9  # closed hand maps to ~1
        assert seq.metadata['grasp']['calibrated_to_droid'] is False

    no_fingers = tmp_path / 'wrist_only.h5'
    _write_h5(no_fingers, {}, include_fingers=False)
    hands, issues = EgoWristAdapter().read({**row, 'hdf5_path': str(no_fingers)})
    assert not issues and set(hands) == {'left', 'right'}
    assert hands['left'].grasp_signal is None
    assert hands['left'].metadata['grasp']['status'] == 'joints_missing'

    disabled = EgoWristAdapter().read({**row, 'hdf5_path': str(path)}, grasp=None)
    assert disabled[0]['left'].grasp_signal is None
    assert disabled[0]['left'].metadata['grasp']['status'] == 'disabled'
