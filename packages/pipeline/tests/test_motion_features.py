import numpy as np
from retrieval.common import load_config
from retrieval.features.motion_features import extract_window
from retrieval.features.pose_transform import wrist_local_features
from retrieval.search.motion_ranking import motion_distances

def features(w, f):
    result = wrist_local_features(w, f)
    result['translation'] = np.linalg.norm(result['translation_delta'], axis=1)*30
    result['rotation'] = np.linalg.norm(result['rotation_delta'], axis=1)*30
    return {'right_'+k: v for k, v in result.items()}

def distance(a, b):
    return motion_distances(a, b, {'left': False, 'right': True}, {'left': False, 'right': True})

def test_identity_shapes(poses):
    a = features(*poses)
    assert a['right_translation_delta'].shape == (30, 3)
    assert a['right_rotation_delta'].shape == (30, 3)
    assert a['right_fingertip'].shape == (30, 15)
    assert a['right_wrist'].shape == (31, 3)
    assert all(v == 0 for v in distance(a, a).values())

def test_hand_shape_changes_only_fingers(poses):
    w, f = poses
    changed = f.copy()
    changed[:, :, :3, 3] += np.einsum('tij,j->ti', w[:, :3, :3], [.03, 0, 0])[:, None, :]
    d = distance(features(w, f), features(w, changed))
    assert d['wrist'] == d['translation'] == d['rotation'] == 0
    assert d['fingertip'] > .01

def test_motion_changes_preserve_finger_pose(poses):
    w, f = poses
    w2, f2 = w.copy(), f.copy()
    offset = np.arange(31)[:, None]*np.array([.01, .02, .03])
    w2[:, :3, 3] += offset
    f2[:, :, :3, 3] += offset[:, None]
    d = distance(features(w, f), features(w2, f2))
    assert d['fingertip'] < 1e-14
    assert d['wrist'] > .1 and d['translation'] > .1

def test_missing_hand_confidence_and_invalid_pose(poses):
    w, f = poses
    cfg = load_config()
    hands = {'right': {'wrist': w, 'tips': f, 'confidence': {}}}
    out, flags, issues = extract_window(hands, 0, 30, cfg)
    assert out is not None and flags == {'left': False, 'right': True}
    hands['right']['confidence']['rightHand'] = np.zeros(31)
    assert extract_window(hands, 0, 30, cfg)[0] is None
    hands['right']['confidence'] = {}
    hands['right']['wrist'] = w.copy()
    hands['right']['wrist'][5, 0, 3] = np.nan
    assert extract_window(hands, 0, 30, cfg)[0] is None

def test_direction_blind_spot_is_documented_by_behavior():
    w = np.tile(np.eye(4),(31,1,1))
    w[:,0,3] = np.arange(31)*.01
    tips = np.tile(np.eye(4),(31,5,1,1))
    tips[:,:,0,3] = w[:,None,0,3]
    reverse, rtips = w.copy(), tips.copy()
    reverse[:,0,3] *= -1
    rtips[:,:,0,3] *= -1
    d = distance(features(w,tips),features(reverse,rtips))
    assert d['translation'] < 1e-14 and d['wrist'] > .1

def test_speed_change_increases_pointwise_distance(poses):
    w,f = poses
    w2,f2 = w.copy(),f.copy()
    offset = w[:,:3,3].copy()
    w2[:,:3,3] += offset
    f2[:,:,:3,3] += offset[:,None]
    d = distance(features(w,f),features(w2,f2))
    assert d['translation'] > 0 and d['wrist'] > 0 and d['fingertip'] < 1e-14
