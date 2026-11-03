import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from retrieval.features.pose_transform import wrist_local_features, validate_se3

def test_se3_invariance(poses):
    w, f = poses
    g = np.eye(4)
    g[:3, :3] = Rotation.from_euler('xyz', [1.2, -.8, 2.1]).as_matrix()
    g[:3, 3] = [10, -3, 8]
    a, b = wrist_local_features(w, f), wrist_local_features(g@w, g@f)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-12)

@pytest.mark.parametrize('kind', ['nan', 'inf', 'reflection', 'nonorthogonal', 'homogeneous'])
def test_invalid_se3(poses, kind):
    w = poses[0].copy()
    if kind == 'nan':
        w[2, 0, 3] = np.nan
    elif kind == 'inf':
        w[1, 1, 1] = np.inf
    elif kind == 'reflection':
        w[0, :3, 0] *= -1
    elif kind == 'nonorthogonal':
        w[0, 0, 0] = 8
    else:
        w[0, 3, 0] = 1
    with pytest.raises(ValueError):
        validate_se3(w)
