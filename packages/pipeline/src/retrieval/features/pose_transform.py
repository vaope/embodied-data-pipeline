import numpy as np
from scipy.spatial.transform import Rotation

def validate_se3(poses, tolerance=1e-3):
    poses = np.asarray(poses)
    if poses.shape[-2:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError('NaN/Inf or invalid SE(3) shape')
    if not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=tolerance, rtol=0):
        raise ValueError('invalid homogeneous row')
    r = poses[..., :3, :3]
    if not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=tolerance, rtol=0):
        raise ValueError('non-orthogonal rotation')
    if not np.allclose(np.linalg.det(r), 1, atol=tolerance, rtol=0):
        raise ValueError('improper rotation determinant')

def wrist_local_features(wrist, tips, tolerance=1e-3):
    validate_se3(wrist, tolerance)
    validate_se3(tips, tolerance)
    if wrist.ndim != 3 or tips.shape != (len(wrist), 5, 4, 4) or len(wrist) < 2:
        raise ValueError('expected [H+1,4,4] wrist and [H+1,5,4,4] tips')
    r, p = wrist[:, :3, :3], wrist[:, :3, 3]
    rt = np.swapaxes(r, 1, 2)
    delta = np.einsum('tij,tj->ti', rt[:-1], np.diff(p, axis=0))
    rot = Rotation.from_matrix(rt[:-1] @ r[1:]).as_rotvec()
    fingers = np.einsum('tij,tkj->tki', rt, tips[:, :, :3, 3]-p[:, None, :])
    trajectory = (p-p[0]) @ r[0]
    return {'translation_delta': delta, 'rotation_delta': rot,
            'fingertip': fingers[:-1].reshape(len(wrist)-1, 15), 'wrist': trajectory}
