import numpy as np
from scipy.spatial.transform import Rotation


def relative_pose(sequence):
    """Fixed first VALID effector frame. Does NOT align wrist axes to robot axes."""
    valid = sequence.valid_mask
    r = np.full((len(valid), 3, 3), np.nan)
    r[valid] = Rotation.from_quat(sequence.rotation_xyzw[valid]).as_matrix()
    first = np.flatnonzero(valid)[0]
    p = (sequence.position_m - sequence.position_m[first]) @ r[first]
    p[~valid] = np.nan
    return p, np.einsum('ij,tjk->tik', r[first].T, r)
