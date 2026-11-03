import numpy as np
import pytest
from scipy.spatial.transform import Rotation

@pytest.fixture
def poses():
    t = np.arange(31)/30
    wrist = np.tile(np.eye(4), (31, 1, 1))
    wrist[:, :3, :3] = Rotation.from_rotvec(np.stack([t*.2, t*.3, t*.5], axis=1)).as_matrix()
    wrist[:, :3, 3] = np.stack([t*.1, t*t*.2, np.sin(t)*.1], axis=1)
    tips = np.tile(np.eye(4), (31, 5, 1, 1))
    local = np.tile(np.array([[i*.02, .1, .03] for i in range(5)]), (31, 1, 1))
    local[:, :, 1] += .005*np.sin(t*4)[:, None]
    tips[:, :, :3, 3] = np.einsum('tij,tkj->tki', wrist[:, :3, :3], local)+wrist[:, None, :3, 3]
    return wrist, tips
