from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class MotionSequence:
    source: str
    sequence_id: str
    timestamps_s: np.ndarray
    position_m: np.ndarray
    rotation_xyzw: np.ndarray
    valid_mask: np.ndarray
    source_frame: str
    effector_frame: str
    reference_point: str
    grasp_signal: np.ndarray | None = None
    grasp_valid_mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ('timestamps_s', 'position_m', 'rotation_xyzw'):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool).copy()
        n = len(self.timestamps_s)
        if n < 3 or self.timestamps_s.shape != (n,) or not np.isfinite(self.timestamps_s).all() or np.any(np.diff(self.timestamps_s) <= 0):
            raise ValueError('at least 3 strictly increasing finite timestamps required')
        if self.position_m.shape != (n, 3) or self.rotation_xyzw.shape != (n, 4) or self.valid_mask.shape != (n,):
            raise ValueError('motion shape mismatch')
        norm = np.linalg.norm(self.rotation_xyzw, axis=1)
        self.valid_mask &= np.isfinite(self.position_m).all(axis=1) & np.isfinite(self.rotation_xyzw).all(axis=1) & (norm > 1e-8)
        if self.valid_mask.sum() < 3:
            raise ValueError('fewer than 3 valid poses')
        self.rotation_xyzw[self.valid_mask] /= norm[self.valid_mask, None]
        if self.grasp_signal is not None:
            self.grasp_signal = np.asarray(self.grasp_signal, float)
            if self.grasp_signal.shape != (n,):
                raise ValueError('grasp shape mismatch')
            self.grasp_valid_mask = np.isfinite(self.grasp_signal) if self.grasp_valid_mask is None else np.asarray(self.grasp_valid_mask, bool) & np.isfinite(self.grasp_signal)
            if self.grasp_valid_mask.shape != (n,):
                raise ValueError('grasp mask shape mismatch')


def matrix_poses(transforms, tolerance=1e-3):
    a = np.asarray(transforms, float)
    if a.ndim != 3 or a.shape[1:] != (4, 4):
        raise ValueError('expected [T,4,4] transforms')
    r = a[:, :3, :3]
    valid = np.isfinite(a).all(axis=(1, 2))
    valid &= np.max(abs(r.transpose(0, 2, 1) @ r - np.eye(3)), axis=(1, 2)) < tolerance
    valid &= abs(np.linalg.det(r) - 1) < tolerance
    valid &= np.max(abs(a[:, 3] - [0, 0, 0, 1]), axis=1) < tolerance
    q = np.full((len(a), 4), np.nan)
    if valid.any():
        q[valid] = Rotation.from_matrix(r[valid]).as_quat()
    return a[:, :3, 3].copy(), q, valid
