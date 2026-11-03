import csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..motion.representation import MotionSequence


class DROIDAdapter:
    def read(self, steps_csv, sequence_id=None, nominal_fps=15.0):
        with Path(steps_csv).open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        fields = [f'observation_cartesian_position_{i}' for i in range(6)]
        if not rows or not all(k in rows[0] for k in fields):
            raise ValueError('missing observed cartesian_position; commanded_only is not enabled')
        pose = np.array([[float(r[k]) for k in fields] for r in rows])
        finite = np.isfinite(pose).all(axis=1)
        quat = np.full((len(rows), 4), np.nan)
        quat[finite] = Rotation.from_euler('xyz', pose[finite, 3:], degrees=False).as_quat()
        # Only accept an explicitly named seconds column as measured time.
        if 'robot_timestamp_seconds' in rows[0]:
            times = np.array([float(r['robot_timestamp_seconds']) + float(r.get('robot_timestamp_nanos', 0))*1e-9 for r in rows])
            time_source = 'robot_timestamp_seconds+nanos'
        else:
            if nominal_fps <= 0:
                raise ValueError('nominal_fps must be positive')
            times = np.array([int(r['step']) for r in rows]) / nominal_fps
            time_source = f'nominal_{nominal_fps:g}hz'
        grip = np.array([float(r.get('observation_gripper_position', 'nan')) for r in rows])
        return MotionSequence('droid', sequence_id or Path(steps_csv).parent.name, times-times[0], pose[:, :3], quat, finite,
                              'droid_robot_base', 'observed_ee', 'observed_ee', grip,
                              np.isfinite(grip) & (grip >= 0) & (grip <= 1),
                              {'timestamp_source': time_source, 'rotation_convention': 'scipy extrinsic xyz radians; xyzw',
                               'pose_origin': 'observation', 'gripper_polarity': '0=open,1=closed; not grasp success',
                               'tcp_calibration': 'unavailable', 'steps_csv': str(Path(steps_csv).resolve())})
