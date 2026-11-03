import h5py
import numpy as np
from ..motion.representation import MotionSequence, matrix_poses
from ..motion.hand_grasp import (GRASP_VERSION, REQUIRED_JOINTS, grasp_from_positions,
                                 load_grasp_config)


class EgoWristAdapter:
    def read(self, row, confidence_threshold=0.2, rotation_tolerance=1e-3,
             grasp=GRASP_VERSION, grasp_config=None):
        """Read both wrist-root poses, plus an optional gripper-like grasp signal.

        ``grasp=None`` disables the hand-grasp mapping (old behaviour).  The
        default ``'ego-grasp-proxy-v1'`` adds ``grasp_signal``/``grasp_valid_mask``
        in the same 0=open / 1=closed convention as DROID's observed gripper.
        The mapping is a labelled proxy, not a calibrated gripper.
        """
        start, end, fps = int(row['start_frame']), int(row['end_frame']), float(row['fps'])
        if start < 0 or end <= start or fps <= 0:
            raise ValueError('invalid original frame interval/fps')
        if not np.isclose(start/fps, row['start_time']) or not np.isclose(end/fps, row['end_time']):
            raise ValueError('frame/time mapping mismatch')
        hands, issues = {}, {}
        grasp_cfg = load_grasp_config() if grasp is not None and grasp_config is None else grasp_config
        if grasp is not None and grasp != GRASP_VERSION:
            raise ValueError('unsupported grasp mapping: ' + str(grasp))
        with h5py.File(row['hdf5_path'], 'r') as h:
            for side in ('left', 'right'):
                key = side+'Hand'
                name = 'transforms/'+key
                try:
                    if name not in h:
                        raise ValueError('missing wrist transform')
                    if end >= len(h[name]):
                        raise ValueError('frame interval outside source HDF5')
                    p, q, valid = matrix_poses(h[name][start:end+1], rotation_tolerance)
                    conf_key = 'confidences/'+key
                    missing_conf = conf_key not in h
                    conf = None
                    if not missing_conf:
                        conf = np.asarray(h[conf_key][start:end+1], float)
                        if conf.shape != valid.shape:
                            raise ValueError('confidence shape mismatch')
                        valid &= np.isfinite(conf) & (conf >= confidence_threshold) & (conf <= 1)
                    signal, signal_valid, grasp_info = _grasp_signal(
                        h, side, start, end, grasp_cfg, conf, confidence_threshold, grasp)
                    hands[side] = MotionSequence(
                        'egodex', row['clip_id']+':'+side, np.arange(end-start+1)/fps,
                        p, q, valid, 'egodex_arkit_world', side+'_hand', 'wrist_root',
                        grasp_signal=signal, grasp_valid_mask=signal_valid,
                        metadata={
                            'confidence_missing': missing_conf,
                            'timestamp_source': 'original_frame/fps',
                            'original_frames_inclusive': [start, end],
                            'hdf5_path': row['hdf5_path'],
                            'grasp': grasp_info,
                        })
                except ValueError as exc:
                    issues[side] = str(exc)
        return hands, issues


def _grasp_signal(h, side, start, end, grasp_cfg, conf, confidence_threshold, grasp):
    """Best-effort grasp proxy: never turns a usable wrist pose into a failure."""
    if grasp is None:
        return None, None, {'status': 'disabled', 'version': None,
                            'calibrated_to_droid': False}
    try:
        available = [j for j in REQUIRED_JOINTS if 'transforms/'+side+j in h]
        if len(available) != len(REQUIRED_JOINTS):
            missing = [j for j in REQUIRED_JOINTS if j not in available]
            return None, None, {'status': 'joints_missing', 'missing_joints': missing,
                                'version': GRASP_VERSION, 'calibrated_to_droid': False}
        if any(len(h['transforms/'+side+j]) <= end for j in REQUIRED_JOINTS):
            return None, None, {'status': 'joints_short', 'version': GRASP_VERSION,
                                'calibrated_to_droid': False}
        positions = {j: np.asarray(h['transforms/'+side+j][start:end+1], float)[:, :3, 3]
                     for j in REQUIRED_JOINTS}
        result = grasp_from_positions(positions, grasp_cfg, confidence=conf,
                                      confidence_threshold=confidence_threshold)
    except (KeyError, ValueError, IndexError) as exc:
        return None, None, {'status': 'error', 'error': str(exc), 'version': GRASP_VERSION,
                            'calibrated_to_droid': False}
    info = {'status': result['status'], 'version': result.get('version', GRASP_VERSION),
            'metric': grasp_cfg['metric'], 'open_ref': grasp_cfg['open_ref'],
            'closed_ref': grasp_cfg['closed_ref'], 'median_kernel': grasp_cfg['median_kernel'],
            'calibrated_to_droid': False, 'stats': result['stats'],
            'mapping': result.get('mapping')}
    if result['signal'] is None:
        return None, None, info
    return result['signal'], result['valid_mask'], info
