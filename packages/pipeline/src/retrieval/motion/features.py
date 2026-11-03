import numpy as np
from scipy.spatial.transform import Rotation
from .relative_pose import relative_pose
from .temporal_align import sequence_distance


def extract_features(sequence, cfg):
    if not sequence.valid_mask.all():
        raise ValueError('resample valid windows before feature extraction')
    p, r = relative_pose(sequence)
    delta = np.diff(p, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    dt = np.diff(sequence.timestamps_s)
    turn_valid = (lengths[:-1] > cfg['stationary_epsilon_m']) & (lengths[1:] > cfg['stationary_epsilon_m'])
    turn = np.full(len(lengths)-1, np.nan)
    turn[turn_valid] = np.arccos(np.clip(np.sum(delta[:-1][turn_valid]*delta[1:][turn_valid], axis=1)/(lengths[:-1][turn_valid]*lengths[1:][turn_valid]), -1, 1))
    angle = Rotation.from_matrix(r[:-1].transpose(0, 2, 1) @ r[1:]).magnitude()
    return {'speed': lengths/dt, 'turn': turn, 'omega': angle/dt,
            'total_path_length_m': float(lengths.sum()), 'net_displacement_m': float(np.linalg.norm(p[-1]-p[0]))}


def feature_distances(a, b, cfg):
    speed = sequence_distance(a['speed']/cfg['speed_scale_m_s'], b['speed']/cfg['speed_scale_m_s'], cfg)
    # Preserve temporal position of turn masks; never compare compressed turn lists.
    turns_b = np.interp(np.linspace(0, 1, len(a['turn'])), np.linspace(0, 1, len(b['turn'])), b['turn'])
    good = np.isfinite(a['turn']) & np.isfinite(turns_b)
    turn = float(np.mean(abs(a['turn'][good]-turns_b[good]))/cfg['turn_scale_rad']) if good.any() else None
    translation = speed if turn is None else (speed+cfg['turn_weight']*turn)/(1+cfg['turn_weight'])
    rotation = sequence_distance(a['omega']/cfg['rotation_scale_rad_s'], b['omega']/cfg['rotation_scale_rad_s'], cfg)
    return translation, rotation, turn
