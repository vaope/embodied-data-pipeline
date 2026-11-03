"""Opt-in prototype: fixed event spans, caption-first score, activity-selected wrist.

No object frame, contact inference, signed cross-domain axes, or free robot windows.
"""
from dataclasses import replace
import numpy as np
from ..motion.temporal_align import resample
from ..motion.features import extract_features

MAX_BEHAVIOR_PENALTY = 0.03  # cosine units; exploratory, not calibrated


def crop_event(sequence, start_frame, end_frame_exclusive):
    if not 0 <= start_frame < end_frame_exclusive <= len(sequence.timestamps_s):
        raise ValueError('event frame interval outside original sequence')
    sl = slice(start_frame, end_frame_exclusive)
    return replace(sequence, timestamps_s=sequence.timestamps_s[sl].copy()-sequence.timestamps_s[start_frame],
                   position_m=sequence.position_m[sl].copy(), rotation_xyzw=sequence.rotation_xyzw[sl].copy(),
                   valid_mask=sequence.valid_mask[sl].copy(),
                   grasp_signal=None if sequence.grasp_signal is None else sequence.grasp_signal[sl].copy(),
                   grasp_valid_mask=None if sequence.grasp_valid_mask is None else sequence.grasp_valid_mask[sl].copy(),
                   metadata=dict(sequence.metadata, original_frames_half_open=[start_frame,end_frame_exclusive],
                                 original_span_s=[float(sequence.timestamps_s[start_frame]),float(sequence.timestamps_s[end_frame_exclusive-1])]))


def features(sequence, config):
    sampled = resample(sequence, 0., float(sequence.timestamps_s[-1]), config)
    f = extract_features(sampled, config)
    f['valid_ratio'] = sampled.metadata['source_valid_ratio']
    f['activity'] = float(np.mean(f['speed'])+0.05*np.mean(f['omega']))
    return f


def choose_active_hand(hands, config):
    available, errors = {}, {}
    for hand, sequence in hands.items():
        try:
            available[hand] = features(sequence, config)
        except ValueError as exc:
            errors[hand] = str(exc)
    if not available:
        raise ValueError('no valid wrist: '+str(errors))
    hand = min(available, key=lambda h: (-available[h]['activity'], h))
    return hand, available[hand], {h: f['activity'] for h,f in available.items()}, errors


def behavior_distance(query, ego):
    grid = np.linspace(0,1,32)
    def profile(x):
        return np.interp(grid,np.linspace(0,1,len(x)),x)
    speed = float(np.mean(abs(profile(query['speed'])-profile(ego['speed'])))/0.1)
    rotation = float(np.mean(abs(profile(query['omega'])-profile(ego['omega']))))
    stop = float(abs(np.mean(query['speed']<0.01)-np.mean(ego['speed']<0.01)))
    path = float(abs(np.log((query['total_path_length_m']+0.001)/(ego['total_path_length_m']+0.001))))
    parts = dict(speed=speed, rotation=rotation, stop_fraction=stop, log_path_ratio=path)
    return 0.4*speed+0.2*rotation+0.2*stop+0.2*path, parts


def fused_score(cosine, distance):
    if distance is not None and (not np.isfinite(distance) or distance < 0):
        raise ValueError('invalid behavior distance')
    penalty = MAX_BEHAVIOR_PENALTY if distance is None else MAX_BEHAVIOR_PENALTY*distance/(1+distance)
    return float(cosine-penalty), float(penalty)
