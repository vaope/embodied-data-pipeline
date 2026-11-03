import numpy as np
from .pose_transform import wrist_local_features, validate_se3

def extract_window(hands, start, end, cfg):
    features, flags, issues, fatal = {}, {}, {}, False
    for side in ('left', 'right'):
        flags[side] = False
        if side not in hands:
            issues[side] = 'missing_hand_transforms'
            continue
        hand = hands[side]
        w, tips = hand['wrist'][start:end+1], hand['tips'][start:end+1]
        try:
            validate_se3(w, cfg['rotation_tolerance'])
            validate_se3(tips, cfg['rotation_tolerance'])
        except ValueError as exc:
            issues[side], fatal = str(exc), True
            continue
        bad = []
        for key, values in hand['confidence'].items():
            c = values[start:end+1]
            if not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
                bad.append(key+':invalid_confidence')
            elif np.any(c < cfg['confidence_threshold']):
                bad.append(key+':low_confidence')
        if bad:
            issues[side] = ','.join(bad)
            continue
        f = wrist_local_features(w, tips, cfg['rotation_tolerance'])
        f['translation'] = np.linalg.norm(f['translation_delta'], axis=1)*cfg['fps']
        f['rotation'] = np.linalg.norm(f['rotation_delta'], axis=1)*cfg['fps']
        features.update({f'{side}_{key}': value for key, value in f.items()})
        flags[side] = True
    if fatal or not any(flags.values()):
        return None, flags, issues
    return features, flags, issues
