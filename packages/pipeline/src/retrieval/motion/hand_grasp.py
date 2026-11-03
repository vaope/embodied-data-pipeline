"""Derive a gripper-like open/close signal from EgoDex hand joints.

``ego-grasp-proxy-v1`` maps a *hand* to the same 0=open / 1=closed range the
DROID adapter already reports for its parallel-jaw gripper:

    ratio    = |thumbTip - indexTip| / |indexMetacarpal - littleMetacarpal|
    closure  = clip((open_ref - ratio) / (open_ref - closed_ref), 0, 1)

The palm width normalises for hand size; the thumb/index tips are the closest
human analogue of two jaws.  This is a **proxy**, not a calibrated mapping:

* reference values are empirical percentiles of 40 EgoDex clips, not fitted to
  the DROID gripper;
* the human hand can grasp in ways a parallel jaw cannot (power grasp, in-hand
  manipulation), so equal numeric values do not mean equal physical state;
* frames with missing joints, implausible palm width, outlier pinch ratio or
  low tracking confidence are marked invalid, never filled with a value.

The produced signal is not used by ``local-directed-v1`` scoring.  Callers must
keep it separate and label it as a proxy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

GRASP_VERSION = 'ego-grasp-proxy-v1'
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'ego_grasp_proxy_v1.yaml'

REQUIRED_JOINTS = ('IndexFingerMetacarpal', 'LittleFingerMetacarpal', 'ThumbTip', 'IndexFingerTip')


def load_grasp_config(overrides=None):
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    if cfg.get('version') != GRASP_VERSION:
        raise ValueError('grasp config version must be ' + GRASP_VERSION)
    overrides = overrides or {}
    if set(overrides) - set(cfg):
        raise ValueError('unknown grasp configuration key(s): ' + str(sorted(set(overrides) - set(cfg))))
    cfg.update(overrides)
    if not cfg['open_ref'] > cfg['closed_ref'] > 0:
        raise ValueError('grasp references must satisfy open_ref > closed_ref > 0')
    if cfg['median_kernel'] < 1 or cfg['median_kernel'] % 2 == 0:
        raise ValueError('median_kernel must be a positive odd integer')
    if not 0 <= cfg['confidence_threshold'] <= 1:
        raise ValueError('confidence_threshold must be in [0,1]')
    if not 0 < cfg['min_palm_width_m'] < cfg['max_palm_width_m']:
        raise ValueError('palm width bounds are invalid')
    return cfg


def _median_filter(values, valid, kernel):
    """Median filter over contiguous valid runs only; invalid frames stay invalid."""
    out = np.asarray(values, float).copy()
    idx = np.flatnonzero(valid)
    if not len(idx):
        return out
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    for run in np.split(idx, splits):
        if len(run) < kernel:
            continue
        seg = out[run]
        pad = kernel // 2
        padded = np.concatenate([np.full(pad, seg[0]), seg, np.full(pad, seg[-1])])
        out[run] = np.array([np.median(padded[i:i + kernel]) for i in range(len(seg))])
    return out


def grasp_from_positions(wrist_to_joint_positions, cfg, confidence=None, confidence_threshold=None):
    """Compute the gripper-like closure signal.

    ``wrist_to_joint_positions`` maps joint name -> [T,3] world positions for the
    four ``REQUIRED_JOINTS``.  ``confidence`` is an optional [T] hand confidence.
    Returns a dict; ``signal`` is NaN where invalid.
    """
    missing = [k for k in REQUIRED_JOINTS if k not in wrist_to_joint_positions]
    if missing:
        return {'status': 'joints_missing', 'missing_joints': missing, 'signal': None,
                'valid_mask': None, 'ratio': None, 'stats': {}, 'version': GRASP_VERSION}
    a = {k: np.asarray(wrist_to_joint_positions[k], float) for k in REQUIRED_JOINTS}
    n = len(a['IndexFingerTip'])
    if any(len(v) != n or v.ndim != 2 or v.shape[1] != 3 for v in a.values()):
        raise ValueError('grasp joints must share shape [T,3]')
    palm_width = np.linalg.norm(a['IndexFingerMetacarpal'] - a['LittleFingerMetacarpal'], axis=1)
    pinch = np.linalg.norm(a['ThumbTip'] - a['IndexFingerTip'], axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = pinch / palm_width
    valid = np.isfinite(ratio) & np.isfinite(palm_width) & (palm_width > 0)
    valid &= (palm_width >= cfg['min_palm_width_m']) & (palm_width <= cfg['max_palm_width_m'])
    valid &= ratio <= cfg['max_pinch_ratio']
    if confidence is not None:
        conf = np.asarray(confidence, float)
        if conf.shape != (n,):
            raise ValueError('confidence length mismatch')
        thr = cfg['confidence_threshold'] if confidence_threshold is None else confidence_threshold
        valid &= np.isfinite(conf) & (conf >= thr)
    if valid.sum() < 2:
        return {'status': 'no_valid_frames', 'missing_joints': [], 'signal': None,
                'valid_mask': None, 'ratio': None, 'stats': {'valid_frames': int(valid.sum()),
                                                             'total_frames': int(n)},
                'version': GRASP_VERSION}
    smoothed = _median_filter(ratio, valid, int(cfg['median_kernel']))
    span = cfg['open_ref'] - cfg['closed_ref']
    closure = np.clip((cfg['open_ref'] - smoothed) / span, 0.0, 1.0)
    signal = np.where(valid, closure, np.nan)
    return {
        'status': 'ok',
        'missing_joints': [],
        'version': GRASP_VERSION,
        'signal': signal,
        'valid_mask': valid,
        'median_kernel': int(cfg['median_kernel']),
        'stats': {
            'valid_frames': int(valid.sum()),
            'total_frames': int(n),
            'valid_ratio': float(valid.mean()),
            'ratio_median': float(np.nanmedian(ratio[valid])),
            'ratio_min': float(np.nanmin(ratio[valid])),
            'ratio_max': float(np.nanmax(ratio[valid])),
            'closure_median': float(np.nanmedian(signal)),
            'closure_min': float(np.nanmin(signal)),
            'closure_max': float(np.nanmax(signal)),
            'palm_width_median_m': float(np.nanmedian(palm_width[valid])),
        },
        'mapping': {
            'version': GRASP_VERSION,
            'metric': cfg['metric'],
            'open_ref': float(cfg['open_ref']),
            'closed_ref': float(cfg['closed_ref']),
            'direction': '0=open, 1=closed (same polarity as DROID observation_gripper_position)',
            'calibrated_to_droid': False,
            'reference_provenance': 'P95/P5 of pinch/palm_width over 40 EgoDex clips, 103,662 frames',
        },
    }


def grasp_series_at_progress(signal, valid_mask, progress, start, end_exclusive):
    """Sample a per-frame signal at normalised progress for one half-open event.

    Returns (values, valid) with NaN where the event sample is invalid; no
    interpolation across invalid frames is attempted.
    """
    values = np.asarray(signal, float)[start:end_exclusive]
    mask = np.asarray(valid_mask, bool)[start:end_exclusive]
    if len(values) < 2:
        raise ValueError('event too short for a grasp series')
    idx = np.rint(np.linspace(0, len(values) - 1, len(progress))).astype(int)
    out = values[idx]
    out_valid = mask[idx] & np.isfinite(out)
    out = np.where(out_valid, out, np.nan)
    return out, out_valid
