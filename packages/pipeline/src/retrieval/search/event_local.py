"""Event-level local directed motion scoring (new, independent of the 8785 scalar prototype).

Keeps the old modules untouched: the old ``event_behavior`` path is still the
``old_scalar`` baseline, this module is ``local-directed-v1``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..motion.local_frame import (FEATURE_VERSION, axis_mapping_sensitivity,
                                  build_local_trajectory, local_motion_distance,
                                  mapping_name, procrustes_shape_only)
from ..search.event_behavior import crop_event  # noqa: F401  (re-exported for callers)

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'droid_ego_local_frame_v1.yaml'

REQUIRED = (
    'progress_points', 'min_valid_ratio', 'max_gap_s', 'confidence_threshold',
    'rotation_tolerance', 'position_scale_m', 'delta_scale_m', 'min_displacement_m',
    'path_floor_m', 'rotation_scale_rad', 'weight_traj_axis', 'weight_direction',
    'weight_delta_axis', 'fusion_lambda', 'nominal_droid_fps',
)


def local_config(overrides=None):
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    if cfg.get('feature_version') != FEATURE_VERSION:
        raise ValueError('config feature_version must be '+FEATURE_VERSION)
    overrides = overrides or {}
    if set(overrides) - set(cfg):
        raise ValueError('unknown local-frame configuration key(s): '+str(sorted(set(overrides) - set(cfg))))
    cfg.update(overrides)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError('missing local-frame configuration: '+str(missing))
    for key in ('position_scale_m', 'delta_scale_m', 'min_displacement_m', 'path_floor_m',
                'rotation_scale_rad', 'nominal_droid_fps'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError('configuration must be positive: '+key)
    if int(cfg['progress_points']) < 3:
        raise ValueError('progress_points must be >= 3')
    for key in ('weight_traj_axis', 'weight_direction', 'weight_delta_axis', 'fusion_lambda'):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('configuration must be non-negative: '+key)
    if cfg['weight_traj_axis'] + cfg['weight_direction'] + cfg['weight_delta_axis'] <= 0:
        raise ValueError('at least one local component weight must be positive')
    if not 0 < cfg['min_valid_ratio'] <= 1:
        raise ValueError('min_valid_ratio must be in (0,1]')
    return cfg


def build_event_representation(sequence, cfg, side=None, mapping=None, mapping_declared=False,
                               reference_point=None, axis_convention=None):
    extra = {'side': side} if side else {}
    return build_local_trajectory(
        sequence, n_points=int(cfg['progress_points']),
        min_valid_ratio=float(cfg['min_valid_ratio']), max_gap_s=float(cfg['max_gap_s']),
        frame_mapping=mapping, mapping_declared=mapping_declared,
        reference_point=reference_point, axis_convention=axis_convention,
        extra=dict(extra, local_config_path=str(CONFIG_PATH)),
    )


def hand_representations(hands, cfg):
    """Build local representations for every valid hand; failures are reported, not filled."""
    built, errors = {}, {}
    for hand, sequence in hands.items():
        try:
            built[hand] = build_event_representation(sequence, cfg, side=hand)
        except ValueError as exc:
            errors[hand] = str(exc)
    return built, errors


def pair_scores(query_trajectory, hand_trajectories, cfg, selected_hand):
    """Local distance for the activity-selected hand plus every other valid hand."""
    if selected_hand not in hand_trajectories:
        raise ValueError(f'selected hand {selected_hand!r} has no local representation')
    scores = {}
    for hand, traj in hand_trajectories.items():
        result = local_motion_distance(query_trajectory, traj, cfg)
        result['hand'] = hand
        scores[hand] = result
    return scores


def fuse(similarity, distance, cfg):
    """Caption cosine minus a bounded local-motion penalty (exploratory lambda)."""
    lam = float(cfg['fusion_lambda'])
    if distance is None:
        return float(similarity) - lam, lam
    if not np.isfinite(distance) or distance < 0:
        raise ValueError('invalid local distance')
    penalty = lam * distance / (1.0 + distance)
    return float(similarity) - penalty, float(penalty)


def sensitivity(query_trajectory, ego_trajectory, cfg):
    return axis_mapping_sensitivity(query_trajectory, ego_trajectory, cfg)


def shape_only(query_trajectory, ego_trajectory, cfg):
    return procrustes_shape_only(query_trajectory, ego_trajectory, cfg)


__all__ = ['FEATURE_VERSION', 'CONFIG_PATH', 'local_config', 'build_event_representation',
           'hand_representations', 'pair_scores', 'fuse', 'sensitivity', 'shape_only',
           'mapping_name', 'crop_event']
