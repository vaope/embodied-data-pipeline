from pathlib import Path
import numpy as np
import yaml
from ..motion.temporal_align import resample, query_windows
from ..motion.features import extract_features, feature_distances


def load_config(overrides=None):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/droid_ego_motion_v1.yaml').read_text(encoding='utf-8'))
    overrides = overrides or {}
    if set(overrides)-set(cfg):
        raise ValueError('unknown motion configuration')
    cfg.update(overrides)
    for key in ('nominal_droid_fps', 'resample_fps', 'stride_s', 'max_windows', 'max_candidates', 'max_gap_s', 'min_duration_s', 'speed_scale_m_s', 'turn_scale_rad', 'rotation_scale_rad_s', 'dtw_max_run'):
        if cfg[key] <= 0:
            raise ValueError('configuration must be positive: '+key)
    if not 0 < cfg['min_valid_ratio'] <= 1 or not 0 <= cfg['confidence_threshold'] <= 1:
        raise ValueError('invalid quality threshold')
    if not cfg['duration_ratios'] or any(not .75 <= x <= 1.5 for x in cfg['duration_ratios']):
        raise ValueError('duration ratios must be bounded [0.75,1.5]')
    if not 0 <= cfg['dtw_band_fraction'] <= .5 or cfg['turn_weight'] < 0:
        raise ValueError('invalid distance configuration')
    return cfg


class GripperEgoMotionRanker:
    def __init__(self, config=None):
        self.config = load_config(config)

    def rank(self, query, candidates, motions):
        cfg = self.config
        records, qcache = [], {}
        for candidate in candidates:
            cid = candidate['clip_id']
            hands, issues = motions[cid]
            choices = []
            for hand, ego in hands.items():
                try:
                    duration = ego.timestamps_s[-1]
                    target = resample(ego, 0., duration, cfg)
                    ef = extract_features(target, cfg)
                    spans = query_windows(query, duration, cfg)
                    for start, length in spans:
                        try:
                            key = (start, length)
                            if key not in qcache:
                                segment = resample(query, start, length, cfg)
                                qcache[key] = (segment, extract_features(segment, cfg))
                            segment, qf = qcache[key]
                            tr, ro, turn = feature_distances(qf, ef, cfg)
                            # Fixed unit scales select one coherent hand/window for both channels.
                            choices.append({'hand': hand, 'span': [start, start+length], 'translation': tr, 'rotation': ro,
                                            'turn': turn, 'window_count': len(spans),
                                            'valid_ratio': min(target.metadata['source_valid_ratio'], segment.metadata['source_valid_ratio']),
                                            'ego_path_length_m': ef['total_path_length_m'], 'query_path_length_m': qf['total_path_length_m'],
                                            'ego_net_displacement_m': ef['net_displacement_m'],
                                            'confidence_missing': ego.metadata.get('confidence_missing', False)})
                        except ValueError:
                            continue
                except ValueError as exc:
                    issues = dict(issues, **{hand: str(exc)})
            row = dict(candidate, retrieval_strategy='droid_ego_motion_dtw' if cfg['use_dtw'] else 'droid_ego_motion_v1',
                       fallback_reason=None, hand_issues=issues)
            if choices:
                best = min(choices, key=lambda x: (x['translation']+x['rotation'], x['hand'], x['span'][0]))
                hand_scores = {h: min((c for c in choices if c['hand'] == h), key=lambda x: x['translation']+x['rotation']) for h in hands if any(c['hand'] == h for c in choices)}
                row.update(motion_distances={'translation': best['translation'], 'rotation': best['rotation']},
                           matched_droid_span_s=best['span'], matched_ego_hand=best['hand'], valid_ratio=best['valid_ratio'],
                           matched_window={'start_time': candidate['start_time'], 'end_time': candidate['end_time']},
                           hand_scores=hand_scores, diagnostics=best,
                           missing_motion_channel=['turn'] if best['turn'] is None else [])
            else:
                row.update(motion_distances={}, fallback_reason='no_valid_motion_window', missing_motion_channel=['translation', 'rotation'],
                           matched_droid_span_s=None, matched_ego_hand=None, valid_ratio=0.)
            records.append(row)
        for channel in ('translation', 'rotation'):
            values = [r['motion_distances'][channel] for r in records if channel in r['motion_distances']]
            for row in records:
                value = row['motion_distances'].get(channel)
                row[channel+'_rank'] = 1+sum(v < value for v in values) if value is not None else None
        for row in records:
            ranks = [row[c+'_rank'] for c in ('translation', 'rotation') if row[c+'_rank'] is not None]
            row['rank_sum'] = sum(ranks) if ranks else None
        # Missing motion goes last, preserving baseline order for fallback rows.
        records.sort(key=lambda r: (r['rank_sum'] is None, r['rank_sum'] or 0,
                                    -r.get('similarity', r.get('semantic_similarity', 0)), r.get('rank', 0), r['clip_id']))
        for i, row in enumerate(records, 1):
            row['baseline_rank'] = row.get('rank')
            row['rank'] = i
        return records
