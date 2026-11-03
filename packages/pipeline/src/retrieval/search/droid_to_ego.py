"""Explicit opt-in router. Existing Ego and text baseline code stays unchanged."""
from pathlib import Path
import json
import time
from ..adapters.droid_motion import DROIDAdapter
from ..adapters.egodex_wrist_motion import EgoWristAdapter
from ..rankers.gripper_ego_ranker import GripperEgoMotionRanker




def retrieve_droid_to_ego(droid_episode_id, query_text=None, candidate_ids=None, config=None, *,
                          droid_root=None, ego_root=None, candidates=None):
    """Fixed candidates preserve E0 ranking; absent IDs reuse existing text recall."""
    started = time.perf_counter()
    ranker = GripperEgoMotionRanker(config)
    cfg = ranker.config
    if droid_root is None or ego_root is None:
        raise ValueError('droid_root and ego_root must be supplied explicitly')
    if candidates is None and candidate_ids is None:
        raise ValueError('Supply a frozen candidate pool; automatic V2 text recall is not yet packaged')
    droid_root, ego_root = Path(droid_root), Path(ego_root)
    if not droid_episode_id.startswith('droid_') or not droid_episode_id[6:].isdigit():
        raise ValueError('this route requires a DROID query')
    index = int(droid_episode_id[6:])
    query = DROIDAdapter().read(droid_root/f'episode_{index:03d}'/'steps.csv', droid_episode_id, cfg['nominal_droid_fps'])
    if candidates is not None:
        candidates = [dict(c) for c in candidates]
        ids = [c['clip_id'] for c in candidates]
        if candidate_ids is not None and ids != candidate_ids:
            raise ValueError('candidate IDs/order mismatch')
    elif candidate_ids is not None:
        ids = list(candidate_ids)
        candidates = [{'clip_id': cid, 'rank': i} for i, cid in enumerate(ids, 1)]
    if len(ids) != len(set(ids)) or not ids or len(ids) > cfg['max_candidates']:
        raise ValueError('empty/duplicate/excessive candidate pool')
    wanted, found = set(ids), {}
    with (ego_root/'manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['clip_id'] in wanted:
                found[row['clip_id']] = row
            if len(found) == len(wanted):
                break
    if set(found) != wanted:
        raise ValueError('candidate IDs missing from Ego manifest')
    reader, motions, enriched = EgoWristAdapter(), {}, []
    for candidate in candidates:
        cid = candidate['clip_id']
        row = found[cid]
        enriched.append(dict(candidate, metadata=row, start_time=row['start_time'], end_time=row['end_time']))
        try:
            motions[cid] = reader.read(row, cfg['confidence_threshold'], cfg['rotation_tolerance'])
        except (OSError, KeyError, ValueError) as exc:
            motions[cid] = ({}, {'source': str(exc)})
    results = ranker.rank(query, enriched, motions)
    return {'query_id': droid_episode_id, 'retrieval_strategy': 'droid_to_ego', 'config': cfg,
            'query_motion_metadata': query.metadata, 'results': results, 'raw_ranking': results,
            'stats': {'candidate_count': len(results), 'fallback_count': sum(r['fallback_reason'] is not None for r in results),
                      'elapsed_s': time.perf_counter()-started}}
