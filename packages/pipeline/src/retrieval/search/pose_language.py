"""Frozen local retrieval recipe: pose/language recall, activity-weighted RMSE.

The recipe is a project decision, not an exact implementation of SiMDex.
"""
from functools import lru_cache
from pathlib import Path
import uuid
import numpy as np
import h5py
from scipy.stats import rankdata
from ..common import file_hash, read_json
from ..data.egodex_reader import HAND_KEYS
from ..features.motion_features import extract_window
from .rank_fusion import rank_fusion

SIDES = ('left', 'right')
ROUTES = {'translation': (0, 30), 'rotation': (30, 60),
          'fingertip': (60, 510), 'wrist': (510, 603)}
RECIPE = 'pose-language-active-v1'


def pack(motion, flags):
    result = np.full((2, 603), np.nan)
    for h, side in enumerate(SIDES):
        if flags[side]:
            for route, (begin, end) in ROUTES.items():
                result[h, begin:end] = motion[side+'_'+route].reshape(-1)
    return result


def energy(features):
    fingers = features[..., 60:510].reshape(*features.shape[:-1], 30, 5, 3)
    return (features[..., :30].mean(axis=-1)
            + .05*features[..., 30:60].mean(axis=-1)
            + np.linalg.norm(np.diff(fingers, axis=-3), axis=-1).mean(axis=(-1, -2))*30)


def hand_weights(query, flags):
    weights = np.zeros(2)
    values = energy(query[flags])
    weights[flags] = np.maximum(values/max(values.sum(), 1e-12), .05)
    return weights/weights.sum()


def distances(query, candidate, flags):
    result = np.full((2, 4), np.nan)
    for h in np.flatnonzero(flags):
        for k, (begin, end) in enumerate(ROUTES.values()):
            result[h, k] = np.sqrt(np.mean((query[h, begin:end]-candidate[h, begin:end])**2))
    if not np.isfinite(result[flags]).all():
        raise ValueError('nonfinite motion features')
    return result


class PoseLanguageSearch:
    def __init__(self, store, config):
        self.store, self.config = store, config
        if store.meta['config']['horizon'] != 30 or store.meta['config']['fps'] != 30:
            raise ValueError('pose-language-active-v1 requires 30 FPS and a 30-frame horizon')
        self.ids = np.array([r['clip_id'] for r in store.rows])
        self.flags = np.array([[r['valid_hands'][s] for s in SIDES] for r in store.rows])
        self.matrix = None
        candidate = config.get('packed_motion_path')
        manifest_hash = file_hash(store.generation/'manifest.jsonl')
        if candidate and Path(candidate).is_file():
            meta_path = Path(candidate).with_name('pack_meta.json')
            if meta_path.is_file():
                meta = read_json(meta_path)
                if (meta.get('status') == 'complete' and meta.get('manifest_sha256') == manifest_hash
                        and store.root/meta['generation'] == store.generation):
                    matrix = np.load(candidate, mmap_mode='r', allow_pickle=False)
                    if matrix.shape != (len(store.rows), 2, 603) or matrix.dtype != np.float64:
                        raise ValueError('invalid packed motion shape/dtype')
                    self.matrix = matrix
        cache = store.root/'search_cache'/f'initial_pose_{manifest_hash}.npy'
        if cache.is_file():
            initial = np.load(cache, allow_pickle=False)
        elif self.matrix is not None:
            initial = np.array(self.matrix[:, :, 60:75])
        else:
            initial = np.full((len(store.rows), 2, 15), np.nan)
            for i, row in enumerate(store.rows):
                motion = store.get_clip_features(row['clip_id'])['motion']
                for h, side in enumerate(SIDES):
                    if self.flags[i, h]:
                        initial[i, h] = motion[side+'_fingertip'][0]
        if initial.shape != (len(store.rows), 2, 15) or not np.isfinite(initial[self.flags]).all():
            raise ValueError('invalid initial pose cache')
        if not cache.is_file():
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Cache raw positions; normalize exactly once after loading.
            temporary = cache.with_name(cache.stem+'_'+uuid.uuid4().hex+'.npy')
            np.save(temporary, initial, allow_pickle=False)
            temporary.replace(cache)
        self.initial = initial/np.maximum(np.linalg.norm(initial, axis=2, keepdims=True), 1e-12)

    def packed(self, index):
        if self.matrix is not None:
            return np.asarray(self.matrix[index])
        value = self.store.get_clip_features(str(self.ids[index]))
        return pack(value['motion'], value['metadata']['valid_hands'])

    def retrieve(self, query_id, top_k, options):
        qi = self.store.by_id[query_id]['embedding_index']
        qm = self.store.rows[qi]
        qflags = self.flags[qi]
        query = self.packed(qi)
        weights = hand_weights(query, qflags)
        pool = np.array([i for i, r in enumerate(self.store.rows) if r['clip_id'] != query_id
                         and (options.get('allow_same_episode') or r['episode_id'] != qm['episode_id'])], dtype=int)
        scores = np.clip(self.store.embeddings @ self.store.embeddings[qi], -1, 1)
        pose = np.linalg.norm(self.initial[:, qflags]-self.initial[qi, qflags], axis=2).mean(axis=1)
        valid = np.all(self.flags[:, qflags], axis=1)
        pose[~valid] = np.inf
        language_rank = rankdata(-scores[pool], method='min')
        pose_rank = rankdata(pose[pool], method='min')
        recall_sum = language_rank+pose_rank
        order = np.lexsort((self.ids[pool], -scores[pool], recall_sum))
        count = options.get('recall_top_n', self.config['recall_top_n'])
        recalled = pool[order[:count]]
        candidates, excluded = [], []
        ranks = {int(pool[i]): (int(language_rank[i]), int(pose_rank[i]), int(recall_sum[i])) for i in order[:count]}
        for ci in recalled:
            if not valid[ci]:
                excluded.append({'clip_id': str(self.ids[ci]), 'reason': 'candidate missing required query hand'})
                continue
            try:
                per_hand = distances(query, self.packed(ci), qflags)
                weighted = (per_hand[qflags]*weights[qflags, None]).sum(axis=0)
                lr, pr, rr = ranks[int(ci)]
                candidates.append({'clip_id': str(self.ids[ci]), 'metadata': self.store.rows[ci],
                                   'semantic_similarity': float(scores[ci]), 'semantic_rank': lr,
                                   'pose_initial_distance': float(pose[ci]), 'pose_rank': pr, 'recall_rank_sum': rr,
                                   'motion_distances': dict(zip(ROUTES, map(float, weighted))),
                                   'per_hand_distances': {s: dict(zip(ROUTES, map(float, per_hand[h])))
                                                          for h, s in enumerate(SIDES) if qflags[h]}})
            except (ValueError, OSError, KeyError) as exc:
                excluded.append({'clip_id': str(self.ids[ci]), 'reason': str(exc)})
        ranked = rank_fusion(candidates)
        results, counts = [], {}
        for row in ranked:
            eid = row['metadata']['episode_id']
            maximum = options.get('max_per_episode', 1)
            if maximum is not None and counts.get(eid, 0) >= maximum:
                continue
            counts[eid] = counts.get(eid, 0)+1
            results.append(row)
            if len(results) == top_k:
                break
        payload = {'query_id': query_id, 'query': qm, 'method': 'pose_language_active',
                   'recipe_version': RECIPE, 'query_hand_weights': dict(zip(SIDES, map(float, weights))),
                   'options': options, 'index_signature': self.store.meta['signature'],
                   'results': results, 'raw_ranking': ranked, 'excluded': excluded,
                   'stats': {'index_windows': len(self.ids), 'eligible_windows': len(pool),
                             'semantic_recall_count': len(recalled), 'hand_pose_valid_count': len(candidates),
                             'excluded_missing_features': len(excluded), 'motion_ranking_count': len(ranked),
                             'final_count': len(results), 'distinct_recalled_episodes': len({self.store.rows[i]['episode_id'] for i in recalled})}}
        if options.get('refine_phase'):
            payload['phase_diagnostic'] = self.refine(query, qflags, weights, payload)
        return payload

    def refine(self, query, qflags, weights, coarse):
        reference = np.sort(np.array([list(r['motion_distances'].values()) for r in coarse['raw_ranking']]), axis=0)
        # Fixed source shortlist, independent of requested display top_k.
        shortlist, used = [], set()
        for row in coarse['raw_ranking']:
            if row['metadata']['episode_id'] not in used:
                shortlist.append(row)
                used.add(row['metadata']['episode_id'])
                if len(shortlist) == 10:
                    break
        optimized, tested, excluded = [], 0, 0
        cfg = self.store.meta['config']
        for old in shortlist:
            meta = old['metadata']
            hands = read_hands(meta['hdf5_path'])
            count = len(next(iter(hands.values()))['wrist'])
            candidates = []
            for offset in range(-14, 15):
                start, end = meta['start_frame']+offset, meta['end_frame']+offset
                if start < 0 or end >= count:
                    continue
                tested += 1
                motion, flags, issues = extract_window(hands, start, end, cfg)
                if motion is None or any(qflags[h] and not flags[s] for h, s in enumerate(SIDES)):
                    excluded += 1
                    continue
                per_hand = distances(query, pack(motion, flags), qflags)
                dist = (per_hand[qflags]*weights[qflags, None]).sum(axis=0)
                ranks = [int(np.searchsorted(reference[:, k], dist[k], side='left'))+1 for k in range(4)]
                clip_id = f"{meta['episode_id']}_f{start:06d}_{end:06d}"
                m = dict(meta, clip_id=clip_id, start_frame=start, end_frame=end,
                         start_time=meta['start_time']+offset/cfg['fps'], end_time=meta['end_time']+offset/cfg['fps'],
                         valid_hands=flags, hand_issues=issues,
                         semantic_text_note='Refinement retains coarse-window semantic recall; no re-encoding.')
                candidates.append(dict(old, clip_id=clip_id, metadata=m, playback_clip_id=old['clip_id'],
                                       frame_offset=offset, refined_motion={k: v.tolist() for k, v in motion.items()},
                                       motion_distances=dict(zip(ROUTES, map(float, dist))),
                                       motion_ranks=dict(zip(ROUTES, ranks)), rank_sum=sum(ranks),
                                       per_hand_distances={s: dict(zip(ROUTES, map(float, per_hand[h]))) for h, s in enumerate(SIDES) if qflags[h]}))
            if candidates:
                optimized.append(min(candidates, key=lambda r: (r['rank_sum'], abs(r['frame_offset']), r['clip_id'])))
        optimized.sort(key=lambda r: (r['rank_sum'], -r['semantic_similarity'], r['clip_id']))
        for i, row in enumerate(optimized, 1):
            row['raw_rank'] = i
        return {'method': 'phase_refined', 'results': optimized, 'tested': tested, 'excluded': excluded,
                'scope': 'Top10 distinct sources, +/-14 frames; fixed coarse-candidate rank distributions; no time warping or semantic re-encoding'}


@lru_cache(maxsize=4)
def read_hands(path):
    hands = {}
    with h5py.File(path, 'r') as source:
        for side, mapping in HAND_KEYS.items():
            names = [mapping['wrist'], *mapping['tips']]
            if not all('transforms/'+n in source for n in names):
                continue
            poses = [np.asarray(source['transforms/'+n], dtype=np.float64) for n in names]
            hands[side] = {'wrist': poses[0], 'tips': np.stack(poses[1:], axis=1),
                           'confidence': {n: np.asarray(source['confidences/'+n]) for n in names if 'confidences/'+n in source}}
    return hands
