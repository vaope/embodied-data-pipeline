from pathlib import Path
from ..common import write_json, load_config
from ..features.feature_store import FeatureStore
from .semantic_recall import semantic_recall
from .motion_ranking import motion_distances
from .rank_fusion import rank_fusion

class RetrievalEngine:
    def __init__(self, artifact_root, config=None):
        self.store = FeatureStore(artifact_root)
        self.config = load_config(config)
        self.pose_search = None

    def get_clip_features(self, clip_id):
        return self.store.get_clip_features(clip_id)

    def retrieve(self, query_clip_id, top_k=10, options=None):
        if top_k < 1:
            raise ValueError('top_k must be positive')
        options = dict(options or {})
        unknown = set(options)-{'allow_same_episode', 'max_per_episode', 'recall_top_n', 'method', 'refine_phase'}
        if unknown:
            raise ValueError(f'unknown options: {unknown}')
        max_per = options.get('max_per_episode', 1)
        if max_per is not None and max_per < 1:
            raise ValueError('max_per_episode must be positive or None')
        if options.get('recall_top_n', 1) < 1:
            raise ValueError('recall_top_n must be positive')
        method = options.get('method', self.config['retrieval_method'])
        if method not in ('pose_language_active', 'legacy_v1'):
            raise ValueError('unknown retrieval method')
        if method == 'pose_language_active':
            from .pose_language import PoseLanguageSearch
            if self.pose_search is None:
                self.pose_search = PoseLanguageSearch(self.store, self.config)
            options = dict(options, method=method, recall_top_n=options.get('recall_top_n', self.config['recall_top_n']))
            payload = self.pose_search.retrieve(query_clip_id, top_k, options)
            write_json(self.store.root/'retrieval_results'/'stable'/f'{query_clip_id}.json', payload)
            return payload
        if options.get('refine_phase'):
            raise ValueError('phase refinement requires pose_language_active')
        options = dict(options, method='legacy_v1', recall_top_n=options.get('recall_top_n', 100))
        query = self.get_clip_features(query_clip_id)
        qm = query['metadata']
        eligible = [r for r in self.store.rows if r['clip_id'] != query_clip_id and
                    (options.get('allow_same_episode', False) or r['episode_id'] != qm['episode_id'])]
        indices = [r['embedding_index'] for r in eligible]
        recalled = semantic_recall(query['semantic'], self.store.embeddings[indices], eligible,
                                  options.get('recall_top_n', self.store.meta['config']['recall_top_n']))
        candidates, excluded = [], []
        for candidate in recalled:
            try:
                feature = self.get_clip_features(candidate['clip_id'])
                distances = motion_distances(query['motion'], feature['motion'], qm['valid_hands'],
                                             feature['metadata']['valid_hands'])
                candidates.append(dict(candidate, motion_distances=distances,
                                       metadata=feature['metadata']))
            except (OSError, ValueError, KeyError) as exc:
                excluded.append({'clip_id': candidate['clip_id'], 'reason': str(exc)})
        ranked = rank_fusion(candidates)
        results, counts = [], {}
        for r in ranked:
            eid = r['metadata']['episode_id']
            if max_per is not None and counts.get(eid, 0) >= max_per:
                continue
            counts[eid] = counts.get(eid, 0)+1
            results.append(r)
            if len(results) == top_k:
                break
        payload = {'query_id': query_clip_id, 'query': qm, 'options': options,
                   'index_signature': self.store.meta['signature'], 'results': results,
                   'raw_ranking': ranked, 'excluded': excluded,
                   'stats': {'index_windows': len(self.store.rows), 'eligible_windows': len(eligible),
                             'semantic_recall_count': len(recalled), 'hand_pose_valid_count': len(candidates),
                             'excluded_missing_features': len(excluded), 'motion_ranking_count': len(ranked),
                             'final_count': len(results),
                             'distinct_recalled_episodes': len({self.store.by_id[r['clip_id']]['episode_id'] for r in recalled})}}
        output = self.store.root/'retrieval_results'/f'{query_clip_id}.json'
        write_json(output, payload)
        return payload

    def generate_report(self, query_clip_id, results, output_path):
        from ..visualization.report_generator import generate_report
        return generate_report(self, query_clip_id, results, output_path)
