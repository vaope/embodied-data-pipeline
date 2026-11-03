from pathlib import Path
import json

def generate_report(engine, query_clip_id, results, output_path):
    if query_clip_id != results['query_id']:
        raise ValueError('query/result mismatch')
    # Keep reports bounded even when the search reranks 10,000 windows.
    all_results = results
    raw = {r['clip_id']: r for r in [*results['raw_ranking'][:100], *results['results']]}
    results = dict(results, raw_ranking=list(raw.values()), report_raw_limit=100,
                   report_raw_total=len(all_results['raw_ranking']))
    if results.get('method') == 'phase_refined':
        results['stats'] = dict(results['stats'], final_count=len(results['results']),
                                motion_ranking_count=len(results['raw_ranking']))
    ids = [query_clip_id, *raw]
    features = {}
    for clip_id in ids:
        if clip_id in raw and 'refined_motion' in raw[clip_id]:
            features[clip_id] = raw[clip_id]['refined_motion']
        else:
            features[clip_id] = {k: v.tolist() for k, v in engine.get_clip_features(clip_id)['motion'].items()}
    payload = json.dumps({'retrieval': results, 'features': features}, ensure_ascii=False,
                         allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = Path(__file__).with_name('report.html').read_text(encoding='utf-8')
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__RETRIEVAL_PAYLOAD__', payload), encoding='utf-8')
    return str(output.resolve())
