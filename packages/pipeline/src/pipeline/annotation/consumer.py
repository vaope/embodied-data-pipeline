"""Import verified AV output into Stage5; reject retired labels and source mismatch."""
from pathlib import Path
from .core import ROOT, digest, file_hash, project_v1, read


def load_for_stage5(path, expected_hash, episode_id, source_video_hash, frame_count):
    path = Path(path)
    annotation = read(path)
    variant = 'AV-' + annotation['method']
    provenance_path = path.parent / f'{variant}.provenance.json'
    provenance = read(provenance_path)
    if provenance.get('experiment_mode') != 'predicted_boundaries':
        raise ValueError('S5_AV_CONTROL_IS_EVALUATION_ONLY')
    native_path = path.parent / f'{variant}.native.json'
    native = read(native_path)
    run = path.parents[2]
    plan = read(run / 'plan.json')
    if provenance.get('schema_version') != 'annotation_av.provenance.v1' or provenance.get('variant') != variant:
        raise ValueError('S5_AV_PROVENANCE_INVALID')
    if expected_hash != file_hash(path) or provenance['annotation_sha256'] != expected_hash or provenance['native_sha256'] != file_hash(native_path):
        raise ValueError('S5_AV_OUTPUT_HASH_MISMATCH')
    if provenance['plan_sha256'] != digest(plan):
        raise ValueError('S5_AV_PLAN_HASH_MISMATCH')
    record = next(r for r in plan['episodes'] if r['job']['id'] == episode_id)
    if (annotation['episode_id'] != episode_id or provenance['episode_id'] != episode_id
            or provenance['source_video_sha256'] != source_video_hash
            or record['media']['source_video_sha256'] != source_video_hash
            or annotation['frame_count'] != frame_count or record['media']['frame_count'] != frame_count):
        raise ValueError('S5_AV_SOURCE_MISMATCH')
    boundary_path = path.parent / f'{provenance["boundary_stage"]}.cache.json'
    task_path = path.parent / 'task.cache.json'
    if provenance['boundary_cache_sha256'] != file_hash(boundary_path) or provenance['task_cache_sha256'] != file_hash(task_path):
        raise ValueError('S5_AV_DEPENDENCY_HASH_MISMATCH')
    from .runner import validated_cache
    boundaries = validated_cache(path.parent, provenance['boundary_stage'], plan, episode_id)['segments']
    caption_stage = f'caption-{variant[-1].lower()}'
    caption_cache = path.parent / f'{caption_stage}.cache.json'
    if file_hash(caption_cache) != provenance['caption_cache_sha256'] or validated_cache(path.parent, caption_stage, plan, episode_id)['segments'] != native['segments']:
        raise ValueError('S5_AV_CAPTION_CACHE_MISMATCH')
    if [(s['start_s'], s['end_s']) for s in boundaries] != [(s['start_s'], s['end_s']) for s in native['segments']]:
        raise ValueError('S5_AV_BOUNDARY_MISMATCH')
    if project_v1(episode_id, variant, record['media'], validated_cache(path.parent, 'task', plan, episode_id), native['segments']) != annotation:
        raise ValueError('S5_AV_PROJECTION_MISMATCH')
    return annotation, provenance, [provenance_path, native_path, boundary_path, task_path, caption_cache, run / 'plan.json']
