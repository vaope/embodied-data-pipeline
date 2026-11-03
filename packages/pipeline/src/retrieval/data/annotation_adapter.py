"""Schema v1 uses [start_frame,end_frame_exclusive); pose endpoints are inclusive."""
from pathlib import Path
import jsonschema
from ..common import read_json

def load_annotation(row, cfg, frame_count):
    path = row.get('annotation_path')
    errors = []
    if path:
        try:
            ann = read_json(path)
            jsonschema.Draft202012Validator(read_json(cfg['schema_path'])).validate(ann)
            if ann['episode_id'] != row['episode_id'] or ann['frame_count'] != frame_count:
                raise ValueError('annotation episode/frame count mismatch')
            if abs(ann['fps'] - cfg['fps']) > 1e-3:
                raise ValueError('annotation FPS mismatch')
            previous = 0
            for step in ann['steps']:
                if not previous <= step['start_frame'] < step['end_frame_exclusive'] <= frame_count:
                    raise ValueError('invalid/overlapping annotation intervals')
                previous = step['end_frame_exclusive']
            if not (ann['auto_qa']['schema_valid'] and ann['auto_qa']['temporal_valid']):
                raise ValueError('source annotation QA failed')
            return ann, errors
        except (ValueError, OSError, jsonschema.ValidationError) as exc:
            errors.append({'code': 'annotation_fallback', 'path': str(path), 'reason': str(exc)[:500]})
    else:
        errors.append({'code': 'annotation_missing', 'reason': 'use official language metadata'})
    return None, errors

def window_text(annotation, start, end, official_text, row):
    # Motion support is [start,end] samples, duration [start/fps,end/fps).
    # A step starting at the last pose endpoint has zero duration overlap.
    if annotation:
        task = annotation['task']['goal'] or annotation['task']['description']
        selected = [s for s in annotation['steps'] if s['start_frame'] < end
                    and s['end_frame_exclusive'] > start]
        text = ' | '.join(s['description'] for s in selected)
        source = 'stage3_schema_v1'
        version = row.get('annotation_version', 'schema_v1_method_A')
    else:
        task, selected, text = official_text, [], ''
        source, version = 'egodex_official_metadata', 'hdf5_attributes'
    task = task.strip()
    if not task:
        raise ValueError('empty task/official annotation')
    return {'task_text': task, 'step_text': text or task,
            'semantic_granularity': 'step_overlap' if text else 'task_only',
            'overlap_step_ids': [s['step_id'] for s in selected],
            'annotation_source': source, 'annotation_version': version,
            'annotation_path': row.get('annotation_path') if annotation else None,
            'semantic_text': f'Task: {task}\nStep: {text or task}'}
