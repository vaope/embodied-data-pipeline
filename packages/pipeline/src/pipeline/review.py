"""Prepare a pending visual review, or apply an explicit human decision file.

Preparation never accepts a gate. Finalization requires a separately supplied
human decision bound to the exact candidate index hash; it cannot infer approval
from a successful model run, an existing video, or a previous episode review.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from .common import read_json, write_json, ref, verify_ref, utc_now
from .visual import SCENE_SCHEMA, finalize_visual


def prepare(candidate_path: Path, output: Path) -> dict:
    index = read_json(candidate_path)
    if index.get('schema_version') != 'learning.stage5.visual_index.v1':
        raise ValueError('Expected a real visual candidate index')
    alignment = read_json(index['scale_alignment_ref']['path'])
    evidence = [ref(candidate_path), index['scale_alignment_ref']]
    evidence.extend(row for row in index['frames'] if row['image_kind'] == 'depth_composite_rgb')
    for row in alignment['frames']:
        evidence.extend(row[k] for k in ('depth_ref', 'decision_ref') if row.get(k))
    for item in evidence:
        verify_ref(item)
    value = {'schema_version': 'learning.stage5.pending_review_request.v1',
             'source_key': index['source_key'], 'source_video_sha256': index['source_video_sha256'],
             'candidate_index_ref': ref(candidate_path), 'created_at': utc_now(),
             'accepted': False, 'review_status': 'pending', 'reviewer': None,
             'required_human_decisions': ['episode_human_arm_removal', 'episode_depth_composite'],
             'evidence': evidence, 'note': 'A model or program success cannot supply the human decisions.'}
    write_json(output, value)
    return value


def finalize(request_path: Path, decision_path: Path, output_dir: Path) -> dict:
    request, decision = read_json(request_path), read_json(decision_path)
    verify_ref(request['candidate_index_ref'])
    if (request.get('schema_version') != 'learning.stage5.pending_review_request.v1' or
            decision.get('reviewer') != 'user' or decision.get('source_key') != request['source_key'] or
            decision.get('candidate_index_sha256') != request['candidate_index_ref']['sha256'] or
            decision.get('m5_accepted') is not True or decision.get('m6_accepted') is not True or
            not str(decision.get('user_statement', '')).strip() or not decision.get('decided_at')):
        raise ValueError('Both scoped, explicit human approvals of this exact candidate are required')
    original = Path(request['candidate_index_ref']['path'])
    index = read_json(original)
    output_dir = output_dir.resolve()
    if original.parent == output_dir or original.parent in output_dir.parents:
        raise ValueError('Use a separate review output directory; keep the reviewed candidate immutable')
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in index.get('input_refs', []):
        verify_ref(item)
    bg = index['background_index_ref']
    external = read_json(index['external_artifacts_manifest_ref']['path']) if index.get('external_artifacts_manifest_ref') else None
    human = external['human_index_ref'] if external else ref(Path(index['episode_ref']['path']).parent/'segmentation/bundle/index.json')
    for item in (bg, human):
        verify_ref(item)
    shared = {'source_key': index['source_key'], 'source_video_sha256': index['source_video_sha256'],
              'reviewer': 'user', 'accepted': True, 'user_statement': decision['user_statement'],
              'decided_at': decision['decided_at'], 'human_decision_ref': ref(decision_path)}
    m5_path = write_json(output_dir/'m5-human-decision.json', {**shared, 'scope': 'episode_human_arm_removal', 'evidence': [bg, human]})
    # Only metadata changes here; every reviewed RGB/depth/decision remains
    # unchanged and is still covered by the original human-bound request.
    required = {str(Path(v['path']).resolve()): v['sha256'] for v in (bg, human)}
    index['gates']['m5'] = {'status': 'accepted', 'scope': 'episode_human_arm_removal',
                           'review_ref': ref(m5_path), 'review_base': str(output_dir),
                           'required_evidence': required}
    index['human_review_request_ref'] = ref(request_path)
    metadata_candidate = write_json(output_dir/'candidate-with-human-m5.json', index)
    alignment = read_json(index['scale_alignment_ref']['path'])
    evidence = [ref(metadata_candidate), index['scale_alignment_ref']]
    evidence.extend(row for row in index['frames'] if row['image_kind'] == 'depth_composite_rgb')
    for row in alignment['frames']:
        evidence.extend(row[k] for k in ('depth_ref', 'decision_ref') if row.get(k))
    m6_path = write_json(output_dir/'m6-human-decision.json', {**shared, 'scope': 'episode_depth_composite', 'evidence': evidence,
                         'reviewed_candidate_parent_ref': request['candidate_index_ref']})
    result = finalize_visual(metadata_candidate, m6_path, output_dir/'visual-reviewed.json')
    # Publish a new wrapper rather than editing the depth manifest reviewed by
    # the user. prepare_visual can consume these receipts without regenerating
    # or changing the reviewed images, avoiding circular manifest hashes.
    scenes = []
    for item in index.get('input_refs', []):
        path = Path(item['path'])
        if path.suffix.lower() == '.json':
            payload = read_json(path)
            if payload.get('schema_version') == SCENE_SCHEMA:
                scenes.append((path, payload))
    if len(scenes) != 1:
        raise ValueError('Expected one hash-bound source scene-depth manifest')
    scene_path, scene = scenes[0]
    scene['reviewed_depth_parent_ref'] = ref(scene_path)
    scene['reviews'] = {**scene.get('reviews', {}), 'm5': ref(m5_path),
                        'm6': {**ref(m6_path), 'candidate_index_ref': ref(metadata_candidate)}}
    scene_out = write_json(output_dir/'scene-depth-reviewed.json', scene)
    episode = read_json(index['episode_ref']['path'])
    episode_id = episode.get('episode_id')
    if not episode_id:
        raise ValueError('Reviewed source episode has no episode_id')
    mapping = write_json(output_dir/'scene-manifests.json', {episode_id: str(scene_out.resolve())})
    result['scene_manifest_ref'] = ref(scene_out)
    result['scene_manifest_map_ref'] = ref(mapping)
    if index.get('external_artifacts_manifest_ref'):
        visual_mapping = write_json(output_dir/'visual-artifacts.json',
            {episode_id: index['external_artifacts_manifest_ref']['path']})
        result['visual_artifact_map_ref'] = ref(visual_mapping)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--visual-index', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('finalize'); p.add_argument('--request', type=Path, required=True); p.add_argument('--decision-file', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.visual_index, args.output) if args.command == 'prepare' else finalize(args.request, args.decision_file, args.output)
    print(json.dumps({'status': value.get('review_status', value.get('index', {}).get('review_status')), 'accepted_by_preparation': False}))


if __name__ == '__main__':
    main()
