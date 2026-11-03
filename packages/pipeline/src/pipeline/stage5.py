"""Single entry point for read-only reuse, typed packaging, and explicit failure states.

No SAM/VLM/IK is loaded by this adapter. Source stages retain their own gates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import jsonschema
import numpy as np

from .common import EPS_TIME, REPRO, ROOT, code_fingerprint, read_json, ref, schema_name, sha256, source_key, stable_hash, utc_now, verify_ref, write_json, write_npz
from .supervision import STRICT_STEPS, SUPERVISION_PROFILES, annotation_diagnostics, build_supervision, plain

SCHEMA = Path(__file__).parent / 'schemas/bundle.schema.json'
ANNOTATION_SCHEMA = Path(__file__).parent / 'annotation/schemas/step.schema.json'


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _missing(reason: str) -> dict:
    return {'availability': 'missing', 'reason': reason}


class EpisodeBuilder:
    def __init__(self, sample: dict, output: Path, run_id: str, audit: dict, scene_manifest: Path | None = None, statistics_path: Path | None = None,
                 supervision_profile: str = STRICT_STEPS, external_artifacts_manifest: Path | None = None, execution_manifest: Path | None = None):
        self.sample, self.output, self.run_id, self.audit = sample, output, run_id, audit
        self.scene_manifest = scene_manifest
        self.statistics_path = statistics_path
        self.supervision_profile = supervision_profile
        self.external_artifacts_manifest = external_artifacts_manifest
        self.execution_manifest = execution_manifest
        self.artifacts: dict[str, dict] = {}
        self.status: dict[str, dict] = {}
        self.records: list[dict] = []
        self.started = time.perf_counter()
        self.episode_id = sample['episode_id']
        self.key = source_key(sample['source_video']['sha256'], sample['source_hdf5']['sha256'])
        self.episode_root: Path | None = REPRO / 'data/episodes' / self.episode_id
        if not (self.episode_root / 'episode.json').is_file():
            self.episode_root = None

    def add(self, name: str, path: Path, schema: str | None = None, producer: str = 'stage5') -> dict:
        value = ref(path, schema_version=schema, producer=producer, source_key=self.key)
        # A known historical code/data snapshot must not silently change.
        expected = self.audit.get('files', {}).get(str(path.resolve()))
        if expected and expected['sha256'] != value['sha256']:
            raise ValueError(f'S5_SOURCE_HASH_MISMATCH against audit: {path}')
        self.artifacts[name] = value
        return value

    def save(self, name: str, relative: str, payload: dict, schema: str) -> dict:
        return self.add(name, write_json(self.output / relative, payload), schema)

    def finish_stage(self, name: str, counts: dict, reasons: list[str] | None = None, program_status: str = 'succeeded', evidence_kind: str = 'real_source_annotation'):
        value = {'program_status': program_status, 'evidence_kind': evidence_kind,
                 'artifact_integrity': 'verified', 'review_status': 'not_accepted_by_this_run',
                 'reason_codes': reasons or [], 'counts': counts}
        self.status[name] = value
        record = {'schema_version': 'learning.stage5.module_manifest.v1', 'module': name,
                  'episode_id': self.episode_id, 'source_key': self.key, 'run_id': self.run_id,
                  'status': value, 'recorded_at': utc_now(),
                  'input_refs': {k:v for k,v in self.artifacts.items() if not k.startswith('manifests/')},
                  'code_hashes': code_fingerprint()}
        self.save('manifests/' + name, 'manifests/' + name + '.json', record, record['schema_version'])
        self.records.append({'module': name, **value})

    def source(self) -> tuple[np.ndarray, np.ndarray, dict]:
        video, hdf = Path(self.sample['source_video']['path']), Path(self.sample['source_hdf5']['path'])
        verify_ref(self.sample['source_video']); verify_ref(self.sample['source_hdf5'])
        self.video_ref = self.add('refs/source_video', video, producer='egodex_original')
        self.hdf_ref = self.add('refs/source_hdf5', hdf, producer='egodex_original')
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
               '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,time_base:frame=best_effort_timestamp_time,pkt_duration_time,duration_time', '-of', 'json', str(video)]
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(process.stdout)
        frames = probe['frames']; stream = probe['streams'][0]
        if not frames or any('best_effort_timestamp_time' not in frame for frame in frames):
            raise ValueError('S5_TIMEBASE_MISSING: original PTS required')
        ts = np.asarray([float(f['best_effort_timestamp_time']) for f in frames], np.float64)
        n = len(ts)
        if n != self.sample['num_frames'] or not np.isfinite(ts).all() or not np.all(np.diff(ts) > 0):
            raise ValueError('S5_TIMEBASE_MISMATCH: unexpected count or nonmonotonic PTS')
        fraction = stream.get('avg_frame_rate') or stream['r_frame_rate']
        numerator, denominator = map(float, fraction.split('/')); fps = numerator / denominator
        last_duration = frames[-1].get('duration_time') or frames[-1].get('pkt_duration_time')
        if last_duration is None or float(last_duration) <= 0:
            raise ValueError('S5_TIMEBASE_MISSING: last frame duration unavailable')
        edges = np.r_[ts, ts[-1] + float(last_duration)]
        with h5py.File(hdf, 'r') as fh:
            if 'transforms' not in fh or 'camera/intrinsic' not in fh:
                raise ValueError('S5_SOURCE_SCHEMA_MISMATCH: not EgoDex Path A')
            for joint, array in fh['transforms'].items():
                if array.shape != (n, 4, 4):
                    raise ValueError(f'S5_SOURCE_LENGTH_MISMATCH: {joint} {array.shape}')
            K = np.asarray(fh['camera/intrinsic'][()], float)
            if K.shape != (3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1]) <= 0:
                raise ValueError('S5_CAMERA_INVALID')
        if self.episode_root:
            old = read_json(self.episode_root / 'episode.json')
            if not np.allclose(old['timebase']['timestamps_s'], ts, atol=EPS_TIME, rtol=0):
                raise ValueError('S5_TIMEBASE_MISMATCH: episode manifest PTS')
            if old['source']['sha256'] != self.video_ref['sha256'] or old['source']['hdf5_sha256'] != self.hdf_ref['sha256']:
                raise ValueError('S5_SOURCE_HASH_MISMATCH: episode source')
        source_records = []
        capture = cv2.VideoCapture(str(video))
        try:
            for i in range(n):
                ok, bgr = capture.read()
                if not ok or bgr.shape[:2] != (stream['height'], stream['width']):
                    raise ValueError(f'S5_SOURCE_DECODE_FAILED: frame={i}')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                source_records.append({'frame_index': i, 'timestamp_s': float(ts[i]), 'image_kind': 'source_rgb',
                    'path': str(video), 'sha256': self.video_ref['sha256'], 'storage': 'video_frame',
                    'bytes': self.video_ref['bytes'],
                    'decoded_frame_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                    'width': stream['width'], 'height': stream['height'], 'color_space': 'RGB',
                    'valid': True, 'preview_only': False, 'evidence_kind': 'real_source_annotation',
                    'source_key': self.key, 'producer': 'egodex_original'})
            if capture.read()[0]:
                raise ValueError('S5_SOURCE_LENGTH_MISMATCH: extra decoded frames')
        finally:
            capture.release()
        path = write_npz(self.output / 'timebase.npz', frame_index=np.arange(n, dtype=np.int64),
                         timestamp_s=ts, frame_edge_s=edges, source_frame_valid=np.ones(n, bool),
                         __schema_version__=np.asarray('learning.stage5.timebase.v1'))
        self.timebase_ref = self.add('timebase', path, 'learning.stage5.timebase.v1')
        self.save('source/refs', 'source/refs.json', {'schema_version': 'learning.stage5.source_refs.v1',
                  'source_key': self.key, 'video': self.video_ref, 'hdf5': self.hdf_ref,
                  'source_dataset': 'egodex', 'source_license': 'inherited_from_egodex_manifest',
                  'publishable': False, 'probe_command': cmd, 'timestamp_source': 'ffprobe_best_effort_timestamp_time',
                  'nominal_fps': fps, 'frame_edge_source': 'pts_and_final_packet_duration',
                  'K': K.tolist(), 'image_width': stream['width'], 'image_height': stream['height']}, 'learning.stage5.source_refs.v1')
        self.finish_stage('source', {'frames': n, 'decoded_frames': n, 'source_pairs': 1})
        return ts, edges, {'source_records': source_records, 'fps': fps, 'stream': stream}

    def admission(self) -> dict:
        if not self.sample.get('stage1_report') or not self.sample.get('stage1_config'):
            raise ValueError('S5_EXPLICIT_ADMISSION_INPUT_REQUIRED: provide stage1_report and stage1_config in the sample')
        p = Path(self.sample['stage1_report']).expanduser().resolve()
        report = read_json(p)
        original = next((row for row in report['episodes'] if row['episode_id'] == self.episode_id), None)
        payload = {'schema_version': 'learning.stage5.admission.v1', 'source_key': self.key,
                   'status': original['status'] if original else 'not_evaluated', 'historical_record': original,
                   'historical_report_ref': self.add('inputs/stage1_report', p, producer='stage1_historical'),
                   'threshold_config_ref': self.add('inputs/stage1_config', Path(self.sample['stage1_config']).expanduser().resolve(), producer='stage1_historical'),
                   'source_video_sha256': self.video_ref['sha256'], 'source_hdf5_sha256': self.hdf_ref['sha256'],
                   'reason_codes': [] if original else ['S5_ADMISSION_UNREVIEWED'],
                   'note': 'Historical result reused; no new manual visibility labels or threshold calibration.'}
        report_ref = self.save('admission/report', 'admission/stage1.json', payload, payload['schema_version'])
        self.finish_stage('admission', {'pass': int(payload['status']=='pass'), 'not_evaluated': int(original is None)}, payload['reason_codes'], program_status='succeeded' if original else 'not_evaluated')
        return {**payload, 'report_ref': report_ref}

    def stage2(self, ts: np.ndarray) -> dict:
        masks, track = Path(self.sample['stage2_masks']), Path(self.sample['stage2_track'])
        if sha256(masks) != self.sample['stage2_masks_sha256'] or sha256(track) != self.sample['stage2_track_sha256']:
            raise ValueError('S5_SOURCE_HASH_MISMATCH: selected Stage2 version')
        with np.load(masks, allow_pickle=False) as data:
            mm = data['masks']
            if mm.dtype != bool or len(mm) != len(ts) or not np.array_equal(data['frame_indices'], np.arange(len(ts))):
                raise ValueError('S5_STAGE2_MASK_SCHEMA_MISMATCH')
            present = mm.reshape(len(ts), -1).any(axis=1)
            shape = list(mm.shape)
        stage2_root = masks.parent.parent if self.episode_id == 'egodex_pour_000' else masks.parent
        refs = {'object_masks_ref': self.add('stage2/object_masks', masks, 'learning.stage2.object_masks.legacy', 'stage2_historical'),
                'object_track_ref': self.add('stage2/object_track', track, producer='stage2_historical')}
        for name in ['hand_pose', 'features', 'hoi_candidates']:
            refs[name + '_ref'] = self.add('stage2/' + name, stage2_root / (name + '.csv'), producer='stage2_historical')
        feature_rows, hoi_rows, track_rows = _csv(stage2_root / 'features.csv'), _csv(stage2_root / 'hoi_candidates.csv'), _csv(track)
        for rows in [feature_rows, hoi_rows, track_rows]:
            if len(rows) != len(ts) or [int(r['frame_index']) for r in rows] != list(range(len(ts))):
                raise ValueError('S5_TIMEBASE_MISMATCH: Stage2 CSV count/index')
        error = max(abs(float(row['timestamp_s']) - ts[i]) for i,row in enumerate(feature_rows))
        if error > EPS_TIME:
            raise ValueError('S5_TIMEBASE_MISMATCH: Stage2 producer clock')
        from collections import Counter
        payload = {'schema_version': 'learning.stage5.stage2_evidence.v1', 'source_key': self.key,
                   'mask_kind': 'object', 'mask_shape': shape, 'nonempty_frames': int(present.sum()),
                   'measurement_valid': present.tolist(), 'producer_timebase': 'episode_pts' if 'pour' in self.episode_id else 'frame_index/nominal_fps',
                   'canonical_timebase_ref': self.timebase_ref, 'max_producer_time_error_s': error,
                   'candidate_state_counts': dict(Counter(row['candidate_state'] for row in hoi_rows)),
                   'track_ids': sorted({int(row['track_id']) for row in track_rows}),
                   'hoi_semantics': 'candidate_only_not_physical_fact', **refs}
        result_ref = self.save('stage2/evidence', 'evidence/stage2.json', payload, payload['schema_version'])
        self.finish_stage('stage2', {'frames': len(ts), 'nonempty_masks': int(present.sum()), 'missing_masks': int((~present).sum())},
                          ['S5_OBJECT_TRACK_MISSING'] if not present.all() else [],
                          program_status='partial' if not present.all() else 'succeeded', evidence_kind='real_model')
        return {**payload, 'evidence_ref': result_ref}

    def stage3(self, ts: np.ndarray, edges: np.ndarray, s2: dict) -> dict:
        annotation_path = self.sample.get('stage3_annotation_av')
        if self.sample.get('stage3_annotation_v1') and not annotation_path:
            raise ValueError('S5_STAGE3_LEGACY_RETIRED: use stage3_annotation_av from pipeline.annotation')
        a = None
        av_provenance = None
        schema_ref = self.add('stage3/schema', ANNOTATION_SCHEMA, producer='stage3_schema_v1')
        if annotation_path:
            source_path = Path(annotation_path)
            if sha256(source_path) != self.sample['stage3_sha256']:
                raise ValueError('S5_SOURCE_HASH_MISMATCH: Stage3 annotation')
            from .annotation.consumer import load_for_stage5
            a, av_provenance, av_refs = load_for_stage5(source_path, self.sample['stage3_sha256'],
                self.episode_id, self.video_ref['sha256'], len(ts))
            for i, evidence in enumerate(av_refs):
                self.add(f'stage3/av_evidence_{i}', evidence, producer='annotation_av')
            if a['episode_id'] != self.episode_id or a['frame_count'] != len(ts):
                raise ValueError('S5_STAGE3_SOURCE_MISMATCH')
            source_manifest = read_json(source_path.parents[2] / 'manifest.json')
            manifest_row = next(row for row in source_manifest['episodes'] if row['episode_id']==self.episode_id)
            if manifest_row['source_video_sha256'] != self.video_ref['sha256']:
                raise ValueError('S5_STAGE3_SOURCE_HASH_MISMATCH')
            self.add('stage3/run_manifest', source_path.parents[2] / 'manifest.json', producer='annotation_av')
            destination = self.output / 'annotations/stage3-v1.json'; destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source_path.read_bytes())
            annotation_ref = self.add('stage3/annotation_copy', destination, 'stage3.v1', av_provenance['variant'])
            self.add('stage3/original', source_path, 'stage3.v1', av_provenance['variant'])
        else:
            annotation_ref = None
        steps = []
        for step in a['steps'] if a else []:
            lo, hi = step['start_frame'], step['end_frame_exclusive']
            if not 0 <= lo < hi <= len(ts):
                raise ValueError('S5_STAGE3_INTERVAL_INVALID')
            steps.append({**step, 'start_s': float(edges[lo]), 'end_s': float(edges[hi]),
                          'robot_hand_coverage': 'full' if step['acting_hand'] in {'right','none'} else 'partial'})
        frame_map = [{'frame_index':i,'timestamp_s':float(ts[i]), 'step_ids':[s['step_id'] for s in steps if s['start_frame']<=i<s['end_frame_exclusive']]} for i in range(len(ts))]
        step_ref = self.save('stage3/step_map','annotations/step-frame-map.json',
                             {'schema_version':'learning.stage5.step_frame_map.v1','source_key':self.key,
                              'availability':'available' if a else 'missing','missing_reason':None if a else 'S5_STAGE3_MISSING',
                              'steps':steps,'frames':frame_map}, 'learning.stage5.step_frame_map.v1')
        associations = [{'vlm_object_id':o['id'],'stage2_track_id':None,'status':'unverified',
                         'evidence_ref':s2['evidence_ref'], 'note':'No cross-modal identity verification; no automatic name matching.'} for o in a['objects']] if a else []
        assoc_ref = self.save('stage3/associations','annotations/object-associations.json',
                              {'schema_version':'learning.stage5.object_associations.v1','source_key':self.key,'associations':associations}, 'learning.stage5.object_associations.v1')
        with h5py.File(self.hdf_ref['path'],'r') as hdf:
            attributes = plain(dict(hdf.attrs))
        reasons, difference_evidence = annotation_diagnostics(attributes,a)
        supervision = build_supervision(self.supervision_profile,attributes,self.hdf_ref,a)
        supervision_ref = self.save('annotations/supervision','annotations/supervision.json',
                                   {'source_key':self.key,**supervision},supervision['schema_version'])
        self.finish_stage('stage3', {'annotations':int(a is not None),'steps':len(steps),'mapped_frames':sum(bool(f['step_ids']) for f in frame_map)}, reasons,
                          program_status='succeeded' if a else 'missing', evidence_kind='real_model' if a else 'unavailable')
        return {'availability':'available' if a else 'missing','stage3_ref':annotation_ref,
                'annotation_schema_ref':schema_ref,'step_map_ref':step_ref,'object_associations_ref':assoc_ref,
                'method':a['method'] if a else None,'algorithm_variant':av_provenance['variant'] if av_provenance else None,
                'auto_qa':a['auto_qa'] if a else None,'review':a['review'] if a else None,
                'reason_codes':reasons,'annotation':a,'inventory_difference_evidence':difference_evidence,
                'supervision_profile':self.supervision_profile,'supervision_ref':supervision_ref,
                'supervision_blocking_reasons':supervision['blocking_reason_codes']}

    def geometry(self, ts: np.ndarray) -> dict:
        refs = {}
        for logical, relative, version in [('hand','hand/sequence.npz','ego2robot.hand.v1'),('canonical','canonical/actions.npz','ego2robot.canonical_action.v1'),('robot','robot/franka_panda/trajectory.npz','ego2robot.robot_trajectory.v1')]:
            p = self.episode_root / relative if self.episode_root else None
            if p is None or not p.is_file():
                refs[logical] = _missing('S5_' + logical.upper() + '_MISSING'); continue
            if schema_name(p) != version: raise ValueError(f'S5_SCHEMA_MISMATCH: {p}')
            with np.load(p,allow_pickle=False) as arrays:
                if not np.array_equal(arrays['frame_index'], np.arange(len(ts))) or not np.allclose(arrays['timestamp_s'],ts,atol=EPS_TIME,rtol=0):
                    raise ValueError(f'S5_TIMEBASE_MISMATCH: {p}')
            refs[logical] = self.add('refs/'+logical,p,version,'ego2robot_historical')
        if self.episode_root:
            for logical,relative in [('episode','episode.json'),('canonical_delta','canonical/actions-delta.npz'),('base','robot/franka_panda/base.json'),('hand_provenance','hand/provenance.json'),('resolved_config','configs/resolved.json')]:
                p = self.episode_root / relative
                if p.is_file(): refs[logical] = self.add('refs/'+logical,p,producer='ego2robot_historical')
        self.finish_stage('geometry',{'hand_frames':len(ts) if refs['hand'].get('path') else 0,
                         'canonical_frames':len(ts) if refs['canonical'].get('path') else 0,
                         'robot_frames':len(ts) if refs['robot'].get('path') else 0},
                         [r['reason'] for r in refs.values() if r.get('availability')=='missing'],
                         program_status='succeeded' if refs['robot'].get('path') else 'missing',evidence_kind='numeric_solver' if refs['robot'].get('path') else 'unavailable')
        return refs

    def build(self) -> dict:
        self.output.mkdir(parents=True,exist_ok=True)
        ts, edges, source_info = self.source()
        admission = self.admission(); s2 = self.stage2(ts); annotations = self.stage3(ts,edges,s2); geometry = self.geometry(ts)
        from .visual import prepare_visual
        visual_kwargs = {'external_artifacts_manifest':self.external_artifacts_manifest} if self.external_artifacts_manifest else {}
        visual_result = prepare_visual(episode_root=self.episode_root,output_dir=self.output,timestamps=ts,
                                       source_video_sha256=self.video_ref['sha256'],source_key=self.key,
                                       scene_depth_manifest=self.scene_manifest,**visual_kwargs)
        index_ref = visual_result.get('index_ref')
        index_path = Path(index_ref['path']) if index_ref else Path(visual_result.get('index_path', self.output/'visual/index.json'))
        visual_index = read_json(index_path)
        visual_index['frames'] = source_info['source_records'] + [record for record in visual_index.get('frames',[]) if record['image_kind']!='source_rgb']
        visual_index.update(source_video_ref=self.video_ref,source_key=self.key)
        visual_index['preferred_kind'] = 'depth_composite_rgb' if visual_index.get('training_ready') else 'source_rgb'
        write_json(index_path,visual_index)
        visual_ref = self.add('visual/index',index_path,visual_index.get('schema_version','learning.stage5.visual_index.v1'))
        for i,value in enumerate(visual_index.get('input_refs',[])):
            if value.get('path'):
                self.add(f'visual/input/{i}',Path(value['path']),producer='visual_input')
        if visual_index.get('scale_alignment_ref'):
            self.add('visual/alignment',Path(visual_index['scale_alignment_ref']['path']),'learning.stage5.depth_alignment.v1')
        for i,record in enumerate(visual_index['frames']):
            if record.get('storage')=='video_frame': continue
            path = record.get('path')
            if path and Path(path).is_file():
                v = self.add(f'visual/frame/{i}',Path(path),producer=record.get('producer','historical_visual'))
                if record.get('sha256') and record['sha256'] != v['sha256']: raise ValueError('S5_VISUAL_HASH_MISMATCH')
        self.finish_stage('visual',{'source_rgb_frames':len(ts),'indexed_frames':len(visual_index['frames']),
                          'formal_composite_frames':sum(f['image_kind']=='depth_composite_rgb' and not f.get('preview_only',True) for f in visual_index['frames'])},
                          visual_result.get('reasons',visual_index.get('reason_codes',[])),
                          program_status='succeeded' if visual_index.get('training_ready') else 'partial',evidence_kind='real_model' if self.episode_root else 'unavailable')
        from .quality import build_quality_and_actions
        qr = build_quality_and_actions(episode_root=self.episode_root,source_hdf5=Path(self.hdf_ref['path']),timestamp_s=ts,
                                      source_frame_valid=np.ones(len(ts),bool),source_key=self.key,output_dir=self.output,
                                      visual_index=visual_index,admission=admission,annotations=annotations)
        if self.statistics_path is not None:
            qr = self.apply_l2(qr,ts,geometry)
        quality_ref = self.add('quality/frames',Path(qr['quality_ref']['path']),'learning.stage5.quality_frames.v1')
        report_path = qr.get('quality_report_ref',{}).get('path') or self.output/'quality/report.json'
        quality_report_ref = self.add('quality/report',Path(report_path),'learning.stage5.quality_report.v1')
        transition_ref = self.add('views/transitions',Path(qr['transition_ref']['path']),'learning.stage5.transitions.v1')
        if qr.get('camera_motion_ref'):
            self.add('quality/camera_motion',Path(qr['camera_motion_ref']['path']),'learning.stage5.camera_motion.v1')
        quality_report = read_json(Path(report_path))
        self.finish_stage('quality',qr.get('counts',{}),quality_report.get('reason_codes',[]),evidence_kind='numeric_solver')
        with np.load(transition_ref['path'],allow_pickle=False) as t:
            training_transitions = int(np.count_nonzero(t['transition_valid']))
            finite_actions = int(np.isfinite(t['action7']).all(axis=1).sum())
        readiness_reasons = sorted(set(annotations['supervision_blocking_reasons'] + ([] if admission['status']=='pass' else ['S5_ADMISSION_UNREVIEWED']) +
            ([] if visual_index.get('training_ready') else ['S5_VISUAL_NOT_READY']) +
            ([] if training_transitions else ['S5_NO_TRAINING_TRANSITIONS']) +
            ([] if geometry['robot'].get('path') else ['S5_ROBOT_MISSING'])))
        if self.execution_manifest and read_json(self.execution_manifest).get('status') in {'failed','requires_manual_runtime_cleanup'}:
            readiness_reasons.append('S5_EXECUTION_INCOMPLETE')
        if self.statistics_path is not None and quality_report['rule_coverage']['L2'].get('status')!='executed':
            readiness_reasons.append('S5_L2_STATS_NOT_READY')
        # No user acceptance is fabricated by packaging. Formal output requires explicit, scoped review evidence.
        ready = not readiness_reasons and quality_report.get('training_readiness')=='training_ready'
        self.finish_stage('package',{'frames':len(ts),'transitions':len(ts)-1,'finite_action_transitions':finite_actions,'training_transitions':training_transitions},readiness_reasons)
        bundle = {'schema_version':'learning.stage5.bundle.v1','episode_id':self.episode_id,'run_id':self.run_id,
                  'source_key':self.key,'profile':'path_a_right_franka_v1','supervision_profile':self.supervision_profile,'hand_selection':'right','robot_id':'franka_panda',
                  'storage_mode':'local_reference','training_readiness':'training_ready' if ready else 'candidate_only',
                  'readiness_reasons':readiness_reasons,'source_license':'inherited_from_egodex_manifest','source_publishable':False,
                  'refs':{'source_video':self.video_ref,'source_hdf5':self.hdf_ref,**geometry},
                  'timebase':self.timebase_ref,'admission':{'status':admission['status'],'report_ref':admission['report_ref']},
                  'stage2':{'evidence_ref':s2['evidence_ref']},'annotations':{k:v for k,v in annotations.items() if k!='annotation'},
                  'visual':{'index_ref':visual_ref,'preferred_kind':visual_index['preferred_kind'],'training_ready':bool(visual_index.get('training_ready'))},
                  'quality':{'frames_ref':quality_ref,'report_ref':quality_report_ref},
                  'views':{'transitions_ref':transition_ref,'action_profile':qr.get('action_profile','canonical_delta_C_target_width_m.v1')},
                  'artifacts':self.artifacts,'module_status':self.status,
                  'holds':{'stage4_dedup':'skipped_by_user_hold','stage4_rebalance':'skipped_by_user_hold','stage3_new_vlm':'hold','stage3_manual_correction':'hold'},
                  'provenance':{'created_at':utc_now(),'code_hashes':code_fingerprint(),'sample':self.sample,
                                'source_snapshot':'20261019_interface_audit','new_model_calls':None if self.execution_manifest else 0,'new_ik_runs':0,
                                'execution_manifest_ref':self.add('execution/manifest',self.execution_manifest) if self.execution_manifest else None},
                  'counts':{'frames':len(ts),'transitions':len(ts)-1,'finite_action_transitions':finite_actions,
                            'qc_valid_transitions':training_transitions,'training_transitions':training_transitions if ready else 0},
                  'bundle_schema_ref':ref(SCHEMA,schema_version='https://json-schema.org/draft/2020-12/schema')}
        jsonschema.Draft202012Validator(read_json(SCHEMA)).validate(bundle)
        write_json(self.output/'bundle.json',bundle)
        return bundle

    def apply_l2(self, qr: dict, ts: np.ndarray, geometry: dict) -> dict:
        """Optional frozen offline thresholds; never fit or relax them online."""
        from .statistics import apply_statistics
        stats_ref=self.add('quality/dataset_statistics',self.statistics_path,'learning.stage5.dataset_statistics.v1','offline_statistics')
        with np.load(qr['transition_ref']['path'],allow_pickle=False) as a:transitions={k:a[k].copy() for k in a.files}
        with np.load(qr['quality_ref']['path'],allow_pickle=False) as a:frames={k:a[k].copy() for k in a.files}
        report=read_json(qr['quality_report_ref']['path'])
        if geometry['robot'].get('path') and geometry['canonical'].get('path'):
            with np.load(geometry['robot']['path'],allow_pickle=False) as a:robot={k:a[k].copy() for k in a.files}
            provenance=read_json(Path(geometry['hand_provenance']['path']))
            result=apply_statistics(timestamp_s=ts,action7=transitions['action7'],geometry_transition_valid=transitions['geometry_candidate_valid'],
                                    bound_width_m=robot['gripper_qpos'].sum(axis=1),qpos=robot['qpos'],
                                    position_error_m=robot['position_error_m'][:,0],rotation_error_rad=robot['rotation_error_rad'][:,0],
                                    stats=self.statistics_path,source_dataset='egodex',source=provenance['provider'])
            l2_report=result['report'];l2_valid=result['transition_valid'];l2_frame=result['frame_valid'];l2_bits=result['reason_bits']
        else:
            l2_report={'status':'not_evaluated','reason':'missing_geometry','statistics_ref':stats_ref}
            l2_valid=np.zeros(len(ts)-1,bool);l2_frame=np.zeros(len(ts),bool);l2_bits=np.zeros(len(ts)-1,np.uint64)
        transitions['l2_valid']=l2_valid;transitions['l2_reason_bits']=l2_bits
        transitions['transition_valid'] &= l2_valid
        frames['l2_frame_valid']=l2_frame;frames['qc_frame_valid'] &= l2_frame
        write_npz(Path(qr['transition_ref']['path']),**transitions);write_npz(Path(qr['quality_ref']['path']),**frames)
        l2_report['statistics_ref']=stats_ref
        l2_ref=self.save('quality/l2_report','quality/l2.json',l2_report,'learning.stage5.l2_application.v1')
        report['rule_coverage']['L2']={**l2_report,'application_ref':l2_ref}
        report['counts']['training_transitions']=int(transitions['transition_valid'].sum())
        report['training_ready']=bool(transitions['transition_valid'].any()) and report.get('training_ready',False)
        report['training_readiness']='training_ready' if report['training_ready'] else 'candidate_only'
        write_json(Path(qr['quality_report_ref']['path']),report)
        qr.update(quality_ref=ref(qr['quality_ref']['path'],schema_version='learning.stage5.quality_frames.v1',source_key=self.key),
                  transition_ref=ref(qr['transition_ref']['path'],schema_version='learning.stage5.transitions.v1',source_key=self.key),
                  quality_report_ref=ref(qr['quality_report_ref']['path'],schema_version='learning.stage5.quality_report.v1',source_key=self.key),
                  report=report,counts=report['counts'])
        return qr


def _cache_valid(directory: Path, fingerprint: str) -> tuple[bool,str]:
    p = directory/'run-state.json'
    if not p.is_file(): return False,'no_previous_run'
    try:
        state = read_json(p)
        if state.get('fingerprint') != fingerprint or state.get('status')!='succeeded': return False,'input_or_code_changed'
        for v in state['artifacts'].values(): verify_ref(v)
        bundle=read_json(directory/'bundle.json')
        for v in bundle['artifacts'].values(): verify_ref(v)
        return True,'all_hashes_match'
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as exc:
        return False,str(exc)


def _within_new_output(path: Path, run_output: Path) -> Path:
    resolved = path.resolve()
    allowed = [(ROOT/'outputs/stage5/runs').resolve(),
               (ROOT/'outputs/stage5/completion').resolve()]
    if not any(root in resolved.parents for root in allowed):
        raise ValueError('S5_EXECUTION_OUTPUT_OUTSIDE_NEW_TREE: '+str(resolved))
    return resolved


def _uses_wsl_dispatcher(command: list[str]) -> bool:
    dispatcher = any(arg == 'pipeline.model_process' or Path(arg).name == 'model_process.py' for arg in command)
    wsl = '--runtime=wsl' in command or any(arg == '--runtime' and command[i+1:i+2] == ['wsl']
                                          for i,arg in enumerate(command))
    return dispatcher and wsl


def _terminate_worker_tree(process: subprocess.Popen, *, wsl: bool) -> dict:
    """Best-effort host process-tree cleanup; never certify WSL Linux exit."""
    cleanup={'worker_pid':process.pid,'platform':os.name,'started_at':utc_now(),
             'linux_runtime_exit_verified':False if wsl else None,'steps':[]}
    if os.name == 'nt':
        command=['taskkill','/PID',str(process.pid),'/T','/F']
        try:
            result=subprocess.run(command,shell=False,capture_output=True,text=True,errors='replace',timeout=30,check=False)
            cleanup['steps'].append({'command':command,'return_code':result.returncode,
                                     'stdout':result.stdout[-4000:],'stderr':result.stderr[-4000:]})
        except Exception as exc:
            cleanup['steps'].append({'command':command,'error':str(exc),'error_type':type(exc).__name__})
    else:
        # Popen starts a new session, so its PID is also the process-group ID.
        # SIGKILL follows SIGTERM even if the parent exits before a descendant.
        for sig in (signal.SIGTERM,signal.SIGKILL):
            try:
                os.killpg(process.pid,sig)
                cleanup['steps'].append({'process_group':process.pid,'signal':sig.name,'sent':True})
            except ProcessLookupError:
                cleanup['steps'].append({'process_group':process.pid,'signal':sig.name,'already_absent':True})
                break
            except OSError as exc:
                cleanup['steps'].append({'process_group':process.pid,'signal':sig.name,'error':str(exc)})
            if sig == signal.SIGTERM:
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
    try:
        cleanup['parent_return_code']=process.wait(timeout=10)
        cleanup['parent_exited']=True
    except subprocess.TimeoutExpired:
        cleanup['parent_exited']=False
    cleanup['requires_manual_runtime_cleanup']=bool(wsl or not cleanup['parent_exited'])
    if any('error' in step for step in cleanup['steps']):
        cleanup['requires_manual_runtime_cleanup']=True
    if os.name == 'nt' and (not cleanup['steps'] or cleanup['steps'][0].get('return_code')!=0):
        cleanup['requires_manual_runtime_cleanup']=True
    cleanup['completed_at']=utc_now()
    return cleanup


def _runtime_cleanup_confirmation(stage: dict, previous: dict) -> dict | None:
    """Accept explicit, hash-backed runtime cleanup facts, never a visual gate."""
    reference=stage.get('runtime_cleanup_confirmation_ref')
    if not reference:return None
    verify_ref(reference)
    confirmation=read_json(reference['path'])
    required={'schema_version':'learning.stage5.runtime_cleanup_confirmation.v1',
              'status':'runtime_cleanup_confirmed','stage_name':previous['name'],
              'timed_out_fingerprint':previous['fingerprint'],'worker_pid':previous.get('worker_pid'),
              'runtime':'wsl' if previous.get('runtime_is_wsl') else 'native'}
    if any(confirmation.get(key)!=value for key,value in required.items()):
        raise ValueError('S5_RUNTIME_CLEANUP_CONFIRMATION_MISMATCH')
    evidence=confirmation.get('evidence_refs',[])
    if not confirmation.get('confirmed_at') or not confirmation.get('confirmed_by') or not evidence:
        raise ValueError('S5_RUNTIME_CLEANUP_CONFIRMATION_REQUIRES_EXPLICIT_EVIDENCE')
    for value in evidence:verify_ref(value)
    return reference


def execute_missing(sample: dict, directory: Path, configuration: dict | None,
                    configuration_ref: dict | None, *, resume: bool = True) -> dict:
    """Run only explicitly declared, bounded worker commands without a shell.

    This orchestrates separate worker processes. Worker configs retain their
    own model/environment/gate checks. A worker exit code never grants review.
    Failed/pending execution is persisted and the caller still packages all
    independently available evidence.
    """
    base = directory/'execution'
    base.mkdir(parents=True,exist_ok=True)
    payload = {'schema_version':'learning.stage5.execution_manifest.v1','episode_id':sample['episode_id'],
               'source_key':source_key(sample['source_video']['sha256'],sample['source_hdf5']['sha256']),
               'configuration_ref':configuration_ref,'created_at':utc_now(),'stages':[],
               'review_effect':'none; execution cannot grant M3/M5/M6 acceptance'}
    episode = (configuration or {}).get('episodes',{}).get(sample['episode_id'])
    if not episode:
        payload.update(status='not_configured',reason='No worker configured; available artifacts are still packaged.')
    else:
        if sample['episode_id']!='egodex_vertical_pick_place_000' or sample['num_frames']>52:
            raise ValueError('S5_EXECUTION_SCOPE: model execution is limited to the 52-frame episode 000')
        limit = episode.get('max_frames')
        if not isinstance(limit,int) or isinstance(limit,bool) or not sample['num_frames']<=limit<=52:
            raise ValueError('S5_EXECUTION_SCOPE: explicitly declare max_frames <= 52 covering the selected episode')
        if episode.get('allow_download',False):
            raise ValueError('S5_EXECUTION_DOWNLOAD_UNDECLARED: install models explicitly outside this runner')
        names = set()
        for stage in episode.get('stages',[]):
            name = stage.get('name','')
            if not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}',name) or name in names:
                raise ValueError('Invalid/duplicate execution stage name')
            names.add(name)
            command = stage.get('command')
            if not isinstance(command,list) or len(command)<2 or not all(isinstance(v,str) and v for v in command):
                raise ValueError('Execution command must be a nonempty argv list')
            if Path(command[0]).name.lower() not in {'python.exe','python','python3','python3.exe'} or '-c' in command:
                raise ValueError('Worker execution requires an explicit Python file/module command, no shell or inline code')
            stage_output = _within_new_output(Path(stage['output_dir']),directory)
            expected=[]
            for item in stage.get('expected_artifacts',[]):
                expected_path=Path(item['path'] if isinstance(item,dict) else item)
                path=_within_new_output(expected_path if expected_path.is_absolute() else stage_output/expected_path,directory)
                if path!=stage_output and stage_output not in path.parents:
                    raise ValueError('Expected worker artifacts must stay under declared output_dir')
                expected.append(path)
            if not expected:
                raise ValueError('Worker must declare at least one expected artifact')
            inputs=stage.get('input_refs',[])
            for value in inputs:verify_ref(value)
            state_path=base/(name+'.json')
            fingerprint=stable_hash({'stage':stage,'source':payload['source_key'],'code':code_fingerprint()})
            cached=False
            old=read_json(state_path) if state_path.is_file() else None
            cleanup_confirmation=None
            if old and (old.get('requires_manual_runtime_cleanup') or old.get('status')=='requires_manual_runtime_cleanup'):
                cleanup_confirmation=_runtime_cleanup_confirmation(stage,old)
                if cleanup_confirmation is None:
                    payload['stages'].append({**old,'status':'requires_manual_runtime_cleanup',
                        'resume_blocked':True,'cache_hit':False,
                        'resume_blocked_reason':'The previous runtime may still be running. Supply hash-backed runtime_cleanup_confirmation_ref or an explicitly new stage name; --no-resume does not clear this state.'})
                    break
            if resume and old:
                if old.get('fingerprint')==fingerprint and old.get('status')=='succeeded':
                    try:
                        for value in old['output_refs']:verify_ref(value)
                        cached=True
                    except (OSError,ValueError):pass
            if cached:
                record={**old,'cache_hit':True}
            else:
                stage_output.mkdir(parents=True,exist_ok=True)
                attempt_started=utc_now()
                attempt_token=stable_hash({'stage':name,'started_at':attempt_started,'parent_pid':os.getpid()})[:16]
                log_path=base/(name+'-'+attempt_token+'.log')
                record={'schema_version':'learning.stage5.execution_stage.v1','name':name,
                        'fingerprint':fingerprint,'command':command,'cwd':str(ROOT),
                        'output_dir':str(stage_output),'input_refs':inputs,'max_frames':limit,
                        'started_at':attempt_started,'status':'failed','cache_hit':False,'output_refs':[],
                        'process_started':False,'runtime_is_wsl':_uses_wsl_dispatcher(command),
                        'runtime_cleanup_confirmation_ref':cleanup_confirmation,
                        'prior_timeout_snapshot_ref':old.get('timeout_snapshot_ref') if old and cleanup_confirmation else None}
                process=None
                try:
                    timeout=stage.get('timeout_s',7200)
                    if not isinstance(timeout,(int,float)) or not 1<=timeout<=14400:
                        raise ValueError('Worker timeout_s must be 1..14400')
                    with log_path.open('w',encoding='utf-8',newline='\n') as stream:
                        options={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
                        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                                 shell=False,**options)
                        record.update(worker_pid=process.pid,process_started=True)
                        try:process.wait(timeout=timeout)
                        except subprocess.TimeoutExpired:
                            record['cleanup']=_terminate_worker_tree(process,wsl=record['runtime_is_wsl'])
                            record['requires_manual_runtime_cleanup']=record['cleanup']['requires_manual_runtime_cleanup']
                            if record['requires_manual_runtime_cleanup']:record['status']='requires_manual_runtime_cleanup'
                            raise
                    record['return_code']=process.returncode
                    if process.returncode!=0:raise ValueError('S5_WORKER_FAILED: nonzero worker exit code')
                    record['output_refs']=[ref(path) for path in expected]
                    for item,value in zip(stage['expected_artifacts'],record['output_refs']):
                        if isinstance(item,dict) and item.get('sha256') and item['sha256']!=value['sha256']:
                            raise ValueError('S5_WORKER_OUTPUT_HASH_MISMATCH')
                    record['status']='succeeded'
                except Exception as exc:
                    record.update(error=str(exc),error_type=type(exc).__name__)
                record['completed_at']=utc_now()
                if log_path.is_file():record['log_ref']=ref(log_path)
                if record.get('requires_manual_runtime_cleanup'):
                    snapshot=base/(name+'.timeout-'+str(record.get('worker_pid'))+'-'+stable_hash(record['started_at'])[:12]+'.json')
                    write_json(snapshot,record)
                    record['timeout_snapshot_ref']=ref(snapshot)
                write_json(state_path,record)
            payload['stages'].append(record)
            if record['status'] in {'failed','requires_manual_runtime_cleanup'}:break
        payload['status']=('requires_manual_runtime_cleanup' if any(s['status']=='requires_manual_runtime_cleanup' for s in payload['stages'])
                           else ('failed' if any(s['status']=='failed' for s in payload['stages']) else 'succeeded'))
    payload['worker_processes_executed']=sum(not s.get('cache_hit',False) and not s.get('resume_blocked',False)
                                           and s.get('process_started',True) for s in payload['stages'])
    payload['new_model_calls']=None if payload['worker_processes_executed'] else 0
    path=base/'manifest.json'
    def identity(value):
        return {'configuration_ref':value.get('configuration_ref'),'source_key':value['source_key'],'status':value['status'],
                'stages':[{key:stage.get(key) for key in ('name','fingerprint','status','output_refs','return_code')}
                          for stage in value['stages']]}
    # Keep the immutable execution evidence byte-for-byte stable on resume;
    # report this invocation's cache activity separately.
    if not path.is_file() or identity(read_json(path))!=identity(payload):
        write_json(path,payload)
    write_json(base/'latest-resume.json',{'created_at':utc_now(),'status':payload['status'],
               'worker_processes_executed':payload['worker_processes_executed'],
               'stage_cache_hits':{s['name']:s.get('cache_hit',False) for s in payload['stages']}})
    return {'manifest_path':path,'manifest':payload,'episode_configuration':episode or {}}


def run(samples_path: Path, output: Path, run_id: str, *, export: bool=True, scene_manifests: dict[str,str] | None=None, resume: bool=True, statistics_path: Path | None=None,
        supervision_profile: str=STRICT_STEPS, export_mode: str='inspect', mode: str='reuse-only',
        visual_artifacts: dict[str,str] | None=None, execution_config: Path | None=None, audit_inventory: Path | None=None) -> dict:
    selection=read_json(samples_path)
    if selection.get('schema_version')!='learning.stage5.audit_samples.v1' or not 1<=len(selection['samples'])<=3:
        raise ValueError('Only the fixed 1-3 audit samples are supported in this bounded integration profile')
    output=output.resolve()
    # All mutations are confined to a new Stage5 run tree, never raw/history/reproduction.
    _within_new_output(output,output)
    if mode not in {'reuse-only','execute-missing'}:raise ValueError('Unsupported run mode')
    if supervision_profile not in SUPERVISION_PROFILES:raise ValueError('Unsupported supervision profile')
    if export_mode not in {'inspect','train','auto'}:raise ValueError('Unsupported export mode')
    configuration=read_json(execution_config) if execution_config else None
    if configuration and configuration.get('schema_version')!='learning.stage5.execution_config.v1':
        raise ValueError('Invalid execution config schema')
    if mode=='reuse-only' and execution_config:raise ValueError('execution-config requires --mode execute-missing')
    output.mkdir(parents=True,exist_ok=True)
    if audit_inventory is None:
        raise ValueError('S5_EXPLICIT_INVENTORY_REQUIRED: provide --audit-inventory')
    audit=read_json(audit_inventory)
    summary={'schema_version':'learning.stage5.run_report.v1','run_id':run_id,'samples_ref':ref(samples_path),
             'mode':mode,'supervision_profile':supervision_profile,'export_mode':export_mode,
             'created_at':utc_now(),'episodes':[],'new_model_calls':None if mode=='execute-missing' else 0,'new_ik_runs':0,
             'command':[sys.executable,*sys.argv]}
    for sample in selection['samples']:
        directory=output/sample['episode_id']; directory.mkdir(parents=True,exist_ok=True)
        execution=None
        if mode=='execute-missing':
            try:
                execution=execute_missing(sample,directory,configuration,ref(execution_config) if execution_config else None,resume=resume)
            except Exception as exc:
                failure={'schema_version':'learning.stage5.execution_manifest.v1','episode_id':sample['episode_id'],
                         'source_key':source_key(sample['source_video']['sha256'],sample['source_hdf5']['sha256']),
                         'status':'failed','error':str(exc),'error_type':type(exc).__name__,'stages':[],
                         'created_at':utc_now(),'review_effect':'none; declaration rejected before unapproved execution'}
                execution={'manifest_path':write_json(directory/'execution/manifest.json',failure),
                           'manifest':failure,'episode_configuration':{}}
        declared=execution['episode_configuration'] if execution else {}
        scene=(scene_manifests or {}).get(sample['episode_id'],declared.get('scene_manifest'))
        external=(visual_artifacts or {}).get(sample['episode_id'],declared.get('visual_artifacts'))
        pending_inputs=[]
        if mode=='execute-missing':
            if scene and not Path(scene).is_file():pending_inputs.append({'kind':'scene_manifest','path':scene});scene=None
            if external and not Path(external).is_file():pending_inputs.append({'kind':'visual_artifacts','path':external});external=None
        fp=stable_hash({'sample':sample,'code':code_fingerprint(),'export':export,'scene':ref(Path(scene)) if scene else None,
                        'statistics':ref(statistics_path) if statistics_path else None,
                        'supervision_profile':supervision_profile,'export_mode':export_mode,'mode':mode,
                        'external_visual':ref(Path(external)) if external else None,
                        'execution':ref(execution['manifest_path']) if execution else None})
        hit,reason=_cache_valid(directory,fp) if resume else (False,'force_rebuild')
        if hit:
            b=read_json(directory/'bundle.json')
            summary['episodes'].append({'episode_id':sample['episode_id'],'status':'cached','cache_reason':reason,
                                      'bundle_path':str(directory/'bundle.json'),'training_readiness':b['training_readiness'],'counts':b['counts'],'readiness_reasons':b['readiness_reasons'],
                                      'export_status':'blocked' if export and export_mode=='train' and b['training_readiness']!='training_ready' else ('cached' if export else 'disabled')})
            print(json.dumps({'episode_id':sample['episode_id'],'status':'cached'},ensure_ascii=False),flush=True)
            continue
        try:
            builder=EpisodeBuilder(sample,directory,run_id,audit,Path(scene) if scene else None,statistics_path,
                                   supervision_profile,Path(external) if external else None,
                                   execution['manifest_path'] if execution else None)
            b=builder.build()
            from .reader import open_bundle
            with open_bundle(directory/'bundle.json',mode='inspect') as reader:
                evidence=reader.get_transition(min(5,b['counts']['transitions']-1))
            from .report import episode_report
            episode_report(directory,b,evidence)
            export_paths=[]; export_status='disabled'; actual_export_mode=None
            if export:
                from .export import export_bundle_hdf5, export_bundle_jsonl, resolve_export_mode
                from .reader import TrainingReadinessError
                h5path=directory/'export/episode.h5'; jsonlpath=directory/'export/frames.jsonl'
                actual_export_mode=resolve_export_mode(directory/'bundle.json',export_mode)
                try:
                    export_bundle_hdf5(directory/'bundle.json',h5path,mode=export_mode,overwrite=True)
                    export_bundle_jsonl(directory/'bundle.json',jsonlpath,mode=export_mode,overwrite=True)
                    export_paths=[str(h5path),str(jsonlpath)];export_status='succeeded'
                except TrainingReadinessError as exc:
                    export_status='blocked'
                    write_json(directory/'export/blocked.json',{'schema_version':'learning.stage5.export_blocked.v1',
                        'requested_mode':export_mode,'resolved_mode':actual_export_mode,'reason':str(exc),
                        'bundle_ref':ref(directory/'bundle.json'),'created_at':utc_now(),
                        'candidate_evidence_preserved':True,'training_ready':False})
            if (directory/'failure.json').is_file(): (directory/'failure.json').unlink()
            state={'schema_version':'learning.stage5.run_state.v1','fingerprint':fp,'status':'succeeded',
                   'artifacts':{str(p.relative_to(directory)):ref(p) for p in directory.rglob('*') if p.is_file() and p.name not in {'run-state.json','latest-resume.json'} and not p.name.endswith('.partial')},
                   'recovered_from':reason,'duration_s':round(time.perf_counter()-builder.started,3)}
            write_json(directory/'run-state.json',state)
            row={'episode_id':sample['episode_id'],'status':'succeeded','cache_reason':reason,'bundle_path':str(directory/'bundle.json'),
                 'training_readiness':b['training_readiness'],'counts':b['counts'],'readiness_reasons':b['readiness_reasons'],
                 'exports':export_paths,'export_status':export_status,'resolved_export_mode':actual_export_mode,
                 'pending_execution_inputs':pending_inputs,'execution_status':execution['manifest']['status'] if execution else 'not_requested',
                 'duration_s':state['duration_s']}
        except Exception as exc:
            failure={'schema_version':'learning.stage5.failure.v1','episode_id':sample['episode_id'],'status':'failed','type':type(exc).__name__,
                     'error':str(exc),'created_at':utc_now(),'reason_code':str(exc).split(':')[0]}
            write_json(directory/'failure.json',failure)
            row={**failure,'cache_reason':reason}
        summary['episodes'].append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    summary['counts']={'episodes':len(summary['episodes']),'succeeded':sum(r['status']=='succeeded' for r in summary['episodes']),
                       'cached':sum(r['status']=='cached' for r in summary['episodes']),'failed':sum(r['status']=='failed' for r in summary['episodes']),
                       'training_ready_episodes':sum(r.get('training_readiness')=='training_ready' for r in summary['episodes'])}
    summary['counts']['blocked_training_exports']=sum(r.get('export_status')=='blocked' for r in summary['episodes'])
    write_json(output/'run-report.json',summary)
    from .report import run_report
    run_report(output,summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run'); p.add_argument('--samples',type=Path,required=True); p.add_argument('--audit-inventory',type=Path,required=True)
    p.add_argument('--mode',choices=['reuse-only','execute-missing'],default='reuse-only')
    p.add_argument('--profile',choices=['path_a_right_franka_v1'],default='path_a_right_franka_v1')
    p.add_argument('--run-id',default='20261020_completion_v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene-manifests',type=Path); p.add_argument('--no-export',action='store_true'); p.add_argument('--no-resume',action='store_true')
    p.add_argument('--visual-artifacts',type=Path,help='JSON mapping episode_id to external visual-artifacts manifest path; immutable old episode is reused for geometry.')
    p.add_argument('--supervision-profile',choices=SUPERVISION_PROFILES,default=STRICT_STEPS,
                   help='Explicit egodex_source_task.v1 uses original HDF5 Task only; never creates Step labels.')
    p.add_argument('--export-mode',choices=['inspect','train','auto'],default='inspect',
                   help='auto chooses train only for a fully training_ready bundle; explicit train preserves blocked evidence.')
    p.add_argument('--execution-config',type=Path,help='Explicit bounded worker argv/config/expected-output manifest; execute-missing only, episode 000 <=52 frames.')
    p.add_argument('--statistics',type=Path,help='Optional frozen dataset statistics. No online fitting; not-ready profiles fail closed.')
    p=sub.add_parser('inspect');p.add_argument('bundle',type=Path);p.add_argument('--transition',type=int,default=0)
    args=parser.parse_args(argv)
    if args.command=='run':
        result=run(args.samples,args.output,args.run_id,export=not args.no_export,
                   scene_manifests=read_json(args.scene_manifests) if args.scene_manifests else None,resume=not args.no_resume,statistics_path=args.statistics,
                   supervision_profile=args.supervision_profile,export_mode=args.export_mode,mode=args.mode,
                   visual_artifacts=read_json(args.visual_artifacts) if args.visual_artifacts else None,execution_config=args.execution_config,audit_inventory=args.audit_inventory)
        raise SystemExit(1 if result['counts']['failed'] else (2 if result['counts']['blocked_training_exports'] else 0))
    from .reader import open_bundle
    result=open_bundle(args.bundle,mode='inspect').get_transition(args.transition)
    print(json.dumps({k:(list(v.shape) if isinstance(v,np.ndarray) else v) for k,v in result.items()},ensure_ascii=False,default=str,indent=2))


if __name__=='__main__': main()
