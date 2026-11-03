import copy
import json
from pathlib import Path
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pipeline.annotation.core import ROOT, cached, digest, file_hash, load_jobs, probe, project_v1, read, validate_segments, write
from pipeline.annotation.runner import execute_stage, prepare, verify
from pipeline.annotation.consumer import load_for_stage5

# A/B/C machinery is exercised with the historical all-variant rental profile; the default
# config.json is the single-route Chutes Qwen3.8 batch profile (see test_chutes_batch.py).
LOCAL_CONFIG = ROOT / 'configs/local_rental_qwen36.json'


class Backend:
    def __init__(self):
        self.calls = []

    def task(self, job, directory):
        self.calls.append(('task', job['id']))
        return dict(instruction='Manipulate the visible object.', task_source='fixture_generator', instruction_source='fixture')

    def segment(self, job, media, task, directory):
        self.calls.append(('segment', job['id']))
        return dict(segments=[dict(start_s=0., end_s=1.), dict(start_s=1., end_s=2.)])

    def caption(self, job, media, task, boundaries, directory):
        self.calls.append(('caption', job['id']))
        return dict(segments=[dict(s, caption='Move the object.', status='complete') for s in boundaries])


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    video = tmp_path / 'input.mp4'
    video.write_bytes(b'fixture-video')
    manifest = tmp_path / 'jobs.jsonl'
    manifest.write_text(json.dumps(dict(id='sample', video='input.mp4', instruction='Move the object.', instruction_source='official_fixture'))+'\n')
    media = dict(source_video_sha256=file_hash(video), frame_count=8, fps=4., duration_s=2.,
                 timestamps_s=[i/4 for i in range(8)], first_source_pts_s=10., cfr=True, width=64, height=64)
    monkeypatch.setattr('pipeline.annotation.runner.probe', lambda p: media)
    out = tmp_path / 'run'
    prepare(manifest, out, config=LOCAL_CONFIG)
    return out, manifest, video


def run_all(out, backend):
    for stage in ['task', 's2', 'caption-a', 'tempseg', 'caption-b', 'caption-c']:
        execute_stage(out, stage, backend=backend)


def test_batch_round_trip_shares_s2_and_task_without_fake_metadata(prepared):
    out, _, _ = prepared
    backend = Backend()
    run_all(out, backend)
    assert verify(out)['ok']
    a = read(out/'episodes/sample/AV-A.provenance.json')
    b = read(out/'episodes/sample/AV-B.provenance.json')
    assert a['boundary_cache_sha256'] == b['boundary_cache_sha256']
    assert len([c for c in backend.calls if c[0] == 'segment']) == 2  # one S2, one TempSeg
    assert not [c for c in backend.calls if c[0] == 'task']  # official instruction
    annotation = read(out/'episodes/sample/annotation_B.json')
    assert annotation['task']['outcome'] == 'unknown'
    assert annotation['steps'][0]['acting_hand'] == 'unknown'
    assert annotation['objects'] == []
    assert not b['user_accepted']


def test_resume_zero_model_calls(prepared):
    out, _, _ = prepared
    run_all(out, Backend())
    b = Backend()
    run_all(out, b)
    assert b.calls == []
    assert read(out/'stage_reports/caption-b.json')['counts']['cached'] == 1


def test_source_mutation_rejected(prepared):
    out, _, video = prepared
    video.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SOURCE_CHANGED'):
        execute_stage(out, 'task', backend=Backend())


def test_incomplete_batch_is_failure(prepared):
    out, _, _ = prepared
    execute_stage(out, 'task', backend=Backend())
    report = verify(out)
    assert not report['ok'] and report['checked'] == 0 and len(report['failures']) == 3


def test_corrupt_output_rejected_by_verifier_and_stage5(prepared):
    out, _, _ = prepared
    run_all(out, Backend())
    p = out/'episodes/sample/annotation_B.json'
    p.write_text('{}')
    assert not verify(out)['ok']
    with pytest.raises((KeyError, ValueError)):
        load_for_stage5(p, file_hash(p), 'sample', 'wrong', 8)


def test_stage5_consumer_real_contract(prepared):
    out, _, video = prepared
    run_all(out, Backend())
    p = out/'episodes/sample/annotation_B.json'
    ann, prov, refs = load_for_stage5(p, file_hash(p), 'sample', file_hash(video), 8)
    assert ann['method'] == 'B' and prov['variant'] == 'AV-B' and all(p.is_file() for p in refs)
    with pytest.raises(ValueError, match='SOURCE_MISMATCH'):
        load_for_stage5(p, file_hash(p), 'sample', 'bad-video', 8)


def test_caption_boundary_change_is_rejected(prepared):
    out, _, _ = prepared
    backend = Backend()
    execute_stage(out, 'task', backend=backend)
    execute_stage(out, 's2', backend=backend)
    original = backend.caption
    def wrong(*args):
        result = original(*args)
        result['segments'][0]['end_s'] = .9
        return result
    backend.caption = wrong
    with pytest.raises(ValueError, match='CAPTION_BOUNDARY_MISMATCH'):
        execute_stage(out, 'caption-a', backend=backend)
    assert not (out/'episodes/sample/caption-a.cache.json').exists()
    assert read(out/'stage_reports/caption-a.json')['counts']['failed'] == 1


def test_failure_can_resume(prepared):
    out, _, _ = prepared
    backend = Backend()
    execute_stage(out, 'task', backend=backend)
    def fail(*args):
        raise RuntimeError('fixture OOM')
    backend.segment = fail
    with pytest.raises(RuntimeError):
        execute_stage(out, 's2', backend=backend)
    assert not (out/'episodes/sample/s2.cache.json').exists()
    execute_stage(out, 's2', backend=Backend())
    assert read(out/'stage_reports/s2.json')['status'] == 'complete'


@pytest.mark.parametrize('segments', [[], [dict(start_s=-1.,end_s=1.)], [dict(start_s=0.,end_s=float('nan'))],
    [dict(start_s=1.,end_s=1.)], [dict(start_s=0.,end_s=4.)]])
def test_invalid_boundaries_rejected(segments):
    with pytest.raises(ValueError):
        validate_segments(segments, 2.)


def test_native_overlap_and_gaps_are_preserved():
    rows = [dict(start_s=0.,end_s=1.,caption='A',status='complete'),
            dict(start_s=.75,end_s=1.25,caption='B',status='complete'),
            dict(start_s=1.5,end_s=2.,caption='C',status='complete')]
    media = dict(timestamps_s=[i/4 for i in range(8)],duration_s=2.,fps=4.)
    ann = project_v1('a','AV-C',media,dict(instruction='Do the task'),rows)
    assert [(s['start_frame'],s['end_frame_exclusive']) for s in ann['steps']] == [(0,4),(3,5),(6,8)]


def test_cache_signature_and_integrity(tmp_path):
    p = tmp_path/'cache.json'
    cached(p, 'sig', lambda: dict(x=1))
    with pytest.raises(ValueError, match='CACHE_MISMATCH'):
        cached(p, 'changed', lambda: None)
    record = read(p); record['result']['x'] = 2; write(p, record)
    with pytest.raises(ValueError, match='CACHE_MISMATCH'):
        cached(p, 'sig', lambda: None)


def test_no_gold_or_duplicate_ids_in_generation_manifest(prepared):
    _, manifest, _ = prepared
    row = json.loads(manifest.read_text())
    row['segments'] = []
    manifest.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='FIELDS_INVALID'):
        load_jobs(manifest)
    row.pop('segments')
    manifest.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError, match='ID_INVALID'):
        load_jobs(manifest)


def test_pilot_limit_enforced(prepared):
    out, manifest, _ = prepared
    with pytest.raises(ValueError, match='PILOT_REVIEW_REQUIRED'):
        prepare(manifest, out.parent/'larger', limit=100)


def test_fixed_gt_caption_control_is_separate_and_cannot_enter_stage5(prepared):
    out, manifest, video = prepared
    gt_dir = out.parent/'gt';gt_dir.mkdir()
    write(gt_dir/'sample.json',dict(episode_id='sample',duration_s=2.,segments=[dict(start_s=.25,end_s=1.75,caption='gold-secret')]))
    control = out.parent/'control'
    prepare(manifest,control,control_dir=gt_dir)
    backend = Backend();run_all(control,backend)
    assert verify(control)['ok']
    assert not [c for c in backend.calls if c[0]=='segment']
    assert 'gold-secret' not in (control/'plan.json').read_text()
    p=control/'episodes/sample/annotation_B.json'
    with pytest.raises(ValueError,match='CONTROL_IS_EVALUATION_ONLY'):
        load_for_stage5(p,file_hash(p),'sample',file_hash(video),8)


def test_missing_official_task_uses_one_shared_generator(prepared):
    out, manifest, _ = prepared
    row = json.loads(manifest.read_text());row.pop('instruction');row.pop('instruction_source')
    manifest.write_text(json.dumps(row)+'\n')
    new = out.parent/'generated'
    prepare(manifest, new)
    backend = Backend()
    run_all(new, backend)
    assert len([c for c in backend.calls if c[0] == 'task']) == 1
    assert verify(new)['ok']


def test_real_ffmpeg_and_official_egoant_adapter_against_http_fixture(tmp_path):
    """Exercise actual contact sheets, request protocol, S2, captions and audit files."""
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg unavailable')
    pytest.importorskip('cv2')
    pytest.importorskip('requests')
    video = tmp_path/'test.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x64:rate=8:duration=6',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
    media = probe(video)
    assert media['frame_count'] == 48 and media['cfr']
    responses = [dict(segments=[dict(start_sec=0.,end_sec=6.,subtask='Move object')]),
                 dict(segments=[dict(start_sec=0.,end_sec=3.,subtask='Pick up object'),dict(start_sec=3.,end_sec=6.,subtask='Place object')]),
                 dict(label='Pick up the object.'),dict(label='Place the object.')]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert request['model'] == read(LOCAL_CONFIG)['qwen']['served_model']
            body = json.dumps(dict(choices=[dict(message=dict(content=json.dumps(responses.pop(0))))],usage={'prompt_tokens':2,'completion_tokens':3})).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        from pipeline.annotation.backends import EgoANTBackend
        config = read(LOCAL_CONFIG);config['qwen']['base_url'] = f'http://127.0.0.1:{server.server_port}/v1'
        engine = EgoANTBackend(config)
        job,task = dict(id='fixture',video=str(video)),dict(instruction='Move the object.')
        result = engine.segment(job,media,task,tmp_path/'raw/segment')
        captions = engine.caption(job,media,task,result['segments'],tmp_path/'raw/caption')
        assert len(captions['segments']) == 2 and not responses
        assert len(list((tmp_path/'raw').rglob('*_request.json'))) == 4
        assert 'Authorization' not in (tmp_path/'raw/caption/00001_request.json').read_text()
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)


def test_missing_dependency_is_counted(prepared):
    out, _, _ = prepared
    with pytest.raises(FileNotFoundError):
        execute_stage(out, 'caption-b', backend=Backend())
    report = read(out/'stage_reports/caption-b.json')
    assert report['status'] == 'failed' and report['counts']['failed'] == 1


def test_copied_upstream_cache_rejected(prepared):
    out, _, _ = prepared
    b = Backend()
    execute_stage(out,'task',backend=b)
    execute_stage(out,'s2',backend=b)
    p = out/'episodes/sample/s2.cache.json'
    record = read(p);record['signature'] = 'another-run';write(p,record)
    with pytest.raises(ValueError,match='UPSTREAM_CACHE_MISMATCH'):
        execute_stage(out,'caption-b',backend=b)


def test_segment_ending_at_last_frame_maps_to_full_grid():
    """ffprobe duration (1e-6 quantised) can be marginally shorter than the OpenCV frame grid, so
    a full-cover segment may end slightly past it; the mapping must clamp onto the grid, not refuse."""
    from pipeline.annotation.core import project_v1
    media = dict(duration_s=2.0, timestamps_s=[0., .5, 1., 1.5], fps=2., frame_count=4)
    rows = [dict(start_s=0., end_s=2.0000007, caption='Move the object.', status='complete')]
    ann = project_v1('e', 'AV-A', media, dict(instruction='Move the object.'), rows)
    assert ann['steps'][0]['start_frame'] == 0 and ann['steps'][0]['end_frame_exclusive'] == 4


def test_segment_spanning_no_frame_is_rejected():
    from pipeline.annotation.core import project_v1
    media = dict(duration_s=2.0, timestamps_s=[0., .5, 1., 1.5], fps=2., frame_count=4)
    rows = [dict(start_s=.1, end_s=.2, caption='Move the object.', status='complete')]
    with pytest.raises(ValueError, match='SUBFRAME_SEGMENT'):
        project_v1('e', 'AV-A', media, dict(instruction='Move the object.'), rows)


def test_temporal_evaluation_uses_one_to_one_matches():
    pytest.importorskip('scipy')
    from pipeline.annotation.evaluate import temporal_score
    p = [dict(start_s=0.,end_s=1.),dict(start_s=0.,end_s=1.)]
    gt = [dict(start_s=0.,end_s=1.)]
    result = temporal_score(p,gt,2.)
    assert result['thresholds']['0.5']['tp'] == 1
    assert result['thresholds']['0.5']['precision'] == .5
    assert result['caption_accuracy'] is None


def test_official_flora_native_input_plans_on_cpu(tmp_path):
    pytest.importorskip('av')
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg unavailable')
    video = tmp_path/'native.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x64:rate=8:duration=6',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
    from pipeline.annotation.backends import flora_modules
    pipeline, api, _ = flora_modules()
    inspection = api().inspect_inputs(video)
    assert inspection['new_model_calls'] == 0 and inspection['duration_s'] == 6.
    videos, _ = api._inputs(video,'human',None,None,False,'Visible task')
    plans, dispositions = pipeline.single_plans(videos,[dict(start_s=1.,end_s=3.)],'Human')
    assert len(plans) == 1
    assert [b['role'] for b in plans[0]['blocks']] == ['BEFORE','TARGET','AFTER']
    assert sum(len(b['frame_indices']) for b in plans[0]['blocks']) <= 64


def test_flora_audit_sets_are_json_projected():
    """transformers 5.8.1 returns loading-info keys as sets; the cache JSON must still serialize.

    Regression for "TypeError: Object of type set is not JSON serializable" raised while writing
    tempseg.cache.json *after* a successful FP32 load and generation.
    """
    from pipeline.annotation.backends import _json_safe
    audit = {'tempseg': {'MODEL_LOADED.json': {
        'gdn_values_exact': True,
        'base_loading_info': {'missing_keys': set(), 'unexpected_keys': set(),
                              'mismatched_keys': set(), 'error_msgs': []},
        'parameter_dtypes': {'torch.bfloat16': 700},
    }}}
    projected = _json_safe(audit)
    info = projected['tempseg']['MODEL_LOADED.json']['base_loading_info']
    assert info['missing_keys'] == [] and info['unexpected_keys'] == [] and info['mismatched_keys'] == []
    assert info['error_msgs'] == [] and projected['tempseg']['MODEL_LOADED.json']['gdn_values_exact'] is True
    json.dumps(projected)
    # Numeric and boundary values must pass through untouched.
    payload = _json_safe({'segments': [{'start_s': 0.0, 'end_s': 3.05}]})
    assert payload == {'segments': [{'start_s': 0.0, 'end_s': 3.05}]}
    assert json.dumps(_json_safe({'x': (1.5, 2)})) == '{"x": [1.5, 2]}'
