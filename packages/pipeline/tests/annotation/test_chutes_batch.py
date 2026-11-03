"""Zero-cost tests for the Chutes/Qwen3.8 batch fixes (local HTTP fixture, no real API).

1. the default profile config.json is loadable and selects 3.8 + AUX + isolation; the historical
   local-vLLM profile stays available for rental weight pinning
2. AUX bare-array tolerance lives in our call layer: vendor bytes/hashes unchanged, official mode
   unchanged, every boundary carries json_wrapper
3. per-episode failure isolation: one bad episode does not stop the batch; re-run retries it
4. request-level cache: a resumed episode does not re-send requests that already succeeded
5. transport retry (429/502/connection reset) and budget stop; thread-safe audit with workers
"""
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pipeline.annotation.core import ROOT, read, source_identity, verify_vendor, write
from pipeline.annotation.runner import execute_stage, prepare, verify

CONFIG = ROOT / 'config.json'
LOCAL_CONFIG = ROOT / 'configs/local_rental_qwen36.json'
MODEL = 'Qwen/Qwen3.8-27B-TEE'


def label_reply(i):
    return json.dumps({'label': f'Do step {i}.'})


class Fixture:
    """Programmable OpenAI-compatible server. Segmentation calls (max_tokens 8192) get either the
    official object form or the Qwen3.8 bare-array form; label calls get a JSON label."""

    def __init__(self, coarse_form='object', fail_video=None, failures=()):
        self.coarse_form, self.fail_video = coarse_form, fail_video
        self.failures = list(failures)          # e.g. [429, 'reset', 502] served before normal replies
        self.calls = []
        self.lock = threading.Lock()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with fixture.lock:
                    fixture.calls.append(body)
                    failure = fixture.failures.pop(0) if fixture.failures else None
                if failure == 'reset':
                    self.connection.close()
                    return
                if failure:
                    self.send_response(failure); self.send_header('Content-Length', '2'); self.end_headers()
                    self.wfile.write(b'{}'); return
                assert body['model'] == MODEL
                text = fixture.reply(body)
                data = json.dumps(dict(choices=[dict(message=dict(content=text))],
                                       usage=dict(prompt_tokens=1000, completion_tokens=20))).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def reply(self, body):
        prompt = json.dumps(body['messages'])
        if body['max_tokens'] <= 256:
            return label_reply(len(self.calls))
        if self.fail_video and self.fail_video in prompt:
            return 'I cannot segment this video.'
        segs = [dict(start_sec=0., end_sec=3., subtask='Pick up object'),
                dict(start_sec=3., end_sec=6., subtask='Place object')]
        is_coarse = 'coarse_hint' not in prompt and 'Refine' not in prompt and 'refine' not in prompt
        if is_coarse and self.coarse_form == 'bare_array':
            return '```json\n' + json.dumps(segs) + '\n```'
        return json.dumps(dict(segments=segs))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=3)

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}/v1'


def make_video(path, seconds=6):
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'testsrc2=size=64x64:rate=8:duration={seconds}',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)


@pytest.fixture
def batch(tmp_path, monkeypatch):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg unavailable')
    pytest.importorskip('cv2')
    monkeypatch.setenv('CHUTES_API_KEY', 'fixture-key')
    monkeypatch.delenv('AV_BUDGET_USD', raising=False)
    rows = []
    for i in range(3):
        v = tmp_path / f'ep{i}.mp4'
        make_video(v)
        rows.append(dict(id=f'ep{i}', video=v.name, instruction='Move the object.', instruction_source='fixture'))
    manifest = tmp_path / 'jobs.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def make(server, **overrides):
        config = read(CONFIG)
        config['qwen']['base_url'] = server.url
        config['execution']['episode_workers'] = 3
        for path, value in overrides.items():
            section, key = path.split('.')
            config[section][key] = value
        cfg = tmp_path / f'config_{len(list(tmp_path.glob("config_*.json")))}.json'
        write(cfg, config)
        out = tmp_path / f'run_{cfg.stem}'
        prepare(manifest, out, config=cfg, limit=3)
        return out
    return make


def run_chain(out):
    return [execute_stage(out, s) for s in ('task', 's2', 'caption-a')]


def test_profile_selects_qwen38_aux_and_isolation():
    c = read(CONFIG)
    assert c['qwen']['served_model'] == MODEL and c['qwen']['base_url'] == 'https://llm.chutes.ai/v1'
    assert c['qwen']['api_key_env'] == 'CHUTES_API_KEY' and c['variants'] == ['AV-A']
    assert c['egoant']['json_wrapper_tolerance'] == 'aux_bare_array'
    assert c['execution']['episode_workers'] == 48 and c['execution']['continue_on_error'] is True
    assert c['execution']['scheduler'] == 'dag' and c['execution']['api_concurrency'] == 48
    assert c['batch_limit'] == 300 and c['qwen']['request_cache'] is True
    # historical rental profile kept for local-inference reproduction and weight pinning
    d = read(LOCAL_CONFIG)
    assert d['qwen']['served_model'] == 'Qwen/Qwen3.6-27B' and 'execution' not in d
    assert d['egoant'].get('json_wrapper_tolerance', 'official') == 'official'
    assert d['qwen']['repo'] == 'Qwen/Qwen3.6-27B' and d['qwen']['revision']


def test_assets_refuse_remote_profile(tmp_path):
    from pipeline.annotation.cli import assets
    with pytest.raises(ValueError, match='ASSETS_REQUIRE_LOCAL_PROFILE'):
        assets(tmp_path / 'cache', 'qwen', config=CONFIG)


def test_vendor_bytes_unchanged_by_aux_patch():
    before = source_identity()
    from pipeline.annotation.backends import egoant_modules
    css, _, _ = egoant_modules()
    assert getattr(css.parse_segments, '_av_dispatch', False)
    verify_vendor()                       # SOURCES.json hashes still match on disk
    assert source_identity() == before


def test_aux_rescues_bare_array_and_marks_every_segment(batch):
    with Fixture(coarse_form='bare_array') as srv:
        out = batch(srv)
        reports = run_chain(out)
    assert [r['status'] for r in reports] == ['complete'] * 3
    assert verify(out)['ok']
    s2 = read(out / 'episodes/ep0/s2.cache.json')['result']
    assert s2['json_wrapper']['coarse'] == 'bare_array_aux' and s2['json_wrapper']['parse_mode'] == 'aux_bare_array'
    assert all(seg['json_wrapper'] in ('object', 'bare_array_aux') for seg in s2['segments'])
    native = read(out / 'episodes/ep0/AV-A.native.json')
    assert all('json_wrapper' in seg for seg in native['segments'])


def test_official_mode_still_rejects_bare_array(batch):
    with Fixture(coarse_form='bare_array') as srv:
        out = batch(srv, **{'egoant.json_wrapper_tolerance': 'official'})
        r = execute_stage(out, 'task'), execute_stage(out, 's2')
    assert r[1]['status'] == 'partial' and r[1]['counts']['failed'] == 3
    assert 'EGOANT_COARSE_FAILED' in read(out / 'episodes/ep0/s2.failure.json')['reason']


def test_failed_episode_is_isolated_and_retried_on_rerun(batch, monkeypatch):
    from pipeline.annotation.backends import EgoANTBackend
    original = EgoANTBackend.segment
    state = dict(fail=True)

    def flaky(self, job, media, task, directory):
        if state['fail'] and job['id'] == 'ep1':
            raise RuntimeError('fixture: coarse unparsable')
        return original(self, job, media, task, directory)
    monkeypatch.setattr(EgoANTBackend, 'segment', flaky)
    with Fixture() as srv:
        out = batch(srv)
        reports = run_chain(out)
        assert reports[1]['status'] == 'partial'
        assert reports[1]['counts'] == dict(succeeded=2, cached=0, failed=1, skipped_upstream=0)
        assert reports[2]['counts']['skipped_upstream'] == 1 and reports[2]['counts']['succeeded'] == 2
        assert (out / 'episodes/ep1/s2.failure.json').exists() and not verify(out)['ok']
        state['fail'] = False
        before = len(srv.calls)
        reports = run_chain(out)
    assert [r['status'] for r in reports] == ['complete'] * 3
    assert reports[1]['counts']['cached'] == 2 and reports[1]['counts']['succeeded'] == 1
    assert verify(out)['ok'] and not (out / 'episodes/ep1/s2.failure.json').exists()
    assert len(srv.calls) - before == 1 + 2 + 2      # only ep1: coarse + 2 S2 windows + 2 captions


def test_request_cache_prevents_rebilling_partial_episode(batch):
    with Fixture() as srv:
        out = batch(srv)
        execute_stage(out, 'task'); execute_stage(out, 's2')
        # simulate a crash after 1 of 2 captions of ep0: run caption once, then drop its success cache
        execute_stage(out, 'caption-a')
        for p in (out / 'episodes/ep0').glob('caption-a.cache.json'):
            p.unlink()
        cache_dir = out / 'episodes/ep0/raw/caption-a/request_cache'
        kept = sorted(cache_dir.glob('*.json'))
        assert len(kept) == 2
        kept[1].unlink()                         # second interval never returned
        before = len(srv.calls)
        report = execute_stage(out, 'caption-a')
    assert report['status'] == 'complete'
    assert len(srv.calls) - before == 1          # only the missing interval went to the model
    assert report['usage']['cached_replies'] >= 1
    assert verify(out)['ok']


def test_transport_retry_absorbs_429_502_and_reset(batch):
    with Fixture(failures=[429, 'reset', 502]) as srv:
        out = batch(srv, **{'execution.episode_workers': 1})
        reports = run_chain(out)
    assert [r['status'] for r in reports] == ['complete'] * 3
    assert verify(out)['ok']
    labels = [s['caption'] for p in out.glob('episodes/*/AV-A.native.json') for s in read(p)['segments']]
    assert labels and not [l for l in labels if l.startswith('[FAILED]')]


def test_budget_stop_halts_stage(batch, monkeypatch):
    monkeypatch.setenv('AV_BUDGET_USD', '0.001')   # below one worst-case reservation
    with Fixture() as srv:
        out = batch(srv)
        execute_stage(out, 'task')
        from pipeline.annotation.backends import BudgetExceeded
        with pytest.raises(BudgetExceeded):
            execute_stage(out, 's2')
    assert len(srv.calls) == 0
    assert read(out / 'stage_reports/s2.json')['status'] == 'failed'


def test_parallel_audit_files_stay_per_episode(batch):
    with Fixture() as srv:
        out = batch(srv)
        run_chain(out)
    for i in range(3):
        reqs = list((out / f'episodes/ep{i}/raw/caption-a').glob('*_request.json'))
        assert len(reqs) == 2
        for r in reqs:
            assert 'Authorization' not in r.read_text()
    ledger = (out / 'usage_ledger.jsonl').read_text().splitlines()
    assert len(ledger) == 3 * (1 + 2 + 2)
