"""Offline concurrency, dependency, replay-equivalence and budget regressions."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace

import pytest

from pipeline.annotation import backends
from pipeline.annotation.core import digest, read, write
from pipeline.annotation.runner import execute_dag, execute_stage, verify
from pipeline.annotation.scheduling import Runtime, use_runtime
from .test_chutes_batch import Fixture, batch  # noqa: F401


def test_actual_48_inflight_and_bounded_decode(tmp_path):
    """96 ready network calls must refill 48 slots; decode has its own smaller cap."""
    stop = threading.Event()
    config = dict(execution=dict(api_concurrency=48, request_workers=96, decode_workers=3))
    release, saturated = threading.Event(), threading.Event()
    lock = threading.Lock()
    active = peak = 0
    with use_runtime(tmp_path, config, stop) as runtime:
        def request(i):
            nonlocal active, peak
            with runtime.network(tmp_path / 'caption-a'):
                with lock:
                    active += 1
                    peak = max(peak, active)
                    if active == 48:
                        saturated.set()
                assert release.wait(5)
                with lock:
                    active -= 1
        with ThreadPoolExecutor(max_workers=96) as pool:
            futures = [pool.submit(request, i) for i in range(96)]
            try:
                assert saturated.wait(5)
                assert runtime.summary()['active'] == 48
            finally:
                release.set()
            for f in futures:
                f.result()
        def decode(_):
            with runtime.decoding():
                time.sleep(.01)
        runtime.map(decode, range(24))
        assert peak == 48
        summary = runtime.summary()
        assert summary['network_attempts'] == 96 and summary['peak_inflight'] == 48
        assert summary['active'] == summary['queued'] == 0
        assert summary['decode_peak'] == 3
    assert read(tmp_path / 'scheduler_summary.json')['peak_inflight'] == 48


def test_single_episode_caption_can_fill_48_slots(batch, monkeypatch):
    """Small episode count no longer limits request concurrency (prepared-image fixture)."""
    import requests
    release = threading.Event()
    lock = threading.Lock()
    active = peak = 0
    bodies = []
    def fake(url, **kwargs):
        nonlocal active, peak
        with lock:
            bodies.append(kwargs['json'])
            active += 1
            peak = max(peak, active)
            if active == 48:
                release.set()
        try:
            assert release.wait(10), 'single episode did not supply 48 independent requests'
            return backends._response_from_body(dict(choices=[dict(message=dict(
                content=json.dumps(dict(label='Move the object.'))))],
                usage=dict(prompt_tokens=10, completion_tokens=5)), url)
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(requests, 'post', fake)
    _, _, labeler = backends.egoant_modules()
    monkeypatch.setattr(labeler, 'build_images_for_segment',
                        lambda cap, fps, count, segs, i, mode, hand: ([f'image-{i}'.encode()], 'raw'))
    with Fixture() as srv:
        out = batch(srv)
        plan = read(out / 'plan.json')
        rec = plan['episodes'][0]
        boundaries = [dict(start_s=i / 8, end_s=(i + 1) / 8) for i in range(48)]
        ledger = backends.Ledger.for_run(out, plan['config'])
        with use_runtime(out, plan['config'], backends.STOP), backends.use_ledger(ledger):
            result = backends.EgoANTBackend(plan['config']).caption(
                rec['job'], rec['media'], dict(instruction='Move the object.'),
                boundaries, out / 'raw/caption-a')
        assert peak == 48 and len(bodies) == 48 and len(result['segments']) == 48
        assert [s['start_s'] for s in result['segments']] == [s['start_s'] for s in boundaries]
        assert len(list((out / 'raw/caption-a').glob('*_request.json'))) == 48
        assert ledger.summary()['network_replies'] == 48
        write(out / 'saturation_evidence.json', dict(
            fixture='prepared image bytes, synchronous fake HTTP, zero external API',
            episodes=1, captions=48, peak_inflight=peak, summary=read(out / 'scheduler_summary.json')))


def test_priority_prefers_segmentation_over_waiting_caption(tmp_path):
    config = dict(execution=dict(api_concurrency=1, request_workers=2))
    stop, release = threading.Event(), threading.Event()
    order = []
    with use_runtime(tmp_path, config, stop) as runtime:
        def request(stage):
            with runtime.network(tmp_path / stage):
                order.append(stage)
                if stage == 'hold':
                    assert release.wait(5)
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(request, 'hold')
            deadline = time.monotonic() + 5
            def await_state(active, queued):
                with runtime.cv:
                    while runtime.active != active or len(runtime.pending) != queued:
                        assert time.monotonic() < deadline
                        runtime.cv.wait(.01)
            try:
                await_state(1, 0)
                low = pool.submit(request, 'caption-a')
                await_state(1, 1)
                high = pool.submit(request, 's2')
                await_state(1, 2)
            finally:
                release.set()
            for f in (first, low, high):
                f.result()
    assert order == ['hold', 's2', 'caption-a']


class DeterministicFixture(Fixture):
    def reply(self, body):
        if body['max_tokens'] <= 256:
            return json.dumps(dict(label='Move object ' + digest(body)[:8] + '.'))
        return super().reply(body)


def requests_of(out):
    return Counter(digest(read(p)['body']) for p in out.glob('episodes/*/raw/*/*_request.json'))


def test_dag_request_and_output_equivalence_and_resume(batch):
    with DeterministicFixture(coarse_form='bare_array') as srv:
        serial = batch(srv)
        for stage in ('task', 's2', 'caption-a'):
            execute_stage(serial, stage)
        dag = batch(srv)
        reports = execute_dag(dag)
        assert [r['status'] for r in reports] == ['complete'] * 3
        assert verify(serial)['ok'] and verify(dag)['ok']
        # Prompt, full image bytes, model, options: exact multiset, order may change.
        assert requests_of(serial) == requests_of(dag)
        for i in range(3):
            a, b = serial / f'episodes/ep{i}', dag / f'episodes/ep{i}'
            assert read(a / 'annotation_A.json') == read(b / 'annotation_A.json')
            assert read(a / 's2.cache.json')['result'] == read(b / 's2.cache.json')['result']
            assert len(list((b / 'raw/caption-a').glob('*_request.json'))) == 2
        before = len(srv.calls)
        again = execute_dag(dag)
        assert len(srv.calls) == before
        assert all(r['counts']['cached'] == 3 for r in again)
        assert read(dag / 'scheduler_summary.json')['network_attempts'] == 0
    assert len((dag / 'usage_ledger.jsonl').read_text().splitlines()) == 15


def test_per_episode_dependency_no_stage_barrier(batch):
    entered_slow, caption_done = threading.Event(), threading.Event()

    class Backend:
        def segment(self, job, media, task, directory):
            if job['id'] == 'ep1':
                entered_slow.set()
                assert caption_done.wait(5), 'caption waited for whole-batch segmentation'
            return dict(segments=[dict(start_s=0., end_s=media['duration_s'])])

        def caption(self, job, media, task, boundaries, directory):
            if job['id'] == 'ep0':
                assert entered_slow.wait(5)
                caption_done.set()
            return dict(segments=[dict(**s, caption='Move the object.', status='complete') for s in boundaries])

    with Fixture() as srv:
        out = batch(srv)
        reports = execute_dag(out, backend=Backend())
    assert all(r['status'] == 'complete' for r in reports)
    assert verify(out)['ok'] and not srv.calls


def test_dag_failure_isolation_and_partial_request_resume(batch, monkeypatch):
    original = backends.EgoANTBackend.segment
    def fail_one(self, job, *args):
        if job['id'] == 'ep1':
            raise RuntimeError('bad episode')
        return original(self, job, *args)
    monkeypatch.setattr(backends.EgoANTBackend, 'segment', fail_one)
    with DeterministicFixture() as srv:
        out = batch(srv)
        reports = execute_dag(out)
        assert reports[1]['counts']['failed'] == 1
        assert reports[2]['counts']['skipped_upstream'] == 1
        assert reports[2]['counts']['succeeded'] == 2
        monkeypatch.setattr(backends.EgoANTBackend, 'segment', original)
        assert all(r['status'] == 'complete' for r in execute_dag(out))
        directory = out / 'episodes/ep0'
        (directory / 'caption-a.cache.json').unlink()
        cache = sorted((directory / 'raw/caption-a/request_cache').glob('*.json'))
        cache[0].unlink()
        before = len(srv.calls)
        assert execute_dag(out)[2]['status'] == 'complete'
        assert len(srv.calls) - before == 1
        assert verify(out)['ok']


def test_dag_budget_stop_drains_and_reports(batch, monkeypatch):
    monkeypatch.setenv('AV_BUDGET_USD', '0.001')
    with Fixture() as srv:
        out = batch(srv)
        with pytest.raises(backends.BudgetExceeded):
            execute_dag(out)
        assert not srv.calls
    summary = read(out / 'scheduler_summary.json')
    assert summary['active'] == summary['queued'] == summary['network_attempts'] == 0
    assert read(out / 'stage_reports/s2.json')['status'] == 'failed'
    assert backends._LEDGER['current'] is None and backends._PATCH['depth'] == 0


def test_retry_backoff_releases_network_slot(tmp_path, monkeypatch):
    """A sleeping retry cannot hold the only slot or prevent another request."""
    sleeping, other_done = threading.Event(), threading.Event()
    calls = Counter()
    lock = threading.Lock()
    def fake(url, **kwargs):
        with lock:
            calls[url] += 1
            n = calls[url]
        response = backends._response_from_body(dict(choices=[dict(message=dict(content='ok'))]), url)
        if url == 'retry' and n == 1:
            response.status_code = 429
        return response
    def sleep(_):
        sleeping.set()
        assert other_done.wait(5), 'retry backoff retained API slot'
    monkeypatch.setattr(backends, 'time', SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=sleep))
    config = dict(execution=dict(api_concurrency=1, request_workers=2, decode_workers=1))
    with use_runtime(tmp_path, config, threading.Event()), backends.use_ledger(None):
        def post(name):
            state = backends._AuditState(tmp_path / name, transport_retries=1)
            return state.post(fake, name, json=dict(name=name))
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(post, 'retry')
            assert sleeping.wait(5)
            second = pool.submit(post, 'other')
            try:
                assert second.result(timeout=5).status_code == 200
            finally:
                other_done.set()
            assert first.result(timeout=5).status_code == 200
    summary = read(tmp_path / 'scheduler_summary.json')
    assert summary['peak_inflight'] == 1 and summary['network_attempts'] == 3
