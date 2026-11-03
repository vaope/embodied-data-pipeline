"""Adapters for pinned upstream implementations; imports/models are lazy."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time

from .core import ROOT, digest, read, write, validate_segments
from .scheduling import current as current_runtime

_IMPORT_LOCK = threading.Lock()
_LOCAL = threading.local()          # per-thread audit state and parse-event sink
_PATCH_LOCK = threading.Lock()
_PATCH = dict(depth=0, original=None)
RETRY_STATUS = {429, 500, 502, 503, 504}


def _add_path(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def egoant_modules():
    scripts = ROOT / 'vendor/egoant/scripts'
    with _IMPORT_LOCK:
        _add_path(scripts)
        _add_path(scripts / 'wgo_bench')
        import contact_sheet_segment as css
        import seg_opt_ideas as s2
        import label_gold_segments as labeler
        _install_parse_dispatch(css)
    return css, s2, labeler


def flora_modules():
    _add_path(ROOT / 'vendor/flora')
    from batch_utils import pipeline
    from batch_utils.api import DataEnginePipeline
    from batch_utils.models import Session
    return pipeline, DataEnginePipeline, Session


# ---------------------------------------------------------------- AUX JSON wrapper tolerance
def extract_array(text):
    """A fenced or bare JSON array of objects, else None (same rule as the WGO-Bench AUX run)."""
    if not text:
        return None
    for pattern in (r"```(?:json)?\s*(\[.*?\])\s*```", r"(\[.*\])"):
        m = re.search(pattern, text, re.S)
        if not m:
            continue
        try:
            value = json.loads(m.group(1))
        except ValueError:
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return None


def _install_parse_dispatch(css):
    """Wrap the vendored parser in memory (vendor bytes unchanged, so source_identity is stable).

    Official behaviour unless the calling thread enabled 'aux_bare_array': then a reply that the
    official parser rejects and that is a bare JSON array is re-wrapped as {"segments": [...]} and
    handed to the SAME official parser. Every parse is logged to the thread's event sink."""
    if getattr(css.parse_segments, '_av_dispatch', False):
        return
    official = css.parse_segments

    def parse_segments(text):
        segs = official(text)
        mode = getattr(_LOCAL, 'json_mode', 'official')
        wrapper = 'object' if segs else 'unparsed'
        if not segs and mode == 'aux_bare_array':
            arr = extract_array(text)
            if arr is not None:
                segs = official('{"segments": ' + json.dumps(arr) + '}')
                wrapper = 'bare_array_aux' if segs else 'unparsed'
        sink = getattr(_LOCAL, 'parse_events', None)
        if sink is not None:
            sink.append(wrapper)
        return segs

    parse_segments._av_dispatch = True
    parse_segments.official = official
    css.parse_segments = parse_segments


@contextmanager
def parse_events(mode):
    if mode not in ('official', 'aux_bare_array'):
        raise ValueError(f'JSON_WRAPPER_TOLERANCE_INVALID: {mode}')
    old = getattr(_LOCAL, 'json_mode', None), getattr(_LOCAL, 'parse_events', None)
    _LOCAL.json_mode, _LOCAL.parse_events = mode, []
    try:
        yield _LOCAL.parse_events
    finally:
        _LOCAL.json_mode, _LOCAL.parse_events = old


def _take_wrapper(events):
    """Wrapper of the parse that ended a call (official loops stop at the first success)."""
    value = events[-1] if events else 'unparsed'
    events.clear()
    return value


# ---------------------------------------------------------------- usage ledger and budget
class BudgetExceeded(BaseException):
    """BaseException on purpose: vendored retry loops catch Exception and would turn a budget
    stop into a [FAILED] label. This must stop the stage instead."""


STOP = threading.Event()
_LEDGER = dict(current=None)


class Ledger:
    """Persistent per-run spend record (<run>/usage_ledger.jsonl, one line per network reply).

    Budget (config qwen.budget_usd, prices qwen.price_usd_per_million) uses a worst-case
    reservation per in-flight request (context_tokens_reserve input + max_tokens output); when the
    reservations do not fit, new requests wait for in-flight ones to settle; if a single request
    cannot fit, BudgetExceeded stops the stage. Without prices only tokens are counted."""

    def __init__(self, path, prices=None, budget=None, context_reserve=262144):
        self.path, self.prices, self.budget, self.context_reserve = Path(path), prices, budget, context_reserve
        self.cv = threading.Condition()
        self.reserved = 0.
        self.budget_wait_s = 0.
        self.totals = dict(network_replies=0, cached_replies=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.)
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.totals['network_replies'] += 1
                    self.totals['prompt_tokens'] += row.get('prompt_tokens') or 0
                    self.totals['completion_tokens'] += row.get('completion_tokens') or 0
                    self.totals['cost_usd'] += row.get('cost_usd') or 0.

    @classmethod
    def for_run(cls, out, config):
        """Budget: env AV_BUDGET_USD overrides config qwen.budget_usd (runtime cap, not run identity)."""
        q = config.get('qwen', {})
        budget = os.environ.get('AV_BUDGET_USD')
        budget = float(budget) if budget else q.get('budget_usd')
        return cls(Path(out) / 'usage_ledger.jsonl', q.get('price_usd_per_million'), budget,
                   q.get('context_tokens_reserve', 262144))

    def worst_case(self, body):
        if not self.prices:
            return 0.
        return (self.context_reserve * self.prices['input']
                + int((body or {}).get('max_tokens') or 8192) * self.prices['output']) / 1e6

    def reserve(self, amount):
        with self.cv:
            while self.budget is not None and self.totals['cost_usd'] + self.reserved + amount > self.budget:
                if STOP.is_set():
                    raise BudgetExceeded('BUDGET_STOP: stage stopping')
                if self.reserved <= 0:
                    STOP.set()
                    raise BudgetExceeded(f'BUDGET_STOP: spent ${self.totals["cost_usd"]:.4f} + next worst case '
                                         f'${amount:.4f} > budget ${self.budget:.2f}')
                start = time.monotonic()
                self.cv.wait(timeout=1.)
                self.budget_wait_s += time.monotonic() - start
            self.reserved += amount

    def settle(self, amount, directory, status, parsed):
        usage = (parsed or {}).get('usage') or {} if isinstance(parsed, dict) else {}
        pt, ct = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if self.prices and pt is not None:
            cost, basis = (pt * self.prices['input'] + (ct or 0) * self.prices['output']) / 1e6, 'usage'
        else:
            cost, basis = amount, 'reservation'           # no usage reported: assume the worst case
        row = dict(t=round(time.time(), 3), dir=str(directory), status=status, prompt_tokens=pt,
                   completion_tokens=ct, cost_usd=cost, cost_basis=basis)
        with self.cv:
            self.reserved -= amount
            self.totals['network_replies'] += 1
            self.totals['prompt_tokens'] += pt or 0
            self.totals['completion_tokens'] += ct or 0
            self.totals['cost_usd'] += cost
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            self.cv.notify_all()

    def cached(self):
        with self.cv:
            self.totals['cached_replies'] += 1

    def summary(self):
        with self.cv:
            return dict(self.totals, cost_usd=round(self.totals['cost_usd'], 6), budget_usd=self.budget,
                        reserved_usd=self.reserved, budget_wait_s=self.budget_wait_s,
                        cost_scope='whole run directory, all invocations; cached replies cost 0')


@contextmanager
def use_ledger(ledger):
    previous, _LEDGER['current'] = _LEDGER['current'], ledger
    STOP.clear()
    try:
        yield ledger
    finally:
        _LEDGER['current'] = previous


# ---------------------------------------------------------------- HTTP audit, request cache, transport retry
def _response_from_body(body, url):
    import requests
    r = requests.models.Response()
    r.status_code, r.url, r.reason, r.encoding = 200, url, 'OK', 'utf-8'
    r._content = json.dumps(body).encode('utf-8')
    r.headers['Content-Type'] = 'application/json'
    return r


class _AuditState:
    """One episode-stage invocation in one thread. Files keep the original layout:
    <dir>/<nnnnn>_request.json / _response.json; cached replies live in <dir>/request_cache/."""

    def __init__(self, directory, transport_retries=0, request_cache=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.count = len(list(self.directory.glob('*_request.json')))
        self.retries = int(transport_retries)
        self.cache = bool(request_cache)
        self.seen = {}
        self.lock = threading.Lock()

    def _record(self, url, body, response=None, error=None, wall=0., attempt=1, from_cache=False):
        with self.lock:
            self.count += 1
            prefix = self.directory / f'{self.count:05d}'
        write(str(prefix) + '_request.json', dict(url=url, body=body))
        if error is not None:
            # Exception messages may contain URLs; no headers/tokens are persisted.
            write(str(prefix) + '_response.json', dict(error_type=type(error).__name__, wall_s=wall,
                                                       attempt=attempt))
            return
        try:
            parsed = response.json()
        except ValueError:
            parsed = {'unparsed_body': response.text}
        write(str(prefix) + '_response.json', dict(status_code=response.status_code, body=parsed, wall_s=wall,
                                                   attempt=attempt, from_cache=from_cache))
        return parsed

    def post(self, original, url, **kwargs):
        import requests
        body = kwargs.get('json')
        key = digest(dict(url=url, body=body))
        with self.lock:
            self.seen[key] = self.seen.get(key, 0) + 1
            first = self.seen[key] == 1
        path = self.directory / 'request_cache' / f'{key}.json'
        # Only the first identical request of an invocation may be served from cache: a repeat is
        # the upstream code retrying after an unusable answer and must reach the model again.
        ledger = _LEDGER['current']
        if self.cache and first and path.exists():
            record = read(path)
            if record.get('key') == key and record.get('response_sha256') == digest(record.get('response_body')):
                response = _response_from_body(record['response_body'], url)
                self._record(url, body, response, from_cache=True)
                if ledger:
                    ledger.cached()
                return response
        for attempt in range(1, self.retries + 2):
            if STOP.is_set():
                raise BudgetExceeded('BUDGET_STOP: stage stopping')
            amount = ledger.worst_case(body) if ledger else 0.
            if ledger:
                ledger.reserve(amount)
            start = time.monotonic()
            try:
                runtime = current_runtime()
                with runtime.network(self.directory) if runtime else nullcontext() as attempt_info:
                    response = original(url, **kwargs)
                    if runtime:
                        attempt_info['status_code'] = response.status_code
            except BudgetExceeded:
                # Admission was cancelled before sending; release, do not bill an unsent request.
                if ledger:
                    with ledger.cv:
                        ledger.reserved -= amount
                        ledger.cv.notify_all()
                raise
            except (requests.ConnectionError, requests.Timeout) as exc:
                if ledger:
                    ledger.settle(amount, self.directory, 'connection_error', None)
                self._record(url, body, error=exc, wall=time.monotonic() - start, attempt=attempt)
                if attempt <= self.retries:
                    time.sleep(min(30., 2. ** attempt) * (.5 + random.random()))
                    continue
                raise
            except Exception as exc:
                if ledger:
                    ledger.settle(amount, self.directory, type(exc).__name__, None)
                self._record(url, body, error=exc, wall=time.monotonic() - start, attempt=attempt)
                raise
            parsed = self._record(url, body, response, wall=time.monotonic() - start, attempt=attempt)
            if ledger:
                ledger.settle(amount, self.directory, response.status_code, parsed)
            if response.status_code in RETRY_STATUS and attempt <= self.retries:
                time.sleep(min(30., 2. ** attempt) * (.5 + random.random()))
                continue
            if self.cache and response.status_code == 200 and isinstance(parsed, dict) and parsed.get('choices'):
                with self.lock:
                    write(path, dict(key=key, url=url, response_sha256=digest(parsed), response_body=parsed))
            return response


def _dispatch_post(url, **kwargs):
    state = getattr(_LOCAL, 'audit', None)
    if state is None:                    # threads outside an audit context: untouched
        return _PATCH['original'](url, **kwargs)
    return state.post(_PATCH['original'], url, **kwargs)


@contextmanager
def audit_http(directory, *, transport_retries=0, request_cache=False, state=None):
    """Capture actual upstream JSON bodies/responses without Authorization headers.

    Thread-safe: requests.post is patched once while any audit context is open and routes to the
    calling thread's own directory. Optional (config-driven): transport retries for 429/5xx and
    connection errors, and a per-request response cache so a resumed episode does not pay again
    for requests that already succeeded."""
    import requests
    with _PATCH_LOCK:
        if _PATCH['depth'] == 0:
            _PATCH['original'] = requests.post
            requests.post = _dispatch_post
        _PATCH['depth'] += 1
    previous = getattr(_LOCAL, 'audit', None)
    _LOCAL.audit = state if state is not None else _AuditState(directory, transport_retries, request_cache)
    try:
        yield _LOCAL.audit
    finally:
        _LOCAL.audit = previous
        with _PATCH_LOCK:
            _PATCH['depth'] -= 1
            if _PATCH['depth'] == 0:
                requests.post = _PATCH['original']


class EgoANTBackend:
    def __init__(self, config):
        self.config = config
        self.json_mode = config['egoant'].get('json_wrapper_tolerance', 'official')
        if self.json_mode not in ('official', 'aux_bare_array'):
            raise ValueError(f'JSON_WRAPPER_TOLERANCE_INVALID: {self.json_mode}')

    def _audit(self, directory, state=None):
        q = self.config['qwen']
        return audit_http(directory, transport_retries=q.get('transport_retries', 0),
                          request_cache=q.get('request_cache', False), state=state)

    def sheets(self, job):
        css, _, _ = egoant_modules()
        c = self.config['egoant']
        runtime = current_runtime()
        with runtime.decoding() if runtime else nullcontext():
            return css.build_sheets(Path(job['video']), c['sample_sec'], c['tiles_per_sheet'], c['tile_px'], c['cols'])

    def task(self, job, directory):
        css, _, _ = egoant_modules()
        sheets, _, _ = self.sheets(job)
        import base64
        import requests
        q = self.config['qwen']
        prompt = (ROOT / 'prompts/task.txt').read_text(encoding='utf-8')
        content = [dict(type='text', text=prompt)] + [dict(type='image_url', image_url=dict(
            url='data:image/jpeg;base64,' + base64.b64encode(b).decode())) for b in sheets]
        payload = dict(model=q['served_model'], messages=[dict(role='user', content=content)],
                       temperature=0., max_tokens=512, chat_template_kwargs={'enable_thinking': False})
        with self._audit(directory):
            for attempt in range(q['attempts']):
                try:
                    response = requests.post(q['base_url'].rstrip('/') + '/chat/completions', json=payload,
                                             headers={'Authorization': 'Bearer ' + os.environ.get(q['api_key_env'], 'local-vllm')},
                                             timeout=q['timeout_s'])
                    response.raise_for_status()
                    text = css.message_text(response.json()['choices'][0]['message'])
                    value = json.loads(text)
                    instruction = value['instruction']
                    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 2000:
                        raise ValueError('TASK_GENERATOR_INVALID')
                    return dict(instruction=instruction.strip(), task_source='generated_' + q['served_model'],
                                instruction_source='shared_task_generator.v1')
                except (ValueError, KeyError, requests.RequestException):
                    if attempt + 1 == q['attempts']:
                        raise

    def segment(self, job, media, task, directory):
        if not media['cfr']:
            raise ValueError('EGOANT_VFR_UNSUPPORTED: upstream OpenCV uses frame/fps; no implicit resampling')
        css, refine, _ = egoant_modules()
        c, q = self.config['egoant'], self.config['qwen']
        sheets, spans, duration = self.sheets(job)
        api_key = os.environ.get(q['api_key_env'], 'local-vllm')
        args = [q['base_url'], q['served_model'], api_key]
        with self._audit(directory) as audit_state, parse_events(self.json_mode) as events:
            coarse, meta = css.call_segment(*args, sheets, spans, task['instruction'], c['sample_sec'],
                                            q['timeout_s'], q['attempts'], 0, prompt_variant=c['coarse_prompt'],
                                            max_tokens=c['max_tokens'], failed_fallback=False, context_fallback=False)
            coarse_wrapper = _take_wrapper(events)
            if not coarse or meta.get('error') or any(m.get('empty') or m.get('error') for m in meta.get('calls', [])):
                raise RuntimeError(f'EGOANT_COARSE_FAILED: raw response retained (json_wrapper={coarse_wrapper})')
            coarse = css.postprocess_segments(coarse, duration, c['min_seg_sec'])
            def refine_one(seg):
                a, b = float(seg['start_sec']), float(seg['end_sec'])
                if b - a < c['min_refine_sec']:
                    return [dict(seg)], None, (a, b, coarse_wrapper)
                wa, wb = max(0., a - c['pad_sec']), min(duration, b + c['pad_sec'])
                indices = [i for i, (x, y) in enumerate(spans) if not (y < wa or x > wb)]
                subspans = [(max(spans[i][0], wa), min(spans[i][1], wb)) for i in indices]
                if not indices:
                    raise RuntimeError('EGOANT_S2_NO_INPUT')
                with self._audit(directory, audit_state), parse_events(self.json_mode) as local_events:
                    rows, call_meta = refine._one_call_custom(*args, [sheets[i] for i in indices], subspans,
                        task['instruction'], c['sample_sec'], q['timeout_s'], q['attempts'], c['refine_prompt'],
                        c['max_tokens'], failed_fallback=False,
                        extra_format={'coarse_hint': f'{a:.2f}-{b:.2f}: {seg.get("subtask", "")}'})
                    call_meta['json_wrapper'] = _take_wrapper(local_events)
                if not rows or call_meta.get('empty') or call_meta.get('error'):
                    raise RuntimeError(f'EGOANT_S2_FAILED: no fallback to coarse labels (json_wrapper={call_meta["json_wrapper"]})')
                clipped = []
                for row in rows:
                    x, y = max(wa, min(wb, float(row['start_sec']))), max(wa, min(wb, float(row['end_sec'])))
                    if y > x + .001:
                        clipped.append(dict(start_sec=x, end_sec=y, subtask=row['subtask']))
                return clipped, call_meta, (wa, wb, call_meta['json_wrapper'])
            runtime = current_runtime()
            parts = runtime.map(refine_one, coarse) if runtime else [refine_one(s) for s in coarse]
            fine = [row for rows, _, _ in parts for row in rows]
            audit = [metadata for _, metadata, _ in parts if metadata is not None]
            windows = [window for _, _, window in parts]
            fine = css.postprocess_segments(css.merge_adjacent(sorted(fine, key=lambda x: x['start_sec'])),
                                            duration, c['min_seg_sec'])
        meta['json_wrapper'] = coarse_wrapper

        def wrapper_of(a, b):
            # A final segment may span several windows after merging: AUX if any source used AUX.
            used = {w for x, y, w in windows if min(b, y) - max(a, x) > 1e-6}
            return 'bare_array_aux' if 'bare_array_aux' in used else 'object'

        result = [dict(start_s=s['start_sec'], end_s=s['end_sec'], segmentation_label=s['subtask'],
                       json_wrapper=wrapper_of(s['start_sec'], s['end_sec'])) for s in fine]
        validate_segments(result, media['duration_s'])
        calls = [coarse_wrapper] + [m['json_wrapper'] for m in audit]
        parse = dict(json_wrapper_tolerance=self.json_mode, coarse=coarse_wrapper,
                     s2=[m['json_wrapper'] for m in audit], aux_rescued_calls=calls.count('bare_array_aux'),
                     parse_mode='aux_bare_array' if 'bare_array_aux' in calls else 'official')
        return dict(segments=result, coarse=coarse, coarse_audit=meta, refinement_audit=audit,
                    json_wrapper=parse, recipe='EgoANT_GEPA_S2_pad0_full_cover', no_selector=True)

    def caption(self, job, media, task, boundaries, directory):
        if not media['cfr']:
            raise ValueError('EGOANT_VFR_UNSUPPORTED')
        _, _, labeler = egoant_modules()
        import cv2
        q = self.config['qwen']
        segments = [dict(start_sec=s['start_s'], end_sec=s['end_s']) for s in boundaries]
        runtime = current_runtime()
        if runtime:
            # Independent VideoCapture per interval: OpenCV handles are not thread safe.
            # Tasks/image payloads are bounded by Runtime.map, decoding by its own semaphore.
            with self._audit(directory) as audit_state:
                def one(i):
                    s = boundaries[i]
                    with runtime.decoding():
                        cap = cv2.VideoCapture(job['video'])
                        try:
                            frames, mode = labeler.build_images_for_segment(
                                cap, media['fps'], media['frame_count'], segments, i, 'raw', None)
                        finally:
                            cap.release()
                    if not frames:
                        raise ValueError('EGOANT_CAPTION_NO_FRAMES')
                    with self._audit(directory, audit_state):
                        caption = labeler.call_label(q['base_url'], q['served_model'],
                            os.environ.get(q['api_key_env'], 'local-vllm'), frames, task['instruction'],
                            'raw', q['timeout_s'], q['attempts'])
                    row = dict(start_s=s['start_s'], end_s=s['end_s'], caption=caption,
                               status='complete', input_mode=mode)
                    if 'json_wrapper' in s:
                        row['json_wrapper'] = s['json_wrapper']
                    return row
                rows = runtime.map(one, range(len(boundaries)))
            validate_segments(rows, media['duration_s'], captions=True)
            return dict(segments=rows)
        rows = []
        cap = cv2.VideoCapture(job['video'])
        try:
            with self._audit(directory):
                for i, s in enumerate(boundaries):
                    frames, mode = labeler.build_images_for_segment(cap, media['fps'], media['frame_count'], segments, i, 'raw', None)
                    if not frames:
                        raise ValueError('EGOANT_CAPTION_NO_FRAMES')
                    caption = labeler.call_label(q['base_url'], q['served_model'],
                        os.environ.get(q['api_key_env'], 'local-vllm'), frames, task['instruction'], 'raw', q['timeout_s'], q['attempts'])
                    row = dict(start_s=s['start_s'], end_s=s['end_s'], caption=caption,
                               status='complete', input_mode=mode)
                    if 'json_wrapper' in s:      # boundary parse provenance (object / bare_array_aux)
                        row['json_wrapper'] = s['json_wrapper']
                    rows.append(row)
        finally:
            cap.release()
        validate_segments(rows, media['duration_s'], captions=True)
        return dict(segments=rows)


def _json_safe(value):
    """Project sets to sorted lists for audit payloads that get written to cache JSON.

    transformers 5.8.1 `output_loading_info=True` returns `missing_keys` / `unexpected_keys` /
    `mismatched_keys` as Python sets, and the official Flora TempSeg loader stores that dict verbatim
    in MODEL_LOADED.json. The runner persists model_verification into `*.cache.json` with
    `json.dumps`, which then fails with "Object of type set is not JSON serializable" *after* a
    successful load and generation. Only bookkeeping types are coerced here: sets become sorted
    lists. Tensors, prompts, boundaries, floats and numerics are returned untouched, and the vendor
    files stay byte-identical so `verify_vendor()`'s 52-file hash lock still holds.
    """
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(v) for v in value)
    return value


class FloraBackend:
    def __init__(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('FLORA_CUDA_REQUIRED: CPU preparation does not need --execute')
        pipeline, api, base = flora_modules()

        class ReuseSession(base):
            def load(self, kind):
                if getattr(self, 'kind', None) == kind and self.model is not None:
                    return self.model, self.proc
                result = super().load(kind)
                self.kind = kind
                return result

        self.pipeline, self.api, self.session = pipeline, api, ReuseSession()

    def inputs(self, job, task):
        videos, _ = self.api._inputs(job['video'], 'human', None, None, False, task['instruction'])
        return videos

    @contextmanager
    def audit_generation(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        original = self.pipeline.generate
        count = len(list(directory.glob('*_generation.json')))
        def generate(model, proc, inputs, budget):
            nonlocal count
            count += 1
            path = directory / f'{count:05d}_generation.json'
            start = time.monotonic()
            try:
                text = original(model, proc, inputs, budget)
                write(path, dict(text=text, status='complete', output_limit=budget, wall_s=time.monotonic()-start))
                return text
            except Exception as exc:
                write(path, dict(status='failed', error_type=type(exc).__name__, reason=str(exc),
                                 output_limit=budget, wall_s=time.monotonic()-start))
                raise
        self.pipeline.generate = generate
        try:
            yield
        finally:
            self.pipeline.generate = original

    def segment(self, job, media, task, directory):
        with self.audit_generation(directory):
            rows, audit = self.pipeline.segment(self.inputs(job, task), 'human', self.session,
                                                task['instruction'], lambda *a, **k: None)
        validate_segments(rows, media['duration_s'])
        return _json_safe(dict(segments=rows, temporal_windows=audit, model_verification=self.session.audit))

    def caption(self, job, media, task, boundaries, directory):
        with self.audit_generation(directory):
            rows, audit = self.pipeline.caption_single(self.inputs(job, task), boundaries, 'Human', self.session,
                                                       task['instruction'], lambda *a, **k: None)
        validate_segments(rows, media['duration_s'], captions=True)
        expected = [(s['start_s'], s['end_s']) for s in boundaries]
        if expected != [(r['start_s'], r['end_s']) for r in rows]:
            raise ValueError('FLORA_CAPTION_CHANGED_BOUNDARIES')
        return _json_safe(dict(segments=rows, caption_inputs=audit, model_verification=self.session.audit))

    def close(self):
        self.session.close()
