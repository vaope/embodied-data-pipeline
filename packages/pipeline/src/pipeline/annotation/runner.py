"""Stage-wise rental runner. A/B share exactly one content-addressed S2 cache."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import threading
import time

from .core import ROOT, VARIANTS, STAGES, cached, digest, file_hash, load_jobs, probe, project_v1, read, source_identity, validate_segments, verify_vendor, write


def prepare(manifest, out, *, data_root=None, config=None, limit=10, pilot_review=None, control_dir=None):
    verify_vendor()
    config = read(config or ROOT / 'config.json')
    if control_dir:
        config['variants'] = ['AV-A', 'AV-B']
    if not config.get('variants') or set(config['variants']) - set(VARIANTS):
        raise ValueError('VARIANTS_INVALID')
    batch_limit = config.get('batch_limit', 100)
    if not 1 <= limit <= batch_limit:
        raise ValueError(f'BATCH_LIMIT_INVALID: 1..{batch_limit}')
    if limit > config['pilot_limit']:
        if not pilot_review:
            raise ValueError('PILOT_REVIEW_REQUIRED: prepare <=10 first; supply genuine reviewed pilot receipt for expansion')
        review = read(pilot_review)
        verified = verify(Path(review['pilot_run']))
        pilot_plan = read(Path(review['pilot_run']) / 'plan.json')
        if len(pilot_plan['episodes']) != config['pilot_limit'] or pilot_plan['config'] != config:
            raise ValueError('PILOT_SCOPE_OR_CONFIG_MISMATCH')
        if review.get('reviewer') != 'user' or review.get('accepted') is not True or not verified['ok']:
            raise ValueError('PILOT_REVIEW_INVALID')
        if review.get('plan_sha256') != digest(read(Path(review['pilot_run']) / 'plan.json')):
            raise ValueError('PILOT_REVIEW_HASH_MISMATCH')
    jobs = load_jobs(manifest, data_root)[:limit]
    records = []
    for job in jobs:
        media = probe(job['video'])
        if not media['cfr'] and set(config['variants']) & {'AV-A', 'AV-B'}:
            raise ValueError(f'EGOANT_VFR_UNSUPPORTED: {job["id"]}; explicit preprocessing required')
        record = dict(job=job, media=media)
        if control_dir:
            gt_path = Path(control_dir) / f'{job["id"]}.json'
            gt = read(gt_path)
            if gt['episode_id'] != job['id'] or abs(gt['duration_s']-media['duration_s']) > 1e-6:
                raise ValueError('CONTROL_REFERENCE_SOURCE_MISMATCH')
            boundaries = [dict(start_s=s['start_s'], end_s=s['end_s']) for s in gt['segments']]
            validate_segments(boundaries,media['duration_s'])
            record.update(control_boundaries=boundaries, reference_sha256=file_hash(gt_path))
        records.append(record)
    plan = dict(schema_version='annotation_av.plan.v1', implementation_sha256=source_identity(),
                config=config, episodes=records, requested_limit=limit,
                source_manifest_sha256=file_hash(manifest), pilot_review=read(pilot_review) if pilot_review else None,
                experiment_mode='gt_boundary_caption_control' if control_dir else 'predicted_boundaries')
    out = Path(out)
    if (out / 'plan.json').exists() and read(out / 'plan.json') != plan:
        raise ValueError('RUN_PLAN_CHANGED: use a new output directory')
    write(out / 'plan.json', plan)
    write(out / 'manifest.json', dict(schema_version='annotation_av.manifest.v1',
        variants=config['variants'], plan_sha256=digest(plan), episodes=[dict(episode_id=r['job']['id'],
        source_video_sha256=r['media']['source_video_sha256'], frame_count=r['media']['frame_count'],
        duration_s=r['media']['duration_s']) for r in records]))
    return dict(episodes=len(records), variants=config['variants'], new_model_calls=0, plan_sha256=digest(plan))


def bound_plan(out):
    plan = read(Path(out) / 'plan.json')
    if plan['implementation_sha256'] != source_identity():
        raise ValueError('IMPLEMENTATION_CHANGED: prepare a new run')
    verify_vendor()
    for rec in plan['episodes']:
        if file_hash(rec['job']['video']) != rec['media']['source_video_sha256']:
            raise ValueError(f'SOURCE_CHANGED: {rec["job"]["id"]}')
    return plan


def cache_result(path):
    record = read(path)
    if record.get('result_sha256') != digest(record['result']):
        raise ValueError(f'CACHE_CORRUPT: {path}')
    return record['result']


def validated_cache(directory, stage, plan, eid):
    """Also bind upstream caches to this exact source, config and run, recursively."""
    directory = Path(directory)
    deps = {}
    if stage != 'task':
        validated_cache(directory, 'task', plan, eid)
        deps['task'] = file_hash(directory / 'task.cache.json')
    if stage.startswith('caption-'):
        key = 'tempseg' if stage == 'caption-c' else 's2'
        validated_cache(directory, key, plan, eid)
        deps[key] = file_hash(directory / f'{key}.cache.json')
    record = read(directory / f'{stage}.cache.json')
    expected = digest(dict(plan=digest(plan), episode=eid, stage=stage, dependencies=deps))
    if record['signature'] != expected:
        raise ValueError(f'UPSTREAM_CACHE_MISMATCH: {eid}/{stage}')
    return cache_result(directory / f'{stage}.cache.json')


class _ThreadStdout:
    """sys.stdout router: threads that registered a log write there, others pass through.
    Replaces contextlib.redirect_stdout, which is process-global and unsafe with episode workers."""

    def __init__(self, fallback):
        self.fallback, self.local = fallback, threading.local()

    def write(self, text):
        return (getattr(self.local, 'stream', None) or self.fallback).write(text)

    def flush(self):
        (getattr(self.local, 'stream', None) or self.fallback).flush()

    def __getattr__(self, name):
        return getattr(self.fallback, name)


_STDOUT_LOCK = threading.Lock()


@contextmanager
def thread_stdout(stream):
    with _STDOUT_LOCK:
        if not isinstance(sys.stdout, _ThreadStdout):
            sys.stdout = _ThreadStdout(sys.stdout)
        router = sys.stdout
    previous = getattr(router.local, 'stream', None)
    router.local.stream = stream
    try:
        yield
    finally:
        router.local.stream = previous


class UpstreamFailed(RuntimeError):
    """Dependency of this episode failed or is missing; not an error of this stage."""


def execute_stage(out, stage, *, backend=None, _plan=None, _records=None, _ledger=None, _report_path=None):
    """Run one stage over the batch.

    config['execution'] (all optional; defaults keep the original strict behaviour):
      episode_workers   episodes processed concurrently (default 1)
      continue_on_error isolate failures per episode; the stage finishes, failed episodes keep no
                        success cache and are retried by re-running the stage (default False)
    Budget errors always stop the stage."""
    if stage not in STAGES:
        raise ValueError('STAGE_INVALID')
    out = Path(out)
    plan = _plan if _plan is not None else bound_plan(out)
    records = _records if _records is not None else plan['episodes']
    execution = plan['config'].get('execution', {})
    workers = int(execution.get('episode_workers', 1))
    if _records is not None:
        workers = 1
    isolate = bool(execution.get('continue_on_error', False))
    if workers < 1:
        raise ValueError('EPISODE_WORKERS_INVALID')
    variants = set(plan['config']['variants'])
    needed = {'task': variants, 's2': variants & {'AV-A', 'AV-B'}, 'tempseg': variants & {'AV-C'},
              'caption-a': variants & {'AV-A'}, 'caption-b': variants & {'AV-B'}, 'caption-c': variants & {'AV-C'}}
    if not needed[stage]:
        return dict(stage=stage, status='skipped', reason='variant_not_requested')
    from . import backends as backend_module
    ledger = _ledger if _ledger is not None else backend_module.Ledger.for_run(out, plan['config'])
    # Backend creation is lazy: cached/official-task stages require no CUDA/API.
    instance = backend
    instance_lock = threading.Lock()
    counts = dict(succeeded=0, cached=0, failed=0, skipped_upstream=0)
    report = dict(stage=stage, total_episodes=len(records), counts=counts, episodes=[], status='running',
                  episode_workers=workers, continue_on_error=isolate)
    report_lock = threading.Lock()
    start = time.monotonic()

    def get_backend():
        nonlocal instance
        with instance_lock:
            if instance is None:
                from .backends import EgoANTBackend, FloraBackend
                instance = EgoANTBackend(plan['config']) if stage in ('task', 's2', 'caption-a') else FloraBackend()
            return instance

    def flush_report():
        report['wall_s'] = time.monotonic() - start
        report['usage'] = ledger.summary()
        write(_report_path or out / 'stage_reports' / f'{stage}.json', report)

    def one(record):
        job, media = record['job'], record['media']
        eid = job['id']
        from .scheduling import current
        runtime = current()
        if runtime:
            runtime.event('stage_start', episode=eid, stage=stage)
        directory = out / 'episodes' / eid
        directory.mkdir(parents=True, exist_ok=True)
        dependencies, task, boundaries = {}, None, None
        try:
            try:
                if stage != 'task':
                    dependencies['task'] = file_hash(directory / 'task.cache.json')
                    task = validated_cache(directory, 'task', plan, eid)
                if stage.startswith('caption-'):
                    key = 'tempseg' if stage == 'caption-c' else 's2'
                    dependencies[key] = file_hash(directory / f'{key}.cache.json')
                    boundaries = validated_cache(directory, key, plan, eid)['segments']
                    validate_segments(boundaries, media['duration_s'])
            except FileNotFoundError as exc:
                if not isolate:
                    raise
                raise UpstreamFailed(f'missing upstream cache: {Path(exc.filename).name}') from exc
            signature = digest(dict(plan=digest(plan), episode=eid, stage=stage, dependencies=dependencies))

            def compute():
                # `instruction` is None only when the manifest omitted the field (then a task may be
                # synthesised). An explicit blank string is a real dataset value and must be kept.
                if stage == 'task' and job.get('instruction') is not None:
                    return dict(instruction=job['instruction'], task_source='official_instruction',
                                instruction_source=job['instruction_source'])
                if stage == 's2' and plan['experiment_mode'] == 'gt_boundary_caption_control':
                    return dict(segments=record['control_boundaries'], boundary_origin='human_reference_control',
                                reference_sha256=record['reference_sha256'])
                engine = get_backend()
                with (directory / f'{stage}.runtime.log').open('a', encoding='utf-8') as log, thread_stdout(log):
                    if stage == 'task':
                        return engine.task(job, directory / 'raw' / stage)
                    if stage in ('s2', 'tempseg'):
                        result = engine.segment(job, media, task, directory / 'raw' / stage)
                        validate_segments(result['segments'], media['duration_s'])
                        return result
                    result = engine.caption(job, media, task, boundaries, directory / 'raw' / stage)
                    validate_segments(result['segments'], media['duration_s'], captions=True)
                    if [(s['start_s'], s['end_s']) for s in boundaries] != [(s['start_s'], s['end_s']) for s in result['segments']]:
                        raise ValueError('CAPTION_BOUNDARY_MISMATCH')
                    return result

            result, hit = cached(directory / f'{stage}.cache.json', signature, compute)
            if stage.startswith('caption-'):
                variant = 'AV-' + stage[-1].upper()
                write_outputs(directory, variant, job, media, task, result, dependencies, plan)
            (directory / f'{stage}.failure.json').unlink(missing_ok=True)
            row = dict(id=eid, status='cached' if hit else 'succeeded')
        except backend_module.BudgetExceeded:
            raise
        except UpstreamFailed as exc:
            row = dict(id=eid, status='skipped_upstream', reason=str(exc))
        except Exception as exc:
            row = dict(id=eid, status='failed', reason=type(exc).__name__ + ': ' + str(exc))
            write(directory / f'{stage}.failure.json', row)
            if not isolate:
                with report_lock:
                    counts['failed'] += 1
                    report['episodes'].append(row)
                    flush_report()
                raise
        with report_lock:
            counts[{'cached': 'cached', 'succeeded': 'succeeded', 'failed': 'failed',
                    'skipped_upstream': 'skipped_upstream'}[row['status']]] += 1
            report['episodes'].append(row)
            flush_report()
        return row

    try:
        with backend_module.use_ledger(ledger) if _ledger is None else nullcontext():
            if workers == 1:
                for record in records:
                    one(record)
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f'av-{stage}') as pool:
                    futures = [pool.submit(one, r) for r in records]
                    first = None
                    for f in futures:
                        try:
                            f.result()
                        except BaseException as exc:      # strict mode or budget: stop queued work
                            if first is None:
                                first = exc
                                if isinstance(exc, backend_module.BudgetExceeded):
                                    backend_module.STOP.set()
                                for g in futures:
                                    g.cancel()
                    if first is not None:
                        raise first
    except BaseException as exc:
        with report_lock:
            done = {r['id'] for r in report['episodes']}
            report['status'] = 'failed'
            report['error'] = type(exc).__name__ + ': ' + str(exc)
            report['not_attempted'] = len([r for r in records if r['job']['id'] not in done])
            flush_report()
        raise
    finally:
        if instance is not None and hasattr(instance, 'close'):
            instance.close()
    # Keep the report in manifest order for readability.
    order = {r['job']['id']: i for i, r in enumerate(plan['episodes'])}
    report['episodes'].sort(key=lambda r: order[r['id']])
    report['status'] = 'complete' if not (counts['failed'] or counts['skipped_upstream']) else 'partial'
    flush_report()
    return report


def execute_dag(out, stages=('task', 's2', 'caption-a'), *, backend=None):
    """Per-episode dependencies, shared HTTP gate, bounded independent interval work.

    Initial validation/hashing happens once. Each episode retains the same bound
    stage signatures, audit layout, isolation and output validation as run-stage.
    """
    from . import backends
    from .scheduling import use_runtime
    out = Path(out)
    plan = bound_plan(out)
    if plan['config']['variants'] != ['AV-A'] or plan['config']['qwen'].get('provider') != 'chutes':
        raise ValueError('DAG_REQUIRES_CHUTES_AV_A_PROFILE')
    if tuple(stages) != ('task', 's2', 'caption-a'):
        raise ValueError('DAG_STAGES_REQUIRED: task s2 caption-a; use --scheduler stage for other subsets')
    c = plan['config'].get('execution', {})
    segment_workers = int(c.get('segment_workers', 64))
    caption_workers = int(c.get('caption_episode_workers', 16))
    if min(segment_workers, caption_workers) < 1:
        raise ValueError('DAG_EPISODE_WORKERS_INVALID')
    ledger = backends.Ledger.for_run(out, plan['config'])
    reports = {s: dict(stage=s, total_episodes=len(plan['episodes']),
                      counts=dict(succeeded=0, cached=0, failed=0, skipped_upstream=0),
                      episodes=[], status='running', scheduler='dag') for s in stages}
    start = time.monotonic()
    fatal = None
    pending = {}
    order = {r['job']['id']: i for i, r in enumerate(plan['episodes'])}

    def flush():
        for stage, report in reports.items():
            report['wall_s'] = time.monotonic() - start
            report['usage'] = ledger.summary()
            report['episodes'].sort(key=lambda r: order[r['id']])
            write(out / 'stage_reports' / f'{stage}.json', report)

    @contextmanager
    def cancellation_guard():
        nonlocal fatal
        try:
            yield
        except BaseException as exc:
            fatal = fatal or exc
            backends.STOP.set()
            for f in pending:
                f.cancel()
            for report in reports.values():
                report['status'] = 'failed'
                report['error'] = type(exc).__name__ + ': ' + str(exc)
                report['not_attempted'] = report['total_episodes'] - len(report['episodes'])
            flush()
            raise

    # Separate coordinator pools: caption fanout cannot consume segmentation coordinators.
    # All actual child work and image payloads are bounded by the shared runtime.
    with use_runtime(out, plan['config'], backends.STOP) as runtime, backends.use_ledger(ledger), \
            ThreadPoolExecutor(max_workers=segment_workers, thread_name_prefix='av-segment') as segments, \
            ThreadPoolExecutor(max_workers=caption_workers, thread_name_prefix='av-caption') as captions, \
            cancellation_guard():
        def submit(record, index):
            stage = stages[index]
            pool = captions if stage.startswith('caption-') else segments
            report_path = out / 'episode_reports' / record['job']['id'] / f'{stage}.json'
            runtime.event('stage_queued', episode=record['job']['id'], stage=stage)
            f = pool.submit(execute_stage, out, stage, backend=backend, _plan=plan,
                            _records=[record], _ledger=ledger, _report_path=report_path)
            pending[f] = record, index

        # Longest first, stable tie break. Prompt/input/output order remains unchanged per episode.
        for record in sorted(plan['episodes'], key=lambda r: (-r['media']['duration_s'], r['job']['id'])):
            submit(record, 0)
        flush()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda f: (-pending[f][0]['media']['duration_s'], pending[f][0]['job']['id'])):
                record, index = pending.pop(future)
                stage = stages[index]
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                    row = result['episodes'][0]
                    reports[stage]['episodes'].append(row)
                    reports[stage]['counts'][row['status']] += 1
                    runtime.event('stage_end', episode=record['job']['id'], stage=stage, status=row['status'])
                    if fatal is None and index + 1 < len(stages):
                        # Failed dependencies proceed to the existing skipped_upstream path.
                        submit(record, index + 1)
                except BaseException as exc:
                    if fatal is None:
                        fatal = exc
                        backends.STOP.set()
                        for f in pending:
                            f.cancel()
                    runtime.event('stage_error', episode=record['job']['id'], stage=stage,
                                  error=type(exc).__name__)
            flush()
        for report in reports.values():
            report['not_attempted'] = report['total_episodes'] - len(report['episodes'])
            report['status'] = ('failed' if fatal else 'partial' if
                report['counts']['failed'] or report['counts']['skipped_upstream'] else 'complete')
            if fatal:
                report['error'] = type(fatal).__name__ + ': ' + str(fatal)
        flush()
    if fatal:
        raise fatal
    return [reports[s] for s in stages]


def write_outputs(directory, variant, job, media, task, result, dependencies, plan):
    native_path = directory / f'{variant}.native.json'
    write(native_path, dict(schema_version='annotation_av.native.v1', variant=variant,
                           experiment_mode=plan['experiment_mode'],
                           episode_id=job['id'], duration_s=media['duration_s'], task=task,
                           segments=result['segments'], interval='[start_s,end_s)',
                           source_clock='relative to first source presentation timestamp',
                           source_first_pts_s=media['first_source_pts_s'], audit=result))
    annotation = project_v1(job['id'], variant, media, task, result['segments'])
    annotation_path = directory / f'annotation_{variant[-1]}.json'
    write(annotation_path, annotation)
    write(directory / f'{variant}.provenance.json', dict(schema_version='annotation_av.provenance.v1',
        variant=variant, episode_id=job['id'], source_video_sha256=media['source_video_sha256'],
        experiment_mode=plan['experiment_mode'],
        plan_sha256=digest(plan), implementation_sha256=plan['implementation_sha256'],
        boundary_stage='tempseg' if variant == 'AV-C' else 's2',
        boundary_cache_sha256=dependencies['tempseg' if variant == 'AV-C' else 's2'],
        task_cache_sha256=dependencies['task'], task_source=task['task_source'],
        annotation_sha256=file_hash(annotation_path), native_sha256=file_hash(native_path),
        caption_cache_sha256=file_hash(directory / f'caption-{variant[-1].lower()}.cache.json'),
        method_enum_note='Schema v1 A/B/C maps to AV-A/B/C only with this provenance file',
        program_status='succeeded', user_accepted=False, semantic_accuracy_evaluated=False))


def verify(out):
    """Read-back checks; missing/failed outputs make the whole requested batch fail."""
    out = Path(out)
    plan = bound_plan(out)
    import jsonschema
    validator = jsonschema.Draft202012Validator(read(ROOT / 'schemas/step.schema.json'))
    checks, failures = [], []
    for rec in plan['episodes']:
        directory = out / 'episodes' / rec['job']['id']
        for variant in plan['config']['variants']:
            try:
                prov = read(directory / f'{variant}.provenance.json')
                ann_path = directory / f'annotation_{variant[-1]}.json'
                native_path = directory / f'{variant}.native.json'
                if prov['plan_sha256'] != digest(plan) or prov['annotation_sha256'] != file_hash(ann_path) or prov['native_sha256'] != file_hash(native_path):
                    raise ValueError('OUTPUT_HASH_MISMATCH')
                native = read(native_path)
                rows = validate_segments(native['segments'], rec['media']['duration_s'], captions=True)
                ann = read(ann_path)
                validator.validate(ann)
                task = validated_cache(directory, 'task', plan, rec['job']['id'])
                expected = project_v1(rec['job']['id'], variant, rec['media'], task, rows)
                if ann != expected:
                    raise ValueError('PROJECTION_MISMATCH')
                boundary = directory / f'{prov["boundary_stage"]}.cache.json'
                if prov['boundary_cache_sha256'] != file_hash(boundary) or prov['task_cache_sha256'] != file_hash(directory / 'task.cache.json'):
                    raise ValueError('DEPENDENCY_HASH_MISMATCH')
                src = validated_cache(directory, prov['boundary_stage'], plan, rec['job']['id'])['segments']
                caption_stage = f'caption-{variant[-1].lower()}'
                caption_cache = directory / f'{caption_stage}.cache.json'
                if file_hash(caption_cache) != prov['caption_cache_sha256'] or validated_cache(directory, caption_stage, plan, rec['job']['id'])['segments'] != rows:
                    raise ValueError('CAPTION_CACHE_MISMATCH')
                if [(r['start_s'], r['end_s']) for r in rows] != [(r['start_s'], r['end_s']) for r in src]:
                    raise ValueError('BOUNDARY_MISMATCH')
                checks.append(dict(id=rec['job']['id'], variant=variant, segments=len(rows)))
            except Exception as exc:
                failures.append(dict(id=rec['job']['id'], variant=variant, reason=str(exc)))
    report = dict(ok=not failures, checked=len(checks), expected=len(plan['episodes']) * len(plan['config']['variants']),
                  checks=checks, failures=failures, semantic_accuracy_evaluated=False, user_accepted=False)
    write(out / 'verification.json', report)
    return report
