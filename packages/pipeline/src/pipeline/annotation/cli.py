from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

from .core import ROOT, STAGES, digest, file_hash, load_jobs, read, verify_vendor, write
from .runner import execute_stage, execute_dag, prepare, verify


def assets(cache, kind, config=None):
    """Run only on the rental host. Hash all downloaded assets and pin revisions.

    Weight pinning is a local-inference concern: the default annotation profile now points at a
    hosted endpoint with no HF repo/revision, so this command reads the local rental profile and
    refuses to run against a profile without a pinned repo+revision.
    """
    cache = Path(cache).resolve()
    profile = read(config or ROOT / 'configs/local_rental_qwen36.json')
    specs = read(ROOT / 'vendor/flora/batch_utils/models.json')
    specs['qwen'] = {k: v for k, v in profile['qwen'].items() if k in ('repo', 'revision')}
    if specs['qwen'].get('repo') is None:
        raise ValueError('ASSETS_REQUIRE_LOCAL_PROFILE: this profile serves the model remotely '
                         '(qwen.repo/revision null); pass --config configs/local_rental_qwen36.json')
    from huggingface_hub import snapshot_download
    groups = ['qwen'] if kind == 'qwen' else ['tempseg', 'tempseg_base', 'captioner']
    lock_path = cache / f'{kind}.assets.json'
    record = dict(schema_version='annotation_av.assets.v1', models={})
    for name in groups:
        spec = specs[name]
        patterns = ['*.json', '*.jinja', '*.safetensors', '*.txt', 'README.md', 'MODEL_PROVENANCE.json', 'SOURCE_MANIFEST.sha256']
        location = Path(snapshot_download(spec['repo'], revision=spec['revision'],
            token=os.environ.get('HF_TOKEN'), local_dir=None if name == 'tempseg_base' else cache / name, allow_patterns=patterns))
        files = {str(p.relative_to(location)).replace('\\', '/'): file_hash(p) for p in sorted(location.rglob('*'))
                 if p.is_file() and '.cache' not in p.relative_to(location).parts}
        if not any(n.endswith('.safetensors') for n in files):
            raise ValueError(f'WEIGHTS_MISSING: {name}')
        record['models'][name] = dict(spec, path=str(location), files=files)
    write(lock_path, record)
    return dict(lock=str(lock_path), models=list(record['models']))


def pack(manifest, output, data_root=None, with_videos=False):
    """Portable code+manifest bundle; never include credentials, outputs or environments."""
    jobs = load_jobs(manifest, data_root)
    rows = []
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identities = {}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('pipeline/__init__.py', '"""Portable annotation package."""\n')
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts:
                continue
            arcname = 'pipeline/annotation/' + path.relative_to(ROOT).as_posix()
            z.write(path, arcname)
            identities[arcname] = file_hash(path)
        for job in jobs:
            rel = f'data/{job["id"]}.mp4'
            rows.append(dict(job, video=rel))
            identities[rel] = file_hash(job['video'])
            if with_videos:
                z.write(job['video'], rel)
        z.writestr('jobs.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        z.writestr('BUNDLE.json', json.dumps(dict(with_videos=with_videos, files=identities), indent=2))
    write(str(output) + '.sha256.json', dict(path=output.name, sha256=file_hash(output), bytes=output.stat().st_size,
                                           with_videos=with_videos, episodes=len(jobs)))
    return dict(bundle=str(output), bytes=output.stat().st_size, episodes=len(jobs), videos_included=with_videos)


def doctor():
    vendors = verify_vendor()
    packages = {}
    from importlib.metadata import version, PackageNotFoundError
    for name in ['torch', 'transformers', 'peft', 'av', 'opencv-python-headless', 'vllm', 'jsonschema']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(platform=platform.platform(), python=sys.version, ffmpeg=shutil.which('ffmpeg'),
                ffprobe=shutil.which('ffprobe'), packages=packages, vendor_files_verified=len(vendors['files']),
                new_model_calls=0, gpu_inference_verified=False)


def stage_summary(report):
    """Console view of a stage report; the full report stays in stage_reports/<stage>.json."""
    failed = [r for r in report.get('episodes', []) if r['status'] in ('failed', 'skipped_upstream')]
    return dict(stage=report.get('stage'), status=report.get('status'), counts=report.get('counts'),
                wall_s=round(report.get('wall_s', 0), 1), usage=report.get('usage'),
                failed=failed[:20], failed_total=len(failed))


def main(argv=None):
    p = argparse.ArgumentParser(description='New AV-A/B/C annotation; old Stage3-RGBabc is retired.')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('doctor', help='Offline environment and vendored-source check')
    d.add_argument('--out', type=Path)
    q = sub.add_parser('prepare', help='Decode/probe/hash inputs; no models or downloads')
    q.add_argument('--manifest', type=Path, required=True)
    q.add_argument('--data-root', type=Path)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--config', type=Path)
    q.add_argument('--limit', type=int, default=10)
    q.add_argument('--pilot-review', type=Path)
    c = sub.add_parser('prepare-control', help='Separate fixed-human-boundary A/B caption evaluation run')
    c.add_argument('--manifest', type=Path, required=True)
    c.add_argument('--reference-dir', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--data-root', type=Path)
    c.add_argument('--limit', type=int, default=10)
    r = sub.add_parser('run-stage', help='Execute one model stage on rental hardware; cached results resume')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--stage', choices=STAGES, required=True)
    r.add_argument('--execute', action='store_true', help='Required for actual inference')
    rc = sub.add_parser('run-chain', help='Run several stages in order, then verify (Chutes/API runs)')
    rc.add_argument('--out', type=Path, required=True)
    rc.add_argument('--stages', nargs='+', choices=STAGES, default=['task', 's2', 'caption-a'])
    rc.add_argument('--execute', action='store_true', help='Required for actual inference')
    rc.add_argument('--scheduler', choices=['dag', 'stage'],
                    help='Default from bound config; DAG overlaps episodes and parallelizes intervals')
    v = sub.add_parser('verify', help='Read back every requested variant, hashes, boundaries and Schema v1')
    v.add_argument('--out', type=Path, required=True)
    h = sub.add_parser('handoff', help='Write verified Stage5 sample fields, no training approval')
    h.add_argument('--out', type=Path, required=True)
    h.add_argument('--variant', choices=['AV-A', 'AV-B', 'AV-C'], default='AV-B')
    e = sub.add_parser('score', help='Temporal IoU P/R/F1 against independent GT')
    e.add_argument('--prediction', type=Path, required=True, help='AV native JSON')
    e.add_argument('--reference', type=Path, required=True, help='Independent JSON with segments and duration_s')
    e.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('assets', help='Explicitly download pinned weights on rental machine')
    a.add_argument('--cache', type=Path, required=True)
    a.add_argument('--kind', choices=['qwen', 'flora'], required=True)
    a.add_argument('--config', type=Path,
                   help='profile that pins local weights (default configs/local_rental_qwen36.json)')
    b = sub.add_parser('pack', help='Build a portable upload archive')
    b.add_argument('--manifest', type=Path, required=True)
    b.add_argument('--data-root', type=Path)
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--with-videos', action='store_true')
    args = p.parse_args(argv)
    try:
        if args.command == 'doctor':
            result = doctor()
            if args.out:
                write(args.out, result)
        elif args.command == 'prepare-control':
            result = prepare(args.manifest, args.out, data_root=args.data_root, limit=args.limit,
                             control_dir=args.reference_dir)
        elif args.command == 'prepare':
            result = prepare(args.manifest, args.out, data_root=args.data_root, config=args.config,
                             limit=args.limit, pilot_review=args.pilot_review)
        elif args.command == 'run-stage':
            if not args.execute:
                result = dict(stage=args.stage, status='planned', new_model_calls=0,
                              command_requires='--execute')
            else:
                result = execute_stage(args.out, args.stage)
                print(json.dumps(stage_summary(result), ensure_ascii=False, indent=2))
                return 0 if result.get('status') in ('complete', 'skipped') else 3
        elif args.command == 'run-chain':
            if not args.execute:
                result = dict(stages=args.stages, status='planned', new_model_calls=0, command_requires='--execute')
            else:
                scheduler = args.scheduler or read(args.out / 'plan.json')['config'].get('execution', {}).get('scheduler', 'stage')
                reports = (execute_dag(args.out, args.stages) if scheduler == 'dag' else
                           [execute_stage(args.out, stage) for stage in args.stages])
                summaries = [stage_summary(report) for report in reports]
                checked = verify(args.out)
                result = dict(stages=summaries, verification=dict(ok=checked['ok'], checked=checked['checked'],
                              expected=checked['expected'], failures=len(checked['failures'])))
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if checked['ok'] else 3
        elif args.command == 'verify':
            result = verify(args.out)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['ok'] else 1
        elif args.command == 'score':
            from .evaluate import temporal_score
            pred, gt = read(args.prediction), read(args.reference)
            if pred['episode_id'] != gt['episode_id'] or abs(pred['duration_s']-gt['duration_s']) > 1e-6:
                raise ValueError('EVALUATION_SOURCE_MISMATCH')
            result = temporal_score(pred['segments'], gt['segments'], gt['duration_s'])
            result.update(prediction_sha256=file_hash(args.prediction), reference_sha256=file_hash(args.reference))
            write(args.output, result)
        elif args.command == 'handoff':
            checked = verify(args.out)
            if not checked['ok']:
                raise ValueError('HANDOFF_REQUIRES_VERIFIED_BATCH')
            plan = read(args.out / 'plan.json')
            if plan['experiment_mode'] != 'predicted_boundaries':
                raise ValueError('CONTROL_RUN_IS_EVALUATION_ONLY')
            if args.variant not in plan['config']['variants']:
                raise ValueError('VARIANT_NOT_RUN')
            rows = []
            for rec in plan['episodes']:
                path = (args.out / 'episodes' / rec['job']['id'] / f'annotation_{args.variant[-1]}.json').resolve()
                rows.append(dict(episode_id=rec['job']['id'], stage3_annotation_av=str(path),
                                 stage3_sha256=file_hash(path), stage3_variant=args.variant))
            result = dict(schema_version='annotation_av.stage5_handoff.v1', samples=rows,
                          user_accepted=False, training_ready=False)
            write(args.out / 'stage5_handoff.json', result)
        elif args.command == 'assets':
            result = assets(args.cache, args.kind, config=args.config)
        else:
            result = pack(args.manifest, args.output, args.data_root, args.with_videos)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
