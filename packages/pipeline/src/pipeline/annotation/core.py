"""Portable contracts, content-bound caches and conservative Schema v1 projection."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
VARIANTS = ('AV-A', 'AV-B', 'AV-C')
STAGES = ('task', 's2', 'caption-a', 'tempseg', 'caption-b', 'caption-c')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def duration_limit_s():
    """Source-duration limit for the Human entry (official: 120s).

    WGO-Bench homer has 11 episodes of 121-171s, which the official envelope rejects. An extension
    run may raise the cap explicitly with ANNOTATION_MAX_DURATION_S; anything produced that way is a
    documented NON-official extension and must be reported separately from 120s-compliant runs.
    """
    raw = os.environ.get('ANNOTATION_MAX_DURATION_S', '').strip()
    if not raw:
        return 120.0
    value = float(raw)
    if not 120.0 <= value <= 600.0:
        raise ValueError(f'ANNOTATION_MAX_DURATION_S out of range (120..600): {value}')
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def source_identity():
    """All wrapper/config/prompt/vendor bytes affect cache identity."""
    return digest({str(p.relative_to(ROOT)).replace('\\', '/'): file_hash(p)
                   for p in sorted(ROOT.rglob('*')) if p.is_file()
                   and p.suffix in {'.py', '.json', '.txt', '.sh', '.md'}
                   and '__pycache__' not in p.parts and 'tests' not in p.parts})


def verify_vendor():
    manifest = read(ROOT / 'vendor/SOURCES.json')
    for row in manifest['files']:
        path = ROOT / 'vendor' / row['path']
        if not path.resolve().is_relative_to((ROOT / 'vendor').resolve()):
            raise ValueError('VENDOR_PATH_INVALID')
        if file_hash(path) != row['sha256']:
            raise ValueError(f'VENDOR_HASH_MISMATCH: {row["path"]}')
    return manifest


def load_jobs(path, data_root=None):
    path = Path(path).resolve()
    root = Path(data_root).resolve() if data_root else path.parent
    jobs, seen = [], set()
    for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - {'id', 'video', 'instruction', 'instruction_source', 'source_id'}:
            raise ValueError(f'MANIFEST_FIELDS_INVALID: line {n}; GT/labels are not generation inputs')
        eid = row.get('id', '')
        if not isinstance(eid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', eid) or eid in seen:
            raise ValueError(f'MANIFEST_ID_INVALID: line {n}')
        seen.add(eid)
        if not isinstance(row.get('video'), str) or not row['video'].strip():
            raise ValueError(f'MANIFEST_VIDEO_INVALID: {eid}')
        # Distinguish "field absent" (no instruction was provided -> the task stage may synthesise one)
        # from "field present but blank" (the dataset says there is no instruction -> keep it blank).
        # WGO-Bench homer has episodes with a blank instruction; Flora's own CLI defaults to blank.
        has_instruction = 'instruction' in row
        instruction = row['instruction'] if has_instruction else ''
        if not isinstance(instruction, str) or len(instruction) > 2000:
            raise ValueError(f'MANIFEST_INSTRUCTION_INVALID: {eid}')
        if instruction.strip() and not row.get('instruction_source'):
            raise ValueError(f'MANIFEST_TASK_SOURCE_MISSING: {eid}')
        video = (root / row['video']).resolve(strict=True)
        if not video.is_file():
            raise ValueError(f'MANIFEST_VIDEO_INVALID: {eid}')
        jobs.append(dict(row, video=str(video),
                         instruction=instruction.strip() if has_instruction else None))
    if not jobs:
        raise ValueError('MANIFEST_EMPTY')
    return jobs


def probe(path):
    """ffprobe native relative PTS; A/B require CFR because upstream uses frame/fps."""
    p = Path(path)
    if not 0 < p.stat().st_size <= 256 * 1024**2:
        raise ValueError('MEDIA_SIZE_UNSUPPORTED: require 0 < bytes <= 256 MiB')
    args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
            '-show_entries', 'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time,duration_time,pkt_duration_time',
            '-of', 'json', str(p)]
    result = json.loads(subprocess.run(args, check=True, capture_output=True, text=True, encoding='utf-8').stdout)
    stream = result['streams'][0]
    from fractions import Fraction
    fps = float(Fraction(stream['avg_frame_rate']))
    pts = [float(f['best_effort_timestamp_time']) for f in result['frames']]
    if len(pts) < 2 or not all(math.isfinite(t) for t in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('MEDIA_PTS_INVALID')
    zero = pts[0]
    pts = [t - zero for t in pts]
    tail = result['frames'][-1]
    duration = pts[-1] + float(tail.get('duration_time') or tail.get('pkt_duration_time') or 1 / fps)
    limit = duration_limit_s()
    if fps < 4 or duration > limit + 1e-6 or len(pts) > 7200 or max(stream['width'], stream['height']) > 4096:
        raise ValueError(f'MEDIA_PROFILE_UNSUPPORTED: require >=4fps, <={limit:g}s, <=7200frames, '
                         f'<=4096px; split long sources first')
    cfr = all(abs(t - i / fps) <= 2e-5 for i, t in enumerate(pts))
    return dict(source_video_sha256=file_hash(p), frame_count=len(pts), fps=fps,
                duration_s=duration, timestamps_s=pts, first_source_pts_s=zero,
                cfr=cfr, width=stream['width'], height=stream['height'])


def validate_segments(segments, duration, captions=False):
    if not isinstance(segments, list) or not segments:
        raise ValueError('SEGMENTS_EMPTY')
    previous = -1.
    for row in segments:
        a, b = row['start_s'], row['end_s']
        if isinstance(a, bool) or isinstance(b, bool) or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in (a, b)):
            raise ValueError('SEGMENT_TIME_INVALID')
        if not 0 <= a < b <= duration + 1e-6 or a < previous:
            raise ValueError('SEGMENT_RANGE_INVALID')
        previous = a
        if captions and (row.get('status') != 'complete' or not isinstance(row.get('caption'), str)
                         or not row['caption'].strip() or row['caption'].startswith('[FAILED]')):
            raise ValueError('CAPTION_INCOMPLETE')
    return segments


def cached(path, signature, compute):
    """Never reuse changed/corrupt input/output. Failed computation leaves no success cache."""
    path = Path(path)
    if path.exists():
        record = read(path)
        if record['signature'] != signature or record['result_sha256'] != digest(record['result']):
            raise ValueError(f'CACHE_MISMATCH: {path}; use a new output directory')
        return record['result'], True
    start = time.monotonic()
    result = compute()
    write(path, dict(signature=signature, result_sha256=digest(result), result=result,
                     wall_s=time.monotonic() - start))
    return result, False


def project_v1(eid, variant, media, task, rows):
    """Only caption+time is model evidence; unknown attributes remain unknown."""
    validate_segments(rows, media['duration_s'], captions=True)
    pts = media['timestamps_s']
    edges = pts + [media['duration_s']]
    steps = []
    for i, row in enumerate(rows):
        lo = bisect.bisect_left(edges, row['start_s'] - 1e-7)
        # Upstream segmentation derives its duration from OpenCV frame_count/fps while probe uses
        # ffprobe duration_time (quantised to 1e-6 s), so a full-cover segment can end up to 1e-6 s
        # past the sentinel - a slack validate_segments already accepts. Clamp the end index onto
        # the frame grid (end_frame_exclusive never exceeds frame_count); a segment that still spans
        # no frame is a real sub-frame defect and stays an error. No silent expansion either way.
        hi = min(bisect.bisect_left(edges, row['end_s'] - 1e-7), len(pts))
        if not 0 <= lo < hi <= len(pts):
            raise ValueError('SUBFRAME_SEGMENT: retain raw native output; no silent expansion')
        steps.append(dict(step_id=i, start_frame=lo, end_frame_exclusive=hi, action_type='unknown',
                          description=row['caption'], acting_hand='unknown', target_object_id=None,
                          task_phase='unknown', outcome='unknown', uncertainty=['metadata_not_provided_by_captioner']))
    issue = dict(code='AV_METADATA_UNAVAILABLE', severity='warning',
                 message='Native captioners supply text and time; hand/object/action enums and outcome are not inferred.')
    annotation = dict(episode_id=eid, method=variant[-1], fps=media['fps'], frame_count=len(pts),
                      timebase=dict(interval='[start_frame,end_frame_exclusive)',
                                    timestamp_source='ffprobe_relative_native_pts', duration_s=media['duration_s']),
                      task=dict(goal=task['instruction'], description=task['instruction'], outcome='unknown'),
                      objects=[], steps=steps, annotation_source='vlm',
                      auto_qa=dict(status='warning', schema_valid=True, temporal_valid=True, issues=[issue]),
                      review=dict(status='not_requested', reviewer=None))
    import jsonschema
    jsonschema.Draft202012Validator(read(ROOT / 'schemas/step.schema.json')).validate(annotation)
    return annotation
