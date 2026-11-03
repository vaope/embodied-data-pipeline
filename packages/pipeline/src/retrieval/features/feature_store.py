"""Content-addressed motion/text cache; publish a complete immutable index generation."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid
import numpy as np
from ..common import digest, file_hash, load_config, read_json, write_json
from ..data.egodex_reader import read_episode
from ..data.annotation_adapter import load_annotation, window_text
from ..data.window_builder import build_windows
from .motion_features import extract_window
from .semantic_encoder import SemanticEncoder

def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*.py'))
                   if 'tests' not in p.parts})

class FeatureStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        pointer = read_json(self.root/'current.json')
        self.generation = self.root/pointer['generation']
        self.meta = read_json(self.generation/'index.json')
        self.rows = [__import__('json').loads(line) for line in
                     (self.generation/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        self.by_id = {r['clip_id']: r for r in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError('duplicate clip IDs in index')
        self.embeddings = np.load(self.generation/'semantic_embeddings.npy', allow_pickle=False)
        if len(self.embeddings) != len(self.rows) or not np.isfinite(self.embeddings).all():
            raise ValueError('invalid semantic matrix')

    def get_clip_features(self, clip_id):
        row = self.by_id[clip_id]
        path = self.root/row['motion_path']
        with np.load(path, allow_pickle=False) as data:
            motion = {key: data[key].copy() for key in data.files}
        return {'metadata': row, 'motion': motion,
                'semantic': self.embeddings[row['embedding_index']].copy()}

def _normalize_sources(dataset_root, annotation_root):
    source = Path(dataset_root).resolve()
    if source.is_dir():
        source = source/'sources.json'
    payload = read_json(source)
    rows = payload.get('episodes', []) if isinstance(payload, dict) else payload
    seen = set()
    normalized = []
    for original in rows:
        row = dict(original)
        eid = row['episode_id']
        if eid in seen:
            raise ValueError(f'duplicate episode_id: {eid}')
        seen.add(eid)
        row['video_path'] = row.get('video_path') or row.get('source_mp4')
        row['hdf5_path'] = row.get('hdf5_path') or row.get('source_hdf5')
        if annotation_root and not row.get('annotation_path'):
            p = Path(annotation_root)/'episodes'/eid/'annotation_A.json'
            if p.is_file():
                row['annotation_path'] = str(p.resolve())
        for key in ('video_path', 'hdf5_path', 'annotation_path', 'playback_video_path'):
            if row.get(key):
                p = Path(row[key])
                row[key] = str((source.parent/p).resolve() if not p.is_absolute() else p.resolve())
        normalized.append(row)
    return normalized

def build_index(dataset_root, annotation_root=None, config=None, encoder=None):
    cfg = load_config(config)
    root = Path(cfg['artifact_root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_rows = _normalize_sources(dataset_root, annotation_root)
    signature = digest({'config': cfg, 'implementation': implementation_hash(),
                        'schema': file_hash(cfg['schema_path'])})
    rows, episode_reports, rejected = [], [], []
    stats = {'episodes_requested': len(source_rows), 'episodes_reused': 0, 'windows_reused': 0,
             'motion_generated': 0, 'semantic_generated': 0, 'semantic_reused': 0}
    for row in source_rows:
        try:
            hashes = {key: file_hash(row[key]) for key in
                      ('video_path', 'hdf5_path', 'annotation_path') if row.get(key)}
            fingerprint = digest({'row': row, 'hashes': hashes, 'signature': signature})
            cache = root/'episode_cache'/f'{fingerprint}.json'
            cached = read_json(cache) if cache.is_file() else None
            if cached and all((root/r['motion_path']).is_file() for r in cached['rows']):
                rows.extend(cached['rows'])
                episode_reports.append(cached['report'])
                rejected.extend(cached['rejected'])
                stats['episodes_reused'] += 1
                stats['windows_reused'] += len(cached['rows'])
                continue
            episode = read_episode(row, cfg)
            ann, warnings = load_annotation(row, cfg, len(episode['times']))
            report = dict(episode['report'], source_hashes=hashes, annotation_warnings=warnings,
                          windows_total=0, windows_valid=0, data_warnings=row.get('data_warnings', []))
            local_rows, local_rejected = [], []
            for start, end in build_windows(len(episode['times']), cfg['horizon'], cfg['stride']):
                report['windows_total'] += 1
                clip_id = f"{row['episode_id']}_f{start:06d}_{end:06d}"
                features, flags, issues = extract_window(episode['hands'], start, end, cfg)
                if features is None:
                    local_rejected.append({'clip_id': clip_id, 'reason': issues})
                    continue
                try:
                    text = window_text(ann, start, end, episode['official_text'], row)
                except ValueError as exc:
                    local_rejected.append({'clip_id': clip_id, 'reason': str(exc)})
                    continue
                key = digest([fingerprint, start, end])
                motion_path = Path('motion_features')/f'{key}.npz'
                (root/motion_path).parent.mkdir(exist_ok=True)
                np.savez_compressed(root/motion_path, **features)
                record = {'clip_id': clip_id, 'episode_id': row['episode_id'],
                          'video_path': row['video_path'], 'hdf5_path': row['hdf5_path'],
                          'playback_video_path': row.get('playback_video_path', row['video_path']),
                          'start_frame': start, 'end_frame': end, 'pose_frame_count': end-start+1,
                          'start_time': float(episode['times'][start]),
                          'end_time': float(episode['times'][end]), 'fps': cfg['fps'],
                          'valid_hands': flags, 'hand_issues': issues,
                          'data_warnings': row.get('data_warnings', []),
                          'motion_path': motion_path.as_posix(), 'source_hashes': hashes,
                          'feature_version': cfg['feature_version'], **text}
                local_rows.append(record)
                stats['motion_generated'] += 1
            report['windows_valid'] = len(local_rows)
            write_json(cache, {'rows': local_rows, 'report': report, 'rejected': local_rejected})
            rows.extend(local_rows)
            episode_reports.append(report)
            rejected.extend(local_rejected)
        except (OSError, ValueError, KeyError, __import__('subprocess').CalledProcessError) as exc:
            rejected.append({'episode_id': row['episode_id'], 'reason': str(exc)})
    rows.sort(key=lambda r: r['clip_id'])
    qa = {'stats': stats, 'episodes': episode_reports, 'rejected': rejected,
          'config': cfg, 'signature': signature}
    write_json(root/'last_build_qa.json', qa)
    if not rows:
        raise ValueError('no valid windows; inspect last_build_qa.json; existing index preserved')
    texts = sorted({r['semantic_text'] for r in rows})
    text_root = root/'semantic_cache'
    text_root.mkdir(exist_ok=True)
    # Resolved revisions should be pinned in config; main is deliberately never reused.
    semantic_sig = digest({k: cfg[k] for k in ('semantic_model', 'semantic_revision', 'semantic_chunk_tokens')})
    vectors, pending = {}, []
    for text in texts:
        path = text_root/f'{digest([signature, semantic_sig, text])}.npy'
        if path.is_file() and cfg['semantic_revision'] != 'main':
            vector = np.load(path, allow_pickle=False)
            if vector.ndim == 1 and np.isfinite(vector).all() and np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                vectors[text] = vector
                stats['semantic_reused'] += 1
                continue
        pending.append((text, path))
    if pending:
        encoder = encoder or SemanticEncoder(cfg)
        for offset in range(0, len(pending), 256):
            batch = pending[offset:offset+256]
            encoded = encoder.encode([text for text, _ in batch])
            for (text, path), vector in zip(batch, encoded):
                vector = np.asarray(vector, dtype=np.float32)
                if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                    raise ValueError('encoder returned non-normalized embedding')
                np.save(path, vector, allow_pickle=False)
                vectors[text] = vector
                stats['semantic_generated'] += 1
    gen = Path('generations')/uuid.uuid4().hex
    destination = root/gen
    destination.mkdir(parents=True)
    for index, row in enumerate(rows):
        row['embedding_index'] = index
    import json
    (destination/'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    np.save(destination/'semantic_embeddings.npy', np.stack([vectors[r['semantic_text']] for r in rows]), allow_pickle=False)
    metadata = {'created_at': datetime.now(timezone.utc).isoformat(), 'config': cfg, 'signature': signature,
                'window_count': len(rows), 'stats': stats, 'source_manifest': str(Path(dataset_root).resolve()),
                'encoder': getattr(encoder, 'metadata', {'model': cfg['semantic_model'],
                           'revision': cfg['semantic_revision'], 'cache_reused': True})}
    write_json(destination/'index.json', metadata)
    write_json(destination/'qa.json', qa)
    # Single writer per artifact root; a failed build cannot replace the last complete index.
    write_json(root/'current.tmp.json', {'generation': gen.as_posix()})
    (root/'current.tmp.json').replace(root/'current.json')
    for name in ('manifest.jsonl', 'semantic_embeddings.npy'):
        shutil.copyfile(destination/name, root/name)
    write_json(root/'index.json', metadata)
    write_json(root/'last_build_qa.json', qa)
    return metadata
