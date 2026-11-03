"""Small deterministic artifact utilities; no model dependencies or hidden downloads."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('EDP_WORKSPACE', Path.cwd())).expanduser().resolve()
# Source checkout of the optional robotization package; never assume another drive.
if os.environ.get('EDP_EGO2ROBOT_ROOT'):
    REPRO = Path(os.environ['EDP_EGO2ROBOT_ROOT']).expanduser().resolve()
else:
    REPRO = ROOT / 'packages/ego2robot'
EPS_TIME = 1e-6


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return path


def write_npz(path: Path, **arrays: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    return path


def ref(path: str | Path, *, schema_version: str | None = None, producer: str = 'stage5', source_key: str | None = None) -> dict:
    p = Path(path).resolve()
    value = {'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size,
             'format': p.suffix.lstrip('.'), 'schema_version': schema_version,
             'producer': producer, 'source_key': source_key, 'availability': 'available'}
    return value


def verify_ref(value: dict) -> None:
    p = Path(value['path'])
    if not p.is_file():
        raise ValueError(f'S5_ARTIFACT_MISSING: {p}')
    if value.get('bytes') is not None and p.stat().st_size != value['bytes']:
        raise ValueError(f'S5_ARTIFACT_SIZE_MISMATCH: {p}')
    if sha256(p) != value['sha256']:
        raise ValueError(f'S5_SOURCE_HASH_MISMATCH: {p}')


def schema_name(path: Path) -> str | None:
    with np.load(path, allow_pickle=False) as arrays:
        return str(arrays['__schema_version__'].item()) if '__schema_version__' in arrays else None


def source_key(video_sha: str, hdf5_sha: str) -> str:
    return hashlib.sha256((video_sha + ':' + hdf5_sha).encode('ascii')).hexdigest()


def code_fingerprint() -> dict[str, str]:
    return {str(p.relative_to(PACKAGE_ROOT)): sha256(p) for p in sorted(PACKAGE_ROOT.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.json'} and '__pycache__' not in p.parts}


def utc_now() -> str:
    return datetime.now().astimezone().isoformat()
