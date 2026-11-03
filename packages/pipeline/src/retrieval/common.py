from __future__ import annotations
import hashlib
import json
from pathlib import Path
import yaml

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_config(config=None):
    cfg = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text(encoding='utf-8'))
    base = Path.cwd()
    if isinstance(config, (str, Path)):
        config = Path(config).expanduser().resolve()
        base = config.parent
        cfg.update(yaml.safe_load(config.read_text(encoding='utf-8')) or {})
    elif config:
        cfg.update(config)
    for key in ('artifact_root', 'packed_motion_path', 'schema_path'):
        if cfg.get(key):
            path = Path(cfg[key]).expanduser()
            cfg[key] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    if not cfg.get('schema_path'):
        cfg['schema_path'] = str(Path(__file__).parent / 'schemas' / 'step.schema.json')
    for key in ('fps', 'horizon', 'stride', 'recall_top_n', 'batch_size'):
        if cfg[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 <= cfg['confidence_threshold'] <= 1:
        raise ValueError('confidence_threshold must be in [0,1]')
    return cfg
