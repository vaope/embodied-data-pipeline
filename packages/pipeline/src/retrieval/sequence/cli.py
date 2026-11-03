"""Portable continuous-subtask search CLI, with an identity-checked embedding cache."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time
import numpy as np
from .models import Query, Config, load_manifest
from .engine import search
from .report import render_clips, write_report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class CachedEncoder:
    def __init__(self, config, cache_path=None):
        from ..features import semantic_encoder
        self.config, self.path, self.model = config, Path(cache_path) if cache_path else None, None
        self.identity = dict(config=config, encoder_sha256=digest(semantic_encoder.__file__))
        self.values = {}
        self.hits = self.encoded = 0
        if self.path and self.path.exists():
            with np.load(self.path,allow_pickle=False) as data:
                if json.loads(str(data['identity'])) != self.identity:
                    raise ValueError('embedding cache identity mismatch; use a new cache path')
                texts, v = data['texts'].tolist(), data['vectors']
                if (v.ndim != 2 or len(v) != len(texts) or len(set(texts)) != len(texts)
                        or not np.isfinite(v).all() or np.any(np.linalg.norm(v,axis=1)<1e-12)):
                    raise ValueError('invalid cached embeddings')
                self.values.update(zip(texts,v))

    def encode(self, texts):
        from ..features.semantic_encoder import SemanticEncoder
        missing = sorted(set(texts)-self.values.keys())
        self.hits += len(texts)-len(missing)
        if missing:
            if self.model is None:
                self.model = SemanticEncoder(self.config)
            vectors = self.model.encode(missing)
            self.values.update(zip(missing,vectors))
            self.encoded += len(missing)
        return np.asarray([self.values[t] for t in texts])

    def save(self):
        if self.path and self.values:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            texts = sorted(self.values)
            temporary = self.path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,identity=json.dumps(self.identity,sort_keys=True),
                                texts=np.asarray(texts),vectors=np.asarray([self.values[t] for t in texts]))
            temporary.replace(self.path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Anchored continuous-subtask semantic retrieval')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--query',type=Path,required=True)
    parser.add_argument('--config',type=Path,help='JSON with search and encoder sections')
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--render-clips',type=int,default=0,metavar='N')
    parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--robot-motion',type=Path,help='Observed Robot steps.csv; enables Robot-to-Ego motion reranking')
    args = parser.parse_args(argv)
    if args.render_clips < 0:
        parser.error('--render-clips must be nonnegative')
    if args.out.exists():
        parser.error('--out must be a new directory')
    if not args.allow_download:
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
    encoder_cfg = dict(semantic_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        semantic_revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42',semantic_chunk_tokens=110,batch_size=32,device='cpu')
    encoder_cfg.update(cfg.get('encoder',{}))
    config = Config(**cfg.get('search',{}))
    query = Query.from_dict(json.loads(args.query.read_text(encoding='utf-8-sig')))
    steps = load_manifest(args.manifest)

    def resolve(step, base):
        if step.video_path and not Path(step.video_path).is_absolute():
            return replace(step,video_path=str((base/step.video_path).resolve()))
        return step

    query = replace(query,steps=tuple(resolve(s,args.query.resolve().parent) for s in query.steps))
    if query.video_reference:
        query = replace(query,video_reference=resolve(query.video_reference,args.query.resolve().parent))
    steps = [resolve(s,args.manifest.resolve().parent) for s in steps]
    encoder = CachedEncoder(encoder_cfg,args.cache)
    started = time.perf_counter()
    result = search(query,steps,encoder,config)
    if args.robot_motion:
        from .motion import MotionConfig, MotionReranker
        rows=[json.loads(line) for line in args.manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        for row in rows:
            if row.get('hdf5_path') and not Path(row['hdf5_path']).is_absolute():
                row['hdf5_path']=str((args.manifest.resolve().parent/row['hdf5_path']).resolve())
        result=MotionReranker(args.robot_motion,rows,MotionConfig(**cfg.get('motion',{}))).apply(result,query,config)
    encoder.save()
    result['provenance'] = dict(manifest=str(args.manifest.resolve()),manifest_sha256=digest(args.manifest),
        query_sha256=digest(args.query),encoder=encoder.identity,
        cache_sha256=digest(args.cache) if args.cache and args.cache.exists() else None,
        code_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.html')},
        elapsed_search_s=time.perf_counter()-started,cache_hits=encoder.hits,newly_encoded=encoder.encoded)
    args.out.mkdir(parents=True)
    # Save retrieval before optional media processing, so failures remain inspectable.
    dump(args.out/'results.json',result)
    try:
        clips = render_clips(result,args.out,args.render_clips)
        dump(args.out/'clip_manifest.json',clips)
    except (OSError, subprocess.SubprocessError) as exc:
        result['media_error'] = str(exc)
    dump(args.out/'results.json',result)
    write_report(result,args.out/'index.html')
    print(json.dumps(dict(out=str(args.out.resolve()),stats=result['stats'],
                         complete=len(result['complete']),partial=len(result['partial']),
                         elapsed_s=time.perf_counter()-started),ensure_ascii=False))


if __name__ == '__main__':
    main()
