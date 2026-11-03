"""Synthetic I/O demonstration; neutral embeddings are NOT language inference."""
import argparse
import shutil
import subprocess
from pathlib import Path
import h5py
import numpy as np
from .common import load_config, write_json
from .data.egodex_reader import HAND_KEYS
from .features.feature_store import build_index, FeatureStore
from .search.retrieval_engine import RetrievalEngine


class NeutralDemoEncoder:
    metadata = {'demo_only': True, 'method': 'constant vector; no language inference'}

    def encode(self, texts):
        return np.tile(np.array([1, 0, 0], dtype=np.float32), (len(texts), 1))


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; existing results are never overwritten')
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            raise RuntimeError(f'{tool} must be installed and on PATH')
    output.mkdir(parents=True)
    video = output / 'synthetic.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=gray:s=64x64:r=30', '-frames:v', '61',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
    sources = []
    for index in range(3):
        path = output / f'episode_{index}.hdf5'
        with h5py.File(path, 'w') as handle:
            handle.attrs['llm_description'] = f'Synthetic translation example {index}'
            for mapping in HAND_KEYS.values():
                for key in [mapping['wrist'], *mapping['tips']]:
                    pose = np.tile(np.eye(4), (61, 1, 1))
                    pose[:, 0, 3] = np.arange(61) * .001 * (index + 1)
                    if key != mapping['wrist']:
                        pose[:, 1, 3] = .1
                    handle[f'transforms/{key}'] = pose
        sources.append({'episode_id': f'episode_{index}', 'video_path': video.name,
                        'hdf5_path': path.name, 'data_warnings': ['synthetic fixture; not a real demonstration']})
    write_json(output / 'sources.json', {'episodes': sources})
    cfg = load_config({'artifact_root': str(output / 'artifacts'), 'semantic_revision': 'synthetic-neutral-demo'})
    build = build_index(output / 'sources.json', config=cfg, encoder=NeutralDemoEncoder())
    engine = RetrievalEngine(cfg['artifact_root'], cfg)
    query = engine.store.rows[0]['clip_id']
    result = engine.retrieve(query, 2)
    reader = FeatureStore(cfg['artifact_root'])
    loaded = reader.get_clip_features(query)
    report = {'kind': 'synthetic_io_demo', 'language_inference': False, 'real_data_quality_claim': False,
              'windows': build['window_count'], 'query': query,
              'results': [r['clip_id'] for r in result['results']],
              'reader_motion_fields': sorted(loaded['motion']),
              'reader_embedding_finite': bool(np.isfinite(loaded['semantic']).all())}
    write_json(output / 'demo_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(run(args.out), indent=2))


if __name__ == '__main__':
    main()
