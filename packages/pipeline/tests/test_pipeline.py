"""Real MP4/HDF5 I/O; fake encoder only isolates network/model dependency in unit tests."""
from functools import partial
from pathlib import Path
import subprocess
import threading
import urllib.request
import h5py
import numpy as np
import pytest
from retrieval.common import read_json, write_json, load_config
from retrieval.data.egodex_reader import HAND_KEYS, read_episode
from retrieval.features.feature_store import build_index, FeatureStore
from retrieval.search.retrieval_engine import RetrievalEngine
from retrieval.server import make_server

class FakeEncoder:
    metadata = {'test_double': True}
    def encode(self, texts):
        return np.tile(np.array([1, 0, 0],dtype=np.float32), (len(texts),1))

@pytest.fixture
def dataset(tmp_path):
    video = tmp_path/'test.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=64x64:r=30',
                    '-frames:v','61','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
    rows = []
    for index in range(3):
        hdf = tmp_path/f'{index}.hdf5'
        with h5py.File(hdf,'w') as h:
            h.attrs['llm_description'] = f'task {index}'
            for side, mapping in HAND_KEYS.items():
                for key in [mapping['wrist'], *mapping['tips']]:
                    pose = np.tile(np.eye(4),(61,1,1))
                    pose[:,0,3] = np.arange(61)*.001*(index+1)
                    if key != mapping['wrist']:
                        pose[:,1,3] = .1
                    h[f'transforms/{key}'] = pose
        rows.append({'episode_id':f'episode_{index}', 'video_path':str(video), 'hdf5_path':str(hdf),
                     'data_warnings':['source identity needs review'] if index==0 else []})
    source = tmp_path/'sources.json'
    write_json(source,{'episodes':rows})
    cfg = load_config({'artifact_root':str(tmp_path/'artifacts'), 'semantic_revision':'pinned-test'})
    return source, cfg, rows

def test_reader_real_fields_and_confidence_absence(dataset):
    source,cfg,rows = dataset
    data = read_episode(rows[0],cfg)
    assert len(data['times']) == 61
    assert len(data['report']['hand_issues']['left']['confidence_missing']) == 6


def test_stable_default_refinement_and_legacy_are_separate(dataset):
    source, cfg, _ = dataset
    build_index(source, config=cfg, encoder=FakeEncoder())
    engine = RetrievalEngine(cfg['artifact_root'])
    query_id = engine.store.rows[0]['clip_id']
    stable = engine.retrieve(query_id, 1, {'refine_phase': True})
    assert stable['method'] == 'pose_language_active'
    assert stable['options']['recall_top_n'] == 10000
    assert np.isclose(sum(stable['query_hand_weights'].values()), 1)
    diagnostic = stable['phase_diagnostic']
    assert len(stable['results']) == 1 and len(diagnostic['results']) == 2
    assert diagnostic['tested'] > 0
    for row in diagnostic['results']:
        assert abs(row['frame_offset']) <= 14
        assert row['playback_clip_id'] in engine.store.by_id
        assert row['metadata']['end_frame']-row['metadata']['start_frame'] == 30
        assert all(np.isfinite(v).all() for v in map(np.asarray, row['refined_motion'].values()))
    saved = read_json(engine.store.root/'retrieval_results'/'stable'/f'{query_id}.json')
    assert saved['method'] == 'pose_language_active'
    legacy = engine.retrieve(query_id, options={'method': 'legacy_v1'})
    assert legacy['options']['recall_top_n'] == 100
    assert 'phase_diagnostic' not in legacy
    assert read_json(engine.store.root/'retrieval_results'/'stable'/f'{query_id}.json') == saved
    with pytest.raises(ValueError, match='requires'):
        engine.retrieve(query_id, options={'method': 'legacy_v1', 'refine_phase': True})

def test_end_to_end_cache_filters_report_and_range(dataset):
    source,cfg,rows = dataset
    first = build_index(source, config=cfg, encoder=FakeEncoder())
    assert first['window_count'] == 9
    engine = RetrievalEngine(cfg['artifact_root'])
    query_id = engine.store.rows[0]['clip_id']
    assert engine.store.by_id[query_id]['data_warnings'] == ['source identity needs review']
    result = engine.retrieve(query_id)
    assert len(result['results']) == 2
    assert len(result['raw_ranking']) == 6
    assert len({r['metadata']['episode_id'] for r in result['results']}) == 2
    assert all(r['metadata']['episode_id'] != 'episode_0' for r in result['raw_ranking'])
    own = engine.retrieve(query_id, 20, {'allow_same_episode':True,'max_per_episode':None})
    assert len(own['results']) == 8
    assert query_id not in {r['clip_id'] for r in own['results']}
    report = Path(cfg['artifact_root'])/'reports'/'test.html'
    engine.generate_report(query_id,result,report)
    assert query_id in report.read_text(encoding='utf-8')
    before = {p:p.stat().st_mtime_ns for p in (engine.store.root/'motion_features').glob('*.npz')}
    again = build_index(source,config=cfg,encoder=FakeEncoder())
    assert again['stats']['windows_reused'] == 9 and again['stats']['semantic_generated'] == 0
    assert all(p.stat().st_mtime_ns == m for p,m in before.items())
    assert engine.retrieve(query_id)['results'] == result['results']
    server = make_server(cfg['artifact_root'],0)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/media/{query_id}'
        req = urllib.request.Request(url,headers={'Range':'bytes=0-15'})
        with urllib.request.urlopen(req) as response:
            assert response.status == 206 and len(response.read()) == 16
    finally:
        server.shutdown(); server.server_close(); thread.join()
    changed = build_index(source, config=dict(cfg,feature_version='v2'),encoder=FakeEncoder())
    assert changed['stats']['motion_generated'] == 9 and changed['signature'] != first['signature']

def test_bad_episode_and_nan_are_logged(dataset):
    source,cfg,rows = dataset
    with h5py.File(rows[0]['hdf5_path'],'a') as h:
        h['transforms/rightHand'][0] = np.full((4,4),np.nan)
    with h5py.File(rows[1]['hdf5_path'],'a') as h:
        del h['transforms/leftHand']
    rows[2]['video_path'] = str(source.parent/'absent.mp4')
    write_json(source,{'episodes':rows})
    result = build_index(source,config=cfg,encoder=FakeEncoder())
    qa = read_json(Path(cfg['artifact_root'])/'last_build_qa.json')
    assert result['window_count'] == 5
    assert len(qa['rejected']) == 2
    assert all(not r['valid_hands']['left'] for r in FeatureStore(cfg['artifact_root']).rows if r['episode_id']=='episode_1')

def test_duplicate_ids_rejected(dataset):
    source,cfg,rows = dataset
    write_json(source,{'episodes':[rows[0],rows[0]]})
    with pytest.raises(ValueError,match='duplicate'):
        build_index(source,config=cfg,encoder=FakeEncoder())

def test_frame_mismatch_rejected(dataset):
    source,cfg,rows = dataset
    with h5py.File(rows[0]['hdf5_path'],'a') as h:
        del h['transforms/rightHand']
        h['transforms/rightHand'] = np.tile(np.eye(4),(60,1,1))
    with pytest.raises(ValueError,match='shape mismatch'):
        read_episode(rows[0],cfg)

def test_invalid_schema_falls_back_and_is_logged(dataset):
    from retrieval.data.annotation_adapter import load_annotation
    source,cfg,rows = dataset
    bad = source.parent/'bad_annotation.json'
    write_json(bad,{'steps':[]})
    row = dict(rows[0],annotation_path=str(bad))
    annotation, issues = load_annotation(row,cfg,61)
    assert annotation is None and issues[0]['code']=='annotation_fallback'

def test_changed_input_recomputes_only_changed_episode(dataset):
    source,cfg,rows = dataset
    build_index(source,config=cfg,encoder=FakeEncoder())
    with h5py.File(rows[0]['hdf5_path'],'a') as h:
        h['transforms/rightHand'][1,0,3] += .001
    changed = build_index(source,config=cfg,encoder=FakeEncoder())
    assert changed['stats']['episodes_reused']==2
    assert changed['stats']['motion_generated']==3

def test_failed_build_preserves_published_index(dataset):
    source,cfg,rows = dataset
    build_index(source,config=cfg,encoder=FakeEncoder())
    before = read_json(Path(cfg['artifact_root'])/'current.json')
    write_json(source,{'episodes':[]})
    with pytest.raises(ValueError,match='no valid windows'):
        build_index(source,config=cfg,encoder=FakeEncoder())
    assert read_json(Path(cfg['artifact_root'])/'current.json') == before
