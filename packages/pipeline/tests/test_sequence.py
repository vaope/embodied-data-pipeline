"""Contract tests and an independent exhaustive oracle for ordered alignment."""
from dataclasses import replace
from itertools import combinations
import json
import random
import numpy as np
import pytest
from retrieval.sequence import Step, Query, Config, search, load_manifest
from retrieval.sequence.engine import align, block_text, suppress
from retrieval.sequence.cli import CachedEncoder
from retrieval.sequence.report import write_report


def step(name, index, source='ego', caption=None):
    return Step(name,source,float(index),float(index+1),caption or name)


class Matrix:
    def __init__(self, pairs):
        self.pairs = pairs

    def sim(self,a,b):
        return self.pairs.get((a,b),0.)


def test_dp_matches_exhaustive_subsequence_oracle():
    # Independent enumeration of all ordered query/candidate subsets, including missing steps.
    q = Query('test','q1',tuple(step(f'q{i}',i,'robot') for i in range(3)))
    c = [step(f'c{i}',i) for i in range(5)]
    cfg = Config(allow_merges=False,min_similarity=.4,max_skips=1,skip_penalty=.12,gap_penalty_per_s=.03)
    for seed in range(30):
        rng = random.Random(seed)
        pairs = {(f'q{i}',f'c{j}'):rng.random() for i in range(3) for j in range(5)}
        expected = {}
        for size in range(1,4):
            for qi in combinations(range(3),size):
                if 1 not in qi:
                    continue
                for cj in combinations(range(5),size):
                    if cj[qi.index(1)] != 2:
                        continue
                    skipped = cj[-1]-cj[0]+1-size
                    values = [pairs[f'q{i}',f'c{j}'] for i,j in zip(qi,cj)]
                    if skipped > cfg.max_skips or min(values) < cfg.min_similarity:
                        continue
                    score = (sum(values)-skipped*(cfg.skip_penalty+cfg.gap_penalty_per_s))/3
                    expected[size] = max(expected.get(size,-float('inf')),score)
        found = {}
        for r in align(q,c,2,Matrix(pairs),cfg):
            size = 3-len(r['missing_query_ids'])
            found[size] = max(found.get(size,-float('inf')),r['score'])
        assert found == pytest.approx(expected)


def test_complete_beats_anchor_only_and_order_is_required():
    q = Query('q','place', (step('take',0,'r'),step('place',1,'r')))
    cfg = Config(allow_merges=False,min_similarity=.5)
    c = [step('place',0),step('take',1)]
    scores = Matrix({('take','take'):1,('place','place'):1})
    rows = align(q,c,0,scores,cfg)
    assert rows and all(r['status']=='partial' for r in rows)
    assert all('take' in r['missing_query_ids'] for r in rows)
    c = [step('take',0),step('noise',1),step('place',2)]
    rows = align(q,c,2,scores,cfg)
    best = sorted(rows,key=lambda r:(-r['coverage'],-r['score']))[0]
    assert best['status']=='complete' and best['skipped_candidate_ids']==['noise']
    assert best['unmatched_seconds']==1
    assert not any(r['status']=='complete' for r in align(q,c,2,scores,replace(cfg,max_skips=0)))


def test_merges_and_member_floor():
    q = Query('q','compound',(step('compound',0,'r','Take lid then place lid'),))
    c = [step('take',0,caption='Take lid'),step('place',1,caption='Place lid')]
    scores = Matrix({('Take lid then place lid','Take lid'):.7,('Take lid then place lid','Place lid'):.8,('Take lid then place lid','Take lid Then Place lid'):.95})
    rows = align(q,c,1,scores,Config())
    assert any(len(r['mappings'][0]['candidate_ids'])==2 for r in rows)
    scores.pairs['Take lid then place lid','Take lid']=.1
    assert all(len(r['mappings'][0]['candidate_ids'])==1 for r in align(q,c,1,scores,Config()))
    q = Query('q','place',(step('take',0,'r','Take lid'),step('place',1,'r','Place lid')))
    c = [step('compound',0,caption='Take lid then place lid')]
    scores = Matrix({('Take lid Then Place lid','Take lid then place lid'):.9,('Take lid','Take lid then place lid'):.8,('Place lid','Take lid then place lid'):.8})
    assert any(r['status']=='complete' for r in align(q,c,0,scores,Config()))
    scores.pairs['Take lid','Take lid then place lid']=.1
    assert not any(r['status']=='complete' for r in align(q,c,0,scores,Config()))


def test_real_caption_conflicts_and_merge_coverage():
    from retrieval.sequence.semantics import check_blocks
    take=step('a',0,caption='Pick up the pen from the bowl')
    place=step('b',1,caption='Place the pen on the table')
    band=step('c',0,caption='Remove the rubber band from the pens')
    assert not check_blocks([take],[band])[0]
    assert not check_blocks([take,place],[place])[0]
    assert not check_blocks([take],[step('x',0,caption='Pick up the lid')])[0]
    assert check_blocks([take,place],[step('x',0,caption='Take the pen then place the pen on the table')])[0]
    assert check_blocks([take],[step('x',0,caption='Take the pen from the shelf')])[0]  # location soft
    assert not check_blocks([step('x',0,caption='未知动作'),place],[place])[0]


def test_key_must_hit_selected_anchor_and_temporal_gaps_are_bounded():
    q = Query('q','q1',(step('q0',0,'r'),step('q1',1,'r')))
    c = [step('c0',0),step('c1',10),step('c2',11)]
    scores = Matrix({('q0','c0'):1,('q1','c1'):1,('q1','c2'):1})
    rows = align(q,c,1,scores,Config(allow_merges=False))
    assert not any(r['status']=='complete' for r in rows)
    assert all(r['mappings'][-1]['candidate_ids']==['c1'] for r in rows)
    assert not align(q,c,0,scores,Config(allow_merges=False))
    assert not any(r['status']=='complete' for r in align(q,c,1,scores,Config(allow_merges=False,max_step_gap_s=20,max_span_s=5)))


class Encoder:
    def encode(self,texts):
        # Controlled action vocabulary, not a semantic-quality test.
        return np.array([[float(word in text.lower()) for word in ('take','place','noise')] for text in texts])


def test_search_keeps_multiple_anchors_and_never_crosses_sources():
    q = Query('q','q1',(step('q0',0,'robot','take'),step('q1',1,'robot','place')))
    c = [step('a0',0,'a','place'),step('a1',1,'a','take'),step('a2',2,'a','place'),
         step('b0',0,'b','take')]
    r = search(q,c,Encoder(),Config(allow_merges=False,min_similarity=.9))
    assert len(r['anchors'])==2
    assert r['complete'][0]['start_s']==1
    assert r['complete'][0]['source_id']=='a'
    assert all(len({s['source_id'] for s in row['steps']})==1 for row in r['complete'])


def test_half_open_adapter_and_invalid_contracts(tmp_path):
    p = tmp_path/'manifest.jsonl'
    p.write_text(json.dumps(dict(episode_id='e',clip_id='a',caption='take',start_time=0,end_time=.96,
                                 annotation_start_s=0,annotation_end_s=1)),encoding='utf-8')
    assert load_manifest(p)[0].end_s==1
    with pytest.raises(ValueError):
        Step('x','s',0,float('nan'),'bad')
    with pytest.raises(ValueError):
        Query('q','not-present',(step('a',0),))
    with pytest.raises(ValueError):
        Query('q','a',(step('a',0),step('b',0)))
    with pytest.raises(ValueError):
        Config(max_skips=-1)
    with pytest.raises(ValueError):
        Config(min_similarity=float('nan'))


def test_nms_preserves_disjoint_occurrences():
    rows=[dict(source_id='e',start_s=a,end_s=b,score=s) for a,b,s in [(0,2,.9),(0,2,.8),(3,5,.7)]]
    assert len(suppress(rows,10,.8))==2


def test_cache_identity_and_report_escaping(tmp_path):
    p=tmp_path/'cache.npz'
    encoder=CachedEncoder({'semantic_model':'test'},p)
    encoder.values={'take':np.array([1.,0.])}
    encoder.save()
    assert np.array_equal(CachedEncoder({'semantic_model':'test'},p).values['take'],[1,0])
    with pytest.raises(ValueError,match='identity mismatch'):
        CachedEncoder({'semantic_model':'other'},p)
    out=tmp_path/'index.html'
    write_report({'unsafe':'</script><script>alert(1)</script>'},out)
    assert '</script><script>alert(1)' not in out.read_text(encoding='utf-8')
    assert '\\u003c/script' in out.read_text(encoding='utf-8')


def test_report_server_ranges_and_access_boundary(tmp_path):
    from threading import Thread
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from retrieval.sequence.serve import make_server
    (tmp_path/'clip.mp4').write_bytes(b'0123456789')
    (tmp_path/'private.npz').write_bytes(b'not served')
    server=make_server(tmp_path,0)
    worker=Thread(target=server.serve_forever,daemon=True)
    worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(Request(base+'/clip.mp4',headers={'Range':'bytes=2-4'})) as r:
            assert r.status==206 and r.read()==b'234'
            assert r.headers['Content-Range']=='bytes 2-4/10'
        for path,code in [('/private.npz',403),('/../outside.html',403),('/absent.html',404)]:
            with pytest.raises(HTTPError) as exc:
                urlopen(base+path)
            assert exc.value.code==code
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base+'/clip.mp4',headers={'Range':'bytes=99-'}))
        assert exc.value.code==416
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
