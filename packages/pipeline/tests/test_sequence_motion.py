"""Behavioral checks of real Pose reading and reranking, not score mocks."""
import csv
from dataclasses import replace
import numpy as np
import h5py
import pytest
from retrieval.sequence.models import Query,Step,Config
from retrieval.sequence.motion import MotionReranker,MotionConfig,combine


def test_missing_neutral_and_query_steps_not_block_count():
    p=[dict(status='ok',query_count=2,distance=0),dict(status='unavailable',query_count=1)]
    r=combine(.7,p,3,.03)
    assert r['score']==pytest.approx(.71)
    assert r['motion_coverage']==pytest.approx(2/3)
    assert combine(.7,[dict(status='unavailable',query_count=1)],1,.03)['score']==.7


def test_real_pose_changes_ranking_without_axis_assumption(tmp_path):
    csv_path=tmp_path/'steps.csv'
    names=['step']+[f'observation_cartesian_position_{j}' for j in range(6)]
    with csv_path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(names)
        for i in range(61):w.writerow([i,.1*i/30,0,0,0,0,0])
    rows=[]
    for name,speed in [('bad',1.),('good',.1)]:
        path=tmp_path/f'{name}.hdf5'
        with h5py.File(path,'w') as h:
            for side in ['left','right']:
                poses=np.tile(np.eye(4),(61,1,1))
                # Robot x and human y differ: invariant scoring must still work.
                poses[:,1,3]=np.arange(61)/30*speed if side=='right' else 0.
                h.create_dataset('transforms/'+side+'Hand',data=poses)
        rows.append(dict(clip_id=name,hdf5_path=str(path),start_frame=0,end_frame=59,fps=30.,start_time=0.,end_time=59/30))
    query=Query('q','q1',(Step('q1','r',0.,2.,'Place the lid on the pot'),))
    cfg=Config(top_k=1,rerank_pool_size=2)
    candidates=[dict(step_id='bad',source_id='b',start_s=0.,end_s=2.,score=.9),dict(step_id='good',source_id='g',start_s=0.,end_s=2.,score=.899)]
    result=dict(candidate_pools=dict(complete=[],single_key=candidates),complete=[],partial=[],baselines={})
    scorer=MotionReranker(csv_path,rows,MotionConfig(nominal_robot_fps=30))
    result=scorer.apply(result,query,cfg)
    top=result['baselines']['single_key_motion'][0]
    assert top['step_id']=='good' and top['semantic_rank']==2
    assert top['motion_coverage']==1.
    assert top['motion_pairs'][0]['ego']['hand']=='right'
    assert top['motion_pairs'][0]['ego']['mapping_declared'] is False
    assert top['motion_pairs'][0]['distance']==pytest.approx(0.,abs=1e-10)
    missing=scorer.score(query,dict(candidates[0],step_id='absent'),single=True)
    assert missing['score']==.9 and missing['motion_status']=='unavailable'


def test_weight_validation():
    for value in [float('nan'),-.1,.5]:
        with pytest.raises(ValueError):MotionConfig(weight=value)
