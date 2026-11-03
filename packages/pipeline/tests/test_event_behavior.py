import numpy as np
import pytest
from retrieval.motion.representation import MotionSequence
from retrieval.rankers.gripper_ego_ranker import load_config
from retrieval.search.event_behavior import crop_event, choose_active_hand, fused_score


def sequence(distance):
    t=np.arange(61)/15
    p=np.zeros((61,3));p[:,0]=distance*t/t[-1]
    return MotionSequence('fixture','event',t,p,np.tile([0.,0.,0.,1.],(61,1)),np.ones(61,bool),
                          'world','effector','point',np.linspace(0,1,61))


def test_event_cannot_include_late_motion_or_next_boundary_frame():
    seq=sequence(1.)
    event=crop_event(seq,15,31)
    assert event.metadata['original_span_s']==[1.,2.]
    assert len(event.timestamps_s)==16 and event.timestamps_s[-1]==1.
    assert event.grasp_signal[-1]==seq.grasp_signal[30]
    event.position_m[0,0]=99
    assert seq.position_m[15,0]!=99
    with pytest.raises(ValueError): crop_event(seq,60,63)


def test_stationary_hand_cannot_win_by_matching_a_stationary_robot():
    h,_,activity,_=choose_active_hand({'left':sequence(0.),'right':sequence(0.4)},load_config())
    assert h=='right' and activity['right']>activity['left']


def test_behavior_cannot_overturn_clear_semantic_margin_and_missing_is_penalized():
    assert fused_score(.8,10000)[0]>fused_score(.76,0)[0]
    assert fused_score(.8,None)[0]<fused_score(.8,0)[0]
    with pytest.raises(ValueError): fused_score(.8,float('nan'))
