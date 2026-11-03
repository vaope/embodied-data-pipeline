import numpy as np
import pytest
from retrieval.search.rank_fusion import rank_fusion
from retrieval.search.motion_ranking import motion_distances, ROUTES
from retrieval.search.semantic_recall import semantic_recall

def test_tie_breaks_and_relative_ranks():
    rows = [{'clip_id': key, 'semantic_similarity': sem,
             'motion_distances': dict.fromkeys(ROUTES, dist)}
            for key, sem, dist in [('z', .8, 1), ('b', .9, 1), ('a', .9, 1), ('last', 1, 2)]]
    result = rank_fusion(rows)
    assert [r['clip_id'] for r in result] == ['a', 'b', 'z', 'last']
    assert [r['rank_sum'] for r in result] == [4, 4, 4, 16]

def test_hand_compatibility_and_average():
    q = {f'{hand}_{k}': np.zeros((30,)) for hand in ['left', 'right'] for k in ROUTES}
    c = {k: v+ (2 if k.startswith('left') else 4) for k, v in q.items()}
    both, right = {'left': True, 'right': True}, {'left': False, 'right': True}
    assert set(motion_distances(q,c,both,both).values()) == {3}
    assert set(motion_distances(q,c,right,both).values()) == {4}
    with pytest.raises(ValueError):
        motion_distances(q,c,both,right)

def test_semantic_recall_ties_and_small_pool():
    rows = [{'clip_id': 'b'}, {'clip_id': 'a'}]
    out = semantic_recall(np.array([1., 0]), np.array([[1.,0],[1.,0]]), rows)
    assert [r['clip_id'] for r in out] == ['a', 'b']
