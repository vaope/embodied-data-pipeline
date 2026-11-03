import pytest
from retrieval.data.window_builder import build_windows
from retrieval.data.annotation_adapter import window_text

@pytest.mark.parametrize('n,expected', [(30, []), (31, [(0, 30)]), (45, [(0, 30)]),
                                        (46, [(0, 30), (15, 45)]), (61, [(0,30),(15,45),(30,60)])])
def test_boundaries(n, expected):
    windows = list(build_windows(n))
    assert windows == expected
    assert all(b-a+1 == 31 and b < n for a, b in windows)

def test_step_overlap_not_whole_episode():
    ann = {'task': {'goal': 'task'}, 'steps': [
        {'step_id': 0, 'start_frame': 0, 'end_frame_exclusive': 15, 'description': 'A'},
        {'step_id': 1, 'start_frame': 15, 'end_frame_exclusive': 30, 'description': 'B'},
        {'step_id': 2, 'start_frame': 30, 'end_frame_exclusive': 60, 'description': 'C'}]}
    text = window_text(ann, 0, 30, '', {})
    assert text['step_text'] == 'A | B'
    assert window_text(ann, 15, 45, '', {})['step_text'] == 'B | C'
    assert window_text(ann, 65, 95, '', {})['semantic_granularity'] == 'task_only'

def test_official_fallback_and_empty_annotation():
    result = window_text(None, 0, 30, '倒水', {})
    assert result['annotation_source'] == 'egodex_official_metadata'
    assert result['semantic_granularity'] == 'task_only'
    with pytest.raises(ValueError):
        window_text(None, 0, 30, '', {})
