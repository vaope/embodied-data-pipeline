"""Tests for the V2 vertical slice (selection, eligibility, export roundtrip)."""
from __future__ import annotations

import json

from pipeline.v2.eval import make_template, metrics
from pipeline.v2.pipeline import ego_eligibility, export_lerobot_draft, export_unified, select, verify_export


def _candidate(clip_id, start, end, task="pick", video="v1"):
    return {
        "clip_id": clip_id,
        "episode_id": "ep",
        "task_text": task,
        "start_time": start,
        "end_time": end,
        "source_hashes": {"video_path": video},
    }


def test_select_drops_overlapping_windows_and_caps_per_video():
    candidates = [
        _candidate("a", 0.0, 1.0),
        _candidate("b", 0.1, 1.1),   # overlaps a -> dropped
        _candidate("c", 5.0, 6.0),   # distinct -> kept
        _candidate("d", 9.0, 10.0),  # third window of v1 -> video cap (2)
        _candidate("e", 0.0, 1.0, video="v2"),  # other video -> kept
    ]
    selected, removed = select(candidates, max_total=10, max_per_task=10, max_per_video=2, overlap=0.5)
    assert [row["clip_id"] for row in selected] == ["a", "c", "e"]
    assert removed["overlap"] == 1
    assert removed["video_cap"] == 1


def test_select_respects_total_and_task_cap():
    candidates = [_candidate(f"c{i}", i * 10.0, i * 10.0 + 1.0, task=f"t{i % 2}", video=f"v{i}") for i in range(6)]
    selected, removed = select(candidates, max_total=3, max_per_task=2, max_per_video=1, overlap=0.5)
    assert len(selected) == 3
    assert removed["max_total"] == 3


def test_ego_eligibility_marks_missing_motion_and_pose():
    labels, reasons = ego_eligibility({"valid_hands": {"right": True}, "motion_path": None, "hdf5_path": None})
    assert "retrieval_text" in labels and "retrieval_motion" not in labels
    assert "missing_motion_feature" in reasons and "missing_hdf5_pose" in reasons
    assert "no_robot_action_gt" in reasons

    labels, reasons = ego_eligibility(
        {"valid_hands": {"right": True}, "motion_path": "m.npz", "hdf5_path": "e.hdf5"}
    )
    assert "retrieval_motion" in labels and "robotization_candidate" in labels
    assert "missing_motion_feature" not in reasons


def test_export_and_verify_roundtrip(tmp_path):
    samples = []
    for i in range(2):
        samples.append(
            {
                "sample_id": f"ego:clip{i}",
                "source_episode_id": "ep",
                "labels": {"task": "pick"},
                "segment": {"start_s": float(i), "end_s": float(i) + 1.0},
                "usage_eligibility": ["retrieval_text"],
            }
        )
    export_unified(samples, tmp_path)
    export_lerobot_draft(samples, tmp_path)
    result = verify_export(tmp_path, len(samples))
    assert result["ok"] is True
    assert result["unified_samples"] == 2 and result["lerobot_draft_episodes"] == 2

    info = json.loads((tmp_path / "lerobot_draft" / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["official_lerobot"] is False


def test_metrics_perfect_and_partial():
    rows = [
        {"query_id": "q1", "rank": "1", "label": "1"},
        {"query_id": "q1", "rank": "2", "label": "0"},
        {"query_id": "q2", "rank": "1", "label": "0"},
        {"query_id": "q2", "rank": "2", "label": "1"},
    ]
    result = metrics(rows, ks=(1, 2))
    assert result["queries"] == 2
    assert result["Recall@1"] == 0.5
    assert result["Recall@2"] == 1.0
    assert result["mAP"] == 0.75


def test_make_template_writes_header_and_rows(tmp_path):
    candidates = [{"clip_id": f"c{i}", "task_text": "pick", "start_time": 0.0, "end_time": 1.0} for i in range(3)]
    path = make_template(candidates, "q1", "put", tmp_path, top_n=2)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("query_id,query_text,rank,clip_id")
    assert len(lines) == 3
    assert "put" in lines[1]
