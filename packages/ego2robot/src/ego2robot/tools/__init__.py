"""Project tooling: fixture builders and the multi-case verification runner."""

from .cases import CASES_FILE, CaseDefinition, case_ground_truth, load_cases
from .hand_synth import (
    HandPose,
    expected_gripper_width,
    expected_tcp_frame,
    expected_tcp_position,
    hand_keypoints,
    make_hand_pose,
)
from .synthetic import build_case, build_cases, render_case_video

__all__ = [
    "CASES_FILE",
    "CaseDefinition",
    "HandPose",
    "build_case",
    "build_cases",
    "case_ground_truth",
    "expected_gripper_width",
    "expected_tcp_frame",
    "expected_tcp_position",
    "hand_keypoints",
    "load_cases",
    "make_hand_pose",
    "render_case_video",
]
