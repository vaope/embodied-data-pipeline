"""Retargeting layer: hand -> canonical gripper, filtering, gaps and smoothing.

Public API::

    from ego2robot.retarget import retarget_hand_sequence, reject_outliers,
                                   fill_gaps, smooth_trajectory
"""

from .gaps import GapConfigLike, GapFillResult, blend_length_frames, fill_gaps
from .hand_to_gripper import (
    MANO_INDEX_TIP,
    MANO_MIDDLE_TIP,
    MANO_THUMB_TIP,
    MANO_WRIST,
    FallbackCode,
    RetargetConfigLike,
    RetargetResult,
    retarget_hand_sequence,
    tcp_frame_from_keypoints,
)
from .outliers import OutlierConfigLike, OutlierResult, reject_outliers
from .smoothing import (
    SmoothingConfigLike,
    SmoothingResult,
    gaussian_kernel,
    legalize_savgol_window,
    smooth_trajectory,
)

__all__ = [
    "FallbackCode",
    "GapConfigLike",
    "GapFillResult",
    "MANO_INDEX_TIP",
    "MANO_MIDDLE_TIP",
    "MANO_THUMB_TIP",
    "MANO_WRIST",
    "OutlierConfigLike",
    "OutlierResult",
    "RetargetConfigLike",
    "RetargetResult",
    "SmoothingConfigLike",
    "SmoothingResult",
    "blend_length_frames",
    "fill_gaps",
    "gaussian_kernel",
    "legalize_savgol_window",
    "reject_outliers",
    "retarget_hand_sequence",
    "smooth_trajectory",
    "tcp_frame_from_keypoints",
]
