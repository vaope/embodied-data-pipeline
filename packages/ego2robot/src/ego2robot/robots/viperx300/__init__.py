"""Trossen ViperX-300 6DOF robot adapter backed by MuJoCo Menagerie."""

from .adapter import (
    VIPERX300_EEF_BODY_NAME,
    VIPERX300_GRIPPER_JOINT_NAMES,
    VIPERX300_JOINT_NAMES,
    VIPERX300_MAX_REACH_M,
    VIPERX300_TCP_SITE_NAME,
    ViperX300,
    ViperX300IKSolver,
    build_viperx_config,
    default_viperx_xml,
)

__all__ = [
    "VIPERX300_EEF_BODY_NAME",
    "VIPERX300_GRIPPER_JOINT_NAMES",
    "VIPERX300_JOINT_NAMES",
    "VIPERX300_MAX_REACH_M",
    "VIPERX300_TCP_SITE_NAME",
    "ViperX300",
    "ViperX300IKSolver",
    "build_viperx_config",
    "default_viperx_xml",
]
