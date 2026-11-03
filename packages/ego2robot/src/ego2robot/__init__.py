"""Ego2Robot data pipeline.

Normative design: ``docs/system-design.md``.
Normative implementation order: ``docs/implementation-plan.md``.
"""

__version__ = "0.1.0"

SCHEMA_VERSION_EPISODE = "ego2robot.episode.v1"
SCHEMA_VERSION_HAND = "ego2robot.hand.v1"
SCHEMA_VERSION_ACTION = "ego2robot.canonical_action.v1"
SCHEMA_VERSION_ROBOT = "ego2robot.robot_trajectory.v1"
SCHEMA_VERSION_RENDER = "ego2robot.render_bundle.v1"
SCHEMA_VERSION_QUALITY = "ego2robot.quality.v1"
SCHEMA_VERSION_STAGE = "ego2robot.stage_manifest.v1"

__all__ = [
    "__version__",
    "SCHEMA_VERSION_EPISODE",
    "SCHEMA_VERSION_HAND",
    "SCHEMA_VERSION_ACTION",
    "SCHEMA_VERSION_ROBOT",
    "SCHEMA_VERSION_RENDER",
    "SCHEMA_VERSION_QUALITY",
    "SCHEMA_VERSION_STAGE",
]
