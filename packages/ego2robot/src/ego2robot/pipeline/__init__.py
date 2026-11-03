"""Pipeline layer: stage contract, registry and runner."""

from .registry import StageRegistry, default_registry, register_stage
from .runner import DagInfo, PipelineRunner, RunOutcome
from .stage import (
    RunRecord,
    Stage,
    StageContext,
    StageLogger,
    StageResult,
    StageStatus,
)

__all__ = [
    "DagInfo",
    "PipelineRunner",
    "RunOutcome",
    "RunRecord",
    "Stage",
    "StageContext",
    "StageLogger",
    "StageRegistry",
    "StageResult",
    "StageStatus",
    "default_registry",
    "register_stage",
]
