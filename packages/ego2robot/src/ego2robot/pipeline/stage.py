"""Stage contract (system-design.md 12).

Every pipeline stage implements :class:`Stage`::

    class Stage(Protocol):
        name: str
        version: str
        def validate_inputs(self, ctx) -> ValidationResult: ...
        def fingerprint(self, ctx) -> str: ...
        def run(self, ctx) -> StageResult: ...
        def validate_outputs(self, result) -> ValidationResult: ...

The runner owns everything else: caching, ``--until``, ``--force-stage``,
manifest writing and failure cleanup.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..core import artifacts as A
from ..core.config import PipelineConfig
from ..core.schema import StageManifest, ValidationResult

__all__ = [
    "RunRecord",
    "Stage",
    "StageContext",
    "StageResult",
    "StageStatus",
    "make_stage_manifest",
]


class StageStatus(str):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CACHED = "cached"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageLogger:
    """Structured line logger carrying ``run_id`` / ``episode_id`` / ``stage``.

    system-design.md 15 requires those fields on every structured log record.
    Kept dependency-light so stages can use it without importing a framework.
    """

    def __init__(self, *, run_id: str, episode_id: str, stage: str, sink: Callable[[str], None] | None = None) -> None:
        self.run_id = run_id
        self.episode_id = episode_id
        self.stage = stage
        self._sink = sink
        self.lines: list[str] = []

    def _emit(self, level: str, message: str) -> None:
        line = (
            f"{A.utc_now_iso()} level={level} run_id={self.run_id} "
            f"episode_id={self.episode_id} stage={self.stage} msg={message}"
        )
        self.lines.append(line)
        if self._sink is not None:
            self._sink(line)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warning(self, message: str) -> None:
        self._emit("WARNING", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)


@dataclass
class StageResult:
    """What a stage produced: output paths, metrics and non-fatal warnings."""

    stage: str
    outputs: list[Path] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: in-memory payloads for downstream stages in the same run
    payload: dict[str, Any] = field(default_factory=dict)
    reused_stages: list[str] = field(default_factory=list)

    def add_output(self, path: str | Path) -> Path:
        p = Path(path)
        if p not in self.outputs:
            self.outputs.append(p)
        return p

    def relative_outputs(self, root: Path) -> list[str]:
        out: list[str] = []
        for p in self.outputs:
            try:
                out.append(str(p.resolve().relative_to(root.resolve())).replace("\\", "/"))
            except ValueError:
                out.append(str(p))
        return out


@dataclass
class StageContext:
    """Everything a stage is allowed to know about the run."""

    layout: A.EpisodeLayout
    config: PipelineConfig
    repo_root: Path
    run_id: str
    episode_id: str
    stage_name: str
    force: bool = False
    dry_run: bool = False
    log: StageLogger | None = None
    #: results of upstream stages in this run
    upstream: dict[str, StageResult] = field(default_factory=dict)
    #: resolved config hash, reused for every stage fingerprint in one run
    config_hash: str = ""

    def require_payload(self, stage: str, key: str) -> Any:
        """Fetch an in-memory upstream payload or explain how to get it."""
        result = self.upstream.get(stage)
        if result is None or key not in result.payload:
            raise KeyError(
                f"stage {self.stage_name!r} needs payload {stage}.{key}, which is absent. "
                f"Upstream stages available: {sorted(self.upstream)}. Either run the producer "
                f"stage first, or have it persist the value to disk so this stage can reload it."
            )
        return result.payload[key]

    def load_npz(self, path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}


@runtime_checkable
class Stage(Protocol):
    """The contract every pipeline stage must satisfy."""

    name: str
    version: str

    def validate_inputs(self, ctx: StageContext) -> ValidationResult: ...

    def fingerprint(self, ctx: StageContext) -> str: ...

    def run(self, ctx: StageContext) -> StageResult: ...

    def validate_outputs(self, result: StageResult) -> ValidationResult: ...


# ---------------------------------------------------------------------------
# helpers shared by concrete stages
# ---------------------------------------------------------------------------


def make_stage_manifest(
    *,
    stage: str,
    stage_version: str,
    status: str,
    input_fingerprints: Mapping[str, str] | None = None,
    config_hash: str = "",
    code_revision: str = "",
    started_at: str = "",
    duration_s: float = 0.0,
    outputs: Sequence[str] = (),
    metrics: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    error: str | None = None,
) -> StageManifest:
    return StageManifest(
        stage=stage,
        stage_version=stage_version,
        status=status,
        input_fingerprints=dict(input_fingerprints or {}),
        config_hash=config_hash,
        code_revision=code_revision,
        started_at=started_at or A.utc_now_iso(),
        duration_s=round(float(duration_s), 4),
        outputs=list(outputs),
        metrics=dict(metrics or {}),
        warnings=list(warnings),
        errors=list(errors),
        error=error,
    )


def failure_manifest(
    *,
    stage: str,
    stage_version: str,
    exc: BaseException,
    input_fingerprints: Mapping[str, str] | None = None,
    config_hash: str = "",
    code_revision: str = "",
    started_at: str = "",
    duration_s: float = 0.0,
    warnings: Sequence[str] = (),
) -> StageManifest:
    """Build a readable failure manifest (never a plausible-looking success)."""
    tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return make_stage_manifest(
        stage=stage,
        stage_version=stage_version,
        status=StageStatus.FAILED,
        input_fingerprints=input_fingerprints,
        config_hash=config_hash,
        code_revision=code_revision,
        started_at=started_at,
        duration_s=duration_s,
        warnings=warnings,
        errors=[f"{type(exc).__name__}: {exc}"],
        error=tb,
    )


@dataclass
class RunRecord:
    """Per-stage outcome recorded by the runner for the episode run log."""

    stage: str
    status: str
    fingerprint: str
    duration_s: float
    manifest_path: str
    outputs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {StageStatus.SUCCEEDED, StageStatus.CACHED, StageStatus.SKIPPED}
