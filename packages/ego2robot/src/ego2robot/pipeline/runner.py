"""Linear stage DAG runner with cache/resume, ``--until`` and ``--force-stage``.

System design section 12.  The pipeline is a linear DAG for the MVP
(``ingest -> hand -> retarget -> robot -> render -> projection``), which keeps
resume semantics trivially auditable: forcing stage *k* invalidates exactly
*k* and everything downstream of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..core import artifacts as A
from ..core.config import PipelineConfig, config_hash, dump_resolved
from ..core.schema import StageManifest
from .stage import (
    RunRecord,
    Stage,
    StageContext,
    StageLogger,
    StageResult,
    StageStatus,
    failure_manifest,
    make_stage_manifest,
)

__all__ = ["DagInfo", "PipelineRunner", "RunOutcome"]


@dataclass
class DagInfo:
    """The resolved DAG, printable by ``e2r run --dry-run`` (Gate M0)."""

    order: list[str]
    dependencies: dict[str, list[str]]
    versions: dict[str, str]

    def describe(self) -> str:
        lines = ["stage DAG (linear):"]
        for i, name in enumerate(self.order):
            deps = ", ".join(self.dependencies.get(name, [])) or "-"
            lines.append(f"  {i + 1}. {name} (v{self.versions.get(name, '?')}) <- {deps}")
        return "\n".join(lines)


@dataclass
class RunOutcome:
    """Result of a pipeline run."""

    episode_id: str
    run_id: str
    records: list[RunRecord] = field(default_factory=list)
    stopped_at: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(r.ok for r in self.records)

    def record(self, stage: str) -> RunRecord | None:
        for r in self.records:
            if r.stage == stage:
                return r
        return None

    def summary_rows(self) -> list[tuple[str, str, float, str]]:
        return [
            (r.stage, r.status, r.duration_s, ",".join(r.outputs) if r.outputs else "-")
            for r in self.records
        ]

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "ok": self.ok,
            "stopped_at": self.stopped_at,
            "error": self.error,
            "stages": [
                {
                    "stage": r.stage,
                    "status": r.status,
                    "fingerprint": r.fingerprint,
                    "duration_s": r.duration_s,
                    "manifest": r.manifest_path,
                    "outputs": r.outputs,
                    "metrics": r.metrics,
                    "warnings": r.warnings,
                    "errors": r.errors,
                }
                for r in self.records
            ],
        }


class PipelineRunner:
    """Executes registered stages in declaration order with cache/resume."""

    def __init__(
        self,
        stages: Mapping[str, Stage],
        *,
        log_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.stages: dict[str, Stage] = dict(stages)
        self._log_sink = log_sink

    # -- DAG ---------------------------------------------------------------

    def dag(self, order: Sequence[str] | None = None) -> DagInfo:
        names = list(order) if order is not None else list(self.stages)
        missing = [n for n in names if n not in self.stages]
        if missing:
            raise KeyError(
                f"unknown stage(s) {missing}; registered stages: {sorted(self.stages)}"
            )
        deps: dict[str, list[str]] = {}
        prev: str | None = None
        for name in names:
            deps[name] = [] if prev is None else [prev]
            prev = name
        return DagInfo(
            order=names,
            dependencies=deps,
            versions={n: self.stages[n].version for n in names},
        )

    # -- cache -------------------------------------------------------------

    @staticmethod
    def cache_status(
        layout: A.EpisodeLayout, stage: str, fingerprint: str
    ) -> tuple[str, StageManifest | None]:
        """Return ``('hit'|'miss'|'invalid', manifest)`` for a stage fingerprint."""
        manifest = A.read_stage_manifest(layout.manifest_path(stage))
        if manifest is None:
            return "miss", None
        if manifest.status != StageStatus.SUCCEEDED:
            return "miss", manifest
        if manifest.input_fingerprints.get("self") != fingerprint:
            return "invalid", manifest
        for rel in manifest.outputs:
            if not (layout.root / rel).exists():
                return "invalid", manifest
        return "hit", manifest

    # -- run ---------------------------------------------------------------

    def run(
        self,
        cfg: PipelineConfig,
        layout: A.EpisodeLayout,
        *,
        order: Sequence[str] | None = None,
        until: str | None = None,
        force: Sequence[str] = (),
        dry_run: bool = False,
        run_id: str | None = None,
        code_revision: str = "",
    ) -> RunOutcome:
        dag = self.dag(order)
        names = list(dag.order)

        if until is not None:
            if until not in names:
                raise KeyError(f"--until {until!r} is not a stage in this DAG {names}")
            names = names[: names.index(until) + 1]
        force_set = set(force)
        unknown_force = force_set - set(self.stages)
        if unknown_force:
            raise KeyError(f"--force-stage refers to unknown stage(s) {sorted(unknown_force)}")

        run_id = run_id or A.hash_json(
            {"episode": cfg.episode_id, "config": config_hash(cfg), "ts": A.utc_now_iso()}
        )[:12]
        cfg.run_id = run_id
        layout.ensure()
        cfg_hash = config_hash(cfg)
        dump_resolved(cfg, layout.resolved_config_json)

        outcome = RunOutcome(episode_id=cfg.episode_id, run_id=run_id)
        if dry_run:
            outcome.stopped_at = "dry-run"
            for name in names:
                outcome.records.append(
                    RunRecord(
                        stage=name,
                        status=StageStatus.SKIPPED,
                        fingerprint="(dry-run)",
                        duration_s=0.0,
                        manifest_path=str(layout.manifest_path(name)),
                    )
                )
            return outcome

        # A stage forced by the user also invalidates everything downstream of
        # it in this run: seen via `forced_downstream`.
        forced_downstream = False
        upstream: dict[str, StageResult] = {}

        for name in names:
            stage = self.stages[name]
            ctx = StageContext(
                layout=layout,
                config=cfg,
                repo_root=Path(cfg.repo_root),
                run_id=run_id,
                episode_id=cfg.episode_id,
                stage_name=name,
                log=StageLogger(
                    run_id=run_id, episode_id=cfg.episode_id, stage=name, sink=self._log_sink
                ),
                upstream=upstream,
                config_hash=cfg_hash,
            )

            if name in force_set:
                forced_downstream = True
            effective_force = name in force_set or forced_downstream

            input_check = stage.validate_inputs(ctx)
            if not input_check.ok:
                detail = "; ".join(str(i) for i in input_check.errors)
                outcome.error = f"stage {name}: input validation failed: {detail}"
                self._write_failure(
                    layout, stage, ctx, outcome, ValueError(outcome.error), input_check.warnings
                )
                outcome.stopped_at = name
                return outcome

            fp = stage.fingerprint(ctx)
            if not effective_force:
                status, manifest = self.cache_status(layout, name, fp)
                if status == "hit" and manifest is not None:
                    result = StageResult(
                        stage=name,
                        outputs=[layout.root / rel for rel in manifest.outputs],
                        metrics=dict(manifest.metrics),
                        warnings=list(manifest.warnings),
                    )
                    upstream[name] = result
                    outcome.records.append(
                        RunRecord(
                            stage=name,
                            status=StageStatus.CACHED,
                            fingerprint=fp,
                            duration_s=0.0,
                            manifest_path=str(layout.manifest_path(name)),
                            outputs=list(manifest.outputs),
                            metrics=dict(manifest.metrics),
                            warnings=list(manifest.warnings),
                        )
                    )
                    continue

            timer = A.StageTimer()
            try:
                result = stage.run(ctx)
            except BaseException as exc:  # noqa: BLE001 - re-raised after recording
                outcome.error = f"stage {name} failed: {type(exc).__name__}: {exc}"
                manifest = self._write_failure(layout, stage, ctx, outcome, exc, [], fingerprint=fp, timer=timer)
                outcome.records.append(
                    RunRecord(
                        stage=name,
                        status=StageStatus.FAILED,
                        fingerprint=fp,
                        duration_s=timer.elapsed_s,
                        manifest_path=str(layout.manifest_path(name)),
                        outputs=list(manifest.outputs),
                        errors=list(manifest.errors),
                    )
                )
                outcome.stopped_at = name
                return outcome

            out_check = stage.validate_outputs(result)
            if not out_check.ok:
                detail = "; ".join(str(i) for i in out_check.errors)
                outcome.error = f"stage {name}: output validation failed: {detail}"
                manifest = self._write_failure(
                    layout, stage, ctx, outcome, ValueError(outcome.error), out_check.warnings,
                    fingerprint=fp, timer=timer,
                )
                outcome.records.append(
                    RunRecord(
                        stage=name,
                        status=StageStatus.FAILED,
                        fingerprint=fp,
                        duration_s=timer.elapsed_s,
                        manifest_path=str(layout.manifest_path(name)),
                        errors=list(manifest.errors),
                    )
                )
                outcome.stopped_at = name
                return outcome

            rel_outputs = result.relative_outputs(layout.root)
            manifest = make_stage_manifest(
                stage=name,
                stage_version=stage.version,
                status=StageStatus.SUCCEEDED,
                input_fingerprints={"self": fp},
                config_hash=cfg_hash,
                code_revision=code_revision,
                started_at=timer.started_at,
                duration_s=timer.elapsed_s,
                outputs=rel_outputs,
                metrics=result.metrics,
                warnings=result.warnings,
            )
            A.write_stage_manifest(layout, manifest)
            upstream[name] = result
            outcome.records.append(
                RunRecord(
                    stage=name,
                    status=StageStatus.SUCCEEDED,
                    fingerprint=fp,
                    duration_s=timer.elapsed_s,
                    manifest_path=str(layout.manifest_path(name)),
                    outputs=rel_outputs,
                    metrics=result.metrics,
                    warnings=result.warnings,
                )
            )

        return outcome

    # -- failure handling --------------------------------------------------

    def _write_failure(
        self,
        layout: A.EpisodeLayout,
        stage: Stage,
        ctx: StageContext,
        outcome: RunOutcome,
        exc: BaseException,
        warnings: Sequence[str],
        *,
        fingerprint: str = "",
        timer: A.StageTimer | None = None,
    ) -> StageManifest:
        """Publish a failed manifest and drop partial products for this stage."""
        stage_dir = layout.stage_dir(stage.name)
        A.cleanup_partials(stage_dir)
        manifest = failure_manifest(
            stage=stage.name,
            stage_version=stage.version,
            exc=exc,
            input_fingerprints={"self": fingerprint} if fingerprint else {},
            config_hash=ctx.config_hash,
            started_at=timer.started_at if timer else "",
            duration_s=timer.elapsed_s if timer else 0.0,
            warnings=list(warnings),
        )
        # A failed model stage can still publish explicitly labelled diagnostic
        # images. Keep these inspectable without treating them as model results.
        for output in getattr(exc, "diagnostic_outputs", ()):
            path = Path(output).resolve()
            try:
                relative = path.relative_to(layout.root.resolve())
            except ValueError:
                continue
            if path.is_file():
                manifest.outputs.append(relative.as_posix())
        A.write_stage_manifest(layout, manifest)
        _ = outcome  # kept for symmetry with the success path
        return manifest


def iter_stage_dirs(layout: A.EpisodeLayout) -> Iterator[tuple[str, Path]]:
    """Yield ``(stage, dir)`` pairs that exist in an episode."""
    for name in ("ingest", "hand", "retarget", "robot", "render", "projection"):
        d = layout.stage_dir(name)
        if d.is_dir():
            yield name, d
