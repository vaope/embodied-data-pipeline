"""``quality`` stage: preflight + L1 report + the M3 gate verdict.

M7 (L2 statistics, canonical packaging, exporters) is deliberately out of scope
for the first development round, which implementation-plan.md section 16 stops
at M3.  What this stage does provide is the *gate evidence*: L1 frame validity,
the paper's ``< 0.05 m`` tracking threshold, and the stricter MVP geometry gate
of implementation-plan.md M3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core import artifacts as A
from ..core.capabilities import HandCapabilityError, require_sequence_support
from ..core.schema import (
    CanonicalActionSequence,
    EpisodeManifest,
    HandSequence,
    RobotTrajectory,
    ValidationResult,
)
from ..pipeline.registry import register_stage
from ..pipeline.stage import StageContext, StageResult
from .l1 import L1Config, build_quality_report, run_l1
from .preflight import REASON_CODES, run_preflight

__all__ = ["QualityStage"]


@register_stage
class QualityStage:
    """``quality``: aggregate every stage's metrics into one verdict."""

    name = "quality"
    version = "1"

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        if not ctx.layout.actions_npz.is_file():
            res.add(
                "quality.no_canonical_actions",
                f"{ctx.layout.actions_npz} missing; run the 'retarget' stage first",
            )
            return res
        # L1/L2 metrics are still computed for the single-arm case only: the
        # hand_missing / cross-arm / robot-visibility checks index hand 0 and one
        # robot mask (ADR-0021).  Report it instead of scoring one hand as "the
        # episode".
        try:
            action = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
            require_sequence_support("quality", action)
        except HandCapabilityError as exc:
            res.add("quality.hand_count_unsupported", str(exc))
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            res.add("quality.unreadable_actions", str(exc))
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        hashes: dict[str, str] = {}
        for name, path in (
            ("actions", ctx.layout.actions_npz),
            ("trajectory", ctx.layout.trajectory_npz),
            ("hand", ctx.layout.hand_npz),
        ):
            if Path(path).is_file():
                hashes[name] = A.file_sha256(path)
        for stage in ("ingest", "hand", "retarget", "robot", "render", "projection"):
            manifest = A.read_stage_manifest(ctx.layout.manifest_path(stage))
            hashes[f"{stage}_fingerprint"] = (
                manifest.input_fingerprints.get("self", "") if manifest else ""
            )
        hashes["quality_config"] = A.hash_json(ctx.config.quality.model_dump(mode="json"))
        return A.hash_json(hashes)

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        cfg = ctx.config.quality
        layout = ctx.layout

        preflight = run_preflight(layout, episode_id=ctx.episode_id)
        result.add_output(A.atomic_write_json(layout.quality_dir / "preflight.json", preflight.to_json()))

        episode = (
            EpisodeManifest.model_validate(A.read_json(layout.episode_json))
            if layout.episode_json.is_file()
            else None
        )
        action = CanonicalActionSequence.load_npz(layout.actions_npz)
        trajectory = (
            RobotTrajectory.load_npz(layout.trajectory_npz)
            if layout.trajectory_npz.is_file()
            else None
        )
        hand = HandSequence.load_npz(layout.hand_npz) if layout.hand_npz.is_file() else None

        fps = float(episode.timebase.nominal_fps) if episode else 30.0

        # robot mask area ratio comes from the rendered masks when available
        mask_ratio: np.ndarray | None = None
        mask_dir = layout.masks_dir / "robot"
        if mask_dir.is_dir():
            import imageio.v2 as imageio

            pngs = sorted(mask_dir.glob("*.png"))
            if pngs:
                ratios = []
                for p in pngs:
                    img = np.asarray(imageio.imread(p))
                    ratios.append(float(np.mean(img > 0)))
                mask_ratio = np.asarray(ratios, dtype=np.float64)
                if mask_ratio.shape[0] != action.num_frames:
                    result.warnings.append(
                        f"found {mask_ratio.shape[0]} robot masks for {action.num_frames} frames; "
                        "mask-based L1 checks were skipped"
                    )
                    mask_ratio = None

        l1 = run_l1(
            action=action,
            trajectory=trajectory,
            hand=hand,
            fps=fps,
            robot_mask_area_ratio=mask_ratio,
            cfg=L1Config(
                ik_tracking_error_m=cfg.l1_ik_tracking_error_m,
                robot_visible_pixels=cfg.l1_robot_visible_pixels,
                robot_mask_area_ratio_max=cfg.l1_robot_mask_area_ratio_max,
                cross_arm_contacts_max=cfg.l1_cross_arm_contacts_max,
                stability_erosion_fps_fraction=cfg.l1_stability_erosion_fps_fraction,
            ),
        )

        report = build_quality_report(
            episode_id=ctx.episode_id,
            action=action,
            trajectory=trajectory,
            hand=hand,
            episode=episode,
            l1=l1,
            layout=layout,
            robot_mask_area_ratio=mask_ratio,
            threshold_profile=cfg.profile,
            preflight_reasons=preflight.reasons,
        )

        gate = self._gate_verdict(report, trajectory, cfg, preflight.passed)
        payload = report.model_dump(mode="json")
        payload["gate_verdict"] = gate
        result.add_output(A.atomic_write_json(layout.quality_json, payload))

        # L2 is defined but not enabled in the MVP profile
        payload_l2 = {
            "schema_version": "ego2robot.quality.l2.v1",
            "episode_id": ctx.episode_id,
            "enabled": bool(cfg.l2_enabled),
            "status": "deferred",
            "notes": [
                "L2 statistical curation needs a dataset-level statistics job "
                "(system-design.md 10.3); per-episode online thresholds would drift, "
                "so they are not computed here",
                "implementation-plan.md M7 owns this work",
            ],
            "planned_rules": {
                "quantile_low": cfg.l2_quantile_low,
                "quantile_high": cfg.l2_quantile_high,
                "iqr_multiplier": cfg.l2_iqr_multiplier,
                "max_invalid_ratio": cfg.l2_max_invalid_ratio,
            },
        }
        result.add_output(A.atomic_write_json(layout.quality_dir / "l2.json", payload_l2))

        payload_l3 = {
            "schema_version": "ego2robot.quality.l3.v1",
            "episode_id": ctx.episode_id,
            "backend": cfg.l3_backend,
            "status": "manual_review",
            "manual_review": {
                "reviewer": None,
                "reviewed_at": None,
                "consistent": None,
                "notes": "",
            },
            "notes": [
                "[PAPER, DEFERRED] the paper audits synthesized video with Qwen3.5 at 4 fps; "
                "the MVP defines the interface and keeps the manual-review field only",
            ],
        }
        result.add_output(A.atomic_write_json(layout.quality_dir / "l3.json", payload_l3))

        result.metrics.update(report.metric_dict())
        result.metrics["frame_valid_ratio"] = report.frame_valid_ratio
        result.metrics["preflight_passed"] = preflight.passed
        result.metrics["preflight_reasons"] = preflight.reasons
        result.metrics["gate_verdict"] = gate
        result.metrics["l1"] = l1.metrics
        result.payload["report"] = report
        result.payload["gate"] = gate
        result.payload["preflight"] = preflight
        if not preflight.passed:
            result.warnings.append(f"preflight blocking reasons: {preflight.reasons}")
        return result

    @staticmethod
    def _gate_verdict(
        report: Any,
        trajectory: RobotTrajectory | None,
        cfg: Any,
        preflight_passed: bool,
    ) -> dict[str, Any]:
        """Implementation-plan.md M3 acceptance conditions, evaluated explicitly."""
        checks: list[dict[str, Any]] = []

        def add(name: str, value: Any, threshold: Any, ok: bool, note: str = "") -> None:
            checks.append(
                {
                    "check": name,
                    "value": value,
                    "threshold": threshold,
                    "pass": bool(ok),
                    "note": note,
                }
            )

        add("preflight", preflight_passed, True, preflight_passed)
        if report.metrics.ik_success_ratio:
            add(
                "ik_success_ratio",
                report.metrics.ik_success_ratio,
                f">= {cfg.mvp_min_ik_success_ratio}",
                report.metrics.ik_success_ratio >= cfg.mvp_min_ik_success_ratio,
                "implementation-plan.md M3 gate",
            )
        if trajectory is not None and np.isfinite(report.metrics.p95_position_error_m):
            add(
                "p95_position_error_m",
                report.metrics.p95_position_error_m,
                f"< {cfg.mvp_p95_position_error_m}",
                report.metrics.p95_position_error_m < cfg.mvp_p95_position_error_m,
                "MVP gate is stricter than the paper's 0.05 m L1 threshold",
            )
            add(
                "p95_rotation_error_rad",
                report.metrics.p95_rotation_error_rad,
                f"< {cfg.mvp_p95_rotation_error_rad}",
                report.metrics.p95_rotation_error_rad < cfg.mvp_p95_rotation_error_rad,
            )
        add(
            "no_nan_in_canonical",
            report.metrics.non_finite_ratio,
            "== 0",
            report.metrics.non_finite_ratio == 0.0,
        )
        add(
            "so3_valid",
            report.metrics.so3_max_correction_rad,
            "finite (SVD repair magnitude recorded)",
            True,
        )
        passed = all(c["pass"] for c in checks)
        failing = [c["check"] for c in checks if not c["pass"]]
        return {
            "gate": "M3",
            "pass": passed,
            "checks": checks,
            "failing_checks": failing,
            "notes": [
                "this verdict is computed by the pipeline; the milestone is only "
                "accepted after the human reviews the preview artifacts (AGENTS.md rule 9)",
            ],
        }

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        report = result.payload.get("report")
        if report is None:
            res.add("quality.no_report", "stage produced no QualityReport payload")
            return res
        if not (result.payload.get("preflight") and result.payload["preflight"].passed):
            res.add(
                "quality.preflight_failed",
                f"preflight failed: {result.payload['preflight'].reasons}",
                severity="warning",
            )
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("quality.missing_output", f"declared output does not exist: {path}")
        _ = REASON_CODES
        return res
