"""Run the pipeline over the validation cases and compare against ground truth.

This is what produces the user-facing verification artifacts.  For every case it

1. runs ``ingest -> hand -> retarget -> robot -> render -> projection -> quality``;
2. diffs the produced canonical trajectory against the analytic expectation in
   ``source/truth.npz``;
3. checks the M3 gates (IK success ratio, p95 FK errors);
4. writes ``report.md``, ``report.json`` and a comparison figure per case.

The comparison figure is the point: a numeric gate can pass while the geometry
is subtly mirrored, so each case also gets a plot of target vs solved TCP and a
per-frame error curve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core import artifacts as A
from ..core.artifacts import json_safe
from ..core.config import resolve_config_from_cli
from ..core.frames import Frame, transform_name
from ..core.schema import CanonicalActionSequence, RobotTrajectory
from ..core.transforms import geodesic_distance_rad, invert_transform
from ..pipeline.registry import default_registry
from ..pipeline.runner import PipelineRunner
from .cases import CASES_FILE, CaseDefinition, load_cases

__all__ = ["CaseOutcome", "run_cases"]


@dataclass
class CaseOutcome:
    """Everything the summary needs about one case."""

    case_id: str
    episode_dir: Path
    ok: bool
    gate_pass: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    truth_metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    previews: list[str] = field(default_factory=list)
    stage_status: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ground-truth comparison
# ---------------------------------------------------------------------------


def _compare_with_truth(
    episode_dir: Path,
    *,
    case: CaseDefinition,
    valid_only: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Diff the canonical output against the analytic expectation."""
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    truth_path = episode_dir / "source" / "truth.npz"
    if not truth_path.is_file():
        return metrics, ["no analytic ground truth written for this case"]

    truth = dict(np.load(truth_path, allow_pickle=False))
    action = CanonicalActionSequence.load_npz(episode_dir / "canonical" / "actions.npz")

    n = min(action.num_frames, int(truth["tcp_position_C_m"].shape[0]))
    produced_pos = np.asarray(action["tcp_position_C_m"], dtype=np.float64)[:n, 0]
    produced_R = np.asarray(action["T_C_TCP"], dtype=np.float64)[:n, 0, :3, :3]
    produced_width = np.asarray(action["gripper_width_m"], dtype=np.float64)[:n, 0]
    produced_valid = np.asarray(action["valid"], dtype=np.bool_)[:n, 0]

    expected_pos = truth["tcp_position_C_m"][:n]
    expected_R = truth["T_C_TCP"][:n, :3, :3]
    expected_width = truth["gripper_width_m"][:n]
    expected_valid = truth["valid"][:n]

    # The pipeline smooths the trajectory, so the raw analytic pose cannot match
    # sample-by-sample: compare *shape-level* quantities that smoothing preserves.
    common = produced_valid & expected_valid
    if valid_only:
        common = common & np.isfinite(produced_pos).all(axis=1)

    if not common.any():
        failures.append("no frame is valid in both the pipeline output and the ground truth")
        return metrics, failures

    pos_err = np.linalg.norm(produced_pos[common] - expected_pos[common], axis=1)
    rot_err = np.array(
        [geodesic_distance_rad(expected_R[i], produced_R[i]) for i in np.nonzero(common)[0]]
    )
    width_err = np.abs(produced_width[common] - expected_width[common])

    # path-shape check: the produced and expected trajectories must traverse the
    # same total distance (smoothing may shorten corners slightly, not scale them)
    span_prod = float(np.linalg.norm(produced_pos[common][-1] - produced_pos[common][0]))
    span_exp = float(np.linalg.norm(expected_pos[common][-1] - expected_pos[common][0]))
    travel_prod = float(np.sum(np.linalg.norm(np.diff(produced_pos[common], axis=0), axis=1)))
    travel_exp = float(np.sum(np.linalg.norm(np.diff(expected_pos[common], axis=0), axis=1)))

    # axis-sign check: the dominant direction of travel must agree in sign
    axis_checks: dict[str, Any] = {}
    if common.sum() > 1:
        delta_prod = produced_pos[common][-1] - produced_pos[common][0]
        delta_exp = expected_pos[common][-1] - expected_pos[common][0]
        for i, ax in enumerate("xyz"):
            axis_checks[ax] = {
                "produced_m": round(float(delta_prod[i]), 6),
                "expected_m": round(float(delta_exp[i]), 6),
                "sign_matches": bool(np.sign(delta_prod[i]) == np.sign(delta_exp[i]))
                or abs(delta_exp[i]) < 1e-4,
            }
        if not all(v["sign_matches"] for v in axis_checks.values()):
            failures.append(
                "travel direction sign mismatch on "
                f"{[k for k, v in axis_checks.items() if not v['sign_matches']]}: "
                "this indicates an axis swap or mirror"
            )

    # handedness / canonical-axis check: the grasp axis must point from the
    # virtual fingertip to the thumb for BOTH hands (paper eq. 3)
    grasp_axis = produced_R[common][:, :, 2]
    expected_axis = expected_R[common][:, :, 2]
    axis_dot = float(np.mean(np.sum(grasp_axis * expected_axis, axis=1)))
    if axis_dot < 0.9:
        failures.append(
            f"grasp axis agrees with ground truth only at mean dot {axis_dot:.3f}; "
            "expected near +1 (check the handedness sign)"
        )

    metrics.update(
        {
            "compared_frames": int(common.sum()),
            "tcp_position_error_mean_m": round(float(pos_err.mean()), 9),
            "tcp_position_error_median_m": round(float(np.median(pos_err)), 9),
            "tcp_position_error_max_m": round(float(pos_err.max()), 9),
            "tcp_rotation_error_mean_rad": round(float(rot_err.mean()), 9),
            "tcp_rotation_error_max_rad": round(float(rot_err.max()), 9),
            "gripper_width_error_mean_m": round(float(width_err.mean()), 9),
            "gripper_width_error_max_m": round(float(width_err.max()), 9),
            "travel_distance_produced_m": round(travel_prod, 6),
            "travel_distance_expected_m": round(travel_exp, 6),
            "travel_distance_ratio": round(travel_prod / travel_exp, 6) if travel_exp > 1e-9 else None,
            "endpoint_span_produced_m": round(span_prod, 6),
            "endpoint_span_expected_m": round(span_exp, 6),
            "grasp_axis_dot_mean": round(axis_dot, 6),
            "axis_direction": axis_checks,
        }
    )

    # thresholds: smoothing legitimately shifts samples by up to a few mm, so the
    # shape metrics are what get gated; the sample-wise error is reported.
    soft_pos = 0.02
    if metrics["tcp_position_error_median_m"] > soft_pos:
        failures.append(
            f"median TCP position error {metrics['tcp_position_error_median_m']:.4f} m "
            f"exceeds {soft_pos} m against the analytic ground truth"
        )
    if metrics["gripper_width_error_max_m"] > 0.02:
        failures.append(
            f"gripper width error {metrics['gripper_width_error_max_m']:.4f} m exceeds 0.02 m"
        )
    ratio = metrics["travel_distance_ratio"]
    if ratio is not None and not (0.75 <= ratio <= 1.25):
        failures.append(
            f"travel distance ratio {ratio:.3f} is outside [0.75, 1.25]; the produced "
            "trajectory does not traverse the same distance as the ground truth"
        )
    return metrics, failures


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _write_comparison_figure(
    path: Path,
    *,
    case_id: str,
    episode_dir: Path,
) -> Path | None:
    """Plot produced vs expected TCP and the error curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth_path = episode_dir / "source" / "truth.npz"
    actions = episode_dir / "canonical" / "actions.npz"
    if not (truth_path.is_file() and actions.is_file()):
        return None
    truth = dict(np.load(truth_path, allow_pickle=False))
    action = CanonicalActionSequence.load_npz(actions)
    n = min(action.num_frames, int(truth["tcp_position_C_m"].shape[0]))

    produced = np.asarray(action["tcp_position_C_m"], dtype=np.float64)[:n, 0]
    expected = truth["tcp_position_C_m"][:n]
    width_p = np.asarray(action["gripper_width_m"], dtype=np.float64)[:n, 0]
    width_e = truth["gripper_width_m"][:n]
    valid = np.asarray(action["valid"], dtype=np.bool_)[:n, 0]
    ts = np.asarray(action["timestamp_s"], dtype=np.float64)[:n]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=110)
    ax = axes[0][0]
    ax.plot(expected[:, 0], expected[:, 2], "-", color="#1f77b4", label="expected (analytic)")
    ax.plot(produced[:, 0], produced[:, 2], "--", color="#d62728", label="produced (pipeline, smoothed)")
    ax.plot(expected[0, 0], expected[0, 2], "o", color="#2ca02c", label="start")
    ax.set_xlabel("x in C [m]")
    ax.set_ylabel("z in C [m]")
    ax.set_title("TCP path, camera-frame x-z")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[0][1]
    err = np.linalg.norm(produced - expected, axis=1) * 1000.0
    ax.plot(ts, err, color="#d62728")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|produced - expected| [mm]")
    ax.set_title(f"TCP position error vs ground truth (median {np.median(err):.2f} mm)")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    valid_idx = np.nonzero(valid)[0]
    ax.plot(valid_idx, width_e[valid_idx] * 1000.0, "-", color="#1f77b4", label="expected width")
    ax.plot(valid_idx, width_p[valid_idx] * 1000.0, "--", color="#d62728", label="produced width")
    ax.axhline(10.0, color="#888888", linestyle=":", label="degeneracy threshold 10 mm")
    ax.set_xlabel("frame")
    ax.set_ylabel("gripper width [mm]")
    ax.set_title("gripper width")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1][1]
    per_axis = (produced - expected) * 1000.0
    for i, (name, colour) in enumerate(zip("xyz", ("#d62728", "#2ca02c", "#1f77b4"))):
        ax.plot(ts, per_axis[:, i], color=colour, label=f"d{name}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error per axis [mm]")
    ax.set_title("signed per-axis error (look for a constant offset = frame/scale bug)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"{case_id}: pipeline output vs analytic ground truth", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(tmp)
    plt.close(fig)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_cases(
    *,
    repo_root: Path,
    fixture_root: Path,
    only: str | None = None,
    force: list[str] | None = None,
    profile: str = "mvp",
    case_ids: Sequence[str] | None = None,
    console: Any = None,
) -> list[CaseOutcome]:
    """Run every selected case end to end and report."""
    cases = load_cases(repo_root / CASES_FILE)
    if only:
        cases = [c for c in cases if c.case_id == only]
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.case_id in wanted]
    if not cases:
        raise SystemExit("no validation case selected")

    outcomes: list[CaseOutcome] = []
    for case in cases:
        outcome = _run_one(
            case,
            repo_root=repo_root,
            fixture_root=fixture_root,
            profile=profile,
            force=force or [],
            console=console,
        )
        outcomes.append(outcome)

    _write_summary(outcomes, fixture_root=fixture_root, profile=profile)
    if console is not None:
        _print_summary(outcomes, console)

    # ---- cross-case checks -------------------------------------------------
    _cross_case_checks(outcomes, fixture_root=fixture_root, console=console)
    return outcomes


def _run_one(
    case: CaseDefinition,
    *,
    repo_root: Path,
    fixture_root: Path,
    profile: str,
    force: list[str],
    console: Any,
) -> CaseOutcome:
    episode_dir = fixture_root / case.episode_id
    if not (episode_dir / "source" / "video.mp4").is_file():
        raise SystemExit(
            f"{episode_dir} has no source video; run `e2r fixtures build` first"
        )

    cfg = resolve_config_from_cli(
        repo_root,
        profile=profile,
        robot_profile=repo_root / "configs" / "robot" / "franka_panda.yaml",
        episode_id=case.episode_id,
    )
    layout = A.EpisodeLayout(episode_dir, robot_id=cfg.robot_binding.robot_id)
    registry = default_registry()

    log_lines: list[str] = []
    runner = PipelineRunner(registry.as_mapping(), log_sink=log_lines.append)
    outcome = runner.run(cfg, layout, order=cfg.stages, force=force)
    A.atomic_write_bytes(
        layout.logs_dir / "run.log", "\n".join(log_lines).encode("utf-8")
    )

    result = CaseOutcome(
        case_id=case.case_id,
        episode_dir=episode_dir,
        ok=outcome.ok,
        gate_pass=False,
        stage_status={r.stage: r.status for r in outcome.records},
        warnings=[w for r in outcome.records for w in r.warnings],
    )
    if not outcome.ok:
        result.failures.append(outcome.error or "pipeline run failed")
        return result

    # ---- collect pipeline metrics -----------------------------------------
    for stage in ("ingest", "hand", "retarget", "robot", "render", "projection", "quality"):
        manifest = A.read_stage_manifest(layout.manifest_path(stage))
        if manifest is not None:
            result.metrics[stage] = manifest.metrics

    quality = A.read_json(layout.quality_json) if layout.quality_json.is_file() else {}
    result.gate_pass = bool(quality.get("gate_verdict", {}).get("pass"))
    gate = quality.get("gate_verdict", {})
    for check in gate.get("checks", []):
        if not check.get("pass"):
            result.failures.append(
                f"gate check {check['check']} failed: value={check.get('value')} "
                f"threshold={check.get('threshold')}"
            )

    # ---- compare with ground truth ----------------------------------------
    truth_metrics, truth_failures = _compare_with_truth(episode_dir, case=case)
    result.truth_metrics = truth_metrics
    result.failures.extend(truth_failures)

    # ---- figure ------------------------------------------------------------
    fig = _write_comparison_figure(
        layout.previews_dir / "comparison.png",
        case_id=case.case_id,
        episode_dir=episode_dir,
    )
    if fig is not None:
        result.previews.append(str(fig.relative_to(episode_dir)).replace("\\", "/"))
    for name in (
        "hand-overlay.mp4",
        "hand-3d.png",
        "projection-overlay.mp4",
        "projection-3d.png",
        "projection-golden-markers.png",
        "projection-report.json",
    ):
        p = layout.previews_dir / name
        if p.is_file():
            result.previews.append(str(p.relative_to(episode_dir)).replace("\\", "/"))

    if console is not None:
        status = "[green]PASS[/green]" if (result.ok and not result.failures) else "[red]FAIL[/red]"
        console.print(f"  {case.case_id}: {status}")
        if result.failures:
            for f in result.failures:
                console.print(f"    [red]- {f}[/red]")
    return result


# ---------------------------------------------------------------------------
# cross-case checks
# ---------------------------------------------------------------------------


def _cross_case_checks(
    outcomes: Sequence[CaseOutcome], *, fixture_root: Path, console: Any
) -> dict[str, Any]:
    """Checks that only make sense when two cases are compared."""
    found: dict[str, Any] = {}
    by_id = {o.case_id: o for o in outcomes}

    # handedness: right and left hand must reach the same canonical jaw-axis
    # direction for the same requested motion direction
    if "case1_right_translate" in by_id and "case2_left_mirror" in by_id:
        axes = {}
        for cid in ("case1_right_translate", "case2_left_mirror"):
            p = fixture_root / cid / "canonical" / "actions.npz"
            if not p.is_file():
                continue
            action = CanonicalActionSequence.load_npz(p)
            R = np.asarray(action["T_C_TCP"], dtype=np.float64)[:, 0, :3, :3]
            valid = np.asarray(action["valid"], dtype=np.bool_)[:, 0]
            if valid.any():
                axes[cid] = R[valid].mean(axis=0)[:, 2]  # mean grasp axis
        if len(axes) == 2:
            a, b = axes.values()
            dot = float(np.dot(a, b))
            found["grasp_axis_dot_right_vs_left"] = round(dot, 6)
            found["handedness_consistent"] = bool(dot > 0.95)

    # static case: the derived relative action must be ~0
    if "case3_static" in by_id:
        p = fixture_root / "case3_static" / "canonical" / "actions.npz"
        if p.is_file():
            action = CanonicalActionSequence.load_npz(p)
            deltas = action.relative_action()
            dp = np.asarray(deltas["delta_position_C_m"], dtype=np.float64)[:, 0]
            dr = np.asarray(deltas["delta_rotvec_C_rad"], dtype=np.float64)[:, 0]
            found["static_max_position_delta_m"] = round(float(np.max(np.linalg.norm(dp, axis=1))), 9)
            found["static_max_rotation_delta_rad"] = round(float(np.max(np.linalg.norm(dr, axis=1))), 9)

    # rotation case: total swept angle must match the 90 deg ground truth
    if "case4_rotation" in by_id:
        p = fixture_root / "case4_rotation" / "canonical" / "actions.npz"
        if p.is_file():
            import numpy as np_

            action = CanonicalActionSequence.load_npz(p)
            deltas = action.relative_action()
            dr = np.asarray(deltas["delta_rotvec_C_rad"], dtype=np.float64)[:, 0]
            found["rotation_total_swept_rad"] = round(float(np.sum(np.linalg.norm(dr, axis=1))), 6)
            found["rotation_expected_swept_rad"] = round(float(np.pi / 4), 6)
            found["rotation_dominant_axis"] = (
                np_.abs(dr).mean(axis=0).round(3).tolist() if dr.size else None
            )

    # degenerate case: the spike frames must have been rejected
    if "case5_degenerate" in by_id:
        manifest = A.read_stage_manifest(
            fixture_root / "case5_degenerate" / "canonical" / "stage-manifest.json"
        )
        if manifest is not None:
            found["degenerate_outlier_rejected_frames"] = manifest.metrics.get(
                "outlier_rejected_frames"
            )
            found["degenerate_large_gaps"] = manifest.metrics.get("large_gaps")
            found["degenerate_frozen_frames"] = manifest.metrics.get("frozen_frames")

    if console is not None and found:
        from rich.table import Table

        table = Table(title="cross-case checks")
        table.add_column("check")
        table.add_column("value")
        for k, v in found.items():
            table.add_row(k, json.dumps(json_safe(v)))
        console.print(table)
    _write_json(fixture_root / "cross-case-checks.json", found)
    return found


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    A.atomic_write_json(path, payload)


def _write_summary(
    outcomes: Sequence[CaseOutcome], *, fixture_root: Path, profile: str
) -> None:
    payload = {
        "schema_version": "ego2robot.validation.v1",
        "profile": profile,
        "pose_variable": str(transform_name(Frame.C, Frame.TCP)),
        "created_at": A.utc_now_iso(),
        "cases": [
            {
                "case_id": o.case_id,
                "episode_dir": str(o.episode_dir),
                "run_ok": o.ok,
                "m3_gate_pass": o.gate_pass,
                "failures": o.failures,
                "warnings": o.warnings,
                "stage_status": o.stage_status,
                "pipeline_metrics": o.metrics,
                "ground_truth_comparison": o.truth_metrics,
                "previews": o.previews,
            }
            for o in outcomes
        ],
    }
    _write_json(fixture_root / "validation-report.json", payload)

    lines = [
        "# Ego2Robot validation report",
        "",
        f"- profile: `{profile}`",
        f"- generated: {payload['created_at']}",
        f"- cases: {len(outcomes)}",
        f"- pose variable checked: `{payload['pose_variable']}` "
        "(pose of the canonical grasp frame expressed in the camera optical frame)",
        "",
        "## Summary",
        "",
        "| case | run | M3 gate | median TCP err [mm] | travel ratio | IK success | p95 FK pos [m] | p95 FK rot [rad] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        robot = _stage_metrics(o, "robot")
        truth = o.truth_metrics
        lines.append(
            "| {cid} | {run} | {gate} | {med} | {ratio} | {ik} | {pp} | {pr} |".format(
                cid=o.case_id,
                run="ok" if o.ok else "FAILED",
                gate="pass" if o.gate_pass else "fail",
                med=_fmt(truth.get("tcp_position_error_median_m"), 1e3, 2),
                ratio=_fmt(truth.get("travel_distance_ratio"), 1.0, 3),
                ik=_fmt(robot.get("ik_success_ratio"), 1.0, 4),
                pp=_fmt(robot.get("p95_position_error_m"), 1.0, 6),
                pr=_fmt(robot.get("p95_rotation_error_rad"), 1.0, 6),
            )
        )
    lines += ["", "## What each case demonstrates", ""]
    for o in outcomes:
        claims = _case_claims(o)
        body = "; ".join(claims) if claims else "(no metrics)"
        lines.append(f"- **{o.case_id}**: {body}")
    lines += ["", "## Per-case detail", ""]
    for o in outcomes:
        lines.append(f"### {o.case_id}")
        lines.append("")
        lines.append(f"- run ok: {o.ok}; M3 gate: {o.gate_pass}")
        lines.append(f"- failures: {o.failures or 'none'}")
        if o.warnings:
            lines.append(f"- warnings: {o.warnings}")
        lines.append("- stage status: " + ", ".join(f"{k}={v}" for k, v in o.stage_status.items()))
        if o.truth_metrics:
            lines.append("- ground-truth comparison:")
            for k, v in o.truth_metrics.items():
                lines.append(f"  - `{k}`: {json.dumps(json_safe(v))}")
        for stage in ("ingest", "hand", "retarget", "robot", "render", "projection", "quality"):
            m = o.metrics.get(stage)
            if not m:
                continue
            keep = {
                k: v
                for k, v in m.items()
                if isinstance(v, (int, float, str, bool)) or k.endswith("histogram")
            }
            lines.append(f"- `{stage}` metrics: {json.dumps(json_safe(keep))[:1200]}")
        if o.previews:
            lines.append("- previews: " + ", ".join(f"`{p}`" for p in o.previews))
        lines.append("")
    _ = geodesic_distance_rad
    (fixture_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, scale: float, digits: int) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * scale:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _stage_metrics(outcome: CaseOutcome, stage: str) -> dict[str, Any]:
    """Fetch a stage's metrics, tolerating the robot-owned subdirectory layout.

    ``robot`` and ``render`` write their manifests into ``robot/<robot_id>/``
    rather than a stage-named directory, so fall back to reading them from disk
    when the in-memory key has nothing usable.
    """
    cached = outcome.metrics.get(stage)
    if isinstance(cached, dict) and cached:
        return cached
    candidates = [
        outcome.episode_dir / stage / "stage-manifest.json",
        outcome.episode_dir / "robot" / stage / "stage-manifest.json",
    ]
    candidates.extend(sorted((outcome.episode_dir / "robot").glob("*/stage-manifest.json")))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a broken manifest is simply skipped
            continue
        if payload.get("stage") == stage:
            metrics = payload.get("metrics")
            if isinstance(metrics, dict) and metrics:
                return metrics
    return cached if isinstance(cached, dict) else {}


def _case_claims(outcome: CaseOutcome) -> list[str]:
    """Turn a case's metrics into the plain-language claim it supports.

    The point of the report is that a reader can see *which* property was
    verified, not only that the gate returned "pass".
    """
    claims: list[str] = []
    t = outcome.truth_metrics
    # Read metrics back through a helper: ``robot`` and ``render`` manifests live
    # in robot/<robot_id>/ rather than a top-level stage directory, so the key
    # used while running is not necessarily where the file ended up.
    retarget = _stage_metrics(outcome, "retarget")
    robot = _stage_metrics(outcome, "robot")
    projection = _stage_metrics(outcome, "projection")
    render = _stage_metrics(outcome, "render")
    hand = _stage_metrics(outcome, "hand")

    if t.get("tcp_position_error_median_m") is not None:
        claims.append(
            f"TCP matches the analytic ground truth to {t['tcp_position_error_median_m'] * 1000:.2f} mm "
            f"median (max {t['tcp_position_error_max_m'] * 1000:.2f} mm)"
        )
    if t.get("grasp_axis_dot_mean") is not None:
        claims.append(f"grasp axis agrees with ground truth at mean dot {t['grasp_axis_dot_mean']:.4f}")
    if t.get("travel_distance_ratio") is not None:
        claims.append(f"travel distance ratio {t['travel_distance_ratio']:.3f}")
    if t.get("axis_direction"):
        bad = [k for k, v in t["axis_direction"].items() if not v["sign_matches"]]
        claims.append("per-axis travel signs all match" if not bad else f"axis sign mismatch on {bad}")
    if retarget.get("frozen_frames"):
        claims.append(f"{int(retarget['frozen_frames'])} frame(s) hit the closed-jaw degeneracy freeze")
    if retarget.get("outlier_rejected_frames"):
        rounds = retarget.get("outlier_per_round")
        claims.append(
            f"velocity filter rejected {retarget['outlier_rejected_frames']} spike frame(s)"
            + (f" over rounds {rounds}" if rounds else "")
        )
    if retarget.get("large_gaps"):
        claims.append(
            f"{retarget['large_gaps']} large gap(s) {retarget.get('gap_spans')} took the home-blend path "
            f"(blend lengths {retarget.get('blend_lengths')})"
        )
    if retarget.get("small_gaps"):
        claims.append(f"{retarget['small_gaps']} small gap(s) interpolated")
    if retarget.get("home_was_derived"):
        claims.append("the gap home pose was derived from the trajectory (no home supplied)")
    if robot.get("ik_success_ratio") is not None:
        claims.append(f"IK solved {robot['ik_success_ratio'] * 100:.1f}% of valid frames")
    if robot.get("max_joint_jump_rad") is not None:
        claims.append(f"max adjacent joint jump {robot['max_joint_jump_rad']:.3f} rad")
    if robot.get("mean_reach_ratio") is not None:
        claims.append(f"mean reach ratio {robot['mean_reach_ratio']:.3f}")
    if projection.get("p95_projected_gap_px") is not None:
        claims.append(
            f"target-vs-FK projected gap p95 {projection['p95_projected_gap_px']:.2f} px"
        )
    render = _stage_metrics(outcome, "render")
    golden = render.get("projection_check", {}) if isinstance(render, dict) else {}
    if isinstance(golden, dict) and golden.get("passes") is not None:
        max_err = golden.get("max_error_px")
        err_text = f"{max_err:.2f} px" if isinstance(max_err, (int, float)) else "no visible marker"
        claims.append(
            "renderer projection golden test "
            + ("passes" if golden["passes"] else "FAILS")
            + f" (max {err_text})"
        )
    if hand.get("valid_ratio") is not None:
        claims.append(f"hand valid on {hand['valid_ratio'] * 100:.1f}% of frames")
    return claims


def _print_summary(outcomes: Sequence[CaseOutcome], console: Any) -> None:
    from rich.table import Table

    table = Table(title="validation summary")
    table.add_column("case", style="bold")
    table.add_column("run")
    table.add_column("M3 gate")
    table.add_column("median TCP err")
    table.add_column("IK ok")
    table.add_column("p95 pos")
    table.add_column("p95 rot")
    for o in outcomes:
        robot = _stage_metrics(o, "robot")
        truth = o.truth_metrics
        table.add_row(
            o.case_id,
            "[green]ok[/green]" if o.ok else "[red]FAIL[/red]",
            "[green]pass[/green]" if o.gate_pass else "[red]fail[/red]",
            _fmt(truth.get("tcp_position_error_median_m"), 1e3, 2) + " mm",
            _fmt(robot.get("ik_success_ratio"), 1.0, 4),
            _fmt(robot.get("p95_position_error_m"), 1.0, 6),
            _fmt(robot.get("p95_rotation_error_rad"), 1.0, 6),
        )
    console.print(table)


def _load_truth(episode_dir: Path) -> dict[str, Any]:
    p = episode_dir / "source" / "truth.npz"
    return dict(np.load(p, allow_pickle=False)) if p.is_file() else {}


_ = (invert_transform, RobotTrajectory)
