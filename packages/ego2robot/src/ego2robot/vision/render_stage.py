"""``render`` stage: MuJoCo RGB, robot/gripper masks and metric depth per frame.

Gate M3 requires the renderer to produce all four outputs
(system-design.md 18.5).  PNGs are written per frame rather than one big
``.npz`` so the episode stays inspectable by a human, as section 13 intends.

The stage also runs :func:`~ego2robot.vision.rendering.verify_projection` once
per episode: emissive markers are placed at the episode's own TCP targets and
the rendered marker centroids must match the analytic pinhole projection.  That
is the evidence that the camera basis is right, not an eyeballed overlay.
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
    RobotTrajectory,
    ValidationResult,
)
from ..pipeline.registry import register_stage
from ..pipeline.stage import StageContext, StageResult
from .render import (
    CameraSpec,
    MujocoRenderer,
    camera_spec_from_pinhole,
    verify_projection,
)

__all__ = ["RenderStage"]


def _write_png(path: Path, array: np.ndarray) -> Path:
    """Write a uint8 RGB or boolean mask PNG atomically.

    The temporary file keeps the ``.png`` suffix: imageio picks its codec from
    the extension, so appending ``.tmp`` to the name would break the write.
    """
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    data = array
    if data.dtype == bool:
        data = (data.astype(np.uint8) * 255)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    imageio.imwrite(tmp, data)
    tmp.replace(path)
    return path


@register_stage
class RenderStage:
    """``render``: robot-only RGB + masks + depth, plus a projection check."""

    name = "render"
    version = "1"

    # -- contract ----------------------------------------------------------

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        for path, stage in (
            (ctx.layout.base_json, "robot"),
            (ctx.layout.trajectory_npz, "robot"),
            (ctx.layout.actions_npz, "retarget"),
            (ctx.layout.episode_json, "ingest"),
        ):
            if not path.is_file():
                res.add(
                    "render.missing_input",
                    f"{path} missing; run the {stage!r} stage first",
                )
        if ctx.layout.trajectory_npz.is_file():
            # One MuJoCo scene, one camera, one rendered robot: H=2 has no
            # renderer yet (ADR-0022), so refuse instead of rendering hand 0.
            try:
                trajectory = RobotTrajectory.load_npz(ctx.layout.trajectory_npz)
                require_sequence_support("render", trajectory)
            except HandCapabilityError as exc:
                res.add("render.hand_count_unsupported", str(exc))
            except Exception as exc:  # noqa: BLE001 - unreadable input is reported
                res.add("render.unreadable_trajectory", str(exc))
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        robot_manifest = A.read_stage_manifest(ctx.layout.manifest_path("robot"))
        return A.hash_json(
            {
                "robot_fingerprint": robot_manifest.input_fingerprints.get("self", "")
                if robot_manifest
                else "",
                "base_json": A.file_sha256(ctx.layout.base_json)
                if ctx.layout.base_json.is_file()
                else "",
                "trajectory": A.file_sha256(ctx.layout.trajectory_npz)
                if ctx.layout.trajectory_npz.is_file()
                else "",
                "render_stage_code_sha": A.file_sha256(Path(__file__)),
                "renderer_code_sha": A.file_sha256(Path(__file__).with_name("render.py")),
                "render": ctx.config.render.model_dump(mode="json"),
                "projection": ctx.config.projection.model_dump(mode="json"),
            }
        )

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        cfg = ctx.config.render
        if not cfg.enabled:
            result.warnings.append("render.enabled is false; stage produced nothing")
            result.payload["skipped"] = True
            return result

        episode = EpisodeManifest.model_validate(A.read_json(ctx.layout.episode_json))
        action = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
        trajectory = RobotTrajectory.load_npz(ctx.layout.trajectory_npz)
        base_payload = A.read_json(ctx.layout.base_json)
        T_C_B = np.asarray(base_payload["T_C_B"], dtype=np.float64)

        camera = episode.camera
        if cfg.width is not None or cfg.height is not None:
            camera = camera.model_copy(
                update={
                    "width": int(cfg.width or camera.width),
                    "height": int(cfg.height or camera.height),
                }
            )
            result.warnings.append(
                "render width/height overridden in config; K was not rescaled, so "
                "projection consistency requires the override to be a no-op"
            )
        if cfg.camera_fovy_deg is not None:
            spec = CameraSpec(
                width=camera.width,
                height=camera.height,
                T_C_B=T_C_B,
                fovy_deg=float(cfg.camera_fovy_deg),
                K=camera.K_array,
                source="config override",
                notes=["fovy overridden in render config; K-derived fovy was not used"],
            )
            result.warnings.append("camera fovy was overridden in the render config")
        else:
            spec = camera_spec_from_pinhole(camera, T_C_B)

        from ..robots.factory import build_robot

        robot_config, robot = build_robot(ctx)

        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        gq = np.asarray(trajectory["gripper_qpos"], dtype=np.float64)
        ik_success = np.asarray(trajectory["ik_success"])[:, 0]
        frame_index = np.asarray(trajectory["frame_index"], dtype=np.int64)
        T = qpos.shape[0]

        visual = ctx.layout.visual_dir
        depth_dir = ctx.layout.depth_dir
        rgb_paths: list[Path] = []
        robot_mask_paths: list[Path] = []
        gripper_mask_paths: list[Path] = []
        depths = np.full((T, camera.height, camera.width), np.nan, dtype=np.float32)
        robot_pixel_counts = np.zeros(T, dtype=np.int64)
        gripper_pixel_counts = np.zeros(T, dtype=np.int64)

        with MujocoRenderer(robot, spec, shadows=cfg.shadow) as renderer:
            for t in range(T):
                frame = renderer.render(qpos[t], gripper_qpos=gq[t], frame_index=int(frame_index[t]))
                stem = f"frame-{int(frame_index[t]):06d}"
                rgb_paths.append(_write_png(visual / "robot" / f"{stem}.png", frame["rgb"]))
                robot_mask_paths.append(
                    _write_png(ctx.layout.masks_dir / "robot" / f"{stem}.png", frame["robot_mask"])
                )
                gripper_mask_paths.append(
                    _write_png(ctx.layout.masks_dir / "gripper" / f"{stem}.png", frame["gripper_mask"])
                )
                depths[t] = frame["depth_m"]
                robot_pixel_counts[t] = int(np.count_nonzero(frame["robot_mask"]))
                gripper_pixel_counts[t] = int(np.count_nonzero(frame["gripper_mask"]))

        np.savez_compressed(depth_dir / "robot_depth_m.npz", depth_m=depths, frame_index=frame_index)

        # ---- projection verification ---------------------------------------
        # The golden check uses a generated fan of points spread over the field
        # of view.  It deliberately does NOT use the episode's own TCP targets:
        # those sit exactly where the gripper geometry is, so the arm occludes the
        # markers and there is nothing to measure.  The camera basis being checked
        # is a property of the renderer, not of this trajectory.
        check = verify_projection(robot, spec)
        projection: dict[str, Any] = {
            "status": "checked",
            "max_error_px": check["max_error_px"],
            "rms_error_px": check.get("rms_error_px"),
            "tolerance_px": check["tolerance_px"],
            "passes": check["passes"],
            "visible_points": check["visible_points"],
            "num_points": check["num_points"],
            "generated_points": check.get("generated_points", True),
            "point_frame": check.get("point_frame", "C"),
            "camera_basis": check["camera_basis"],
        }
        _write_png(
            ctx.layout.previews_dir / "projection-golden-markers.png",
            check["marker_rgb"],
        )
        result.add_output(ctx.layout.previews_dir / "projection-golden-markers.png")
        if not check["passes"]:
            max_err = check.get("max_error_px")
            detail = f"{max_err:.2f} px" if isinstance(max_err, (int, float)) else "no visible marker"
            result.warnings.append(
                f"golden projection check failed: max error {detail} "
                f"> {check['tolerance_px']:.2f} px"
            )

        result.metrics.update(
            {
                "num_frames": T,
                "width": camera.width,
                "height": camera.height,
                "fovy_deg": round(spec.fovy_deg, 6),
                "camera_znear_m": spec.znear,
                "camera_zfar_m": spec.zfar,
                "rgb_frames_written": len(rgb_paths),
                "robot_visible_frames": int(np.count_nonzero(robot_pixel_counts + gripper_pixel_counts > 0)),
                "robot_visible_ratio": round(
                    float(np.mean((robot_pixel_counts + gripper_pixel_counts) > 0)), 6
                )
                if T
                else 0.0,
                "robot_mask_pixels_median": int(np.median(robot_pixel_counts)) if T else 0,
                "gripper_mask_pixels_median": int(np.median(gripper_pixel_counts)) if T else 0,
                "robot_mask_area_ratio_max": round(
                    float(np.max(robot_pixel_counts + gripper_pixel_counts) / (camera.width * camera.height)), 6
                )
                if T
                else 0.0,
                "depth_valid_ratio": round(float(np.mean(np.isfinite(depths))), 9),
                "depth_min_m": round(float(np.nanmin(depths)), 6) if np.isfinite(depths).any() else None,
                "depth_max_m": round(float(np.nanmax(depths)), 6) if np.isfinite(depths).any() else None,
                "projection_check": projection,
            }
        )
        result.add_output(depth_dir / "robot_depth_m.npz")
        result.payload["render_spec"] = spec
        result.payload["projection_check"] = projection
        result.payload["robot"] = robot
        return result

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        if result.payload.get("skipped"):
            return res
        metrics = result.metrics
        if metrics.get("rgb_frames_written", 0) != metrics.get("num_frames", -1):
            res.add(
                "render.frame_count",
                f"wrote {metrics.get('rgb_frames_written')} RGB frames for "
                f"{metrics.get('num_frames')} trajectory frames",
            )
        if metrics.get("robot_visible_frames", 0) == 0:
            res.add(
                "render.robot_never_visible",
                "the robot was invisible in every rendered frame; the camera or base "
                "placement is wrong",
            )
        depth_ratio = metrics.get("depth_valid_ratio", 0.0)
        if depth_ratio == 0.0:
            res.add(
                "render.no_valid_depth",
                "no frame had any valid robot depth; system-design.md 18.5 requires "
                "metric depth output",
            )
        if metrics.get("robot_mask_area_ratio_max", 0.0) > 0.99:
            res.add(
                "render.mask_covers_image",
                "the robot mask covers essentially the whole image; camera or base "
                "placement is implausible",
                severity="warning",
            )
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("render.missing_output", f"declared output does not exist: {path}")
        return res
