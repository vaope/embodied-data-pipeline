"""``projection`` stage: the M3 geometric gate, rendered as evidence images.

implementation-plan.md M3 and system-design.md 18.4 require that the robot's
end-effector projection and the target hand/TCP agree on sampled frames, with a
saved evidence video.  This stage produces exactly that:

* ``previews/projection-overlay.mp4``      -- full-length overlay
* ``previews/projection-frame-XXXXXX.png`` -- representative stills
* ``previews/projection-3d.png``           -- camera + hand + TCP + arm in 3-D
* ``previews/projection-report.json``      -- the numbers behind the pictures

Three independent TCP estimates are drawn so a systematic error is attributable:

``target``
    ``T_C_TCP`` from the canonical action sequence, projected with ``K``.
``fk``
    ``FK(qpos)`` from the solved joints, pushed into camera coordinates through ``T_C_B``.
``robot_render``
    the actual MuJoCo render of the arm, i.e. ground truth for what the
    synthetic frame will look like.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core import artifacts as A
from ..core.capabilities import HandCapabilityError, require_sequence_support
from ..core.frames import Frame, transform_name
from ..core.schema import (
    CanonicalActionSequence,
    EpisodeManifest,
    HandSequence,
    RobotTrajectory,
    ValidationResult,
)
from ..core.transforms import invert_transform
from ..pipeline.registry import register_stage
from ..pipeline.stage import StageContext, StageResult
from .frames_3d import (
    draw_camera_frustum,
    draw_frame_axes,
    draw_trajectory_3d,
    render_scene_figure,
)
from .hand_overlay import write_image, write_overlay_video

__all__ = ["ProjectionStage", "draw_projection_overlay"]

TARGET_COLOUR = (255, 64, 64)      # RGB red   - canonical T_C_TCP target
FK_COLOUR = (64, 128, 255)         # RGB blue  - FK of the solved qpos
HAND_COLOUR = (64, 220, 64)        # RGB green - measured hand keypoints


def _project(K: np.ndarray, points_C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection returning ``(uv, in_front)``."""
    P = np.asarray(points_C, dtype=np.float64).reshape(-1, 3)
    z = P[:, 2]
    in_front = z > 1e-6
    uv = np.full((P.shape[0], 2), np.nan)
    safe = np.where(in_front, z, np.nan)
    uv[:, 0] = K[0, 0] * P[:, 0] / safe + K[0, 2]
    uv[:, 1] = K[1, 1] * P[:, 1] / safe + K[1, 2]
    return uv, in_front


def _draw_axes_2d(image_rgb: np.ndarray, T_C_F: np.ndarray, K: np.ndarray, *, length_m: float, colour: tuple[int, int, int]) -> None:
    """Draw the projected axes of a camera-frame pose onto an RGB image.

    ``image_rgb`` is modified in place.  The drawing happens on one contiguous
    BGR buffer that is converted back once: OpenCV rejects a negative-stride
    ``img[:, :, ::-1]`` view ("Layout of the output array img is incompatible
    with cv::Mat"), and repeated per-primitive conversions are also slower.
    """
    import cv2

    origin = T_C_F[:3, 3]
    tips = [origin + T_C_F[:3, i] * length_m for i in range(3)]
    pts = _project(K, np.vstack([origin, *tips]))[0]
    if not np.isfinite(pts).all():
        return
    bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    o = (int(round(pts[0, 0])), int(round(pts[0, 1])))
    axis_rgb = ((255, 0, 0), (0, 255, 0), (0, 0, 255))  # x red, y green, z blue
    for i in range(3):
        tip = (int(round(pts[i + 1, 0])), int(round(pts[i + 1, 1])))
        cv2.arrowedLine(bgr, o, tip, axis_rgb[i][::-1], 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(bgr, o, 4, colour[::-1], -1, cv2.LINE_AA)
    image_rgb[:, :, :] = bgr[:, :, ::-1]


def draw_projection_overlay(
    image_rgb: np.ndarray,
    *,
    K: np.ndarray,
    target_T_C_TCP: np.ndarray,
    fk_position_C_m: np.ndarray,
    hand_keypoints_uv_px: np.ndarray | None = None,
    gripper_width_m: float | None = None,
    robot_label: str | None = None,
    frame_index: int,
    timestamp_s: float,
    position_error_m: float | None = None,
    rotation_error_rad: float | None = None,
    ik_success: bool = True,
    solver_status: str = "",
    axes_length_m: float = 0.05,
    robot_render_rgb: np.ndarray | None = None,
    robot_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compose one evidence frame (returns RGB).

    All OpenCV calls operate on a single contiguous BGR buffer which is flipped
    back to RGB once at the end: ``img[:, :, ::-1]`` is a negative-stride view
    that OpenCV refuses to use as an output array.
    """
    import cv2

    out = np.array(image_rgb, dtype=np.uint8, copy=True)
    if robot_render_rgb is not None and robot_mask is not None:
        # This is a geometry diagnostic, so show both arm and gripper pixels.
        m = robot_mask.astype(bool)
        if m.any():
            out[m] = (
                0.55 * out[m].astype(np.float32) + 0.45 * robot_render_rgb[m].astype(np.float32)
            ).astype(np.uint8)

    bgr = np.ascontiguousarray(out[:, :, ::-1])

    if hand_keypoints_uv_px is not None:
        uv = np.asarray(hand_keypoints_uv_px, dtype=np.float64).reshape(21, 2)
        for j, rgb in ((0, HAND_COLOUR), (4, HAND_COLOUR), (8, HAND_COLOUR), (12, HAND_COLOUR)):
            if np.isfinite(uv[j]).all():
                cv2.circle(
                    bgr,
                    (int(round(uv[j, 0])), int(round(uv[j, 1]))),
                    8,
                    rgb[::-1],
                    2,
                    cv2.LINE_AA,
                )

    origin = np.asarray(target_T_C_TCP, dtype=np.float64)[:3, 3]
    tips = [origin + np.asarray(target_T_C_TCP, dtype=np.float64)[:3, i] * axes_length_m for i in range(3)]
    target_axis_uv = _project(K, np.vstack([origin, *tips]))[0]
    if np.isfinite(target_axis_uv).all():
        o = (int(round(target_axis_uv[0, 0])), int(round(target_axis_uv[0, 1])))
        for i, rgb in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
            cv2.arrowedLine(
                bgr,
                o,
                (int(round(target_axis_uv[i + 1, 0])), int(round(target_axis_uv[i + 1, 1]))),
                rgb[::-1],
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )
        cv2.circle(bgr, o, 5, TARGET_COLOUR[::-1], -1, cv2.LINE_AA)

    fk_uv, in_front = _project(K, np.asarray(fk_position_C_m, dtype=np.float64))
    if in_front[0] and np.isfinite(fk_uv[0]).all():
        centre = (int(round(fk_uv[0, 0])), int(round(fk_uv[0, 1])))
        cv2.drawMarker(
            bgr, centre, FK_COLOUR[::-1], cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA
        )

    target_uv, target_front = _project(K, origin)
    lines = [
        f"frame {frame_index}  t={timestamp_s:.3f}s",
        f"IK {'OK' if ik_success else 'FAIL'} ({solver_status})",
    ]
    if robot_label:
        lines.insert(1, robot_label)
    if position_error_m is not None:
        lines.append(f"pos err {position_error_m * 1000:.2f} mm")
    if rotation_error_rad is not None:
        lines.append(f"rot err {np.degrees(rotation_error_rad):.3f} deg")
    if gripper_width_m is not None and np.isfinite(gripper_width_m):
        lines.append(f"gripper {gripper_width_m * 1000:.1f} mm")
    lines.append("red axes+dot = target T_C_TCP")
    lines.append("blue cross  = FK EEF")

    y = 28
    for text in lines:
        # ONE putText call per line: drawing a dark outline pass and a light fill
        # pass on the same baseline produced visibly doubled glyphs at this font
        # scale, which reads as a rendering bug in the review artifacts
        cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3, cv2.LINE_AA)
        y += 28

    if target_front[0] and np.isfinite(target_uv[0]).all() and np.isfinite(fk_uv[0]).all():
        delta = float(np.linalg.norm(target_uv[0] - fk_uv[0]))
        label = f"projected gap {delta:.1f} px"
        cv2.putText(bgr, label, (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 3, cv2.LINE_AA)
    return np.ascontiguousarray(bgr[:, :, ::-1])


@register_stage
class ProjectionStage:
    """``projection``: the saved evidence for the M3 geometry gate."""

    name = "projection"
    version = "1"

    def validate_inputs(self, ctx: StageContext) -> ValidationResult:
        res = ValidationResult()
        for path, stage in (
            (ctx.layout.episode_json, "ingest"),
            (ctx.layout.actions_npz, "retarget"),
            (ctx.layout.trajectory_npz, "robot"),
            (ctx.layout.base_json, "robot"),
        ):
            if not path.is_file():
                res.add("projection.missing_input", f"{path} missing; run {stage!r} first")
        if ctx.layout.actions_npz.is_file():
            # The overlay draws one target/FK pair per frame (ADR-0022).
            try:
                action = CanonicalActionSequence.load_npz(ctx.layout.actions_npz)
                require_sequence_support("projection", action)
            except HandCapabilityError as exc:
                res.add("projection.hand_count_unsupported", str(exc))
            except Exception as exc:  # noqa: BLE001
                res.add("projection.unreadable_actions", str(exc))
        return res

    def fingerprint(self, ctx: StageContext) -> str:
        robot_manifest = A.read_stage_manifest(ctx.layout.manifest_path("robot"))
        return A.hash_json(
            {
                "robot_fingerprint": robot_manifest.input_fingerprints.get("self", "")
                if robot_manifest
                else "",
                "trajectory": A.file_sha256(ctx.layout.trajectory_npz)
                if ctx.layout.trajectory_npz.is_file()
                else "",
                "projection_code_sha": A.file_sha256(Path(__file__)),
                "projection": ctx.config.projection.model_dump(mode="json"),
            }
        )

    def run(self, ctx: StageContext) -> StageResult:
        result = StageResult(stage=self.name)
        cfg = ctx.config.projection
        layout = ctx.layout

        episode = EpisodeManifest.model_validate(A.read_json(layout.episode_json))
        action = CanonicalActionSequence.load_npz(layout.actions_npz)
        trajectory = RobotTrajectory.load_npz(layout.trajectory_npz)
        base_payload = A.read_json(layout.base_json)
        T_C_B = np.asarray(base_payload["T_C_B"], dtype=np.float64)
        T_B_C = invert_transform(T_C_B)
        K = episode.camera.K_array

        hand = HandSequence.load_npz(layout.hand_npz) if layout.hand_npz.is_file() else None

        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        gq = np.asarray(trajectory["gripper_qpos"], dtype=np.float64)
        pos_err = np.asarray(trajectory["position_error_m"])[:, 0]
        rot_err = np.asarray(trajectory["rotation_error_rad"])[:, 0]
        ik_success = np.asarray(trajectory["ik_success"])[:, 0]
        status = np.asarray(trajectory["solver_status"])
        frame_index = np.asarray(trajectory["frame_index"], dtype=np.int64)
        T = qpos.shape[0]

        T_C_TCP = np.asarray(action["T_C_TCP"], dtype=np.float64)[:, 0]
        valid = np.asarray(action["valid"], dtype=np.bool_)[:, 0]
        width = np.asarray(action["gripper_width_m"], dtype=np.float64)[:, 0]
        timestamps = np.asarray(action["timestamp_s"], dtype=np.float64)

        from ..robots.factory import build_robot
        from ..vision.render import CameraSpec, MujocoRenderer, camera_spec_from_pinhole

        robot_config, robot = build_robot(ctx)

        spec = camera_spec_from_pinhole(episode.camera, T_C_B)
        status_names = {
            int(k): v for k, v in (trajectory.meta.get("status_registry") or {}).items()
        }

        # ---- per-frame FK EEF positions in the camera frame ----------------
        fk_positions_C = np.full((T, 3), np.nan)
        arm_chain_C: list[dict[str, np.ndarray]] = []
        chain_bodies = _chain_bodies(robot, robot_config.eef_body_name, robot_config.base_body_name)
        for t in range(T):
            robot.set_state(qpos[t], gq[t])
            T_B_TCP_fk = robot.tcp_pose() if hasattr(robot, "tcp_pose") else (
                robot.body_pose(robot_config.eef_body_name) @ robot_config.T_E_TCP_array
            )
            fk_positions_C[t] = (T_C_B @ T_B_TCP_fk)[:3, 3]
            arm_chain_C.append(
                {name: (T_C_B @ robot.body_pose(name))[:3, 3] for name in chain_bodies}
            )

        # ---- overlays -------------------------------------------------------
        previews = layout.previews_dir
        overlay_paths: list[Path] = []
        frames_iter: list[np.ndarray] = []
        gap_px: list[float] = []
        per_frame: list[dict[str, Any]] = []

        from ..ingest.video import iter_frames

        video = layout.video_path
        if not video.is_file():
            video = _find_video(layout)
        renderer = MujocoRenderer(robot, spec)
        try:
            for t, (_i, ts, image_rgb) in enumerate(iter_frames(video, start=0, stop=T)):
                if t >= T:
                    break
                rendered = renderer.render(qpos[t], gripper_qpos=gq[t], frame_index=int(frame_index[t]))
                hand_uv = (
                    np.asarray(hand["keypoints_uv_px"])[t, 0] if hand is not None else None
                )
                overlay = draw_projection_overlay(
                    image_rgb,
                    K=K,
                    target_T_C_TCP=T_C_TCP[t],
                    fk_position_C_m=fk_positions_C[t],
                    hand_keypoints_uv_px=hand_uv,
                    gripper_width_m=float(width[t]) if robot_config.gripper_dof else None,
                    robot_label=(
                        robot_config.robot_id
                        + (" (arm only; no gripper)" if not robot_config.gripper_dof else "")
                    ),
                    frame_index=int(frame_index[t]),
                    timestamp_s=float(ts),
                    position_error_m=float(pos_err[t]) if np.isfinite(pos_err[t]) else None,
                    rotation_error_rad=float(rot_err[t]) if np.isfinite(rot_err[t]) else None,
                    ik_success=bool(ik_success[t]),
                    solver_status=status_names.get(int(status[t]), str(int(status[t]))),
                    axes_length_m=float(cfg.draw_axes_m),
                    robot_render_rgb=rendered["rgb"],
                    robot_mask=rendered["robot_mask"] | rendered["gripper_mask"],
                )
                frames_iter.append(overlay)
                target_uv = _project(K, T_C_TCP[t][:3, 3])[0]
                fk_uv = _project(K, fk_positions_C[t])[0]
                gap = float(np.linalg.norm(target_uv[0] - fk_uv[0])) if (
                    np.isfinite(target_uv).all() and np.isfinite(fk_uv).all()
                ) else float("nan")
                gap_px.append(gap)
                per_frame.append(
                    {
                        "frame_index": int(frame_index[t]),
                        "timestamp_s": round(float(ts), 6),
                        "valid": bool(valid[t]),
                        "ik_success": bool(ik_success[t]),
                        "solver_status": status_names.get(int(status[t]), str(int(status[t]))),
                        "position_error_m": round(float(pos_err[t]), 9) if np.isfinite(pos_err[t]) else None,
                        "rotation_error_rad": round(float(rot_err[t]), 9) if np.isfinite(rot_err[t]) else None,
                        "target_uv_px": [round(float(v), 3) for v in target_uv[0]] if np.isfinite(target_uv).all() else None,
                        "fk_uv_px": [round(float(v), 3) for v in fk_uv[0]] if np.isfinite(fk_uv).all() else None,
                        "projected_gap_px": round(gap, 4) if np.isfinite(gap) else None,
                    }
                )
        finally:
            renderer.close()

        if frames_iter:
            video_path = previews / "projection-overlay.mp4"
            write_overlay_video(video_path, frames_iter, fps=episode.timebase.nominal_fps or 30.0)
            result.add_output(video_path)

            n_stills = max(1, int(cfg.sample_frames))
            picks = np.linspace(0, len(frames_iter) - 1, min(n_stills, len(frames_iter))).astype(int)
            for k, i in enumerate(picks):
                p = previews / f"projection-frame-{int(frame_index[i]):06d}.png"
                write_image(p, frames_iter[i])
                result.add_output(p)

        # ---- 3-D view -------------------------------------------------------
        mid = T // 2
        T_C_C = np.eye(4)  # the camera is the origin of frame C

        def _draw(ax: Any) -> None:
            draw_camera_frustum(ax, T_C_C, label="camera C")
            draw_trajectory_3d(
                ax,
                T_C_TCP[:, :3, 3],
                colour="#d62728",
                label="target T_C_TCP",
            )
            draw_trajectory_3d(
                ax,
                fk_positions_C,
                colour="#1f77b4",
                label="FK EEF",
                linewidth=1.0,
                alpha=0.8,
            )
            draw_frame_axes(ax, T_C_TCP[mid], length=float(cfg.draw_axes_m), label="TCP axes", linewidth=2.5)
            T_C_B_pose = np.linalg.inv(T_B_C)
            draw_frame_axes(ax, T_C_B_pose, length=float(cfg.draw_axes_m), label="robot base B", linewidth=2.5)
            for name, pos in arm_chain_C[mid].items():
                ax.plot([pos[0]], [pos[1]], [pos[2]], marker="o", color="#ff7f0e", markersize=4)
            if hand is not None:
                kp = np.asarray(hand["keypoints_C_m"])[mid, 0]
                finite = np.isfinite(kp).all(axis=1)
                ax.scatter(kp[finite, 0], kp[finite, 1], kp[finite, 2], c="#2ca02c", s=10, label="hand keypoints")

        points_for_limits = [T_C_TCP[:, :3, 3], fk_positions_C, np.array([[0.0, 0.0, 0.0]])]
        p3d = render_scene_figure(
            previews / "projection-3d.png",
            title=f"M3 geometry: camera, hand, canonical TCP and {robot_config.robot_id} EEF in frame C",
            draw=_draw,
            point_sets=points_for_limits,
        )
        result.add_output(p3d)

        finite_gap = np.asarray([g for g in gap_px if np.isfinite(g)])
        report = {
            "schema_version": "ego2robot.projection.v1",
            "episode_id": ctx.episode_id,
            "pose_variable": str(transform_name(Frame.C, Frame.TCP)),
            "frames": per_frame,
            "summary": {
                "num_frames": T,
                "ik_success_ratio": round(float(np.mean(ik_success)) if T else 0.0, 6),
                "p95_position_error_m": round(float(np.nanpercentile(pos_err, 95)), 9) if T else None,
                "p95_rotation_error_rad": round(float(np.nanpercentile(rot_err, 95)), 9) if T else None,
                "median_projected_gap_px": round(float(np.median(finite_gap)), 4) if finite_gap.size else None,
                "p95_projected_gap_px": round(float(np.percentile(finite_gap, 95)), 4) if finite_gap.size else None,
                "max_projected_gap_px": round(float(np.max(finite_gap)), 4) if finite_gap.size else None,
            },
            "notes": [
                "projected gap = distance in pixels between the canonical target TCP and the FK EEF",
                "target and FK are projected with the same K, so a systematic gap indicates a "
                "coordinate or IK error, not a camera error",
                "robot_render pixels are blended into the stills as a visual ground truth of "
                "what the synthetic frame contains",
            ],
        }
        result.add_output(A.atomic_write_json(previews / "projection-report.json", report))
        result.metrics.update(report["summary"])
        result.metrics["num_stills"] = int(len(frames_iter))
        result.payload["projection_report"] = report
        return result

    def validate_outputs(self, result: StageResult) -> ValidationResult:
        res = ValidationResult()
        summary = result.metrics
        if not summary:
            res.add("projection.no_metrics", "projection stage produced no metrics")
            return res
        if summary.get("num_frames", 0) == 0:
            res.add("projection.no_frames", "no frames were processed")
        ratio = summary.get("ik_success_ratio", 0.0) or 0.0
        p95_pos = summary.get("p95_position_error_m")
        p95_rot = summary.get("p95_rotation_error_rad")
        gate_pos = result.metrics.get("gate_p95_position_error_m", 0.01)
        gate_rot = result.metrics.get("gate_p95_rotation_error_rad", 0.10)
        if p95_pos is not None and p95_pos >= gate_pos:
            res.add(
                "projection.p95_position",
                f"p95 position error {p95_pos:.6f} m is not below the MVP gate {gate_pos} m",
                severity="warning",
            )
        if p95_rot is not None and p95_rot >= gate_rot:
            res.add(
                "projection.p95_rotation",
                f"p95 rotation error {p95_rot:.6f} rad is not below the MVP gate {gate_rot} rad",
                severity="warning",
            )
        if ratio < 0.95:
            res.add(
                "projection.ik_success_ratio",
                f"IK success ratio {ratio:.4f} is below the MVP gate 0.95",
                severity="warning",
            )
        for path in result.outputs:
            if not Path(path).is_file():
                res.add("projection.missing_output", f"declared output does not exist: {path}")
        return res


def _find_video(layout: Any) -> Path:
    for f in sorted(layout.source_dir.glob("*")):
        if f.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            return f
    raise FileNotFoundError(f"no video under {layout.source_dir}")


def _chain_bodies(robot: Any, eef_name: str, base_name: str) -> tuple[str, ...]:
    """Choose a few named bodies along the model's own base-to-EEF chain."""

    mujoco = robot._mujoco
    model = robot.model
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_name))
    base_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_name))
    if body_id < 0 or base_id < 0:
        raise ValueError(f"model has no base/EEF bodies {base_name!r}/{eef_name!r}")
    ids: list[int] = []
    current = body_id
    while current > 0:
        ids.append(current)
        if current == base_id:
            break
        current = int(model.body_parentid[current])
    if not ids or (base_id == 0 and current != 0) or (base_id != 0 and ids[-1] != base_id):
        raise ValueError(f"{base_name!r} is not an ancestor of {eef_name!r}")
    ids.reverse()
    sampled = sorted(set(int(i) for i in np.linspace(0, len(ids) - 1, min(5, len(ids)))))
    return tuple(str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, ids[i])) for i in sampled)
