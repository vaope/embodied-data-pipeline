"""Fixture builder: render the synthetic validation clips and their annotations.

Output per case (see :mod:`ego2robot.tools.cases`)::

    <fixture_root>/<case_id>/
      episode.json         e2r EpisodeManifest including an explicit T_C_B
      case.json            case definition + analytic ground truth (npz alongside)
      source/video.mp4     synthetic tabletop clip with the hand drawn over it
      source/hands.hdf5    MANO-21 metric keypoints in the camera frame
      source/camera.json   pinhole K derived from the chosen fov
      source/truth.npz     analytic expected T_C_TCP / deltas

The scene is a simple tabletop seen from an ego viewpoint: a table plane, a
graspable object, lighting gradient and mild noise so that the video is a real
decodable clip rather than a flat colour field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..core import artifacts as A
from ..core.schema import (
    CameraFile,
    CameraModel,
    EpisodeManifest,
    EpisodeSource,
    Province,
    TaskSpec,
    Timebase,
)
from ..ingest.fixture import HandAnnotations, write_hand_annotations
from ..viz.hand_overlay import HAND_BONES, HAND_KEYPOINT_COLORS, write_overlay_video
from .cases import CaseDefinition, case_ground_truth
from .hand_synth import HandPose

__all__ = ["build_case", "build_cases", "render_case_video"]


def _camera_for(case: CaseDefinition) -> CameraModel:
    """Pinhole camera with a 60 deg horizontal field of view."""
    fov_x = np.radians(60.0)
    fx = (case.width / 2.0) / np.tan(fov_x / 2.0)
    return CameraModel(
        width=case.width,
        height=case.height,
        K=[[float(fx), 0.0, case.width / 2.0], [0.0, float(fx), case.height / 2.0], [0.0, 0.0, 1.0]],
        distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
        calibration_source="calibrated",
        confidence=1.0,
        debug_only=False,
    )


def _project(camera: CameraModel, points_C: np.ndarray) -> np.ndarray:
    """Project ``[... , 3]`` to ``[..., 2]``, preserving the caller's shape."""
    return camera.project(np.asarray(points_C, dtype=np.float64))


def _project_one(camera: CameraModel, point_C: np.ndarray) -> np.ndarray:
    """Project a single 3-D point to ``(u, v)``."""
    return np.asarray(camera.project(np.asarray(point_C, dtype=np.float64).reshape(1, 3)))[0]


def _render_scene_frame(
    case: CaseDefinition,
    camera: CameraModel,
    frame_index: int,
    keypoints_uv: np.ndarray,
    *,
    object_xy_m: tuple[float, float],
    rng: np.random.Generator,
    occlude_fraction: float,
) -> np.ndarray:
    """A tabletop ego view with the hand drawn on top."""
    import cv2

    W, H = camera.width, camera.height
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # --- sky / wall gradient ------------------------------------------------
    for y in range(H):
        shade = int(210 - 60 * (y / H))
        img[y, :, :] = (shade, shade, min(255, shade + 8))

    # --- table plane --------------------------------------------------------
    horizon = int(H * 0.34)
    table_colour = (150, 138, 122)  # BGR
    img[horizon:, :, :] = table_colour
    cv2.line(img, (0, horizon), (W, horizon), (110, 100, 90), 2, cv2.LINE_AA)

    # --- a graspable object (upright cylinder) ------------------------------
    bottom_C = np.array([object_xy_m[0], object_xy_m[1], 0.38])
    top_C = bottom_C + np.array([0.0, 0.0, -0.055])
    uv_bottom = _project_one(camera, bottom_C)
    uv_top = _project_one(camera, top_C)
    if np.isfinite(uv_bottom).all() and np.isfinite(uv_top).all():
        radius_uv = float(
            abs(
                _project_one(camera, bottom_C + np.array([0.022, 0.0, 0.0]))[0]
                - _project_one(camera, bottom_C - np.array([0.022, 0.0, 0.0]))[0]
            )
            / 2.0
        )
        cv2.rectangle(
            img,
            (int(uv_top[0] - radius_uv), int(uv_top[1])),
            (int(uv_bottom[0] + radius_uv), int(uv_bottom[1])),
            (96, 120, 190),
            -1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            img,
            (int(uv_top[0]), int(uv_top[1])),
            (max(1, int(radius_uv)), max(2, int(radius_uv * 0.32))),
            0,
            0,
            360,
            (140, 160, 220),
            -1,
            cv2.LINE_AA,
        )

    # --- a second static object for scale reference -------------------------
    ref_C = np.array([-0.06, 0.06, 0.42])
    uv_ref = _project_one(camera, ref_C)
    if np.isfinite(uv_ref).all():
        cv2.circle(
            img, (int(uv_ref[0]), int(uv_ref[1])), 18, (170, 190, 120), -1, cv2.LINE_AA
        )

    # --- sensor noise -------------------------------------------------------
    noise = rng.normal(0.0, 2.6, size=img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # --- hand overlay (what a Path A dataset would not have, hence the HDF5) -
    overlay = np.ascontiguousarray(img[:, :, ::-1])
    finite = np.isfinite(keypoints_uv).all(axis=1)
    # soft shadow blob to make the hand read as "in the scene"
    if finite.any():
        centre = np.nanmean(keypoints_uv[finite], axis=0)
        if np.isfinite(centre).all():
            cv2.circle(
                overlay,
                (int(centre[0]), int(centre[1])),
                int(0.09 * W),
                (205, 205, 205),
                -1,
                cv2.LINE_AA,
            )
            img = np.ascontiguousarray(overlay[:, :, ::-1])
    for a, b in HAND_BONES:
        if finite[a] and finite[b]:
            cv2.line(
                overlay,
                (int(keypoints_uv[a, 0]), int(keypoints_uv[a, 1])),
                (int(keypoints_uv[b, 0]), int(keypoints_uv[b, 1])),
                (60, 60, 60),
                5,
                cv2.LINE_AA,
            )
    for j in range(21):
        if finite[j]:
            cv2.circle(
                overlay,
                (int(keypoints_uv[j, 0]), int(keypoints_uv[j, 1])),
                4,
                (80, 80, 80),
                -1,
                cv2.LINE_AA,
            )

    # --- occlusion proxy: the object can cover the hand briefly -------------
    if occlude_fraction > 0.0:
        k = int(occlude_fraction * H)
        overlay[:k, :, :] = overlay[:k, :, :] // 2

    # No case-id/frame HUD is burned into the fixture video itself: every
    # preview and debug overlay already stamps the frame index, and two
    # competing labels in the same corner are unreadable.
    _ = (case, frame_index)
    _ = HAND_KEYPOINT_COLORS
    return np.ascontiguousarray(overlay[:, :, ::-1])


def _corrupt(
    case: CaseDefinition,
    keypoints: np.ndarray,
    confidence: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the case's corruption model, returning ``(kp, conf, valid)``."""
    kp = np.array(keypoints, dtype=np.float32, copy=True)
    conf = np.array(confidence, dtype=np.float32, copy=True)
    valid = np.ones(case.num_frames, dtype=bool)

    spikes = case.corruptions.get("spike_frames") or []
    magnitude = float(case.corruptions.get("spike_position_m", 0.06))
    for f in spikes:
        if 0 <= f < case.num_frames:
            # a detection jump: the whole hand teleports, which is exactly the
            # failure mode the paper's velocity filter targets
            kp[f, :, 0] += magnitude
            kp[f, :, 2] -= magnitude * 0.5
            conf[f] = 0.45

    span = case.corruptions.get("dropout_span")
    if span:
        lo, hi = int(span[0]), int(span[1])
        kp[lo:hi] = np.nan
        conf[lo:hi] = 0.0
        valid[lo:hi] = False

    # mild per-frame jitter everywhere else, so smoothing has something to do
    jitter = case.corruptions.get("jitter_m")
    if jitter:
        noise = rng.normal(0.0, float(jitter), size=kp.shape)
        kp = np.where(np.isfinite(kp), kp + noise, kp)

    return kp, conf, valid


def build_case(
    case: CaseDefinition,
    *,
    root: Path,
    force: bool = False,
    console: Any = None,
    jitter_m: float = 0.0012,
) -> dict[str, Any]:
    """Render one case into ``root/<case_id>`` and return a summary."""
    episode_dir = root / case.episode_id
    video_path = episode_dir / "source" / "video.mp4"
    if video_path.is_file() and not force:
        return {"case_id": case.case_id, "path": str(episode_dir), "skipped": True}

    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "source").mkdir(parents=True, exist_ok=True)
    camera = _camera_for(case)

    if "jitter_m" not in case.corruptions:
        case.corruptions["jitter_m"] = jitter_m

    poses: list[HandPose] = [case.sequence(i) for i in range(case.num_frames)]
    kp_C = np.stack([p.keypoints_C_m for p in poses], axis=0).astype(np.float32)
    uv = camera.project(kp_C.astype(np.float64)).astype(np.float32)

    rng = np.random.default_rng(abs(hash(case.case_id)) % (2**32))
    kp_C_input, conf, valid = _corrupt(case, kp_C, np.full(case.num_frames, 0.95, dtype=np.float32), rng)
    uv_input = camera.project(np.nan_to_num(kp_C_input, nan=np.nan).astype(np.float64)).astype(np.float32)

    # ---- video ------------------------------------------------------------
    frames = []
    for i in range(case.num_frames):
        occlude = 0.0
        span = case.corruptions.get("dropout_span")
        if span and span[0] - 2 <= i < span[1] + 2:
            occlude = 0.18
        frames.append(
            _render_scene_frame(
                case,
                camera,
                i,
                uv[i],
                object_xy_m=case.object_xy_m,
                rng=np.random.default_rng(1000 + i),
                occlude_fraction=occlude,
            )
        )
    write_overlay_video(video_path, frames, fps=case.fps)

    # ---- hand annotations --------------------------------------------------
    annotations = HandAnnotations(
        keypoints_C_m=kp_C_input,
        confidence=conf,
        handedness=np.full(case.num_frames, case.handedness, dtype=np.int8),
        uv_px=uv_input,
        keypoint_confidence=None,
        attributes={
            "case_id": case.case_id,
            "source_frame": "C",
            "length_unit": "m",
            "keypoint_convention": "mano21",
        },
    )
    write_hand_annotations(
        episode_dir / "source" / "hands.hdf5",
        {("left" if case.handedness < 0 else "right"): annotations},
        attributes={"case_id": case.case_id, "generator": "ego2robot.tools.synthetic"},
    )

    # ---- camera.json -------------------------------------------------------
    A.atomic_write_json(
        episode_dir / "source" / "camera.json",
        CameraFile(
            camera=camera,
            camera_frame="C",
            provenance="synthetic fixture: K derived from a 60 deg horizontal fov",
        ),
    )

    # ---- episode.json ------------------------------------------------------
    sha = A.file_sha256(video_path)
    manifest = EpisodeManifest(
        episode_id=case.episode_id,
        source=EpisodeSource(
            video_uri="source/video.mp4",
            sha256=sha,
            source_dataset="synthetic_geometry",
            hdf5_uri="source/hands.hdf5",
            hdf5_sha256=A.file_sha256(episode_dir / "source" / "hands.hdf5"),
            license="project-generated",
            publishable=True,
        ),
        timebase=Timebase(
            num_frames=case.num_frames,
            nominal_fps=case.fps,
            timestamps_s=[float(i / case.fps) for i in range(case.num_frames)],
            timebase_source="nominal_fps",
        ),
        camera=camera,
        task=TaskSpec(instruction=case.instruction),
        provenance=Province(created_at=A.utc_now_iso()),
        T_C_B=np.asarray(case.T_C_B, dtype=np.float64).tolist(),
        notes=[
            "synthetic_geometry fixture: hand poses are analytic, not perceived",
            f"checks: {case.checks}",
            *case.notes,
        ],
    )
    A.atomic_write_json(episode_dir / "episode.json", manifest)

    # ---- analytic ground truth --------------------------------------------
    truth = case_ground_truth(case)
    np.savez_compressed(
        episode_dir / "source" / "truth.npz",
        **{k: v for k, v in truth.items() if isinstance(v, np.ndarray)},
    )
    A.atomic_write_json(
        episode_dir / "case.json",
        {
            "case_id": case.case_id,
            "instruction": case.instruction,
            "checks": case.checks,
            "handedness": case.handedness,
            "num_frames": case.num_frames,
            "fps": case.fps,
            "width": case.width,
            "height": case.height,
            "corruptions": case.corruptions,
            "object_xy_m": list(case.object_xy_m),
            "T_C_B": case.T_C_B.tolist(),
            "notes": case.notes,
        },
    )

    if console is not None:
        console.print(
            f"  [green]built[/green] {case.case_id}: {case.num_frames} frames, "
            f"{case.width}x{case.height} @ {case.fps:g} fps, hand={'right' if case.handedness > 0 else 'left'}"
        )
    return {
        "case_id": case.case_id,
        "path": str(episode_dir),
        "num_frames": case.num_frames,
        "valid_frames": int(np.count_nonzero(valid)),
        "video_sha256": sha,
        "skipped": False,
    }


def build_cases(
    cases: Sequence[CaseDefinition],
    *,
    root: Path,
    force: bool = False,
    console: Any = None,
) -> list[dict[str, Any]]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for case in cases:
        out.append(build_case(case, root=root, force=force, console=console))
    return out


def render_case_video(case: CaseDefinition, out_path: Path, *, console: Any = None) -> Path:
    """Render only the video for a case (debug helper)."""
    camera = _camera_for(case)
    (out_path.parent).mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    for i in range(case.num_frames):
        pose = case.sequence(i)
        uv = camera.project(pose.keypoints_C_m)
        frames.append(
            _render_scene_frame(
                case,
                camera,
                i,
                uv,
                object_xy_m=case.object_xy_m,
                rng=np.random.default_rng(1000 + i),
                occlude_fraction=0.0,
            )
        )
    write_overlay_video(out_path, frames, fps=case.fps)
    if console is not None:
        console.print(f"wrote {out_path} ({len(frames)} frames)")
    return out_path


def iterate_cases(cases: Iterable[CaseDefinition]) -> Iterable[tuple[str, int]]:
    for c in cases:
        yield c.case_id, c.num_frames
