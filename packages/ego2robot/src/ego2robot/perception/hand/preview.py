"""Preview generation for the hand stage (M1 evidence artifacts).

Every verification run must produce user-viewable images with the relevant
overlays (AGENTS.md rule 8), so the hand stage calls this module to write:

* ``previews/hand-overlay.mp4``         -- full clip with the 2-D skeleton(s)
* ``previews/hand-overlay-frame-*.png`` -- representative stills
* ``previews/hand-3d.png``              -- camera + keypoints + TCP frames in 3-D
* ``previews/gripper-frame-*.png``      -- TCP axes, jaw line and gripper width

ADR-0021: when the sequence carries two hands both are drawn on the same frame
with a hand-specific colour (right keeps the per-finger palette, left is drawn in
one amber colour), both TCP frames are projected, and each hand gets its own
``gripper-frame-<frame>-<side>.png`` still.  A two-hand preview must never look
like a one-hand preview with a second skeleton missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...core.schema import HAND_SIDE_SIGN, CameraModel, HandSequence
from ...ingest.video import iter_frames
from ...retarget.hand_to_gripper import (
    MANO_INDEX_TIP,
    MANO_MIDDLE_TIP,
    MANO_THUMB_TIP,
    MANO_WRIST,
    FallbackCode,
    RetargetResult,
    retarget_hand_sequence,
)
from ...viz.frames_3d import (
    draw_camera_frustum,
    draw_frame_axes,
    draw_trajectory_3d,
    render_scene_figure,
)
from ...viz.hand_overlay import draw_hand_overlay, write_image, write_overlay_video

__all__ = ["write_hand_previews"]

RETARGET_MARKERS = {
    MANO_WRIST: "wrist",
    MANO_THUMB_TIP: "thumb tip",
    MANO_INDEX_TIP: "index tip",
    MANO_MIDDLE_TIP: "middle tip",
}

#: RGB axis colours for projected TCP frames (x approach, y normal, z jaw)
TCP_AXIS_RGB = ((255, 60, 60), (60, 220, 60), (60, 120, 255))

#: BGR colour used for the *left* hand when both hands share one frame.
LEFT_HAND_BGR = (0, 150, 255)

#: Line colours for the 3-D figure, keyed by side.
_HAND_COLOURS_3D = {"right": "#1f77b4", "left": "#ff7f0e"}


def _axes_2d_from_camera_pose(
    image_rgb: np.ndarray,
    T_C_F: np.ndarray,
    camera: CameraModel,
    *,
    length_m: float,
    marker: str = "",
) -> None:
    """Draw the projected axes of a camera-frame pose onto an RGB image in place."""
    import cv2

    T = np.asarray(T_C_F, dtype=np.float64)
    origin = T[:3, 3]
    if not np.isfinite(origin).all():
        return
    tips = np.vstack([origin, *[origin + T[:3, i] * length_m for i in range(3)]])
    uv = camera.project(tips)
    if not np.isfinite(uv).all():
        return
    bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    o = (int(round(uv[0, 0])), int(round(uv[0, 1])))
    for i in range(3):
        cv2.arrowedLine(
            bgr,
            o,
            (int(round(uv[i + 1, 0])), int(round(uv[i + 1, 1]))),
            TCP_AXIS_RGB[i][::-1],
            2,
            cv2.LINE_AA,
            tipLength=0.28,
        )
    cv2.circle(bgr, o, 4, (255, 0, 255), -1, cv2.LINE_AA)
    if marker:
        cv2.putText(
            bgr,
            marker,
            (o[0] + 6, o[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    image_rgb[:, :, :] = bgr[:, :, ::-1]


def _tcp_transform(raw: RetargetResult, t: int) -> np.ndarray | None:
    R = raw.rotation_C_TCP[t]
    p = raw.position_C_m[t]
    if not (np.isfinite(R).all() and np.isfinite(p).all()):
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _draw_text_block(image_rgb: np.ndarray, lines: Sequence[str], *, start_y: int) -> None:
    """Draw status lines in place, below the header written by the overlay."""
    import cv2

    bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    y = int(start_y)
    for text in lines:
        cv2.putText(
            bgr,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    image_rgb[:, :, :] = bgr[:, :, ::-1]


def _side_labels(sequence: HandSequence, indices: Sequence[int]) -> list[str]:
    """Stable side name per hand index, from ``hand_ids`` or the handedness sign."""
    ids = sequence.hand_ids
    labels: list[str] = []
    for h in indices:
        if h < len(ids):
            labels.append(ids[h])
            continue
        column = np.asarray(sequence["handedness"])[:, h]
        signs = {int(v) for v in column if int(v) != 0}
        if len(signs) == 1:
            labels.append("right" if signs.pop() > 0 else "left")
        else:
            labels.append(f"hand{h}")
    return labels


def write_hand_previews(
    *,
    sequence: HandSequence,
    camera: CameraModel,
    video_path: Path,
    previews_dir: Path,
    hand_index: int | None = None,
    hand_ids: Sequence[str] | None = None,
    case_id: str = "",
    num_stills: int = 4,
    dry_run: bool = False,
) -> list[Path]:
    """Render the M1 hand previews; returns the paths that were written.

    ``hand_index`` keeps the legacy single-hand behaviour (and artifact names);
    ``None`` draws every hand the sequence carries, which is what the bimanual
    profile uses.
    """
    outputs: list[Path] = []
    if dry_run:
        return outputs

    n_hands = sequence.num_hands
    if n_hands == 0:
        raise ValueError("hand sequence carries no hand axis; cannot build previews")
    indices = [int(hand_index)] if hand_index is not None else list(range(n_hands))
    labels = list(hand_ids) if hand_ids else _side_labels(sequence, indices)
    multi = len(indices) > 1

    kp_uv = np.asarray(sequence["keypoints_uv_px"], dtype=np.float64)
    kp_C = np.asarray(sequence["keypoints_C_m"], dtype=np.float64)
    valid = np.asarray(sequence["valid"])
    conf = np.asarray(sequence["detection_confidence"])
    handedness = np.asarray(sequence["handedness"])
    timestamps = np.asarray(sequence["timestamp_s"], dtype=np.float64)
    frame_index = np.asarray(sequence["frame_index"], dtype=np.int64)
    T = kp_uv.shape[0]

    raw_by_hand: dict[int, RetargetResult] = {}
    codes_by_hand: dict[int, np.ndarray] = {}
    for h in indices:
        raw, _action = retarget_hand_sequence(sequence, hand_index=h)
        raw_by_hand[h] = raw
        codes_by_hand[h] = np.asarray(raw.fallback_code)

    frames: list[np.ndarray] = []
    for i, ts, image_rgb in iter_frames(video_path, start=0, stop=T):
        if i >= T:
            break
        canvas = np.array(image_rgb, dtype=np.uint8, copy=True)
        notes_all: list[str] = []
        for h in indices:
            side = labels[indices.index(h)] if indices.index(h) < len(labels) else f"hand{h}"
            fallback = FallbackCode.decode(int(codes_by_hand[h][i]))
            width = (
                float(raw_by_hand[h].gripper_width_m[i])
                if np.isfinite(raw_by_hand[h].gripper_width_m[i])
                else None
            )
            width_text = f"width {width * 1000:.1f} mm" if width is not None else "width n/a"
            state = "VALID" if bool(valid[i, h]) else "MISSING/INVALID"
            notes_all.append(f"{side}: conf {float(conf[i, h]):.2f} {state} {width_text}")
            if fallback != ["none"]:
                notes_all.append(f"{side} fallback: {','.join(fallback)}")
        for pos, h in enumerate(indices):  # H order: right is drawn under left, stably
            draw_hand_overlay(
                canvas,
                kp_uv[i, h],
                valid=bool(valid[i, h]),
                handedness=int(handedness[i, h]),
                confidence=float(conf[i, h]),
                frame_index=int(frame_index[i]),
                timestamp_s=float(ts),
                case_id=case_id,
                annotation_lines=(),
                in_place=True,
                colour_override=LEFT_HAND_BGR if multi and handedness[i, h] < 0 else None,
                show_header=pos == 0,
            )
        # One shared status block below the header: drawing it inside the per-hand
        # call would stack two headers on top of each other.
        _draw_text_block(canvas, notes_all, start_y=76)
        for h in indices:
            T_C_TCP = _tcp_transform(raw_by_hand[h], i)
            if T_C_TCP is None:
                continue
            width = float(raw_by_hand[h].gripper_width_m[i])
            _axes_2d_from_camera_pose(
                canvas,
                T_C_TCP,
                camera,
                length_m=max(0.01, width if np.isfinite(width) else 0.02),
                marker=(labels[indices.index(h)][0].upper() if multi else ""),
            )
        frames.append(canvas)

    stills = max(num_stills, 6) if multi else num_stills
    if frames:
        vp = previews_dir / "hand-overlay.mp4"
        write_overlay_video(vp, frames, fps=_fps(sequence))
        outputs.append(vp)
        picks = np.linspace(0, len(frames) - 1, min(stills, len(frames))).astype(int)
        for i in picks:
            p = previews_dir / f"hand-overlay-frame-{int(frame_index[i]):06d}.png"
            write_image(p, frames[i])
            outputs.append(p)

    # ---- 3-D view ---------------------------------------------------------
    any_valid = valid.reshape(T, -1).any(axis=1)
    mid = int(np.argmax(any_valid)) if bool(np.any(any_valid)) else 0

    def _draw(ax: Any) -> None:
        draw_camera_frustum(ax, np.eye(4), label="camera C")
        for h in indices:
            side = labels[indices.index(h)] if indices.index(h) < len(labels) else f"hand{h}"
            colour = _HAND_COLOURS_3D.get(side, "#7f7f7f")
            draw_trajectory_3d(ax, kp_C[:, h, MANO_WRIST], colour=colour, label=f"{side} wrist")
            draw_trajectory_3d(
                ax, raw_by_hand[h].position_C_m, colour=colour, label=f"{side} TCP (raw)"
            )
            pts = kp_C[mid, h][list(RETARGET_MARKERS)]
            finite_pts = np.isfinite(pts).all(axis=1)
            if finite_pts.any():
                ax.scatter(
                    pts[finite_pts, 0],
                    pts[finite_pts, 1],
                    pts[finite_pts, 2],
                    c=colour,
                    s=36,
                    label=f"{side} retarget keypoints",
                )
            finite = np.isfinite(kp_C[mid, h]).all(axis=1)
            if finite.any():
                ax.scatter(
                    kp_C[mid, h][finite, 0],
                    kp_C[mid, h][finite, 1],
                    kp_C[mid, h][finite, 2],
                    c=colour,
                    s=8,
                    alpha=0.5,
                    label=f"{side} MANO-21",
                )
            for j, label in RETARGET_MARKERS.items():
                p = kp_C[mid, h, j]
                if np.isfinite(p).all():
                    ax.text(p[0], p[1], p[2], f" {side[0]}:{label}", fontsize=7)
            T_C_TCP = _tcp_transform(raw_by_hand[h], mid)
            if T_C_TCP is not None:
                centre = T_C_TCP[:3, 3]
                ax.text(
                    centre[0],
                    centre[1],
                    centre[2],
                    f" TCP {side[0]}",
                    fontsize=8,
                )
                draw_frame_axes(ax, T_C_TCP, length=0.03, label=f"{side} TCP frame")

    widths = {
        h: float(raw_by_hand[h].gripper_width_m[mid])
        if np.isfinite(raw_by_hand[h].gripper_width_m[mid])
        else None
        for h in indices
    }
    width_text = ", ".join(
        f"{labels[indices.index(h)] if indices.index(h) < len(labels) else h} "
        f"{'n/a' if widths[h] is None else f'{widths[h] * 1000:.1f} mm'}"
        for h in indices
    )
    p3d = render_scene_figure(
        previews_dir / "hand-3d.png",
        title=(
            f"M1 hand geometry in frame C (frame {int(frame_index[mid])}, "
            f"gripper width {width_text})"
        ),
        draw=_draw,
        point_sets=[kp_C[:, h].reshape(-1, 3) for h in indices]
        + [raw_by_hand[h].position_C_m for h in indices],
    )
    outputs.append(p3d)

    # ---- TCP frame stills --------------------------------------------------
    for h in indices:
        side = labels[indices.index(h)] if indices.index(h) < len(labels) else f"hand{h}"
        suffix = "" if not multi else f"-{side}"
        for t in np.linspace(0, T - 1, min(stills, T)).astype(int):
            if not bool(valid[t, h]):
                continue
            _write_tcp_still(
                previews_dir / f"gripper-frame-{int(frame_index[t]):06d}{suffix}.png",
                kp_uv[t, h],
                camera,
                raw_by_hand[h],
                int(t),
                int(frame_index[t]),
                float(timestamps[t]),
                float(conf[t, h]),
                int(handedness[t, h]),
                FallbackCode.decode(int(codes_by_hand[h][t])),
                outputs,
                side=side if multi else "",
            )
    return outputs


def _write_tcp_still(
    path: Path,
    uv: np.ndarray,
    camera: CameraModel,
    raw: RetargetResult,
    t: int,
    frame_index: int,
    timestamp_s: float,
    confidence: float,
    handedness: int,
    fallback: list[str],
    outputs: list[Path],
    *,
    side: str = "",
) -> None:
    """Draw the 2-D skeleton, jaw line and TCP frame for a single frame."""
    import cv2

    from ...viz.hand_overlay import HAND_BONES, HAND_KEYPOINT_COLORS

    W, H = camera.width, camera.height
    bgr = np.full((H, W, 3), 245, dtype=np.uint8)
    kp = np.asarray(uv, dtype=np.float64)
    finite = np.isfinite(kp).all(axis=1)
    for a, b in HAND_BONES:
        if finite[a] and finite[b]:
            cv2.line(
                bgr,
                (int(kp[a, 0]), int(kp[a, 1])),
                (int(kp[b, 0]), int(kp[b, 1])),
                HAND_KEYPOINT_COLORS.get(b, (120, 120, 120)),
                2,
                cv2.LINE_AA,
            )
    for j in range(21):
        if finite[j]:
            cv2.circle(bgr, (int(kp[j, 0]), int(kp[j, 1])), 3, HAND_KEYPOINT_COLORS[j], -1, cv2.LINE_AA)

    if finite[MANO_THUMB_TIP] and finite[MANO_INDEX_TIP] and finite[MANO_MIDDLE_TIP]:
        vf = 0.7 * kp[MANO_INDEX_TIP] + 0.3 * kp[MANO_MIDDLE_TIP]
        tip = kp[MANO_THUMB_TIP]
        cv2.line(bgr, (int(tip[0]), int(tip[1])), (int(vf[0]), int(vf[1])), (0, 0, 255), 2, cv2.LINE_AA)
        mid = 0.5 * (tip + vf)
        cv2.circle(bgr, (int(mid[0]), int(mid[1])), 6, (255, 0, 255), 2, cv2.LINE_AA)

    p = raw.position_C_m[t]
    T_C_TCP = _tcp_transform(raw, t)
    if T_C_TCP is not None:
        width = float(raw.gripper_width_m[t])
        axis_len = max(0.01, width if np.isfinite(width) else 0.02)
        tips = np.vstack([p, *[p + T_C_TCP[:3, i] * axis_len for i in range(3)]])
        uv_axes = camera.project(tips)
        if np.isfinite(uv_axes).all():
            o = (int(uv_axes[0, 0]), int(uv_axes[0, 1]))
            for i in range(3):
                cv2.arrowedLine(
                    bgr,
                    o,
                    (int(uv_axes[i + 1, 0]), int(uv_axes[i + 1, 1])),
                    TCP_AXIS_RGB[i][::-1],
                    2,
                    cv2.LINE_AA,
                    tipLength=0.3,
                )

    width_mm = raw.gripper_width_m[t] * 1000.0
    hand_label = "right" if handedness > 0 else "left"
    if side:
        hand_label = f"{hand_label} ({side})"
    lines = [
        f"frame {frame_index}  t={timestamp_s:.3f}s  hand={hand_label}  conf={confidence:.2f}",
        f"TCP = [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m    gripper width = {width_mm:.1f} mm",
        f"fallback: {','.join(fallback)}",
        "x=approach(red)  y=gripper normal(green)  z=grasp axis(blue)  magenta dot = TCP",
    ]
    y = 26
    for text in lines:
        cv2.putText(bgr, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(bgr, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y += 23

    write_image(path, np.ascontiguousarray(bgr[:, :, ::-1]))
    outputs.append(path)


def _fps(sequence: HandSequence) -> float:
    ts = np.asarray(sequence["timestamp_s"], dtype=np.float64)
    if ts.size > 1:
        dt = float(np.median(np.diff(ts)))
        if dt > 0:
            return 1.0 / dt
    return 30.0
