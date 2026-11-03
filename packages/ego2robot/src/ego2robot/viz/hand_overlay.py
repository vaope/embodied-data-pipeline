"""Hand overlay: 2D keypoints, handedness, confidence and missing state on RGB.

implementation-plan.md M1 requires ``previews/hand-overlay.mp4`` plus still
images, and Gate M1 requires the overlay to line up with the hand pixels with no
x/y mirror.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "HAND_BONES",
    "HAND_KEYPOINT_COLORS",
    "HandOverlayStyle",
    "draw_hand_overlay",
    "write_image",
    "write_overlay_video",
]

#: MANO-21 kinematic chains used for the skeleton drawing.
HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # little
)

#: BGR colours for (thumb, index, middle, ring, little) chains.
HAND_KEYPOINT_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    **{i: (60, 76, 231) for i in range(1, 5)},      # thumb - red
    **{i: (113, 204, 46) for i in range(5, 9)},     # index - green
    **{i: (219, 152, 52) for i in range(9, 13)},    # middle - blue
    **{i: (182, 89, 155) for i in range(13, 17)},   # ring - purple
    **{i: (15, 196, 241) for i in range(17, 21)},   # little - amber
}

#: the four joints the retargeter consumes, drawn larger
RETARGET_HIGHLIGHT = {0: "wrist", 4: "thumb_tip", 8: "index_tip", 12: "middle_tip"}


@dataclass
class HandOverlayStyle:
    radius: int = 4
    bone_thickness: int = 2
    highlight_radius: int = 7
    draw_labels: bool = True
    font_scale: float = 0.5
    text_thickness: int = 1


def _to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a canonical RGB image to OpenCV BGR *inside this adapter*."""
    return np.ascontiguousarray(image[:, :, ::-1])


def draw_hand_overlay(
    image_rgb: NDArray[np.uint8],
    keypoints_uv_px: NDArray[np.floating[Any]],
    *,
    valid: bool = True,
    handedness: int = 1,
    confidence: float = 1.0,
    frame_index: int = 0,
    timestamp_s: float = 0.0,
    case_id: str = "",
    style: HandOverlayStyle | None = None,
    annotation_lines: Sequence[str] = (),
    in_place: bool = False,
    colour_override: tuple[int, int, int] | None = None,
    show_header: bool = True,
) -> NDArray[np.uint8]:
    """Draw a MANO-21 skeleton plus status text.

    Returns an **RGB** image; the BGR conversion is confined to this function.
    Invalid frames are drawn in grey with an explicit "MISSING" banner rather
    than being skipped, so a dropout is visible in the artifact.

    ``colour_override`` (BGR) replaces the per-finger palette with one colour;
    the bimanual preview uses it to keep the second hand visually separate.
    ``show_header=False`` suppresses the status block so two hands can share one
    annotation area instead of stacking two headers.
    """
    import cv2

    style = style or HandOverlayStyle()
    rgb = image_rgb if in_place else np.array(image_rgb, dtype=np.uint8, copy=True)
    # ``ascontiguousarray`` on the reversed view by itself produces a copy, so the
    # drawing must be copied back into ``rgb`` when the caller asked for in-place
    # work -- otherwise every stroke is silently thrown away (found by comparing
    # the two-hand overlay against the video).
    canvas = _to_bgr(rgb)
    uv = np.asarray(keypoints_uv_px, dtype=np.float64).reshape(21, 2)

    finite = np.isfinite(uv).all(axis=1)

    def _colour(joint: int) -> tuple[int, int, int]:
        if colour_override is not None:
            return colour_override if valid else (128, 128, 128)
        if not valid:
            return (128, 128, 128)
        return HAND_KEYPOINT_COLORS.get(joint, (255, 255, 255))

    for a, b in HAND_BONES:
        if not (finite[a] and finite[b]):
            continue
        cv2.line(
            canvas,
            (int(round(uv[a, 0])), int(round(uv[a, 1]))),
            (int(round(uv[b, 0])), int(round(uv[b, 1]))),
            _colour(b),
            style.bone_thickness,
            cv2.LINE_AA,
        )
    for j in range(21):
        if not finite[j]:
            continue
        centre = (int(round(uv[j, 0])), int(round(uv[j, 1])))
        if j in RETARGET_HIGHLIGHT:
            cv2.circle(canvas, centre, style.highlight_radius, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, centre, style.highlight_radius - 2, _colour(j), -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, centre, style.radius, _colour(j), -1, cv2.LINE_AA)

    prefix = f"{case_id}  " if case_id else ""
    head = f"{prefix}frame {frame_index}  t={timestamp_s:.3f}s"
    hand_label = {1: "right", -1: "left", 0: "unknown"}.get(int(handedness), "?")
    state = "VALID" if valid else "MISSING / INVALID"
    lines = [head, f"hand={hand_label} conf={confidence:.2f} {state}", *annotation_lines]
    if not show_header:
        lines = list(annotation_lines)
    if style.draw_labels:
        y = 28
        for i, text in enumerate(lines):
            txt_colour = (255, 255, 255) if i < 2 else (0, 255, 255)
            if not show_header and i == 0:
                txt_colour = (255, 255, 255)
            if show_header and i == 1 and not valid:
                txt_colour = (0, 0, 255)
            # a single stroke pass: the two-pass outline+fill trick rendered as
            # visibly doubled glyphs at this scale, which looks like a bug in the
            # review artifacts
            cv2.putText(
                canvas,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale,
                txt_colour,
                style.text_thickness + 1,
                cv2.LINE_AA,
            )
            y += 24

    if in_place:
        rgb[:, :, :] = canvas[:, :, ::-1]
        return rgb
    return np.ascontiguousarray(canvas[:, :, ::-1])


def write_image(path: str | Path, image_rgb: NDArray[np.uint8]) -> Path:
    """Write an RGB image as PNG (tiny wrapper so callers never handle BGR)."""
    import imageio.v2 as imageio

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # the temp file must keep the real extension: imageio selects its codec by it
    tmp = p.with_name(f".{p.stem}.tmp{p.suffix}")
    imageio.imwrite(tmp, np.asarray(image_rgb, dtype=np.uint8))
    tmp.replace(p)
    return p


def write_overlay_video(
    path: str | Path,
    frames: Iterable[NDArray[np.uint8]],
    *,
    fps: float,
    codec: str = "libx264",
    crf: int = 20,
) -> Path:
    """Encode RGB frames to an MP4 via PyAV (no OpenCV video writer).

    Timestamps are derived from the frame index and the requested rate, so the
    written file carries a real PTS track rather than a frame-count placeholder.
    """
    import fractions

    import av

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp" + p.suffix)
    rate = fractions.Fraction(fps).limit_denominator(100000) if fps > 0 else fractions.Fraction(30, 1)
    container = av.open(str(tmp), mode="w")
    stream = container.add_stream(codec, rate=rate)
    count = 0
    for frame in frames:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected an RGB image [H, W, 3], got {arr.shape}")
        if count == 0:
            stream.width = int(arr.shape[1])
            stream.height = int(arr.shape[0])
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "preset": "medium"}
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
        vf.pts = count
        # frame time base = 1 / stream rate, so pts is a frame index
        vf.time_base = fractions.Fraction(rate.denominator, rate.numerator)
        for packet in stream.encode(vf):
            container.mux(packet)
        count += 1
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    if count == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"no frames were supplied for {p}")
    tmp.replace(p)
    return p
