"""Small, dependency-light visual evidence writers for M5 stages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "make_area_curve",
    "make_inpainting_triptych",
    "make_segmentation_triptych",
    "representative_indices",
    "status_image",
    "write_png",
    "write_video",
]


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional image dependency
        raise RuntimeError("M5 previews require OpenCV (opencv-python-headless)") from exc
    return cv2


def representative_indices(num_frames: int, *, maximum: int = 8) -> list[int]:
    """Choose deterministic, unique frame indices spanning the full sequence."""
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise TypeError("num_frames must be an integer")
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise TypeError("maximum must be an integer")
    if num_frames < 0 or maximum <= 0:
        raise ValueError("num_frames must be non-negative and maximum must be positive")
    if num_frames == 0:
        return []
    count = min(num_frames, maximum)
    return [int(v) for v in np.linspace(0, num_frames - 1, count).round().astype(np.int64)]


def write_png(path: str | Path, image: NDArray[np.generic]) -> Path:
    """Atomically write a PNG, encoding boolean masks as 0/255 grayscale."""
    import imageio.v2 as imageio

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(image)
    if data.dtype == np.bool_:
        data = data.astype(np.uint8) * 255
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    try:
        imageio.imwrite(temporary, data)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def write_video(path: str | Path, frames: Sequence[NDArray[np.uint8]], *, fps: float) -> Path:
    """Write an ordered constant-FPS preview video without changing source PTS data."""
    import imageio.v2 as imageio

    if not frames:
        raise ValueError("preview video requires at least one frame")
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("preview fps must be finite and positive")
    first = np.asarray(frames[0])
    if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("preview frames must be uint8 RGB arrays with shape [H, W, 3]")
    shape = first.shape
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    try:
        with imageio.get_writer(
            temporary,
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        ) as writer:
            for index, frame in enumerate(frames):
                image = np.asarray(frame)
                if image.dtype != np.uint8 or image.shape != shape:
                    raise ValueError(f"preview frame {index} has a different dtype or shape")
                writer.append_data(image)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _require_rgb_mask(
    source_rgb: NDArray[np.generic], mask: NDArray[np.generic], name: str
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    source = np.asarray(source_rgb)
    binary = np.asarray(mask)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source_rgb must be uint8 RGB with shape [H, W, 3]")
    if binary.dtype != np.bool_ or binary.ndim != 2 or binary.shape != source.shape[:2]:
        raise ValueError(f"{name} must be bool [H, W] and match the source image")
    return source, binary


def _resize_panel(image: NDArray[np.uint8], width: int) -> NDArray[np.uint8]:
    cv2 = _cv2()
    height, original_width = image.shape[:2]
    if original_width <= width:
        return image.copy()
    new_height = max(1, int(round(height * width / original_width)))
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)


def _caption(image: NDArray[np.uint8], label: str) -> NDArray[np.uint8]:
    cv2 = _cv2()
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 32), (20, 20, 20), thickness=-1)
    cv2.putText(output, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _overlay(
    source_rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    *,
    color_rgb: tuple[int, int, int],
    alpha: float = 0.42,
) -> NDArray[np.uint8]:
    overlay = source_rgb.copy()
    if np.any(mask):
        source = source_rgb[mask].astype(np.float32)
        color = np.asarray(color_rgb, dtype=np.float32)
        overlay[mask] = np.clip((1.0 - alpha) * source + alpha * color, 0, 255).astype(np.uint8)
    return overlay


def _join_panels(panels: Sequence[tuple[NDArray[np.uint8], str]], *, panel_width: int) -> NDArray[np.uint8]:
    resized = [_caption(_resize_panel(image, panel_width), label) for image, label in panels]
    heights = {panel.shape[0] for panel in resized}
    if len(heights) != 1:
        raise ValueError("preview panels must have matching aspect ratios")
    return np.concatenate(resized, axis=1)


def make_segmentation_triptych(
    source_rgb: NDArray[np.generic],
    raw_mask: NDArray[np.generic],
    refined_mask: NDArray[np.generic],
    *,
    frame_index: int,
    timestamp_s: float,
    panel_width: int = 640,
) -> NDArray[np.uint8]:
    """Build a source/raw/refined overlay for one frame."""
    source, raw = _require_rgb_mask(source_rgb, raw_mask, "raw_mask")
    _, refined = _require_rgb_mask(source, refined_mask, "refined_mask")
    label = f"frame {frame_index:06d}  t={timestamp_s:.3f}s"
    return _join_panels(
        [
            (source, f"SOURCE | {label}"),
            (_overlay(source, raw, color_rgb=(255, 70, 55)), f"RAW mask | {label}"),
            (_overlay(source, refined, color_rgb=(40, 230, 235)), f"REFINED mask | {label}"),
        ],
        panel_width=panel_width,
    )


def make_inpainting_triptych(
    source_rgb: NDArray[np.generic],
    refined_mask: NDArray[np.generic],
    background_rgb: NDArray[np.generic],
    *,
    frame_index: int,
    timestamp_s: float,
    inference_resolution: tuple[int, int] | None = None,
    panel_width: int = 640,
) -> NDArray[np.uint8]:
    """Build a source/mask/background comparison for one frame."""
    source, mask = _require_rgb_mask(source_rgb, refined_mask, "refined_mask")
    background = np.asarray(background_rgb)
    if background.dtype != np.uint8 or background.shape != source.shape:
        raise ValueError("background_rgb must be uint8 RGB and match source dimensions")
    label = f"frame {frame_index:06d}  t={timestamp_s:.3f}s"
    background_caption = "BACKGROUND"
    if inference_resolution is not None and inference_resolution != (source.shape[1], source.shape[0]):
        background_caption = f"BACKGROUND | PREVIEW {inference_resolution[0]}x{inference_resolution[1]}"
    return _join_panels(
        [
            (source, f"SOURCE | {label}"),
            (_overlay(source, mask, color_rgb=(40, 230, 235)), f"REFINED mask | {label}"),
            (background, f"{background_caption} | {label}"),
        ],
        panel_width=panel_width,
    )


def make_area_curve(
    raw_area_pixels: Sequence[int],
    refined_area_pixels: Sequence[int],
    *,
    width: int = 1000,
    height: int = 420,
) -> NDArray[np.uint8]:
    """Render raw and refined mask pixel area by frame as a PNG-ready RGB array."""
    if len(raw_area_pixels) != len(refined_area_pixels):
        raise ValueError("raw and refined area curves must have equal length")
    if not raw_area_pixels:
        raise ValueError("area curves must contain at least one frame")
    if width < 320 or height < 180:
        raise ValueError("area curve canvas is too small for labels")
    raw = np.asarray(raw_area_pixels, dtype=np.float64)
    refined = np.asarray(refined_area_pixels, dtype=np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(refined).all() or np.any(raw < 0) or np.any(refined < 0):
        raise ValueError("area curves must contain finite, non-negative values")

    cv2 = _cv2()
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 70, 28, width - 24, height - 56
    cv2.rectangle(canvas, (left, top), (right, bottom), (225, 225, 225), 1)
    for fraction in (0.25, 0.5, 0.75):
        y = int(round(bottom - fraction * (bottom - top)))
        cv2.line(canvas, (left, y), (right, y), (235, 235, 235), 1, cv2.LINE_AA)

    maximum = max(float(np.max(raw)), float(np.max(refined)), 1.0)
    count = len(raw)

    def points(values: NDArray[np.float64]) -> NDArray[np.int32]:
        if count == 1:
            xs = np.asarray([(left + right) // 2], dtype=np.int32)
        else:
            xs = np.linspace(left, right, count).round().astype(np.int32)
        ys = (bottom - values / maximum * (bottom - top)).round().astype(np.int32)
        return np.stack([xs, ys], axis=1)

    if count > 1:
        cv2.polylines(canvas, [points(raw)], False, (50, 70, 235), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [points(refined)], False, (30, 150, 45), 2, cv2.LINE_AA)
    else:
        cv2.circle(canvas, tuple(int(v) for v in points(raw)[0]), 3, (50, 70, 235), -1)
        cv2.circle(canvas, tuple(int(v) for v in points(refined)[0]), 3, (30, 150, 45), -1)
    cv2.putText(canvas, "Mask area by frame (pixels)", (left, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"0       {count - 1}", (left, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"0 - {int(maximum)}", (4, top + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.line(canvas, (width - 330, 19), (width - 302, 19), (50, 70, 235), 2, cv2.LINE_AA)
    cv2.putText(canvas, "raw", (width - 296, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.line(canvas, (width - 215, 19), (width - 187, 19), (30, 150, 45), 2, cv2.LINE_AA)
    cv2.putText(canvas, "refined", (width - 181, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def status_image(source_rgb: NDArray[np.generic], *, stage: str, reason: str) -> NDArray[np.uint8]:
    """Create a clearly labelled failure diagnostic over an actual source frame."""
    source = np.asarray(source_rgb)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source_rgb must be uint8 RGB with shape [H, W, 3]")
    cv2 = _cv2()
    output = _resize_panel(source, 1280)
    banner_height = 92
    cv2.rectangle(output, (0, 0), (output.shape[1], banner_height), (22, 28, 38), thickness=-1)
    cv2.putText(
        output,
        f"M5 {stage.upper()} - INFERENCE INCOMPLETE",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    message = reason.replace("\n", " ").strip()
    cv2.putText(
        output,
        message[:145],
        (24, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (210, 220, 230),
        1,
        cv2.LINE_AA,
    )
    return output
