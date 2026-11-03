"""Video probing and frame decoding bound to presentation timestamps.

System design 5.3: *the container PTS is the master clock*.  ``frame_index /
nominal_fps`` is explicitly forbidden as an implicit timebase, so every decode
helper here returns timestamps derived from ``pts * time_base``.

PyAV (FFmpeg) is the primary backend; OpenCV is only a fallback for containers
PyAV cannot open, and when it is used the fact is recorded in the probe result
so a reviewer can tell the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

__all__ = [
    "VideoProbe",
    "VideoStreamInfo",
    "frame_at",
    "iter_frames",
    "probe_video",
    "read_frame_by_index",
]


@dataclass
class VideoStreamInfo:
    """Container/stream level facts about a video file."""

    codec: str = ""
    width: int = 0
    height: int = 0
    nominal_fps: float = 0.0
    time_base: str = ""
    num_frames_declared: int | None = None
    duration_s: float | None = None
    pixel_format: str = ""
    rotation_deg: float = 0.0


@dataclass
class VideoProbe:
    """Result of :func:`probe_video`."""

    path: Path
    backend: str
    stream: VideoStreamInfo
    num_frames: int
    timestamps_s: list[float]
    first_frame_sha256: str = ""
    decode_errors: list[str] = field(default_factory=list)
    #: per-frame PTS in stream timebase units, kept for traceability
    pts: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def nominal_fps(self) -> float:
        return self.stream.nominal_fps

    @property
    def width(self) -> int:
        return self.stream.width

    @property
    def height(self) -> int:
        return self.stream.height

    @property
    def monotonic(self) -> bool:
        if len(self.timestamps_s) < 2:
            return True
        d = np.diff(np.asarray(self.timestamps_s, dtype=np.float64))
        return bool(np.all(d > 0.0))

    @property
    def median_dt_s(self) -> float:
        if len(self.timestamps_s) < 2:
            return 1.0 / self.nominal_fps if self.nominal_fps else 0.0
        return float(np.median(np.diff(np.asarray(self.timestamps_s, dtype=np.float64))))

    @property
    def measured_fps(self) -> float:
        dt = self.median_dt_s
        return 0.0 if dt <= 0.0 else 1.0 / dt

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "backend": self.backend,
            "codec": self.stream.codec,
            "width": self.width,
            "height": self.height,
            "nominal_fps": round(self.nominal_fps, 6),
            "measured_fps": round(self.measured_fps, 6),
            "num_frames": self.num_frames,
            "duration_s": round(self.timestamps_s[-1] - self.timestamps_s[0], 6)
            if len(self.timestamps_s) > 1
            else 0.0,
            "monotonic_pts": self.monotonic,
            "time_base": self.stream.time_base,
            "pixel_format": self.stream.pixel_format,
            "warnings": list(self.warnings),
            "decode_errors": list(self.decode_errors),
        }


# ---------------------------------------------------------------------------
# PyAV backend
# ---------------------------------------------------------------------------


def _probe_with_av(path: Path, *, decode_check_frames: int = 3) -> VideoProbe:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream found")
        stream = container.streams.video[0]
        codec_name = stream.codec_context.name
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        rate = stream.average_rate
        nominal_fps = float(rate) if rate else float(stream.base_rate or 0.0)
        time_base = str(stream.time_base)
        tb = float(stream.time_base) if stream.time_base else 0.0
        declared = int(stream.frames) if stream.frames else None
        duration_s: float | None = None
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        pix_fmt = stream.codec_context.pix_fmt or ""
        rotation = 0.0
        try:
            side = stream.metadata.get("rotate")
            if side:
                rotation = float(side)
        except Exception:  # pragma: no cover - metadata is best-effort
            rotation = 0.0

        timestamps: list[float] = []
        pts_list: list[int] = []
        errors: list[str] = []
        warnings: list[str] = []
        first_hash = ""
        count = 0
        for frame in container.decode(stream):
            ts = None
            if frame.pts is not None and tb:
                ts = float(frame.pts) * tb
            elif frame.time is not None:
                ts = float(frame.time)
            if ts is None:
                # No PTS at all. Record it, but never silently fabricate one.
                errors.append(f"frame {count} has no PTS")
                ts = float("nan")
            timestamps.append(ts)
            pts_list.append(int(frame.pts) if frame.pts is not None else -1)
            if count < decode_check_frames:
                arr = frame.to_ndarray(format="rgb24")
                if count == 0:
                    import hashlib

                    first_hash = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
            count += 1

    if count == 0:
        raise ValueError(f"{path}: decoder produced zero frames")

    if any(np.isnan(t) for t in timestamps):
        warnings.append("some frames carried no PTS; timestamps are not usable as a master clock")

    probe = VideoProbe(
        path=path,
        backend="pyav",
        stream=VideoStreamInfo(
            codec=codec_name,
            width=width,
            height=height,
            nominal_fps=nominal_fps,
            time_base=time_base,
            num_frames_declared=declared,
            duration_s=duration_s,
            pixel_format=pix_fmt,
            rotation_deg=rotation,
        ),
        num_frames=count,
        timestamps_s=timestamps,
        first_frame_sha256=first_hash,
        decode_errors=errors,
        pts=pts_list,
        warnings=warnings,
    )
    if declared is not None and declared != count:
        probe.warnings.append(
            f"container declares {declared} frames but the decoder produced {count}"
        )
    return probe


# ---------------------------------------------------------------------------
# OpenCV fallback
# ---------------------------------------------------------------------------


def _probe_with_opencv(path: Path, *, decode_check_frames: int = 3) -> VideoProbe:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"{path}: OpenCV could not open the file")
    warnings = [
        "PyAV unavailable or failed; fell back to OpenCV, whose timestamps are "
        "frame_index / nominal_fps rather than container PTS"
    ]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
    timestamps: list[float] = []
    first_hash = ""
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fps > 0:
            timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
        else:
            timestamps.append(float(count))
        if count == 0:
            import hashlib

            first_hash = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
        count += 1
    cap.release()
    if count == 0:
        raise ValueError(f"{path}: OpenCV decoded zero frames")
    return VideoProbe(
        path=path,
        backend="opencv",
        stream=VideoStreamInfo(
            codec=codec or "unknown",
            width=width,
            height=height,
            nominal_fps=fps,
            time_base="n/a",
            num_frames_declared=count,
            duration_s=timestamps[-1] if timestamps else None,
            pixel_format="bgr24 (opencv native)",
        ),
        num_frames=count,
        timestamps_s=timestamps,
        first_frame_sha256=first_hash,
        warnings=warnings,
    )


def probe_video(path: str | Path, *, decode_check_frames: int = 3, backend: str = "auto") -> VideoProbe:
    """Probe a video: codec, resolution, PTS, fps, frame count, decode checks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"video not found: {p}")
    if backend in {"auto", "pyav"}:
        try:
            return _probe_with_av(p, decode_check_frames=decode_check_frames)
        except ImportError:
            if backend == "pyav":
                raise
        except Exception:
            if backend == "pyav":
                raise
    return _probe_with_opencv(p, decode_check_frames=decode_check_frames)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def iter_frames(
    path: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
    rgb: bool = True,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, timestamp_s, image)`` for a frame range.

    Images are RGB by default; the OpenCV fallback converts from BGR here so no
    BGR array ever escapes this module (system-design.md 5.2).
    """
    p = Path(path)
    try:
        import av

        with av.open(str(p)) as container:
            stream = container.streams.video[0]
            tb = float(stream.time_base) if stream.time_base else 0.0
            for i, frame in enumerate(container.decode(stream)):
                if i < start:
                    continue
                if stop is not None and i >= stop:
                    break
                ts = float(frame.pts) * tb if (frame.pts is not None and tb) else float(frame.time or 0.0)
                yield i, ts, frame.to_ndarray(format="rgb24")
        return
    except ImportError:
        pass

    import cv2

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"{p}: could not open for decoding")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i < start:
                i += 1
                continue
            if stop is not None and i >= stop:
                break
            ts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0 if fps > 0 else float(i)
            yield i, ts, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def read_frame_by_index(path: str | Path, frame_index: int, *, rgb: bool = True) -> np.ndarray:
    """Decode exactly one frame by index."""
    for i, _ts, image in iter_frames(path, start=frame_index, stop=frame_index + 1, rgb=rgb):
        if i == frame_index:
            return image
    raise IndexError(f"{path}: frame {frame_index} could not be decoded")


def frame_at(timestamps_s: list[float], query_s: float) -> tuple[int, float]:
    """Nearest frame index for a timestamp plus the residual in seconds."""
    arr = np.asarray(timestamps_s, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no timestamps to search")
    idx = int(np.argmin(np.abs(arr - query_s)))
    return idx, float(arr[idx] - query_s)
