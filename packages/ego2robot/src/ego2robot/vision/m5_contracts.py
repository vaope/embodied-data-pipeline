"""Typed, versioned contracts for the M5 human-removal model boundary.

These records contain paths and metadata only.  Model libraries (including
PyTorch) belong in isolated adapters and are deliberately not imported here.
"""

from __future__ import annotations

import math
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

__all__ = [
    "BackgroundFrame",
    "BackgroundSequenceIndex",
    "InpaintingConfig",
    "MaskFrame",
    "MaskSequenceIndex",
    "ModelRequest",
    "ModelResult",
    "RuntimeConfig",
    "SegmentationConfig",
    "SourceFrame",
    "SourceFrameIndex",
]


def _episode_relative_path(value: str) -> str:
    """Require a portable path that cannot escape its episode directory."""
    normalized = value.replace("\\", "/")
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(normalized)
    if (
        not value.strip()
        or "\x00" in value
        or normalized.startswith("/")
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or normalized in {"", "."}
        or ".." in posix_path.parts
    ):
        raise ValueError("artifact paths must be non-empty, relative to the episode, and stay within it")
    return normalized


EpisodeRelativePath = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
    AfterValidator(_episode_relative_path),
]
Sha256Hex = Annotated[
    str,
    StringConstraints(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]


class _StrictModel(BaseModel):
    """Shared settings for public M5 contracts."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class RuntimeConfig(_StrictModel):
    """External model runtime location and reproducibility settings."""

    python_executable: str | None = None
    repo_path: str | None = None
    checkpoint_path: str | None = None
    device: str = "cuda"
    seed: int = 0
    revision: str | None = None
    execution: Literal["native", "wsl"] = "native"
    wsl_distribution: str | None = None

    @model_validator(mode="after")
    def _check_runtime(self) -> RuntimeConfig:
        if not self.device.strip():
            raise ValueError("runtime.device must be non-empty")
        if self.execution == "wsl" and not (self.wsl_distribution or "").strip():
            raise ValueError("runtime.wsl_distribution is required when execution='wsl'")
        if self.wsl_distribution is not None and not self.wsl_distribution.strip():
            raise ValueError("runtime.wsl_distribution must be non-empty when set")
        return self


class SegmentationConfig(_StrictModel):
    """SAM 3 settings and deterministic post-processing parameters."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: Literal["sam3"] = "sam3"
    text_prompt: str = "person"
    anchor_frame: Literal["middle"] = "middle"
    chunk_frames: int = 400
    chunk_overlap: int = 50
    gap_fill_max_frames: int = 3
    area_median_window: int = 11
    area_median_fraction: float = 0.5
    close_kernel: int = 5
    approval_record: EpisodeRelativePath = "quality/reviews/m3-video.json"
    rig_approval_record: EpisodeRelativePath = "quality/reviews/rig-geometry.json"
    adr: str = "docs/decisions/0011-m5-runtime-and-mask-contract.md"

    @model_validator(mode="after")
    def _check_segmentation(self) -> SegmentationConfig:
        if self.chunk_frames <= 0:
            raise ValueError("segmentation.chunk_frames must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_frames:
            raise ValueError("segmentation.chunk_overlap must be in [0, chunk_frames)")
        if self.gap_fill_max_frames < 0:
            raise ValueError("segmentation.gap_fill_max_frames must be non-negative")
        if self.area_median_window <= 0 or self.area_median_window % 2 == 0:
            raise ValueError("segmentation.area_median_window must be a positive odd integer")
        if not 0.0 < self.area_median_fraction <= 1.0:
            raise ValueError("segmentation.area_median_fraction must be in (0, 1]")
        if self.close_kernel <= 0 or self.close_kernel % 2 == 0:
            raise ValueError("segmentation.close_kernel must be a positive odd integer")
        if not self.text_prompt.strip():
            raise ValueError("segmentation.text_prompt must be non-empty")
        if not self.approval_record.strip() or not self.adr.strip():
            raise ValueError("segmentation approval_record and adr must be non-empty")
        if not self.rig_approval_record.strip():
            raise ValueError("segmentation rig_approval_record must be non-empty")
        return self


class InpaintingConfig(_StrictModel):
    """ProPainter settings, with optional explicit inference resize."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: Literal["propainter"] = "propainter"
    fp16: bool = True
    neighbor_length: int = 10
    ref_stride: int = 10
    subvideo_length: int = 80
    mask_dilation: int = 4
    raft_iterations: int = 20
    inference_width: int | None = None
    inference_height: int | None = None
    adr: str = "docs/decisions/0011-m5-runtime-and-mask-contract.md"

    @model_validator(mode="after")
    def _check_inpainting(self) -> InpaintingConfig:
        if self.neighbor_length <= 0:
            raise ValueError("inpainting.neighbor_length must be positive")
        if self.ref_stride <= 0:
            raise ValueError("inpainting.ref_stride must be positive")
        if self.subvideo_length <= 0:
            raise ValueError("inpainting.subvideo_length must be positive")
        if self.mask_dilation < 0:
            raise ValueError("inpainting.mask_dilation must be non-negative")
        if self.raft_iterations <= 0:
            raise ValueError("inpainting.raft_iterations must be positive")
        dimensions = (self.inference_width, self.inference_height)
        if (dimensions[0] is None) != (dimensions[1] is None):
            raise ValueError("inpainting inference_width and inference_height must be set together")
        if dimensions[0] is not None and dimensions[1] is not None:
            if dimensions[0] <= 0 or dimensions[1] <= 0:
                raise ValueError("inpainting inference dimensions must be positive")
            if dimensions[0] % 8 != 0 or dimensions[1] % 8 != 0:
                raise ValueError("inpainting inference dimensions must be multiples of 8")
        if not self.adr.strip():
            raise ValueError("inpainting.adr must be non-empty")
        return self


class _FrameMetadata(_StrictModel):
    frame_index: int = Field(ge=0)
    timestamp_s: float

    @model_validator(mode="after")
    def _check_timestamp(self) -> _FrameMetadata:
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        return self


class SourceFrame(_FrameMetadata):
    """One decoded source RGB frame stored relative to its episode root."""

    path: EpisodeRelativePath
    sha256: Sha256Hex


class MaskFrame(_FrameMetadata):
    """One frame's raw and refined mask files."""

    valid: bool
    raw_path: EpisodeRelativePath
    raw_sha256: Sha256Hex
    refined_path: EpisodeRelativePath
    refined_sha256: Sha256Hex


class BackgroundFrame(_FrameMetadata):
    """One inpainted background frame stored relative to its episode root."""

    valid: bool
    path: EpisodeRelativePath
    sha256: Sha256Hex


class _FrameIndexBase(_StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_video_sha256: Sha256Hex
    provenance: dict[str, Any] = Field(default_factory=dict)


def _validate_frame_order(frames: list[_FrameMetadata]) -> list[_FrameMetadata]:
    if not frames:
        raise ValueError("frame index must contain at least one frame")
    previous_frame_index = -1
    previous_timestamp_s = -math.inf
    for frame in frames:
        if frame.frame_index <= previous_frame_index:
            raise ValueError("frame_index values must be strictly ascending")
        if frame.timestamp_s <= previous_timestamp_s:
            raise ValueError("timestamp_s values must be strictly ascending")
        previous_frame_index = frame.frame_index
        previous_timestamp_s = frame.timestamp_s
    return frames


class SourceFrameIndex(_FrameIndexBase):
    """PTS-aligned manifest of decoded source frames."""

    schema_version: Literal["ego2robot.m5_source_frames.v1"] = "ego2robot.m5_source_frames.v1"
    frames: list[SourceFrame] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_frame_order(self) -> SourceFrameIndex:
        _validate_frame_order(self.frames)
        return self


class MaskSequenceIndex(_FrameIndexBase):
    """PTS-aligned manifest of raw and refined segmentation masks."""

    schema_version: Literal["ego2robot.m5_masks.v1"] = "ego2robot.m5_masks.v1"
    frames: list[MaskFrame] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_frame_order(self) -> MaskSequenceIndex:
        _validate_frame_order(self.frames)
        return self


class BackgroundSequenceIndex(_FrameIndexBase):
    """PTS-aligned manifest of inpainted RGB background frames."""

    schema_version: Literal["ego2robot.m5_background.v1"] = "ego2robot.m5_background.v1"
    frames: list[BackgroundFrame] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_frame_order(self) -> BackgroundSequenceIndex:
        _validate_frame_order(self.frames)
        return self


class ModelRequest(_StrictModel):
    """JSON request sent to an isolated model process."""

    schema_version: Literal["ego2robot.m5_model_request.v1"] = "ego2robot.m5_model_request.v1"
    task: Literal["segmentation", "inpainting"]
    episode_root: str = Field(min_length=1)
    source_index_path: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    mask_index_path: str | None = None
    runtime: RuntimeConfig
    parameters: dict[str, Any]


class ModelResult(_StrictModel):
    """Raw JSON response; the calling stage validates count and file contents."""

    schema_version: Literal["ego2robot.m5_model_result.v1"] = "ego2robot.m5_model_result.v1"
    task: Literal["segmentation", "inpainting"]
    frame_paths: list[str]
    provenance: dict[str, Any]
