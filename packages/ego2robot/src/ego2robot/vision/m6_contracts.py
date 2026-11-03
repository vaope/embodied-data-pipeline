"""Typed, versioned contracts for the M6 scene-depth and compositing boundary.

ADR-0008 fixes the rule this module encodes: a scene depth may only reach the
compositor when its ``mode`` is ``metric`` or ``aligned``.  ``relative`` depth is
an affine-invariant quantity and comparing it with MuJoCo's metric robot depth is
a category error, so the compositor raises rather than tolerating it.

The model runtime is the same ``RuntimeConfig`` the M5 stages use, because the
Depth Anything inference runs in the same isolated WSL environment and must be
described the same way (interpreter, repo, checkpoint, device, seed, revision).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .m5_contracts import RuntimeConfig

__all__ = [
    "CompositeConfig",
    "DepthValidationConfig",
    "SceneDepthConfig",
]


class _StrictModel(BaseModel):
    """Shared settings for public M6 contracts."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class DepthValidationConfig(_StrictModel):
    """Thresholds for checking that a metric depth is actually on the metric scale.

    The check is an *anchor* comparison, not a rescale: the anchors are points
    whose metric depth is already known from another part of the pipeline.  A depth
    field that fails it is reported, never silently corrected.
    """

    enabled: bool = True
    #: median |z_scene(u,v) - z_anchor| over all anchors, metres
    max_median_abs_error_m: float = 0.10
    #: 90th percentile of the same residual, metres
    max_p90_abs_error_m: float = 0.20
    #: anchors whose denominator is nearer than this are ignored as unreliable
    min_anchor_depth_m: float = 0.05
    #: fraction of anchor samples that must be finite and in front of the camera
    min_anchor_coverage: float = 0.80

    #: Number of *original* frames to re-estimate depth on so the hand keypoints can
    #: serve as metric anchors.  The TCP anchor is weak evidence — the TCP sits in the
    #: finger gap, so its ray often misses the object it supposedly touches — while a
    #: hand keypoint is a point on the hand surface whose camera-frame z is known.
    #: 0 disables the reference check.
    reference_frames: int = 6

    @model_validator(mode="after")
    def _check_thresholds(self) -> DepthValidationConfig:
        for name in ("max_median_abs_error_m", "max_p90_abs_error_m", "min_anchor_depth_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"validation.{name} must be positive")
        if self.max_p90_abs_error_m < self.max_median_abs_error_m:
            raise ValueError(
                "validation.max_p90_abs_error_m must be >= max_median_abs_error_m"
            )
        if not 0.0 < self.min_anchor_coverage <= 1.0:
            raise ValueError("validation.min_anchor_coverage must be in (0, 1]")
        if self.reference_frames < 0:
            raise ValueError("validation.reference_frames must be non-negative")
        return self


class SceneDepthConfig(_StrictModel):
    """Depth Anything 3 adapter settings and the depth-mode contract."""

    enabled: bool = True
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: Literal["depth_anything_v3"] = "depth_anything_v3"

    #: [ENGINEERING] ADR-0008.  ``relative`` is allowed only as a diagnostic and is
    #: rejected by the compositor.
    mode: Literal["metric", "aligned", "relative"] = "metric"

    #: Which pixels the depth is estimated on.  ``auto`` prefers the inpainted
    #: background (the surface the robot is composited onto) and falls back to the
    #: decoded source frames, recording the choice in the manifest.
    input: Literal["auto", "inpainted", "original"] = "auto"

    #: DA3 is run on a reduced working grid and the result is resampled to native
    #: resolution; the geometry assertion is that the returned model grid equals
    #: this grid exactly, so no crop/letterbox may occur.
    work_width: int = 756
    work_height: int = 420

    #: [PAPER] DA3 Metric-Large emits focal-normalised depth.  Metres are recovered
    #: with ``raw * focal_px / focal_scale_divisor`` on the asserted model grid.
    #: Do not change this without a new ADR: it moves every metric depth.
    focal_scale_divisor: float = 300.0

    seed: int = 0

    #: When set, no model is run: the stage ingests an existing manifest produced by
    #: the same schema and records it as ``imported``.  Used to reuse an already
    #: validated depth run without pretending it was just computed.
    import_manifest: str | None = None

    validation: DepthValidationConfig = Field(default_factory=DepthValidationConfig)
    adr: str = "docs/decisions/0020-scene-depth-and-depth-aware-composite.md"

    @model_validator(mode="after")
    def _check_scene_depth(self) -> SceneDepthConfig:
        if self.work_width <= 0 or self.work_height <= 0:
            raise ValueError("scene_depth work grid must be positive")
        if self.work_width > 4096 or self.work_height > 4096:
            raise ValueError("scene_depth work grid is implausibly large")
        if self.focal_scale_divisor <= 0:
            raise ValueError("scene_depth.focal_scale_divisor must be positive")
        if self.mode == "aligned" and self.import_manifest is not None:
            raise ValueError(
                "an imported manifest declares its own mode; set scene_depth.mode to "
                "match it instead of forcing 'aligned'"
            )
        if not self.adr.strip():
            raise ValueError("scene_depth.adr must be non-empty")
        return self


class CompositeConfig(_StrictModel):
    """Depth-aware composite settings (paper appendix A.4, system-design 9.6)."""

    enabled: bool = True

    #: ``depth_aware`` is the formal product; ``debug_alpha`` draws the robot on top
    #: of everything, writes to a different file name and can never be a valid
    #: episode (ADR-0008 alternative B).
    mode: Literal["depth_aware", "debug_alpha"] = "depth_aware"

    #: [PAPER] appendix A.4: the gripper is exempt from depth occlusion inside a
    #: dilated hand mask, because the hand region is what inpainting replaced and a
    #: depth estimate there describes the background behind the hand.
    hand_mask_dilation_kernel: int = 5

    #: [PAPER] appendix A.4: the arm body is between the camera and the workspace, so
    #: it is always composited and never depth-tested.
    arm_body_always_visible: bool = True

    #: ``auto`` prefers ``inpainting/frames/background`` and falls back to the decoded
    #: source frames with an explicit manifest flag.
    background: Literal["auto", "inpainted", "original"] = "auto"

    #: ``auto`` prefers the M5 refined hand mask and falls back to no exemption,
    #: which is recorded and warned about (never silently assumed clean).
    hand_mask: Literal["auto", "refined", "none"] = "auto"

    #: Frames whose gripper has no depth at all (fully occluded) still get the arm.
    min_visible_pixels: int = 1

    adr: str = "docs/decisions/0020-scene-depth-and-depth-aware-composite.md"

    @model_validator(mode="after")
    def _check_composite(self) -> CompositeConfig:
        if self.hand_mask_dilation_kernel <= 0 or self.hand_mask_dilation_kernel % 2 == 0:
            raise ValueError("composite.hand_mask_dilation_kernel must be a positive odd integer")
        if self.min_visible_pixels < 0:
            raise ValueError("composite.min_visible_pixels must be non-negative")
        if not self.adr.strip():
            raise ValueError("composite.adr must be non-empty")
        return self
