"""Configuration layers, resolution and hashing.

System design section 14.  Precedence, lowest to highest::

    package defaults < pipeline profile < dataset/source profile
                     < robot profile < CLI explicit overrides

The fully resolved document is written to ``configs/resolved.json`` for every
episode and hashed into stage manifests, so a product can always be traced back
to the exact parameters that produced it.

Every block that hard-codes a number the paper does not publish carries a
``provenance`` note pointing at the ADR that justifies it
(see ``docs/decisions/`` and AGENTS.md rule 10).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import config_hash as _config_hash
from ..vision.m5_contracts import InpaintingConfig, SegmentationConfig
from ..vision.m6_contracts import CompositeConfig, SceneDepthConfig

__all__ = [
    "BasePlacementConfig",
    "ConfigError",
    "GapConfig",
    "HandConfig",
    "IngestConfig",
    "InpaintingConfig",
    "OutlierConfig",
    "PipelineConfig",
    "ProjectionConfig",
    "QualityConfig",
    "RegistryConfig",
    "ResolvedConfig",
    "RetargetConfig",
    "RigArmConfig",
    "RigConfig",
    "RigSearchConfig",
    "RobotBindingConfig",
    "RunConfig",
    "SegmentationConfig",
    "SmoothingConfig",
    "deep_merge",
    "load_config_file",
    "load_yaml",
    "resolve_config",
]


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or structurally invalid."""


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class IngestConfig(_Block):
    """Video probe / timebase policy."""

    timebase_source: str = "pts"
    """``pts`` is normative (system-design.md 5.3); ``nominal`` only for fixtures."""
    decode_check_frames: int = 3
    require_monotonic_timestamps: bool = True
    max_frames: int | None = None


class HandConfig(_Block):
    """Path A / Path B hand input policy.

    ``selection`` and ``sides`` are the ADR-0021 hand-identity contract:

    ``legacy``
        Reproduce the historical single-hand behaviour: the provider inspects the
        annotation and keeps one hand (``right`` when present, else ``left``).
        ``hand_count`` must be 1 and ``sides`` must be empty.
    ``both``
        Read both annotated hands into the ``[T, 2, ...]`` sequence in the order
        given by ``sides`` (default ``[right, left]``, the Path A order).
    ``explicit``
        Read exactly the hands listed in ``sides``, in that order.

    ``hand_count`` must agree with ``sides`` in every case, so a profile cannot
    claim two hands while the adapter writes one (or vice versa).  The order
    becomes ``hand_ids`` on the persisted sequence; it is never inferred from
    array position by a reader.
    """

    path: str = "A"
    keypoint_convention: str = "mano21"
    hand_count: int = 1
    selection: str = "legacy"
    """``legacy`` | ``both`` | ``explicit`` (ADR-0021)."""
    sides: list[str] = Field(default_factory=list)
    """Hand sides in persisted ``H`` order; required for ``explicit``."""
    min_detection_confidence: float = 0.3
    provider: str = "egodex_hdf5"
    #: metric scaling applied by the adapter; EgoDex/ARKit annotations are metres
    source_length_unit: str = "m"
    max_hand_depth_m: float = 0.4
    min_hand_depth_m: float = 0.05

    @model_validator(mode="after")
    def _check(self) -> HandConfig:
        if self.path not in {"A", "B"}:
            raise ValueError("hand.path must be 'A' or 'B'")
        if self.source_length_unit not in {"m", "mm", "cm"}:
            raise ValueError("hand.source_length_unit must be one of m, mm, cm")
        if self.selection not in {"legacy", "both", "explicit"}:
            raise ValueError("hand.selection must be 'legacy', 'both' or 'explicit'")
        if self.hand_count not in {1, 2}:
            raise ValueError("hand.hand_count must be 1 or 2")
        unknown = [s for s in self.sides if s not in {"left", "right"}]
        if unknown:
            raise ValueError(f"hand.sides contains unknown side(s) {unknown}; use 'left'/'right'")
        if len(set(self.sides)) != len(self.sides):
            raise ValueError(f"hand.sides must not repeat a side, got {self.sides}")
        if self.selection == "legacy":
            if self.hand_count != 1:
                raise ValueError(
                    "hand.selection 'legacy' is single-hand by definition; set "
                    "hand.selection to 'both' (or 'explicit') for hand_count 2"
                )
            if self.sides:
                raise ValueError(
                    "hand.sides only applies to hand.selection 'both'/'explicit'; "
                    "the legacy provider chooses a hand itself"
                )
        if self.selection == "explicit":
            if not self.sides:
                raise ValueError("hand.selection 'explicit' requires hand.sides")
        if self.selection == "both":
            if self.sides and sorted(self.sides) != ["left", "right"]:
                raise ValueError(
                    "hand.selection 'both' reads both hands, so hand.sides must list "
                    f"both sides (in H order); got {self.sides}"
                )
        if self.selection in {"both", "explicit"} and len(self.resolved_sides) != self.hand_count:
            raise ValueError(
                f"hand.hand_count={self.hand_count} disagrees with hand.sides="
                f"{self.resolved_sides}; they must describe the same hands"
            )
        return self

    @property
    def resolved_sides(self) -> list[str]:
        """Hand sides in persisted ``H`` order (empty for the legacy selection)."""
        if self.selection == "both":
            return list(self.sides) or ["right", "left"]
        if self.selection == "explicit":
            return list(self.sides)
        return []

    @property
    def multi_hand(self) -> bool:
        return self.selection in {"both", "explicit"}


class DegeneracyConfig(_Block):
    """[PAPER] appendix A.3 degeneracy thresholds."""

    min_gripper_width_m: float = 0.01
    parallel_dot_threshold: float = 0.999
    """sin(angle) below which ``z`` and ``d`` count as parallel.
    [PAPER] states ``||z x d|| ~= 0`` without a number; the numeric tolerance is
    an engineering choice recorded in ADR-0004."""


class RetargetConfig(_Block):
    """Hand-to-gripper retargeting (system-design.md 7)."""

    virtual_fingertip_index_weight: float = 0.7
    virtual_fingertip_middle_weight: float = 0.3
    handedness_right_sign: int = 1
    handedness_left_sign: int = -1
    repair_so3: bool = True
    max_so3_correction_rad: float = 1e-3
    degeneracy: DegeneracyConfig = Field(default_factory=DegeneracyConfig)


class OutlierConfig(_Block):
    """[PAPER] appendix A.3 velocity filtering."""

    enabled: bool = True
    position_median_multiplier: float = 5.0
    position_floor_per_fps_m: float = 0.9
    rotation_floor_per_fps_rad: float = 10.0
    rounds: int = 2
    min_samples_for_median: int = 3


class GapConfig(_Block):
    """[PAPER] appendix A.1 gap handling."""

    small_gap_max_frames: int = 10
    blend_min_frames: int = 5
    blend_max_frames: int = 90
    blend_position_speed_m_per_frame: float = 0.00325
    """[PAPER] ``n_pos = dp / 3.25 mm``."""
    blend_rotation_speed_rad_per_frame: float = 0.01885
    """[PAPER] ``n_rot = dtheta / 1.08 deg``, converted to radians."""
    blend_position_weight: float = 0.6
    blend_rotation_weight: float = 0.4
    #: gripper state while a large gap is filled with the home configuration
    home_gripper_width_m: float = 0.08
    home_source: str = "config"
    """[ENGINEERING] the paper does not publish the home TCP pose used for large
    gaps; this project takes it from configuration (ADR-0006)."""


class SmoothingConfig(_Block):
    """[PAPER] appendix A.3 temporal smoothing."""

    position_window: int = 21
    position_polyorder: int = 3
    width_window: int = 21
    width_polyorder: int = 3
    orientation_sigma_frames: float = 10.0
    orientation_kernel_size: int = 21
    enforce_odd_window: bool = True

    @model_validator(mode="after")
    def _check_kernel(self) -> SmoothingConfig:
        if self.orientation_kernel_size % 2 == 0:
            raise ValueError("orientation_kernel_size must be odd")
        return self


class BasePlacementConfig(_Block):
    """[PAPER] appendix A.4 base pose search / [ENGINEERING] MVP fixed base."""

    mode: str = "fixed"
    """``fixed`` (MVP) or ``grid_search`` (M4 reproduction)."""
    lateral_levels: list[float] = Field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
    )
    forward_levels: list[float] = Field(
        default_factory=lambda: [-0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    )
    vertical_levels: list[float] = Field(default_factory=lambda: [0.4, 0.2, 0.0, -0.2, -0.4])
    pitch_deg: list[float] = Field(default_factory=lambda: [30.0, 45.0, 60.0])
    yaw_deg: list[float] = Field(default_factory=lambda: [-45.0, -20.0, 0.0, 20.0, 45.0])
    roll_deg: list[float] = Field(default_factory=lambda: [-15.0, 0.0, 15.0])
    min_camera_distance_m: float = 0.20
    min_base_distance_m: float = 0.08
    max_reach_fraction: float = 0.9
    max_keyframes: int = 20
    reach_ratio_target: float = 0.65
    reach_penalty_weight: float = 5.0
    ik_iterations: int = 100
    ik_tolerance: float = 1e-5


class IKConfig(_Block):
    """IK solver settings (system-design.md 8.3)."""

    solver: str = "mink"
    qp_backend: str = "quadprog"
    iterations: int = 100
    tolerance: float = 1e-5
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    lm_damping: float = 0.0
    warm_start: bool = True
    seed_attempts: int = 12
    max_joint_jump_rad: float = 1.5


class RobotBindingConfig(_Block):
    """Robot binding policy (system-design.md 8).

    The ``mjcf_*`` / ``dof`` / ``joint_names`` / ``eef_body_name`` fields are
    *derived* from the vendored model by the selected robot adapter; a profile may
    restate them so that a changed model shows up as a configuration diff
    instead of a silent behaviour change.
    """

    robot_id: str = "franka_panda"
    mjcf_uri: str | None = None
    mjcf_source: str = ""
    mjcf_license: str = ""
    dof: int | None = None
    gripper_dof: int | None = None
    eef_body_name: str | None = None
    base_body_name: str | None = None
    joint_names: list[str] = Field(default_factory=list)
    gripper_joint_names: list[str] = Field(default_factory=list)
    max_reach_m: float | None = None
    base_placement: BasePlacementConfig = Field(default_factory=BasePlacementConfig)
    ik: IKConfig = Field(default_factory=IKConfig)
    gripper_width_min_m: float = 0.0
    gripper_width_max_m: float = 0.08
    """[PAPER] appendix A.7, Panda gripper range 0-80 mm."""
    gripper_width_epsilon_m: float = 1e-4
    #: maps canonical gripper width (m) to a normalised command; export profile decision
    gripper_command_mode: str = "width_m"
    require_base_from_episode: bool = True


class RigArmConfig(_Block):
    """One arm of the two-arm rig (ADR-0022)."""

    arm_id: str
    #: hand side this arm is bound to; the binding is explicit, not inferred from
    #: the arm's position in the list (ADR-0022 decision 4)
    side: str
    robot_id: str = "franka_panda"
    #: placement side for the per-arm base search: +1 right of the camera axis,
    #: -1 left.  ``None`` falls back to ``+1`` for ``right`` and ``-1`` for ``left``.
    lateral_sign: int | None = None
    #: explicit ``T_C_B`` (camera -> arm base).  Required when ``base_mode: fixed``;
    #: otherwise it is written back after the per-arm search so the next run is fixed.
    T_C_B: list[list[float]] | None = None

    @model_validator(mode="after")
    def _check(self) -> RigArmConfig:
        if self.side not in {"left", "right"}:
            raise ValueError("rig arm side must be 'left' or 'right'")
        if self.lateral_sign not in {None, -1, 1}:
            raise ValueError("rig arm lateral_sign must be -1, +1 or unset")
        if self.T_C_B is not None:
            T = np.asarray(self.T_C_B, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"rig arm T_C_B must be 4x4, got {T.shape}")
            if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
                raise ValueError("rig arm T_C_B bottom row must be [0, 0, 0, 1]")
            R = T[:3, :3]
            if float(np.max(np.abs(R.T @ R - np.eye(3)))) > 1e-6:
                raise ValueError("rig arm T_C_B rotation is not orthonormal")
            if abs(float(np.linalg.det(R)) - 1.0) > 1e-6:
                raise ValueError("rig arm T_C_B rotation must have determinant +1")
        return self

    @property
    def resolved_lateral_sign(self) -> int:
        if self.lateral_sign is not None:
            return int(self.lateral_sign)
        return 1 if self.side == "right" else -1


class RigSearchConfig(_Block):
    """Reduced per-arm base grid for the first two-arm delivery.

    Deliberately smaller than the single-arm M4 grid: the joint acceptance is what
    is under test here, and a 10k-candidate grid per arm would dominate the run
    time.  The full grid remains available for the later two-base search.
    """

    lateral_levels: list[float] = Field(default_factory=lambda: [0.4, 0.6, 0.8])
    forward_levels: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.4])
    vertical_levels: list[float] = Field(default_factory=lambda: [0.2, 0.0, -0.2])
    pitch_deg: list[float] = Field(default_factory=lambda: [45.0])
    yaw_deg: list[float] = Field(default_factory=lambda: [-20.0, 0.0, 20.0])
    roll_deg: list[float] = Field(default_factory=lambda: [0.0])
    max_keyframes: int = 20
    ik_iterations: int = 100
    ik_tolerance: float = 1e-5
    #: how many per-arm candidates the pairwise joint search may pair up
    top_k: int = 5
    #: hard minimum distance between the two bases in the pairwise search
    min_base_separation_m: float = 0.40
    reach_ratio_target: float = 0.65
    reach_penalty_weight: float = 5.0
    max_reach_fraction: float = 0.9
    min_camera_distance_m: float = 0.20
    min_base_distance_m: float = 0.08
    min_forward_depth_m: float = 0.30
    """Base must sit at least this far in front of the camera (camera-frame ``z``).

    The paper's 0.20 m constraint is a distance, which a base *beside* the camera
    satisfies while being almost in the camera plane; such a placement is IK-feasible
    and visually absurd (ADR-0023).
    """


class RigConfig(_Block):
    """Two-arm rig (ADR-0022).  Disabled by default; the single-arm stage is untouched."""

    enabled: bool = False
    #: ``per_arm_search`` searches each arm once and freezes the result in rig.json;
    #: ``fixed`` requires ``T_C_B`` on every arm.
    base_mode: str = "per_arm_search"
    arms: list[RigArmConfig] = Field(default_factory=list)
    search: RigSearchConfig = Field(default_factory=RigSearchConfig)
    max_cross_arm_contacts: int = 1
    """[PAPER] appendix A.5 / ``QualityConfig.l1_cross_arm_contacts_max``."""
    write_previews: bool = True
    sample_stills: int = 8
    #: B3 first half: render both arms in the joint scene (RGB/owner/masks/depth)
    write_render: bool = True
    render_frames: int = 6
    render_plane_height_m: float | None = None

    @model_validator(mode="after")
    def _check(self) -> RigConfig:
        if self.base_mode not in {"per_arm_search", "fixed"}:
            raise ValueError("rig.base_mode must be 'per_arm_search' or 'fixed'")
        if not self.enabled:
            return self
        if len(self.arms) != 2:
            raise ValueError(f"rig.enabled requires exactly two arms, got {len(self.arms)}")
        arm_ids = [a.arm_id for a in self.arms]
        sides = [a.side for a in self.arms]
        if len(set(arm_ids)) != 2:
            raise ValueError(f"rig arm_id values must be unique, got {arm_ids}")
        if sorted(sides) != ["left", "right"]:
            raise ValueError(f"rig arms must cover both hands sides, got {sides}")
        if self.base_mode == "fixed":
            missing = [a.arm_id for a in self.arms if a.T_C_B is None]
            if missing:
                raise ValueError(
                    f"rig.base_mode 'fixed' requires T_C_B on every arm; missing {missing}"
                )
        return self

    def arm_for_side(self, side: str) -> RigArmConfig:
        for arm in self.arms:
            if arm.side == side:
                return arm
        raise KeyError(f"no rig arm is bound to hand side {side!r}")


class RenderConfig(_Block):
    """MuJoCo rendering (system-design.md 9.4)."""

    enabled: bool = True
    width: int | None = None
    height: int | None = None
    camera_fovy_deg: float | None = None
    """[ENGINEERING] leave unset to derive fovy from the calibrated intrinsics."""
    lights: bool = True
    shadow: bool = True
    save_arrays_npz: bool = True


class ProjectionConfig(_Block):
    """Projection debug gate (system-design.md 15, Gate M3)."""

    sample_frames: int = 8
    overlay_scale: float = 1.0
    draw_axes_m: float = 0.05


class QualityConfig(_Block):
    """L1/L2 thresholds (system-design.md 10)."""

    profile: str = "repro_v1"
    l1_ik_tracking_error_m: float = 0.05
    """[PAPER] appendix A.5."""
    l1_robot_visible_pixels: int = 1
    l1_robot_mask_area_ratio_max: float = 0.70
    l1_cross_arm_contacts_max: int = 1
    l1_stability_erosion_fps_fraction: float = 0.3
    #: [ENGINEERING] MVP geometry gate, stricter than the paper's L1 threshold
    mvp_p95_position_error_m: float = 0.01
    mvp_p95_rotation_error_rad: float = 0.10
    mvp_min_ik_success_ratio: float = 0.95
    l2_enabled: bool = False
    l2_quantile_low: float = 0.01
    l2_quantile_high: float = 0.99
    l2_iqr_multiplier: float = 3.0
    l2_max_invalid_ratio: float = 0.60
    l3_backend: str = "manual_review"


class RegistryConfig(_Block):
    """Fallback / reason-code registry location."""

    fallback_codes_file: str = "configs/quality/reason-codes.yaml"


class PipelineConfig(_Block):
    """Top-level resolved configuration document."""

    profile: str = "mvp"
    episode_id: str = ""
    run_id: str = ""
    repo_root: str = "."
    data_root: str = "data/episodes"
    fixture_root: str = "data/fixtures"
    stages: list[str] = Field(
        default_factory=lambda: ["ingest", "hand", "retarget", "robot", "render", "projection"]
    )
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    hand: HandConfig = Field(default_factory=HandConfig)
    retarget: RetargetConfig = Field(default_factory=RetargetConfig)
    outliers: OutlierConfig = Field(default_factory=OutlierConfig)
    gaps: GapConfig = Field(default_factory=GapConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    robot_binding: RobotBindingConfig = Field(default_factory=RobotBindingConfig)
    rig: RigConfig = Field(default_factory=RigConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    inpainting: InpaintingConfig = Field(default_factory=InpaintingConfig)
    scene_depth: SceneDepthConfig = Field(default_factory=SceneDepthConfig)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    #: fixture-specific overrides, e.g. the synthetic case definitions
    case: dict[str, Any] = Field(default_factory=dict)
    #: provenance labels for values the paper does not publish
    provenance: dict[str, str] = Field(default_factory=dict)


ResolvedConfig = PipelineConfig
RunConfig = PipelineConfig


# ---------------------------------------------------------------------------
# YAML loading / merging
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return {} if payload is None else payload


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and require it to be a mapping."""
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level, got {type(payload).__name__}")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either.

    Lists are replaced wholesale (a profile that sets ``stages`` means exactly
    those stages), scalars and mappings are merged key by key.
    """
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce_overrides(overrides: Mapping[str, Any] | Iterable[str] | None) -> list[dict[str, Any]]:
    """Accept ``{"a.b": 1}`` notation or a list of mappings."""
    if overrides is None:
        return []
    if isinstance(overrides, Mapping):
        nested: dict[str, Any] = {}
        for dotted, value in overrides.items():
            cursor = nested
            parts = dotted.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        return [nested]
    return [dict(o) for o in overrides]


def resolve_config(
    *,
    repo_root: str | Path,
    profile: str = "mvp",
    source_profile: str | Path | None = None,
    robot_profile: str | Path | None = None,
    quality_profile: str | Path | None = None,
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
    extra_layers: Iterable[Mapping[str, Any]] = (),
) -> PipelineConfig:
    """Resolve the configuration stack and return the validated document.

    Layer order is exactly the one in system-design.md 14.
    """
    root = Path(repo_root)
    merged: dict[str, Any] = {"profile": profile, "repo_root": str(root)}

    layers: list[tuple[str, Mapping[str, Any]]] = []
    profile_path = root / "configs" / "pipeline" / f"{profile}.yaml"
    layers.append((str(profile_path), load_config_file(profile_path)))
    if source_profile is not None:
        layers.append((str(source_profile), load_config_file(source_profile)))
    if robot_profile is not None:
        layers.append((str(robot_profile), load_config_file(robot_profile)))
    if quality_profile is not None:
        layers.append((str(quality_profile), load_config_file(quality_profile)))
    for i, layer in enumerate(extra_layers):
        layers.append((f"extra_layer[{i}]", layer))
    for i, override in enumerate(_coerce_overrides(overrides)):
        layers.append((f"cli_override[{i}]", override))

    for name, layer in layers:
        try:
            merged = deep_merge(merged, layer)
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigError(f"failed to merge layer {name}: {exc}") from exc

    merged.setdefault("provenance", {})
    cfg = PipelineConfig.model_validate(merged)
    _attach_layer_trace(cfg, [name for name, _ in layers])
    return cfg


def _attach_layer_trace(cfg: PipelineConfig, layers: list[str]) -> None:
    """Record which files took part in resolution (audit trail)."""
    cfg.provenance = dict(cfg.provenance)
    cfg.provenance["_config_layers"] = " < ".join(layers)


def resolve_config_from_cli(
    repo_root: str | Path,
    *,
    profile: str = "mvp",
    profile_file: str | Path | None = None,
    source_profile: str | Path | None = None,
    robot_profile: str | Path | None = None,
    quality_profile: str | Path | None = None,
    set_values: Mapping[str, Any] | None = None,
    case: Mapping[str, Any] | None = None,
    episode_id: str | None = None,
) -> PipelineConfig:
    """CLI-facing wrapper that also injects the fixture case block."""
    extra: list[Mapping[str, Any]] = []
    if case is not None:
        extra.append({"case": dict(case)})
    cfg = resolve_config(
        repo_root=repo_root,
        profile=profile,
        source_profile=profile_file or source_profile,
        robot_profile=robot_profile,
        quality_profile=quality_profile,
        overrides=set_values,
        extra_layers=extra,
    )
    if episode_id is not None:
        cfg.episode_id = episode_id
    return cfg


def config_hash(cfg: PipelineConfig | Mapping[str, Any]) -> str:
    """Stable hash of the resolved configuration."""
    payload = cfg.model_dump(mode="json") if isinstance(cfg, PipelineConfig) else dict(cfg)
    return _config_hash(payload)


def dump_resolved(cfg: PipelineConfig, path: str | Path) -> Path:
    """Persist the resolved document next to the episode artifacts."""
    from .artifacts import atomic_write_json

    return atomic_write_json(path, cfg)
