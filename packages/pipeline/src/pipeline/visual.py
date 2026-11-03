"""Read-only visual adapter and explicit metric-depth compositing.

This module never loads a model, guesses scene depth, or invents a gate approval.
The scene manifest is documented in ``visual.md``. Array-only calls are useful
for numerical inspection; they are not evidence of a model or episode review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import PIL
from PIL import Image, ImageFilter


ADAPTER_VERSION = "learning.stage5.visual.v1"
SCENE_SCHEMA = "learning.stage5.scene_depth.v1"
TIME_TOLERANCE_S = 1e-6
REAL_EVIDENCE = {"real_model", "real_sensor", "real_source_annotation"}


@dataclass(frozen=True)
class DepthAnchor:
    """A measured scene correspondence in native RGB pixels, with provenance.

    Robot-render depth is not a scene measurement at the same pixel. A hand
    landmark also needs an explicit surface correspondence before use here.
    ``evidence_kind=synthetic_fixture`` is only for numerical demonstrations.
    """

    x: int
    y: int
    depth_m: float
    evidence_ref: str
    evidence_kind: str
    surface_correspondence: str
    weight: float = 1.0


@dataclass(frozen=True)
class DepthAlignment:
    depth_m: np.ndarray
    valid_mask: np.ndarray
    mode: str
    metric_valid: bool
    evidence_kind: str
    scale: float
    shift: float
    residual_rmse_m: float | None
    residual_max_m: float | None
    anchor_count: int
    quantity: str
    confidence: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompositeFrame:
    rgb: np.ndarray
    visible_robot_mask: np.ndarray
    occluded_gripper_mask: np.ndarray
    decision_map: np.ndarray
    comparison_valid_mask: np.ndarray
    frame_valid: bool
    diagnostics: dict[str, Any]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path, **extra: Any) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha(path),
            "bytes": path.stat().st_size, **extra}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _positive(value: np.ndarray) -> np.ndarray:
    return np.isfinite(value) & (value > 0)


def align_depth(
    depth: np.ndarray,
    *,
    mode: Literal["metric", "relative"],
    quantity: Literal["z_depth_m", "relative_depth", "relative_inverse_depth"],
    evidence_kind: str,
    anchors: Sequence[DepthAnchor | Mapping[str, Any]] = (),
    max_rmse_m: float | None = None,
    max_residual_m: float | None = None,
    confidence: np.ndarray | None = None,
    min_confidence: float | None = None,
) -> DepthAlignment:
    """Validate metric Z or fit an explicit affine relative representation.

    relative_depth: Z = scale * raw + shift
    relative_inverse_depth: 1/Z = scale * raw + shift

    The representation is mandatory: inverse depth must never be fitted as Z.
    No outlier is removed silently. Three distinct pixel anchors, two distinct
    relative values and explicit residual limits are required. Passing a fit
    only establishes numerical consistency of the supplied anchors; it does
    not verify the measurements or confer visual acceptance.
    """
    raw = np.asarray(depth, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("scene depth must have shape [height,width]")
    if evidence_kind not in REAL_EVIDENCE | {"synthetic_fixture", "mock"}:
        raise ValueError("explicit scene evidence_kind is required")
    confidence_array = None
    input_valid = np.isfinite(raw)
    if confidence is not None:
        confidence_array = np.asarray(confidence, dtype=np.float32)
        if confidence_array.shape != raw.shape:
            raise ValueError("depth confidence shape mismatch")
        finite_confidence = np.isfinite(confidence_array)
        if np.any(finite_confidence & ((confidence_array < 0) | (confidence_array > 1))):
            raise ValueError("depth confidence must be in [0,1]")
        input_valid &= finite_confidence
        if min_confidence is not None:
            if not 0 <= min_confidence <= 1:
                raise ValueError("min_confidence must be in [0,1]")
            input_valid &= confidence_array >= min_confidence
    elif min_confidence is not None:
        raise ValueError("min_confidence requires a confidence array")

    scale, shift = 1.0, 0.0
    residuals = np.empty(0, dtype=np.float64)
    anchor_rows: list[dict[str, Any]] = []
    accepted = True
    if mode == "metric":
        if quantity != "z_depth_m":
            raise ValueError("metric mode requires camera optical Z in metres")
        if anchors:
            raise ValueError("metric mode does not silently refit supplied anchors")
        metric = raw.copy()
    elif mode == "relative":
        if quantity not in {"relative_depth", "relative_inverse_depth"}:
            raise ValueError("relative mode requires an explicit depth representation")
        if max_rmse_m is None or max_residual_m is None:
            raise ValueError("relative alignment requires explicit residual thresholds in m")
        if not np.isfinite([max_rmse_m, max_residual_m]).all() or min(max_rmse_m, max_residual_m) < 0:
            raise ValueError("alignment residual thresholds must be finite and nonnegative")
        parsed = [a if isinstance(a, DepthAnchor) else DepthAnchor(**dict(a)) for a in anchors]
        if len(parsed) < 3 or len({(a.x, a.y) for a in parsed}) != len(parsed):
            raise ValueError("relative alignment requires >=3 distinct scene anchor pixels")
        sample, measured, weights = [], [], []
        for anchor in parsed:
            if not isinstance(anchor.x, int) or not isinstance(anchor.y, int):
                raise ValueError("anchor pixels must be integer native RGB indices")
            if not (0 <= anchor.y < raw.shape[0] and 0 <= anchor.x < raw.shape[1]):
                raise ValueError("anchor lies outside native RGB/depth dimensions")
            if not anchor.evidence_ref or not anchor.surface_correspondence:
                raise ValueError("anchor requires measurement provenance and surface correspondence")
            if anchor.evidence_kind not in REAL_EVIDENCE | {"synthetic_fixture", "mock"}:
                raise ValueError("anchor evidence_kind is required")
            if evidence_kind in REAL_EVIDENCE and anchor.evidence_kind not in REAL_EVIDENCE:
                raise ValueError("real depth cannot acquire metric scale from synthetic/mock anchors")
            if not input_valid[anchor.y, anchor.x] or not np.isfinite(anchor.depth_m) or anchor.depth_m <= 0:
                raise ValueError("anchor measurement and raw prediction must be finite and valid")
            if not np.isfinite(anchor.weight) or anchor.weight <= 0:
                raise ValueError("anchor weight must be finite and positive")
            sample.append(raw[anchor.y, anchor.x])
            measured.append(anchor.depth_m)
            weights.append(anchor.weight)
            anchor_rows.append(dict(vars(anchor), raw_value=float(sample[-1])))
        sample_array = np.asarray(sample)
        measured_array = np.asarray(measured)
        target = measured_array if quantity == "relative_depth" else 1.0 / measured_array
        matrix = np.column_stack([sample_array, np.ones(len(parsed))])
        root_weights = np.sqrt(weights)
        fit, _, rank, singular = np.linalg.lstsq(matrix * root_weights[:, None],
                                                target * root_weights, rcond=None)
        if rank != 2:
            raise ValueError("relative anchor fit is rank deficient; scale is not identifiable")
        scale, shift = map(float, fit)
        if not np.isfinite(fit).all() or scale <= 0:
            raise ValueError("relative prediction/quantity has inconsistent depth ordering")
        affine = scale * raw + shift
        predicted_anchors = scale * sample_array + shift
        if quantity == "relative_inverse_depth":
            with np.errstate(divide="ignore", invalid="ignore"):
                metric = 1.0 / affine
                predicted_anchors = 1.0 / predicted_anchors
            input_valid &= affine > 0
        else:
            metric = affine
        residuals = predicted_anchors - measured_array
        if not np.isfinite(residuals).all():
            raise ValueError("anchor fit produces invalid metric depth")
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        maximum = float(np.max(np.abs(residuals)))
        accepted = rmse <= max_rmse_m and maximum <= max_residual_m
        for row, residual in zip(anchor_rows, residuals):
            row["residual_m"] = float(residual)
    else:
        raise ValueError("scene depth mode must be metric or relative")
    valid = input_valid & _positive(metric) & (metric <= np.finfo(np.float32).max)
    result = np.where(valid, metric, np.nan).astype(np.float32)
    return DepthAlignment(
        depth_m=result, valid_mask=valid, mode="metric" if mode == "metric" else "aligned_metric",
        metric_valid=bool(accepted and np.any(valid)), evidence_kind=evidence_kind,
        scale=scale, shift=shift,
        residual_rmse_m=float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else None,
        residual_max_m=float(np.max(np.abs(residuals))) if residuals.size else None,
        anchor_count=len(anchor_rows), quantity=quantity, confidence=confidence_array,
        diagnostics={"anchors": anchor_rows, "fit": "none" if mode == "metric" else "weighted_affine",
                     "max_rmse_m": max_rmse_m, "max_residual_m": max_residual_m,
                     "input_valid_pixels": int(np.count_nonzero(valid)),
                     "residual_limits_pass": bool(accepted),
                     "measurement_provenance_verified": False,
                     "review_status": "pending"},
    )


def _binary_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != shape:
        raise ValueError(f"{name} shape mismatch")
    if value.dtype != np.bool_ and not np.isin(value, [0, 1, 255]).all():
        raise ValueError(f"{name} must be bool or contain only 0/1/255")
    return value != 0


def compose_frame(
    background_rgb: np.ndarray,
    robot_rgb: np.ndarray,
    arm_mask: np.ndarray,
    gripper_mask: np.ndarray,
    robot_depth_m: np.ndarray,
    scene_depth: DepthAlignment,
    human_arm_mask: np.ndarray,
    *,
    depth_epsilon_m: float = 0.0,
    hand_mask_dilation_kernel: int = 5,
) -> CompositeFrame:
    """Arm always drawn; only gripper uses metric Z and hand protection.

    Invalid scene/robot depth at a gripper pixel leaves background there and
    invalidates the frame. This conservative policy is visible in decision=5;
    it never pretends unknown depth is an infinite scene distance. Equal depth
    keeps the gripper; epsilon is an explicit nonnegative engineering margin.
    Decision IDs: 0 background,1 arm,2 front gripper,3 protected gripper,
    4 occluded gripper,5 unknown comparison.
    """
    background = np.asarray(background_rgb)
    robot = np.asarray(robot_rgb)
    if background.ndim != 3 or background.shape[2] != 3 or background.dtype != np.uint8:
        raise ValueError("background must be RGB uint8[H,W,3]")
    if robot.shape != background.shape or robot.dtype != np.uint8:
        raise ValueError("robot RGB shape/dtype mismatch")
    shape = background.shape[:2]
    arm = _binary_mask(arm_mask, shape, "arm body mask")
    gripper = _binary_mask(gripper_mask, shape, "gripper mask")
    human = _binary_mask(human_arm_mask, shape, "human arm mask")
    if np.any(arm & gripper):
        raise ValueError("arm body and gripper masks must be disjoint")
    if scene_depth.mode not in {"metric", "aligned_metric"} or not scene_depth.metric_valid:
        raise ValueError("compositing requires valid metric/aligned_metric scene depth")
    robot_z = np.asarray(robot_depth_m)
    scene_z = np.asarray(scene_depth.depth_m)
    if robot_z.shape != shape or scene_z.shape != shape or scene_depth.valid_mask.shape != shape:
        raise ValueError("depth shape mismatch")
    if hand_mask_dilation_kernel < 1 or hand_mask_dilation_kernel % 2 != 1:
        raise ValueError("hand dilation kernel must be a positive odd integer")
    if not np.isfinite(depth_epsilon_m) or depth_epsilon_m < 0:
        raise ValueError("depth_epsilon_m must be finite and nonnegative")
    dilated = np.asarray(Image.fromarray(human.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(hand_mask_dilation_kernel))) != 0
    comparison_valid = scene_depth.valid_mask & _positive(scene_z) & _positive(robot_z)
    # Protection bypasses the depth decision for the removed human region.
    protected = gripper & dilated
    unknown = gripper & ~dilated & ~comparison_valid
    occluded = gripper & ~dilated & comparison_valid & (scene_z + depth_epsilon_m < robot_z)
    front = gripper & ~dilated & comparison_valid & ~occluded
    visible = arm | protected | front
    output = background.copy()
    output[visible] = robot[visible]
    decision = np.zeros(shape, dtype=np.uint8)
    for code, selection in [(1, arm), (2, front), (3, protected), (4, occluded), (5, unknown)]:
        decision[selection] = code
    return CompositeFrame(
        rgb=output, visible_robot_mask=visible, occluded_gripper_mask=occluded,
        decision_map=decision, comparison_valid_mask=comparison_valid,
        frame_valid=bool(not np.any(unknown) and np.any(visible)),
        diagnostics={"arm_body_always_visible": True, "hand_dilation_kernel": hand_mask_dilation_kernel,
                     "depth_epsilon_m": depth_epsilon_m,
                     "unknown_depth_policy": "background_and_frame_invalid",
                     "decision_pixel_counts": {str(i): int(np.count_nonzero(decision == i)) for i in range(6)},
                     "evidence_kind": scene_depth.evidence_kind, "review_status": "pending"},
    )


def _checked_ref(base: Path, item: Mapping[str, Any]) -> Path:
    path = _resolve(base, str(item["path"]))
    if not path.is_file() or _sha(path) != item.get("sha256"):
        raise ValueError(f"missing artifact or SHA256 mismatch: {path}")
    return path


def _sequence_index(root: Path, relative: str, timestamps: np.ndarray,
                    source_sha: str, *, external_path: Path | None = None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    path = external_path if external_path is not None else root / relative
    if not path.is_file():
        return None, []
    index = _read_json(path)
    expected_schema = {"segmentation/source-index.json": "ego2robot.m5_source_frames.v1",
                       "segmentation/bundle/index.json": "ego2robot.m5_masks.v1",
                       "inpainting/bundle/index.json": "ego2robot.m5_background.v1"}[relative]
    if index.get("schema_version") != expected_schema:
        raise ValueError(f"unexpected visual sequence schema: {path}")
    if index.get("source_video_sha256") != source_sha:
        raise ValueError(f"source_video_sha256 mismatch: {path}")
    rows = index.get("frames", [])
    if len(rows) != len(timestamps):
        raise ValueError(f"frame count mismatch: {path}")
    for expected, row in enumerate(rows):
        if (row.get("frame_index") != expected or
                not np.isfinite(row.get("timestamp_s", np.nan)) or
                abs(row["timestamp_s"] - timestamps[expected]) > TIME_TOLERANCE_S):
            raise ValueError(f"source frame/PTS mismatch: {path} frame {expected}")
    return index, rows


def _legacy_model_evidence(index: Mapping[str, Any]) -> str:
    provenance = index.get("provenance", {})
    model = provenance.get("model_provenance", provenance)
    if (model.get("mock") is True or model.get("evidence_kind") == "mock" or
            "mock" in str(model.get("model_name", "")).lower()):
        return "mock"
    if model.get("repository_revision") and model.get("checkpoint_sha256"):
        return "real_model"
    return "unavailable"


def _image_record(path: Path, kind: str, frame: int, timestamp: float,
                  width: int, height: int, source_key: str, evidence_kind: str,
                  preview_only: bool, *, expected_sha: str | None = None,
                  valid: bool = True) -> dict[str, Any]:
    if expected_sha is not None and _sha(path) != expected_sha:
        raise ValueError(f"image SHA256 mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.size != (width, height):
            raise ValueError(f"native camera/image dimension mismatch: {path}")
        if kind in {"robot_only_rgb", "arm_removed_background", "source_rgb", "depth_composite_rgb"} and image.mode != "RGB":
            raise ValueError(f"image is not native RGB: {path}")
        if kind.endswith("mask"):
            _binary_mask(np.asarray(image), (height, width), kind)
    return dict(_ref(path), frame_index=frame, timestamp_s=float(timestamp), image_kind=kind,
                width=width, height=height, color_space="RGB" if kind.endswith("rgb") or kind == "arm_removed_background" else "mask",
                valid=bool(valid), source_key=source_key, producer=ADAPTER_VERSION,
                evidence_kind=evidence_kind, preview_only=bool(preview_only))


def _review(root: Path, reference: Mapping[str, Any] | None, source_sha: str,
            expected_scope: str, required_evidence: Mapping[str, str],
            source_key: str | None = None) -> dict[str, Any]:
    if reference is None:
        return {"status": "pending", "reason": "review record missing"}
    path = _checked_ref(root, reference)
    review = _read_json(path)
    scope_match = review.get("scope") == expected_scope
    evidence = review.get("evidence", [])
    actual = {str(_resolve(root, row["path"]).resolve()): row.get("sha256") for row in evidence}
    evidence_match = all(actual.get(str(Path(p).resolve())) == digest for p, digest in required_evidence.items())
    for row in evidence:
        _checked_ref(root, row)
    source_match = source_sha in actual.values() or review.get("source_video_sha256") == source_sha
    key_match = source_key is None or review.get("source_key") == source_key
    accepted = (review.get("accepted") is True and review.get("reviewer") == "user" and
                scope_match and source_match and key_match and evidence_match and bool(review.get("user_statement")))
    return {"status": "accepted" if accepted else "pending", "review_ref": _ref(path),
            "scope_match": scope_match, "source_match": source_match, "source_key_match": key_match,
            "evidence_match": evidence_match, "review_base": str(root.resolve()),
            "required_evidence": dict(required_evidence), "scope": expected_scope}


def finalize_visual(candidate_index: Path, m6_receipt: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Promote a fully reviewed real composite to a NEW visual index.

    Receipt bytes must already contain the user's actual decision. This API
    validates evidence and never creates an accepted review on the user's behalf.
    The accepted candidate stays immutable, so its receipt hash remains valid.
    This only certifies visual readiness; the bundle still needs its own QC.
    """
    candidate_path = Path(candidate_index).resolve()
    receipt_path = Path(m6_receipt).resolve()
    destination = Path(output_path).resolve() if output_path else candidate_path.with_name("index-reviewed.json")
    if destination in {candidate_path, receipt_path}:
        raise ValueError("visual finalization must preserve candidate index and review receipt")
    index = _read_json(candidate_path)
    if index.get("schema_version") != "learning.stage5.visual_index.v1":
        raise ValueError("unexpected candidate visual index schema")
    source_sha, key = index["source_video_sha256"], index["source_key"]
    n = len(index["composite_numerically_valid"])
    if not n or not all(index["composite_numerically_valid"]):
        raise ValueError("formal visual readiness requires N numerically valid composite frames")
    # Rehash every indexed local file; no stale gate dictionary is trusted.
    for row in index["frames"]:
        _checked_ref(candidate_path.parent, row)
    for reference in index.get("input_refs", []):
        _checked_ref(candidate_path.parent, reference)
    for name in ("m3", "m5"):
        gate = index.get("gates", {}).get(name, {})
        if not gate.get("review_ref") or not gate.get("review_base"):
            raise ValueError(f"{name} requires a scoped user review receipt")
        revalidated = _review(Path(gate["review_base"]), gate["review_ref"], source_sha,
            gate["scope"], gate.get("required_evidence", {}), key if name == "m5" else None)
        if revalidated["status"] != "accepted":
            raise ValueError(f"{name} review does not accept these source/evidence hashes")
        index["gates"][name] = revalidated
    episode_path = _checked_ref(candidate_path.parent, index["episode_ref"])
    episode = _read_json(episode_path)
    if episode_path.parent in destination.parents or destination == episode_path:
        raise ValueError("reviewed index must remain outside the read-only source episode")
    expected_key = hashlib.sha256((episode["source"]["sha256"] + ":" + episode["source"]["hdf5_sha256"]).encode("ascii")).hexdigest()
    if expected_key != key or episode["source"]["sha256"] != source_sha:
        raise ValueError("episode MP4/HDF5 source_key identity mismatch")
    if episode["camera"].get("calibration_source") == "assumed" or episode["camera"].get("debug_only") is True:
        raise ValueError("assumed/debug camera cannot be promoted to formal visual observation")
    # Full-resolution real M5 is a prerequisite; upscaled preview is not repaired
    # by approving a later composite. Re-open its original provenance.
    backgrounds = [r for r in index["frames"] if r["image_kind"] == "arm_removed_background"]
    masks = [r for r in index["frames"] if r["image_kind"] == "human_arm_mask"]
    if ([r["frame_index"] for r in backgrounds] != list(range(n)) or
            [r["frame_index"] for r in masks] != list(range(n)) or
            any(r["preview_only"] or not r["valid"] or r["evidence_kind"] != "real_model" for r in backgrounds + masks)):
        raise ValueError("M5 must supply N real-model, valid, non-preview frames and masks")
    background_ref = index.get("background_index_ref")
    if not background_ref:
        raise ValueError("full-resolution M5 provenance index missing")
    background_index = _read_json(_checked_ref(candidate_path.parent, background_ref))
    provenance = background_index.get("provenance", {})
    inference_size = provenance.get("inference_resolution", [provenance.get("inference_width"), provenance.get("inference_height")])
    if inference_size != [index["width"], index["height"]] or provenance.get("preview_only", False):
        raise ValueError("M5 inference resolution must equal native camera resolution")
    composites = [r for r in index["frames"] if r["image_kind"] == "depth_composite_rgb"]
    if ([r["frame_index"] for r in composites] != list(range(n)) or
            any(not r["valid"] or not r.get("metric_valid") or r.get("upstream_preview_only", True) or
                r.get("depth_mode") not in {"metric", "aligned_metric"} or
                r.get("evidence_kind") not in REAL_EVIDENCE for r in composites)):
        raise ValueError("formal composite requires N source-aligned real metric-depth frames")
    alignment_path = _checked_ref(candidate_path.parent, index["scale_alignment_ref"])
    alignment = _read_json(alignment_path)
    if (alignment.get("source_key") != key or alignment.get("source_video_sha256") != source_sha or
            [row["frame_index"] for row in alignment["frames"]] != list(range(n))):
        raise ValueError("alignment source/index coverage mismatch")
    required = {str(candidate_path): _sha(candidate_path), str(alignment_path): _sha(alignment_path)}
    for row in composites:
        required[row["path"]] = row["sha256"]
    for row in alignment["frames"]:
        if not row["metric_valid"] or row["evidence_kind"] not in REAL_EVIDENCE:
            raise ValueError("nonmetric/synthetic depth cannot be promoted")
        for field in ("depth_ref", "decision_ref"):
            path = _checked_ref(alignment_path.parent, row[field])
            required[str(path.resolve())] = row[field]["sha256"]
    review = _review(receipt_path.parent, _ref(receipt_path), source_sha,
                     "episode_depth_composite", required, key)
    if review["status"] != "accepted":
        raise ValueError("M6 receipt must accept candidate index, alignment and every composite/depth/decision hash")
    index["gates"]["m6"] = review
    for row in composites:
        row["preview_only"] = False
        row["review_status"] = "accepted"
    index.update(training_ready=True, review_status="accepted", frame_valid=[True] * n,
                 required_reviews_passed=True, review_parent_index_ref=_ref(candidate_path),
                 review_receipt_ref=_ref(receipt_path))
    removable = {"S5_M3_REVIEW_PENDING", "S5_M5_REVIEW_PENDING", "S5_M6_REVIEW_PENDING", "S5_VISUAL_PREVIEW_ONLY", "S5_VISUAL_MISSING"}
    index["reason_codes"] = [code for code in index["reason_codes"] if code not in removable]
    _write_json(destination, index)
    return {"index_path": str(destination), "index_ref": _ref(destination), "index": index,
            "training_ready": True, "frame_valid": [True] * n, "review_status": "accepted"}


def prepare_visual(
    episode_root: Path | None,
    output_dir: Path,
    timestamps: np.ndarray,
    source_video_sha256: str,
    source_key: str,
    scene_depth_manifest: Path | None = None,
    external_artifacts_manifest: Path | None = None,
) -> dict[str, Any]:
    """Index existing native images and optionally compose supplied scene depth.

    Writes only output_dir/visual; source episode and legacy indexes are read-only.
    Missing depth never falls back to alpha compositing. All generated composite
    observations remain review-pending unless an existing M6 receipt can be
    validated by finalize_visual. No review is generated or inferred here.
    The caller may append independently extracted source_rgb records to index.
    """
    output = Path(output_dir) / "visual"
    root = Path(episode_root) if episode_root is not None else output / "__missing_source_episode__"
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be finite, strictly increasing source PTS")
    if not source_key or len(source_video_sha256) != 64:
        raise ValueError("source key and video SHA256 are required")
    if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError("output_dir must be outside the read-only source episode")
    output.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    records: list[dict[str, Any]] = []
    input_refs: list[dict[str, Any]] = []
    external_indices: dict[str, Path] = {}
    external_manifest_path = None
    if external_artifacts_manifest is not None:
        external_manifest_path = Path(external_artifacts_manifest).resolve()
        external = _read_json(external_manifest_path)
        if (external.get("schema_version") != "learning.stage5.visual_artifacts.v1" or
                external.get("source_key") != source_key or
                external.get("source_video_sha256") != source_video_sha256):
            raise ValueError("external visual artifacts source/schema mismatch")
        input_refs.append(_ref(external_manifest_path))
        for name, key in (("source", "source_index_ref"), ("human", "human_index_ref"), ("background", "background_index_ref")):
            if not external.get(key):
                raise ValueError(f"external visual artifacts missing {key}")
            path = _checked_ref(external_manifest_path.parent, external[key]).resolve()
            external_indices[name] = path
            # External indexes use absolute image paths, keeping episode-relative
            # historical indexes compatible without changing their base directory.
            for row in _read_json(path).get("frames", []):
                fields = ("raw_path", "refined_path") if name == "human" else ("path",)
                if any(not Path(row.get(field, "")).is_absolute() for field in fields):
                    raise ValueError("external visual index image paths must be absolute")
    episode_path = root / "episode.json"
    episode = _read_json(episode_path) if episode_path.is_file() else None
    if episode is None:
        reasons += ["S5_VISUAL_EPISODE_MISSING", "S5_VISUAL_MISSING", "S5_DEPTH_MISSING"]
        camera = None
    else:
        input_refs.append(_ref(episode_path))
        if episode.get("source", {}).get("sha256") != source_video_sha256:
            raise ValueError("episode source video SHA256 mismatch")
        source_video = _checked_ref(root, {"path": episode["source"]["video_uri"], "sha256": source_video_sha256})
        input_refs.append(_ref(source_video))
        hdf_sha = episode.get("source", {}).get("hdf5_sha256")
        if hdf_sha and hashlib.sha256((source_video_sha256 + ":" + hdf_sha).encode("ascii")).hexdigest() != source_key:
            raise ValueError("episode MP4/HDF5 source_key mismatch")
        if hdf_sha:
            source_hdf = _checked_ref(root, {"path": episode["source"]["hdf5_uri"], "sha256": hdf_sha})
            input_refs.append(_ref(source_hdf))
        episode_times = np.asarray(episode["timebase"].get("timestamps_s", []), dtype=np.float64)
        if episode_times.shape != timestamps.shape or not np.isfinite(episode_times).all() or np.any(np.abs(episode_times-timestamps) > TIME_TOLERANCE_S):
            raise ValueError("caller timestamps do not match episode's stored source PTS")
        camera = episode["camera"]
    width = int(camera["width"]) if camera else None
    height = int(camera["height"]) if camera else None
    sequence: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]]]] = {}
    sequence_paths: dict[str, Path] = {}
    if episode:
        for name, relative in [("source", "segmentation/source-index.json"),
                               ("human", "segmentation/bundle/index.json"),
                               ("background", "inpainting/bundle/index.json")]:
            sequence_paths[name] = external_indices.get(name, root / relative)
            index, rows = _sequence_index(root, relative, timestamps, source_video_sha256,
                                           external_path=external_indices.get(name))
            sequence[name] = (index, rows)
            if index:
                input_refs.append(_ref(sequence_paths[name]))
                if (index.get("width"), index.get("height")) != (width, height):
                    raise ValueError(f"{name} index is not in native camera pixels")
                preview = bool(index.get("provenance", {}).get("preview_only", False))
                evidence_kind = "real_source_annotation" if name == "source" else _legacy_model_evidence(index)
                if evidence_kind in {"mock", "unavailable"}:
                    reasons.append("S5_MOCK_ARTIFACT" if evidence_kind == "mock" else "S5_VISUAL_PROVENANCE_UNVERIFIED")
                    preview = True
                for frame, row in enumerate(rows):
                    kind = {"source": "source_rgb", "human": "human_arm_mask", "background": "arm_removed_background"}[name]
                    path_field, sha_field = ("refined_path", "refined_sha256") if name == "human" else ("path", "sha256")
                    records.append(_image_record(_resolve(root, row[path_field]), kind, frame, timestamps[frame],
                        width, height, source_key, evidence_kind,
                        preview, expected_sha=row[sha_field], valid=row.get("valid", True)))
        for frame, timestamp in enumerate(timestamps):
            for directory, kind in [("visual/robot", "robot_only_rgb"), ("masks/robot", "arm_body_mask"), ("masks/gripper", "gripper_mask")]:
                path = root / directory / f"frame-{frame:06d}.png"
                if path.is_file():
                    records.append(_image_record(path, kind, frame, timestamp, width, height,
                                                  source_key, "numeric_solver", False))
    by_frame_kind = {(r["frame_index"], r["image_kind"]): r for r in records}
    geometry_review = root / "quality/reviews/m3-video.json"
    m3_ref = _ref(geometry_review) if geometry_review.is_file() else None
    required_m3 = {str(root / "canonical/actions.npz"): _sha(root / "canonical/actions.npz")} if (root / "canonical/actions.npz").is_file() else {}
    gates = {"m3": _review(root, m3_ref, source_video_sha256, "episode_projection_video", required_m3),
             "m5": {"status": "pending", "reason": "episode-scoped M5 review not supplied"},
             "m6": {"status": "pending", "reason": "new composite requires user inspection"}}
    if gates["m3"]["status"] != "accepted":
        reasons.append("S5_M3_REVIEW_PENDING")
    if not sequence.get("background", (None, []))[0]:
        reasons.append("S5_BACKGROUND_MISSING")
    elif any(r["preview_only"] for r in records if r["image_kind"] == "arm_removed_background"):
        reasons.append("S5_VISUAL_PREVIEW_ONLY")
    if not sequence.get("human", (None, []))[0]:
        reasons.append("S5_HUMAN_ARM_MASK_MISSING")
    if len([r for r in records if r["image_kind"] == "robot_only_rgb"]) != len(timestamps):
        reasons.append("S5_ROBOT_RENDER_MISSING")
    alignment_records: list[dict[str, Any]] = []
    composite_frame_valid = [False] * len(timestamps)
    if scene_depth_manifest is None:
        reasons.append("S5_DEPTH_MISSING")
    elif not episode:
        reasons.append("S5_DEPTH_CAMERA_MISSING")
    else:
        manifest_path = Path(scene_depth_manifest).resolve()
        manifest = _read_json(manifest_path)
        input_refs.append(_ref(manifest_path))
        if manifest.get("schema_version") != SCENE_SCHEMA:
            raise ValueError(f"scene manifest schema must be {SCENE_SCHEMA}")
        if manifest.get("source_key") != source_key or manifest.get("source_video_sha256") != source_video_sha256:
            raise ValueError("scene depth source identity mismatch")
        if manifest.get("reviews", {}).get("m6"):
            # A receipt must refer to the immutable candidate it reviewed.
            # Regenerating after adding the receipt to the manifest would alter
            # the manifest/input hash and create an impossible circular hash.
            m6_reference = manifest["reviews"]["m6"]
            receipt_path = _checked_ref(manifest_path.parent, m6_reference)
            if not m6_reference.get("candidate_index_ref"):
                raise ValueError("M6 review requires candidate_index_ref; alternatively call finalize_visual directly")
            candidate_path = _checked_ref(manifest_path.parent, m6_reference["candidate_index_ref"])
            candidate = _read_json(candidate_path)
            if candidate.get("source_key") != source_key or candidate.get("source_video_sha256") != source_video_sha256:
                raise ValueError("M6 candidate source identity mismatch")
            reviewed = finalize_visual(candidate_path, receipt_path, output / "index-reviewed.json")
            return {**reviewed, "status": "succeeded", "reasons": reviewed["index"]["reason_codes"],
                    "input_refs": reviewed["index"]["input_refs"], "preview_only": False,
                    "observation_kind": "depth_composite_rgb", "evidence_kind": "real_reviewed_inputs"}
        if manifest.get("coordinate_frame") != "C_optical" or manifest.get("depth_axis") != "z":
            raise ValueError("scene depth must use camera optical Z, not ray distance")
        scene_camera = manifest.get("camera", {})
        if ((scene_camera.get("width"), scene_camera.get("height")) != (width, height) or
                not np.allclose(np.asarray(scene_camera.get("K", [])), np.asarray(camera["K"]), atol=1e-8, rtol=0)):
            raise ValueError("scene depth camera must match native episode K and dimensions; no implicit resize")
        measurement = manifest.get("measurement_provenance")
        if not measurement:
            raise ValueError("scene depth requires measurement/model provenance file reference")
        _checked_ref(manifest_path.parent, measurement)
        input_refs.append(_ref(_resolve(manifest_path.parent, measurement["path"])))
        bg_index_path = sequence_paths.get("background", root / "inpainting/bundle/index.json")
        human_index_path = sequence_paths.get("human", root / "segmentation/bundle/index.json")
        required_m5 = {str(p): _sha(p) for p in (bg_index_path, human_index_path) if p.is_file()}
        gates["m5"] = _review(manifest_path.parent, manifest.get("reviews", {}).get("m5"),
                               source_video_sha256, "episode_human_arm_removal", required_m5, source_key)
        if manifest.get("reviews", {}).get("m3"):
            gates["m3"] = _review(manifest_path.parent, manifest["reviews"]["m3"], source_video_sha256,
                                    "episode_projection_video", required_m3, source_key)
        rows = manifest.get("frames", [])
        indices = [r.get("frame_index") for r in rows]
        if any(not isinstance(i, int) or i < 0 or i >= len(timestamps) for i in indices) or indices != sorted(set(indices)):
            raise ValueError("scene depth frame indices must be unique increasing source indices")
        robot_depth_path = root / "depth/robot_depth_m.npz"
        if not robot_depth_path.is_file():
            reasons.append("S5_ROBOT_DEPTH_MISSING")
        else:
            # This optional path is the only place the potentially large render
            # depth array is loaded. Pure reuse-only indexing never loads it.
            input_refs.append(_ref(robot_depth_path))
            with np.load(robot_depth_path, allow_pickle=False) as arrays:
                robot_depth = arrays["depth_m"]
                if (robot_depth.shape != (len(timestamps), height, width) or robot_depth.dtype != np.float32 or
                        arrays["frame_index"].dtype != np.int64 or not np.array_equal(arrays["frame_index"], np.arange(len(timestamps)))):
                    raise ValueError("robot depth shape/index mismatch")
            for row in rows:
                frame = row["frame_index"]
                if not isinstance(row.get("valid"), bool):
                    raise ValueError("scene depth frame requires explicit boolean valid")
                if not row["valid"]:
                    reasons.append("S5_SCENE_DEPTH_FRAME_INVALID")
                    continue
                if abs(float(row["timestamp_s"]) - timestamps[frame]) > TIME_TOLERANCE_S or not np.isfinite(row["timestamp_s"]):
                    raise ValueError("scene depth PTS mismatch")
                raw_path = _checked_ref(manifest_path.parent, row)
                input_refs.append(_ref(raw_path))
                raw = np.load(raw_path, allow_pickle=False)
                if raw.shape != (height, width) or not np.issubdtype(raw.dtype, np.floating):
                    raise ValueError("scene depth frame shape mismatch")
                anchor_values = []
                if manifest["mode"] == "relative":
                    anchor_path = _checked_ref(manifest_path.parent, row["anchors_ref"])
                    input_refs.append(_ref(anchor_path))
                    anchor_json = _read_json(anchor_path)
                    if (anchor_json.get("source_key") != source_key or anchor_json.get("frame_index") != frame or
                            abs(float(anchor_json.get("timestamp_s", np.nan)) - timestamps[frame]) > TIME_TOLERANCE_S or
                            not np.isfinite(anchor_json.get("timestamp_s", np.nan))):
                        raise ValueError("anchor source/time binding mismatch")
                    anchor_values = anchor_json["anchors"]
                    for anchor in anchor_values:
                        evidence = _checked_ref(anchor_path.parent, anchor["evidence_ref"])
                        input_refs.append(_ref(evidence))
                        anchor["evidence_ref"] = str(evidence)
                confidence = None
                if row.get("confidence_ref"):
                    confidence_path = _checked_ref(manifest_path.parent, row["confidence_ref"])
                    input_refs.append(_ref(confidence_path))
                    confidence = np.load(confidence_path, allow_pickle=False)
                aligned = align_depth(raw, mode=manifest["mode"], quantity=manifest["quantity"],
                    evidence_kind=manifest["evidence_kind"], anchors=anchor_values,
                    max_rmse_m=manifest.get("alignment", {}).get("max_rmse_m"),
                    max_residual_m=manifest.get("alignment", {}).get("max_residual_m"),
                    confidence=confidence, min_confidence=manifest.get("min_confidence"))
                alignment = {"frame_index": frame, "timestamp_s": float(timestamps[frame]),
                    "mode": aligned.mode, "metric_valid": aligned.metric_valid,
                    "scale": aligned.scale, "shift": aligned.shift,
                    "residual_rmse_m": aligned.residual_rmse_m, "residual_max_m": aligned.residual_max_m,
                    "quantity": aligned.quantity, "evidence_kind": aligned.evidence_kind, **aligned.diagnostics}
                alignment_records.append(alignment)
                metric_path = output / "depth" / f"frame-{frame:06d}.npz"
                metric_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(metric_path, depth_m=aligned.depth_m, valid=aligned.valid_mask,
                                    frame_index=np.int64(frame), timestamp_s=np.float64(timestamps[frame]))
                alignment["depth_ref"] = _ref(metric_path)
                if not aligned.metric_valid:
                    reasons.append("S5_DEPTH_ALIGNMENT_FAILED")
                    continue
                kinds = ["arm_removed_background", "robot_only_rgb", "arm_body_mask", "gripper_mask", "human_arm_mask"]
                inputs = [by_frame_kind.get((frame, kind)) for kind in kinds]
                if any(record is None or not record["valid"] for record in inputs):
                    reasons.append("S5_COMPOSITE_INPUT_MISSING")
                    continue
                bg, robot, arm, gripper, human = [np.asarray(Image.open(record["path"])) for record in inputs]
                composed = compose_frame(bg, robot, arm, gripper, robot_depth[frame], aligned, human,
                    depth_epsilon_m=float(manifest.get("compositing", {}).get("depth_epsilon_m", 0.0)))
                composite_path = output / "composite" / f"frame-{frame:06d}.png"
                composite_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(composed.rgb).save(composite_path)
                diagnostic_path = output / "decisions" / f"frame-{frame:06d}.npz"
                diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(diagnostic_path, decision_map=composed.decision_map,
                    visible_robot_mask=composed.visible_robot_mask, occluded_gripper_mask=composed.occluded_gripper_mask,
                    comparison_valid_mask=composed.comparison_valid_mask, frame_index=np.int64(frame),
                    timestamp_s=np.float64(timestamps[frame]))
                alignment["composite_diagnostics"] = composed.diagnostics
                alignment["decision_ref"] = _ref(diagnostic_path)
                # Array validity and human approval remain independent.
                composite_frame_valid[frame] = composed.frame_valid
                record = _image_record(composite_path, "depth_composite_rgb", frame, timestamps[frame],
                    width, height, source_key, aligned.evidence_kind, True, valid=composed.frame_valid)
                record.update(depth_mode=aligned.mode, scale_alignment_frame=frame,
                              upstream_preview_only=any(r["preview_only"] for r in inputs),
                              metric_valid=aligned.metric_valid, decision_ref=_ref(diagnostic_path))
                records.append(record)
            del robot_depth
    if gates["m5"]["status"] != "accepted":
        reasons.append("S5_M5_REVIEW_PENDING")
    reasons.append("S5_M6_REVIEW_PENDING")
    if not any(r["image_kind"] == "depth_composite_rgb" for r in records):
        reasons.append("S5_VISUAL_MISSING")
    alignment_path = output / "alignment.json"
    _write_json(alignment_path, {"schema_version": "learning.stage5.depth_alignment.v1", "source_key": source_key,
                                 "source_video_sha256": source_video_sha256, "frames": alignment_records})
    records.sort(key=lambda r: (r["frame_index"], r["image_kind"]))
    # A bundle must not stop being readable when the live implementation changes.
    # Preserve the exact implementation bytes under a content-addressed filename.
    implementation_bytes = Path(__file__).read_bytes()
    implementation_digest = hashlib.sha256(implementation_bytes).hexdigest()
    implementation_snapshot = output / f"implementation-{implementation_digest[:20]}.py"
    if implementation_snapshot.is_file():
        if _sha(implementation_snapshot) != implementation_digest:
            raise ValueError("immutable visual implementation snapshot was altered")
    else:
        implementation_snapshot.write_bytes(implementation_bytes)
    index = {"schema_version": "learning.stage5.visual_index.v1", "adapter_version": ADAPTER_VERSION,
        "source_key": source_key, "source_video_sha256": source_video_sha256,
        "episode_ref": _ref(episode_path) if episode else None,
        "background_index_ref": _ref(sequence_paths["background"]) if sequence.get("background", (None, []))[0] else None,
        "external_artifacts_manifest_ref": _ref(external_manifest_path) if external_manifest_path else None,
        "width": width, "height": height, "camera": camera, "frames": records,
        "training_ready": False, "review_status": "pending", "gates": gates,
        "reason_codes": sorted(set(reasons)), "scale_alignment_ref": _ref(alignment_path),
        "composite_numerically_valid": composite_frame_valid,
        "frame_valid": [False] * len(timestamps),
        "coordinate_frame": "C_optical", "depth_axis": "z", "depth_unit": "m",
        "pixel_transform": "identity_native_camera_pixels; no implicit resize",
        "mask_conversion": "bool/0/1/255 -> bool; arm body and gripper disjoint",
        "color_conversion": "Pillow native RGB; BGR not accepted",
        "input_refs": input_refs, "model_execution": "none",
        "implementation_ref": _ref(implementation_snapshot),
        "implementation_source_path": str(Path(__file__).resolve()),
        "environment": {"numpy": np.__version__, "pillow": PIL.__version__},
        "storage_mode": "local_reference", "program_status": "succeeded"}
    index_path = output / "index.json"
    _write_json(index_path, index)
    result = {"index_path": str(index_path.resolve()), "index_ref": _ref(index_path), "index": index,
            "status": "succeeded", "reasons": index["reason_codes"], "input_refs": input_refs,
            "frame_valid": index["frame_valid"], "training_ready": False, "review_status": "pending",
            "observation_kind": "depth_composite_rgb" if alignment_records else "source_rgb",
            "evidence_kind": "unavailable" if not alignment_records else "mixed_explicit_inputs",
            "preview_only": True}
    return result
