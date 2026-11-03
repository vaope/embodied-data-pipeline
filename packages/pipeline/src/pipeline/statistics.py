"""Offline, source-isolated L2 statistics with immutable calibration artifacts.

No model inference or online threshold fitting occurs here.  Calibration uses
only explicitly listed, hash-verified Stage 5 bundles and their real geometry
candidate transitions.  Diagnostic artifacts cannot confer training validity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .actions import ACTION_PROFILE, CAMERA_FRAME_SEMANTICS
from .common import read_json, ref, sha256, stable_hash, write_json

SCHEMA_VERSION = "learning.stage5.dataset_statistics.v1"
FEATURE_PROFILE = "camera_view_actions_and_panda_state_derivatives.v1"
APPLY_SCHEMA = "learning.stage5.l2_application.v1"
REASON_BITS = {
    "S5_L2_STATS_NOT_READY": 1 << 0,
    "S5_L2_SOURCE_GROUP_MISMATCH": 1 << 1,
    "S5_L2_GEOMETRY_INVALID": 1 << 2,
    "S5_L2_STENCIL_UNAVAILABLE": 1 << 3,
    "S5_L2_OUTLIER": 1 << 4,
    "S5_L2_EPISODE_INVALID_RATIO": 1 << 5,
}
_DIM7 = ("x", "y", "z", "rx", "ry", "rz", "gripper_width")
FEATURE_UNITS = {
    "action7": ["m", "m", "m", "rad", "rad", "rad", "m"],
    "action_velocity": ["m/s"] * 3 + ["rad/s"] * 3 + ["m/s"],
    "action_acceleration": ["m/s^2"] * 3 + ["rad/s^2"] * 3 + ["m/s^2"],
    "action_jerk": ["m/s^3"] * 3 + ["rad/s^3"] * 3 + ["m/s^3"],
    "joint_velocity": ["rad/s"] * 7,
    "joint_acceleration": ["rad/s^2"] * 7,
    "joint_jerk": ["rad/s^3"] * 7,
    "fk_residual": ["m", "rad"],
}


def _group_id(source_dataset: str, source: str, action_profile: str, robot_id: str) -> str:
    return stable_hash({"source_dataset": source_dataset, "source": source,
                        "action_profile": action_profile, "robot_id": robot_id})


def _stats_digest(statistics: dict) -> str:
    return stable_hash({key: value for key, value in statistics.items() if key != "statistics_id"})


def _as_matrix(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {array.shape}")
    return array


def _differentiate(values: np.ndarray, times: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trailing difference; invalid neighbours are never joined or bridged."""
    output = np.full_like(values, np.nan, dtype=np.float64)
    mask = np.zeros(len(values), dtype=bool)
    if len(values) > 1:
        denominator = np.diff(times)
        mask[1:] = valid[1:] & valid[:-1] & np.isfinite(denominator) & (denominator > 0)
        locations = np.flatnonzero(mask)
        output[locations] = (values[locations] - values[locations - 1]) / denominator[locations - 1, None]
        mask &= np.isfinite(output).all(axis=1)
    return output, mask


def compute_features(
    *, timestamp_s: np.ndarray, action7: np.ndarray,
    geometry_transition_valid: np.ndarray, bound_width_m: np.ndarray,
    qpos: np.ndarray, position_error_m: np.ndarray, rotation_error_rad: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Physical finite differences on original PTS; all rows retain source i.

    Action translations/rotvecs are camera-view deltas.  Their first six
    velocity channels divide by dt; width velocity differentiates the *bound*
    frame width rather than dividing the absolute seventh action channel.
    First derivatives live at transition midpoints; acceleration and jerk use
    actual distances between the successive midpoint-derived time samples.
    """
    ts = np.asarray(timestamp_s, dtype=np.float64)
    if ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError("timestamp_s must contain at least two finite increasing PTS")
    n, m = len(ts), len(ts) - 1
    action = _as_matrix(action7, (m, 7), "action7")
    width = _as_matrix(bound_width_m, (n,), "bound_width_m")
    state = _as_matrix(qpos, (n, 7), "qpos")
    pos = _as_matrix(position_error_m, (n,), "position_error_m")
    rot = _as_matrix(rotation_error_rad, (n,), "rotation_error_rad")
    baseline = np.asarray(geometry_transition_valid, dtype=bool)
    if baseline.shape != (m,):
        raise ValueError("geometry_transition_valid must have shape [N-1]")
    dt = np.diff(ts)
    midpoint = (ts[:-1] + ts[1:]) * 0.5
    velocity = np.c_[action[:, :6] / dt[:, None], np.diff(width) / dt]
    joint_velocity = np.diff(state, axis=0) / dt[:, None]
    features = {"action7": action.copy(), "action_velocity": velocity,
                "joint_velocity": joint_velocity,
                "fk_residual": np.c_[np.maximum(pos[:-1], pos[1:]), np.maximum(rot[:-1], rot[1:])]}
    masks = {name: baseline & np.isfinite(value).all(axis=1) for name, value in features.items()}
    for family in ("action", "joint"):
        acceleration, amask = _differentiate(features[family + "_velocity"], midpoint, masks[family + "_velocity"])
        acceleration_time = midpoint.copy()
        acceleration_time[1:] = (midpoint[1:] + midpoint[:-1]) * 0.5
        jerk, jmask = _differentiate(acceleration, acceleration_time, amask)
        features[family + "_acceleration"] = acceleration
        features[family + "_jerk"] = jerk
        masks[family + "_acceleration"] = amask
        masks[family + "_jerk"] = jmask
    # NaN is explicit missingness; thresholds never learn from a false row.
    for name in features:
        features[name][~masks[name]] = np.nan
    return features, masks


def _load_calibration_bundle(path: Path) -> dict[str, Any]:
    from .reader import open_bundle
    reader = open_bundle(path, mode="inspect")
    bundle = reader.bundle
    if reader.canonical is None or reader.robot is None:
        raise ValueError(f"L2 calibration requires real canonical and robot records: {path}")
    quality = reader.quality
    if "geometry_candidate_valid" not in quality or "l1_frame_valid" not in quality:
        raise ValueError(f"Per-frame geometry candidate/L1 evidence missing: {path}")
    frame_mask = quality["geometry_candidate_valid"].astype(bool) & quality["l1_frame_valid"].astype(bool)
    baseline = frame_mask[:-1] & frame_mask[1:]
    transition = reader.transitions
    if "geometry_candidate_valid" in transition:
        baseline &= transition["geometry_candidate_valid"]
    baseline &= np.isfinite(transition["action7"]).all(axis=1)
    episode_ref = bundle.get("refs", {}).get("episode")
    if not episode_ref or not episode_ref.get("path"):
        raise ValueError("Calibration requires the bound original episode manifest to identify the source dataset")
    def artifact_path(reference: dict) -> Path:
        target = Path(reference["path"])
        return target if target.is_absolute() else reader.root / target
    episode = read_json(artifact_path(episode_ref))
    source_record = episode.get("source", {})
    source_dataset = source_record.get("source_dataset")
    # A dataset source identifier is not an episode ID or a guessed filename.
    # Provider is a verified persisted field from the original hand record.
    hand_meta = {}
    hand_ref = bundle.get("refs", {}).get("hand", {})
    if hand_ref.get("path"):
        with np.load(artifact_path(hand_ref), allow_pickle=False) as archive:
            if "__meta_json__" in archive:
                hand_meta = json.loads(str(archive["__meta_json__"].item()))
    source = source_record.get("source") or hand_meta.get("provider")
    if not isinstance(source_dataset, str) or not source_dataset or not isinstance(source, str) or not source:
        raise ValueError("Explicit source_dataset and source/provider are required; unknown sources are not pooled")
    robot_id = bundle.get("robot_id")
    if reader.action_profile != ACTION_PROFILE or robot_id != "franka_panda":
        raise ValueError("This statistics feature profile supports only the declared camera-view Panda action profile")
    arrays = {
        "timestamp_s": reader.timestamp_s, "action7": transition["action7"],
        "geometry_transition_valid": baseline,
        "bound_width_m": np.sum(reader.robot["gripper_qpos"], axis=1),
        "qpos": reader.robot["qpos"], "position_error_m": reader.robot["position_error_m"][:, 0],
        "rotation_error_rad": reader.robot["rotation_error_rad"][:, 0],
    }
    features, masks = compute_features(**arrays)
    group_id = _group_id(source_dataset, source, reader.action_profile, robot_id)
    provenance = {"episode_id": bundle["episode_id"], "source_key": bundle["source_key"],
        "bundle_ref": ref(path), "source_video_ref": bundle["refs"]["source_video"],
        "source_hdf5_ref": bundle["refs"]["source_hdf5"], "episode_ref": episode_ref,
        "quality_frames_ref": bundle["quality"]["frames_ref"],
        "transitions_ref": bundle["views"]["transitions_ref"],
        "canonical_ref": bundle["refs"]["canonical"], "robot_ref": bundle["refs"]["robot"],
        "num_frames": reader.num_frames, "num_transitions": len(baseline),
        "geometry_l1_valid_transitions": int(baseline.sum()),
        "included_source_frame_indices": np.flatnonzero(baseline).tolist(),
        "included_target_frame_indices": (np.flatnonzero(baseline) + 1).tolist(),
        "timestamp_s_range": [float(reader.timestamp_s[0]), float(reader.timestamp_s[-1])],
        "source_dataset": source_dataset, "source": source,
        "source_identifier_provenance": "episode.source.source or hand.__meta_json__.provider",
        "group_id": group_id}
    reader.close()
    return {"group_id": group_id, "source_dataset": source_dataset, "source": source,
            "robot_id": robot_id, "action_profile": ACTION_PROFILE, "features": features,
            "masks": masks, "provenance": provenance}


def _distribution_plot(groups: dict[str, list[dict]], path: Path) -> None:
    rows = []
    for gid in sorted(groups):
        episodes = groups[gid]
        for feature in FEATURE_UNITS:
            rows.append((gid, episodes, feature))
    width, row_height = 1400, 150
    canvas = Image.new("RGB", (width, max(160, 45 + len(rows) * row_height)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), "Real geometry distributions; Q01/Q99 and 3*(Q99-Q01) bounds. Calibration readiness is declared in statistics JSON.", fill="black")
    for row_i, (_, episodes, name) in enumerate(rows):
        y = 45 + row_i * row_height
        values = np.concatenate([episode["features"][name][episode["masks"][name]] for episode in episodes], axis=0)
        title = f"{episodes[0]['source_dataset']} / {episodes[0]['source']} / {name} / samples={len(values)}"
        draw.text((15, y), title, fill="black")
        for dim in range(values.shape[1]):
            x = 20 + dim * 195
            column = values[:, dim]
            if not len(column):
                draw.text((x, y + 30), "no valid stencil", fill="red")
                continue
            qlo, qhi = np.quantile(column, [0.01, 0.99], method="linear")
            lo, hi = float(column.min()), float(column.max())
            hist, _ = np.histogram(column, bins=24, range=(lo, hi) if hi > lo else (lo - 1e-9, hi + 1e-9))
            maximum = max(1, int(hist.max()))
            for bar, count in enumerate(hist):
                bx = x + bar * 7
                height = int(60 * count / maximum)
                draw.rectangle((bx, y + 90 - height, bx + 5, y + 90), fill=(65, 120, 185))
            label = _DIM7[dim] if name.startswith("action") else f"joint{dim+1}" if name.startswith("joint") else ("position", "rotation")[dim]
            draw.text((x, y + 96), f"{label} [{FEATURE_UNITS[name][dim]}]", fill="black")
            draw.text((x, y + 111), f"Q01={qlo:.3g} Q99={qhi:.3g}", fill=(160, 35, 20))
            draw.text((x, y + 126), f"range {lo:.3g}..{hi:.3g}", fill=(50, 50, 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def fit_dataset_statistics(
    bundle_paths: Iterable[str | Path], output_path: str | Path,
    *, min_episodes: int, min_transitions: int, diagnostic_only: bool = False,
) -> dict[str, Any]:
    """Fit immutable Q01/Q99 thresholds from explicit, isolated real sources.

    The minimum episode count and minimum valid observations *per feature*
    are caller-supplied policy.  Diagnostic mode always emits not_ready even
    when counts meet that policy.  Existing differing output is never replaced.
    """
    if isinstance(min_episodes, bool) or min_episodes < 1 or int(min_episodes) != min_episodes:
        raise ValueError("min_episodes must be an explicit positive integer")
    if isinstance(min_transitions, bool) or min_transitions < 2 or int(min_transitions) != min_transitions:
        raise ValueError("min_transitions must be an explicit integer >= 2")
    paths = sorted({Path(p).resolve() / "bundle.json" if Path(p).is_dir() else Path(p).resolve() for p in bundle_paths}, key=str)
    if not paths:
        raise ValueError("Explicit bundle_paths must not be empty")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Statistics are immutable; choose a new versioned output path: {output}")
    groups: dict[str, list[dict]] = {}
    source_keys = set()
    for path in paths:
        episode = _load_calibration_bundle(path)
        key = episode["provenance"]["source_key"]
        if key in source_keys:
            raise ValueError(f"Repeated identical source_key in explicit calibration list: {key}")
        source_keys.add(key)
        groups.setdefault(episode["group_id"], []).append(episode)
    result_groups = {}
    for group_id, episodes in sorted(groups.items()):
        reasons = []
        if len(episodes) < min_episodes:
            reasons.append("minimum_episodes_not_met")
        parameters = {}
        for feature in FEATURE_UNITS:
            values = np.concatenate([ep["features"][feature][ep["masks"][feature]] for ep in episodes], axis=0)
            dimensions = []
            if len(values) < min_transitions:
                reasons.append(f"{feature}:minimum_valid_feature_observations_not_met")
            for dim, unit in enumerate(FEATURE_UNITS[feature]):
                column = values[:, dim]
                if len(column):
                    qlo, qhi = np.quantile(column, [0.01, 0.99], method="linear")
                    spread = float(qhi - qlo)
                    dimensions.append({"dimension": dim, "unit": unit, "count": len(column),
                        "q01": float(qlo), "q99": float(qhi), "q99_minus_q01": spread,
                        "lower": float(qlo - 3 * spread), "upper": float(qhi + 3 * spread)})
                else:
                    dimensions.append({"dimension": dim, "unit": unit, "count": 0,
                        "q01": None, "q99": None, "q99_minus_q01": None, "lower": None, "upper": None})
            parameters[feature] = {"count": len(values), "dimensions": dimensions}
        if diagnostic_only:
            reasons.append("diagnostic_only_requested")
        first = episodes[0]
        result_groups[group_id] = {"group_id": group_id, "source_dataset": first["source_dataset"],
            "source": first["source"], "action_profile": ACTION_PROFILE, "robot_id": first["robot_id"],
            "num_unique_episodes": len(episodes),
            "geometry_l1_valid_transitions": sum(ep["provenance"]["geometry_l1_valid_transitions"] for ep in episodes),
            "status": "not_ready" if reasons else "ready", "not_ready_reasons": reasons,
            "features": parameters, "input_episodes": [ep["provenance"] for ep in episodes]}
    image_path = output.with_name(output.stem + "-distributions.png")
    if image_path.exists():
        raise FileExistsError(f"Diagnostic image exists; choose a new statistics output version: {image_path}")
    _distribution_plot(groups, image_path)
    result = {"schema_version": SCHEMA_VERSION, "feature_profile": FEATURE_PROFILE,
        "status": "ready" if all(group["status"] == "ready" for group in result_groups.values()) else "not_ready",
        "diagnostic_only": bool(diagnostic_only), "calibration_policy": {"min_episodes": min_episodes, "min_transitions_per_feature": min_transitions},
        "threshold_origin": "offline empirical Q01/Q99 from explicitly listed L1/geometry-valid source transitions",
        "quantile_method": "numpy.quantile(method=linear)", "quantiles": [0.01, 0.99],
        "bounds_formula": "[Q01 - 3*(Q99-Q01), Q99 + 3*(Q99-Q01)]",
        "iqr_terminology": "spread is Q99-Q01, not conventional Q75-Q25",
        "feature_semantics": {"camera_frame_semantics": CAMERA_FRAME_SEMANTICS,
            "physical_control_claim": False, "timebase": "original strictly increasing PTS; no resampling",
            "action_velocity": "first six delta channels/dt; seventh is diff(recorded bound frame width)/dt",
            "state_velocity": "diff(recorded robot qpos)/dt",
            "acceleration_jerk": "adjacent midpoint finite differences; unavailable stencil remains NaN",
            "fk_residual": "maximum of source/target recorded FK position and rotation residuals",
            "training_use": "apply_statistics never replaces L1/visual/scoped review gates"},
        "episode_rejection": {"invalid_frame_ratio_gt": 0.60, "comparison": "strictly greater than",
            "mapping": "failed transition marks both endpoint frames; unavailable required derivative stencil is excluded"},
        "groups": result_groups, "plot_ref": ref(image_path, producer=SCHEMA_VERSION),
        "code_ref": ref(Path(__file__), producer=SCHEMA_VERSION),
        "numpy_version": np.__version__}
    result["statistics_id"] = _stats_digest(result)
    write_json(output, result)
    return result


def apply_statistics(
    *, timestamp_s: np.ndarray, action7: np.ndarray, geometry_transition_valid: np.ndarray,
    bound_width_m: np.ndarray, qpos: np.ndarray, position_error_m: np.ndarray,
    rotation_error_rad: np.ndarray, stats: dict | str | Path,
    source_dataset: str, source: str, action_profile: str = ACTION_PROFILE,
    robot_id: str = "franka_panda", geometry_frame_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Apply frozen source-matched thresholds without updating any statistic.

    Returns L2 masks and diagnostics, not overall training readiness.  A caller
    must AND the returned transition mask with its existing visual/review/QC
    masks.  Diagnostic or insufficient statistics always return zero valid.
    """
    statistics = read_json(Path(stats)) if isinstance(stats, (str, Path)) else stats
    if statistics.get("schema_version") != SCHEMA_VERSION or statistics.get("feature_profile") != FEATURE_PROFILE:
        raise ValueError("Unsupported statistics schema/feature profile")
    if statistics.get("statistics_id") != _stats_digest(statistics):
        raise ValueError("Statistics content digest mismatch")
    if statistics.get("code_ref", {}).get("sha256") != sha256(Path(__file__)):
        raise ValueError("Statistics feature implementation differs from current code; fit a new versioned calibration")
    ts = np.asarray(timestamp_s, dtype=np.float64)
    baseline = np.asarray(geometry_transition_valid, dtype=bool)
    features, feature_masks = compute_features(timestamp_s=ts, action7=action7,
        geometry_transition_valid=baseline, bound_width_m=bound_width_m, qpos=qpos,
        position_error_m=position_error_m, rotation_error_rad=rotation_error_rad)
    n, m = len(ts), len(ts) - 1
    reasons = np.zeros(m, dtype=np.uint64)
    reasons[~baseline] |= np.uint64(REASON_BITS["S5_L2_GEOMETRY_INVALID"])
    group = statistics.get("groups", {}).get(_group_id(source_dataset, source, action_profile, robot_id))
    available = group is not None and group.get("status") == "ready" and not statistics.get("diagnostic_only")
    if group is None:
        reasons[:] |= np.uint64(REASON_BITS["S5_L2_SOURCE_GROUP_MISMATCH"])
    if not available:
        reasons[:] |= np.uint64(REASON_BITS["S5_L2_STATS_NOT_READY"])
    outlier = np.zeros(m, bool)
    evaluated = baseline.copy() if available else np.zeros(m, bool)
    feature_outliers = {}
    for name, values in features.items():
        mask = feature_masks[name]
        evaluated &= mask
        rejected = np.zeros_like(values, dtype=bool)
        if available:
            dimensions = group["features"][name]["dimensions"]
            if len(dimensions) != values.shape[1]:
                raise ValueError(f"Threshold dimension mismatch for {name}")
            lower = np.asarray([dim["lower"] for dim in dimensions], dtype=float)
            upper = np.asarray([dim["upper"] for dim in dimensions], dtype=float)
            if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
                raise ValueError(f"Nonfinite or reversed frozen thresholds for {name}")
            rejected[mask] = (values[mask] < lower) | (values[mask] > upper)
            outlier |= rejected.any(axis=1)
        feature_outliers[name] = rejected
    reasons[baseline & ~evaluated] |= np.uint64(REASON_BITS["S5_L2_STENCIL_UNAVAILABLE"])
    reasons[outlier] |= np.uint64(REASON_BITS["S5_L2_OUTLIER"])
    valid = evaluated & ~outlier
    if geometry_frame_valid is None:
        frame_geometry = np.r_[baseline[0], baseline[:-1] & baseline[1:], baseline[-1]]
        frame_mask_source = "conservative reconstruction from adjacent geometry transitions"
    else:
        frame_geometry = np.asarray(geometry_frame_valid, dtype=bool)
        if frame_geometry.shape != (n,):
            raise ValueError("geometry_frame_valid must have shape [N]")
        if np.any(baseline & ~(frame_geometry[:-1] & frame_geometry[1:])):
            raise ValueError("Geometry transition claims invalid endpoint as valid")
        frame_mask_source = "provided original per-frame geometry candidate mask"
    frame_valid = frame_geometry.copy()
    rejected_transitions = ~valid
    frame_valid[:-1] &= ~rejected_transitions
    frame_valid[1:] &= ~rejected_transitions
    invalid_ratio = float(np.mean(~frame_valid)) if available else None
    episode_rejected = bool(invalid_ratio > 0.60) if available else None
    if episode_rejected:
        valid[:] = False
        reasons[:] |= np.uint64(REASON_BITS["S5_L2_EPISODE_INVALID_RATIO"])
    report = {"schema_version": APPLY_SCHEMA, "statistics_id": statistics["statistics_id"],
        "source_dataset": source_dataset, "source": source, "action_profile": action_profile,
        "status": "executed" if available else "not_evaluated", "thresholds_updated": False,
        "num_frames": n, "num_transitions": m, "geometry_valid_transitions": int(baseline.sum()),
        "l2_evaluated_transitions": int(evaluated.sum()), "l2_outlier_transitions": int(outlier.sum()),
        "l2_valid_transitions": int(valid.sum()), "invalid_frame_ratio": invalid_ratio,
        "episode_rejected": episode_rejected, "invalid_frame_ratio_threshold": 0.60,
        "frame_mask_source": frame_mask_source, "overall_training_readiness": "not_decided_by_L2",
        "reason_registry": REASON_BITS,
        "feature_evaluated_counts": {key: int(mask.sum()) for key, mask in feature_masks.items()},
        "feature_outlier_dimension_counts": {key: mask.sum(axis=0).tolist() for key, mask in feature_outliers.items()}}
    return {"transition_valid": valid, "frame_valid": frame_valid, "evaluated": evaluated,
            "reason_bits": reasons, "outlier": outlier, "features": features,
            "feature_masks": feature_masks, "feature_outliers": feature_outliers, "report": report}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit frozen source-isolated Stage 5 L2 statistics from explicit bundles")
    parser.add_argument("bundles", nargs="+", type=Path, help="Explicit bundle.json paths; no recursive discovery")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-episodes", required=True, type=int)
    parser.add_argument("--min-transitions", required=True, type=int, help="Minimum valid observations for every required feature")
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args(argv)
    result = fit_dataset_statistics(args.bundles, args.output, min_episodes=args.min_episodes,
        min_transitions=args.min_transitions, diagnostic_only=args.diagnostic_only)
    print(json.dumps({"path": str(args.output.resolve()), "statistics_id": result["statistics_id"],
        "status": result["status"], "groups": {key: {"status": value["status"],
            "num_unique_episodes": value["num_unique_episodes"],
            "geometry_l1_valid_transitions": value["geometry_l1_valid_transitions"],
            "not_ready_reasons": value["not_ready_reasons"]} for key, value in result["groups"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
