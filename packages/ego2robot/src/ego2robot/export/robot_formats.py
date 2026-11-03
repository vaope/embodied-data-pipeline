"""Small, deterministic exports for a reviewed Ego2Robot episode.

These adapters preserve the project's canonical schemas. They are deliberately
not presented as LeRobot or robomimic exporters.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..core.capabilities import require_hand_count_support

EXPORTER_VERSION = "1.0.0"
HDF5_SCHEMA = "ego2robot.export.canonical_hdf5.v1"
JSONL_SCHEMA = "ego2robot.export.frame_jsonl.v1"
MANIFEST_SCHEMA = "ego2robot.export_manifest.v1"


def _output_names(robot_id: str) -> tuple[str, ...]:
    return (
        f"canonical-{robot_id}.h5",
        "frames.jsonl",
        "representative-overlay.png",
        "field-comparison.png",
    )


def _gripper_raw_semantics(mode: str) -> str:
    if mode == "width_m":
        return (
            "Copied from canonical/actions-delta.npz::gripper_command. The resolved profile "
            "sets gripper_command_mode=width_m, so this is the source-frame physical jaw width "
            "in metres; it is provisional, not normalized or a standardized control command."
        )
    return (
        "Copied without transformation from canonical/actions-delta.npz::gripper_command. "
        f"The resolved profile mode is {mode!r}; no unit or normalized-control interpretation "
        "is inferred by this exporter."
    )


class TransitionRecord(TypedDict):
    frame_index_to: int
    timestamp_s_to: float
    delta_position_C_m: list[float]
    delta_rotvec_C_rad: list[float]
    gripper_channel_raw: dict[str, Any]


class FrameRecord(TypedDict):
    schema_version: str
    episode_id: str
    robot_id: str
    frame_index: int
    timestamp_s: float
    timebase_source: str
    canonical: dict[str, Any]
    robot: dict[str, Any]
    outgoing_transition: TransitionRecord | None


@dataclass(frozen=True, slots=True)
class EpisodeBundle:
    """Typed, aligned view of the existing episode artifacts."""

    episode_root: Path
    repo_root: Path
    episode_id: str
    instruction: str
    source_dataset: str
    source_license: str
    source_publishable: bool
    timebase_source: str
    nominal_fps: float
    frame_index: NDArray[np.int64]
    timestamp_s: NDArray[np.float64]
    camera_K: NDArray[np.float64]
    T_C_B: NDArray[np.float64]
    canonical: dict[str, NDArray[Any]]
    delta: dict[str, NDArray[Any]]
    robot: dict[str, NDArray[Any]]
    robot_id: str
    joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    joint_units: tuple[str, ...]
    gripper_joint_units: tuple[str, ...]
    T_E_TCP: NDArray[np.float64]
    robot_metadata: dict[str, Any]
    confidence_source: str
    gripper_command_mode: str
    input_paths: dict[str, Path]

    @property
    def num_frames(self) -> int:
        return int(self.frame_index.shape[0])

    @property
    def num_transitions(self) -> int:
        return max(self.num_frames - 1, 0)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


def _episode_path(episode_root: Path, uri: str, field_name: str) -> Path:
    candidate_uri = Path(uri)
    if candidate_uri.is_absolute():
        raise ValueError(f"episode source.{field_name} must be a relative episode path")
    root = episode_root.resolve()
    candidate = (root / candidate_uri).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"episode source.{field_name} escapes the episode directory")
    return candidate


def _joint_units_from_mjcf(model_path: Path, names: tuple[str, ...]) -> tuple[str, ...]:
    import xml.etree.ElementTree as ET

    root = ET.parse(model_path).getroot()
    class_type: dict[str, str] = {}
    for default in root.iter("default"):
        class_name = default.get("class")
        joint = default.find("joint")
        if class_name and joint is not None and joint.get("type"):
            class_type[class_name] = str(joint.get("type"))
    elements = {
        str(element.get("name")): element
        for element in root.iter("joint")
        if element.get("name") is not None
    }
    units: list[str] = []
    for name in names:
        element = elements.get(name)
        if element is None:
            raise ValueError(f"joint {name!r} is absent from the recorded MJCF")
        kind = element.get("type") or class_type.get(str(element.get("class")), "hinge")
        if kind == "hinge":
            units.append("rad")
        elif kind == "slide":
            units.append("m")
        else:
            raise ValueError(f"cannot describe qpos unit for MJCF joint {name!r} of type {kind!r}")
    return tuple(units)


def _need_arrays(name: str, arrays: dict[str, NDArray[Any]], required: tuple[str, ...]) -> None:
    missing = sorted(set(required) - arrays.keys())
    if missing:
        raise ValueError(f"{name} is missing arrays: {missing}")


def _load_bundle(episode_root: Path, repo_root: Path) -> EpisodeBundle:
    import yaml

    episode_root = episode_root.resolve()
    repo_root = repo_root.resolve()
    files = {
        "episode": episode_root / "episode.json",
        "resolved_config": episode_root / "configs" / "resolved.json",
        "exporter_module": Path(__file__).resolve(),
        "exporter_cli": repo_root / "tools" / "export_robot_formats.py",
    }
    for key, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"required {key} input is missing: {path}")
    episode = json.loads(files["episode"].read_text(encoding="utf-8"))
    resolved = json.loads(files["resolved_config"].read_text(encoding="utf-8"))
    source = episode["source"]
    robot_id = str(resolved.get("robot_binding", {}).get("robot_id", ""))
    if not robot_id or Path(robot_id).name != robot_id or any(sep in robot_id for sep in ("/", "\\")):
        raise ValueError(f"invalid robot_id in resolved config: {robot_id!r}")
    robot_dir = (episode_root / "robot" / robot_id).resolve()
    if not robot_dir.is_relative_to(episode_root):
        raise ValueError("resolved robot path escapes the episode directory")
    files.update(
        {
            "camera": episode_root / "source" / "camera.json",
            "source_video": _episode_path(episode_root, str(source.get("video_uri", "")), "video_uri"),
            "source_hdf5": _episode_path(episode_root, str(source.get("hdf5_uri", "")), "hdf5_uri"),
            "canonical": episode_root / "canonical" / "actions.npz",
            "delta": episode_root / "canonical" / "actions-delta.npz",
            "robot": robot_dir / "trajectory.npz",
            "base": robot_dir / "base.json",
            "hand_provenance": episode_root / "hand" / "provenance.json",
            "robot_profile": repo_root / "configs" / "robot" / f"{robot_id}.yaml",
            "decision": repo_root / "docs" / "decisions" / "0018-robot-format-trial.md",
        }
    )
    for key, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"required {key} input is missing: {path}")
    camera = json.loads(files["camera"].read_text(encoding="utf-8"))["camera"]
    base = json.loads(files["base"].read_text(encoding="utf-8"))
    hand_provenance = json.loads(files["hand_provenance"].read_text(encoding="utf-8"))
    robot_profile = yaml.safe_load(files["robot_profile"].read_text(encoding="utf-8"))["robot_binding"]
    if str(base.get("robot_id", "")) != robot_id or str(robot_profile.get("robot_id", "")) != robot_id:
        raise ValueError("resolved robot_id, base.json, and robot profile must agree")

    canonical = _load_npz(files["canonical"])
    delta = _load_npz(files["delta"])
    robot = _load_npz(files["robot"])
    _need_arrays(
        "canonical/actions.npz",
        canonical,
        (
            "timestamp_s", "frame_index", "T_C_TCP", "tcp_position_C_m",
            "tcp_quaternion_C_xyzw", "gripper_width_m", "confidence", "valid", "fallback_code",
        ),
    )
    _need_arrays(
        "canonical/actions-delta.npz",
        delta,
        ("delta_position_C_m", "delta_rotvec_C_rad", "gripper_command"),
    )
    _need_arrays(
        "robot/franka_panda/trajectory.npz",
        robot,
        (
            "timestamp_s", "frame_index", "qpos", "qvel", "gripper_qpos",
            "T_B_TCP_target", "T_B_TCP_fk", "position_error_m", "rotation_error_rad",
            "ik_success", "collision_code", "joint_limit_hit", "ik_iterations", "solver_status",
        ),
    )

    frame_index = canonical["frame_index"].astype(np.int64, copy=False)
    timestamp_s = canonical["timestamp_s"].astype(np.float64, copy=False)
    n_frames = int(frame_index.shape[0])
    if n_frames < 2:
        raise ValueError(f"format trial requires at least two frames; got {n_frames}")
    if frame_index.ndim != 1 or timestamp_s.shape != (n_frames,):
        raise ValueError("canonical frame_index and timestamp_s must be 1D and have equal lengths")
    if not np.all(np.diff(frame_index) == 1):
        raise ValueError("canonical frame_index must be contiguous for outgoing transitions")
    if not np.all(np.isfinite(timestamp_s)) or not np.all(np.diff(timestamp_s) > 0.0):
        raise ValueError("canonical timestamps must be finite and strictly increasing")

    source_pts = np.asarray(episode["timebase"]["timestamps_s"], dtype=np.float64)
    if source_pts.shape != timestamp_s.shape or not np.array_equal(source_pts, timestamp_s):
        raise ValueError("canonical timestamps do not exactly match source PTS in episode.json")
    if int(episode["timebase"]["num_frames"]) != n_frames:
        raise ValueError("canonical frame count does not match the source timebase manifest")
    if not np.array_equal(robot["frame_index"], frame_index):
        raise ValueError("Franka trajectory frame_index does not match canonical frame_index")
    if not np.array_equal(robot["timestamp_s"], timestamp_s):
        raise ValueError("Franka trajectory timestamp_s does not match canonical source PTS")
    if delta["delta_position_C_m"].shape[0] != n_frames - 1:
        raise ValueError("relative position deltas must map to exactly T-1 transitions")
    for name in ("delta_position_C_m", "delta_rotvec_C_rad", "gripper_command"):
        if delta[name].shape[0] != n_frames - 1:
            raise ValueError(f"{name} must have one row per source-frame transition")

    T_C_B = np.asarray(base["T_C_B"], dtype=np.float64)
    robot_meta = json.loads(str(robot["__meta_json__"].item()))
    if T_C_B.shape != (4, 4):
        raise ValueError("base.json T_C_B must be a 4x4 transform")
    if "T_C_B" in robot_meta and not np.allclose(T_C_B, np.asarray(robot_meta["T_C_B"]), atol=1e-12):
        raise ValueError("base.json T_C_B differs from the transform recorded with Franka trajectory")
    if np.asarray(camera["K"], dtype=np.float64).shape != (3, 3):
        raise ValueError("camera K must be a 3x3 matrix")

    dof = int(robot_profile["dof"])
    gripper_dof = int(robot_profile["gripper_dof"])
    joint_names = tuple(str(name) for name in robot_profile["joint_names"])
    gripper_joint_names = tuple(str(name) for name in robot_profile["gripper_joint_names"])
    if robot["qpos"].shape != (n_frames, dof) or len(joint_names) != dof:
        raise ValueError(f"{robot_id} qpos shape/order disagrees with the robot profile")
    if robot["gripper_qpos"].shape != (n_frames, gripper_dof) or len(gripper_joint_names) != gripper_dof:
        raise ValueError(f"{robot_id} gripper qpos shape/order disagrees with the robot profile")
    if "T_E_TCP" not in robot_meta:
        raise ValueError("robot trajectory metadata is missing required T_E_TCP")
    T_E_TCP = np.asarray(robot_meta["T_E_TCP"], dtype=np.float64)
    if T_E_TCP.shape != (4, 4):
        raise ValueError("robot trajectory metadata T_E_TCP must be 4x4")
    model_path = (repo_root / str(robot_profile["mjcf_uri"])).resolve()
    if not model_path.is_relative_to(repo_root):
        raise ValueError("robot profile MJCF path escapes the repository")
    if not robot_meta.get("mjcf_sha256"):
        raise ValueError("robot trajectory metadata is missing the MJCF hash required to identify qpos order")
    if not model_path.is_file() or _sha256_file(model_path) != robot_meta["mjcf_sha256"]:
        raise ValueError("current robot MJCF hash differs from the model used by the trajectory")
    files["robot_model"] = model_path

    for source_key, file_key in (("sha256", "source_video"), ("hdf5_sha256", "source_hdf5")):
        expected = str(source.get(source_key, ""))
        if expected and _sha256_file(files[file_key]) != expected:
            raise ValueError(f"source file hash does not match episode.json: {source_key}")

    return EpisodeBundle(
        episode_root=episode_root,
        repo_root=repo_root,
        episode_id=str(episode["episode_id"]),
        instruction=str(episode.get("task", {}).get("instruction", "")),
        source_dataset=str(source.get("source_dataset", "unknown")),
        source_license=str(source.get("license", "unknown")),
        source_publishable=bool(source.get("publishable", False)),
        timebase_source=str(episode["timebase"].get("timebase_source", "unknown")),
        nominal_fps=float(episode["timebase"]["nominal_fps"]),
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        camera_K=np.asarray(camera["K"], dtype=np.float64),
        T_C_B=T_C_B,
        canonical=canonical,
        delta=delta,
        robot=robot,
        robot_id=str(base["robot_id"]),
        joint_names=joint_names,
        gripper_joint_names=gripper_joint_names,
        joint_units=_joint_units_from_mjcf(files["robot_model"], joint_names),
        gripper_joint_units=_joint_units_from_mjcf(files["robot_model"], gripper_joint_names),
        T_E_TCP=T_E_TCP,
        robot_metadata=robot_meta,
        confidence_source=str(hand_provenance.get("confidence_source", "unspecified")),
        gripper_command_mode=str(resolved.get("robot_binding", {}).get("gripper_command_mode", "unspecified")),
        input_paths=files,
    )


def _input_hashes(bundle: EpisodeBundle) -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in sorted(bundle.input_paths.items())}


def _config(bundle: EpisodeBundle) -> dict[str, Any]:
    representative_index = bundle.num_frames // 2
    return {
        "exporter_version": EXPORTER_VERSION,
        "adr": "docs/decisions/0018-robot-format-trial.md",
        "schemas": {"hdf5": HDF5_SCHEMA, "jsonl": JSONL_SCHEMA},
        "outputs": list(_output_names(bundle.robot_id)),
        "representative_frame_index": int(bundle.frame_index[representative_index]),
        "representative_frame_selection": "middle decoded video frame",
        "transition_alignment": "outgoing from frame t to frame t+1 using source PTS",
        "gripper_command_mode_from_resolved_config": bundle.gripper_command_mode,
        "compatibility_claim": "project-local formats only; no LeRobot or robomimic compatibility asserted",
    }


def _output_manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def _resume_manifest(output_dir: Path, fingerprint: str, output_names: tuple[str, ...]) -> dict[str, Any] | None:
    path = _output_manifest_path(output_dir)
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "succeeded" or manifest.get("input_fingerprint") != fingerprint:
            return None
        for entry in manifest.get("outputs", []):
            artifact = output_dir / str(entry["path"])
            if not artifact.is_file():
                return None
            if artifact.stat().st_size != int(entry["size_bytes"]):
                return None
            if _sha256_file(artifact) != entry["sha256"]:
                return None
        produced = {str(entry["path"]) for entry in manifest.get("outputs", [])}
        if produced != set(output_names):
            return None
        return manifest
    except (KeyError, OSError, ValueError, TypeError):
        return None


def _json_value(array: NDArray[Any], index: int, hand_index: int = 0) -> Any:
    value = np.asarray(array[index, hand_index])
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def _frame_record(bundle: EpisodeBundle, index: int) -> FrameRecord:
    transition: TransitionRecord | None = None
    if index < bundle.num_transitions:
        transition = {
            "frame_index_to": int(bundle.frame_index[index + 1]),
            "timestamp_s_to": float(bundle.timestamp_s[index + 1]),
            "delta_position_C_m": _json_value(bundle.delta["delta_position_C_m"], index),
            "delta_rotvec_C_rad": _json_value(bundle.delta["delta_rotvec_C_rad"], index),
            "gripper_channel_raw": {
                "value": float(_json_value(bundle.delta["gripper_command"], index)[0]),
                "unit": "m" if bundle.gripper_command_mode == "width_m" else "unspecified",
                "source_field": "canonical/actions-delta.npz::gripper_command",
                "source_frame_index": int(bundle.frame_index[index]),
                "mode": bundle.gripper_command_mode,
                "semantic_status": _gripper_raw_semantics(bundle.gripper_command_mode),
            },
        }
    return {
        "schema_version": JSONL_SCHEMA,
        "episode_id": bundle.episode_id,
        "robot_id": bundle.robot_id,
        "frame_index": int(bundle.frame_index[index]),
        "timestamp_s": float(bundle.timestamp_s[index]),
        "timebase_source": bundle.timebase_source,
        "canonical": {
            "hand_index": 0,
            "T_C_TCP": _json_value(bundle.canonical["T_C_TCP"], index),
            "tcp_position_C_m": _json_value(bundle.canonical["tcp_position_C_m"], index),
            "tcp_quaternion_C_xyzw": _json_value(bundle.canonical["tcp_quaternion_C_xyzw"], index),
            "gripper_width_m": float(_json_value(bundle.canonical["gripper_width_m"], index)),
            "confidence": float(_json_value(bundle.canonical["confidence"], index)),
            "valid": bool(_json_value(bundle.canonical["valid"], index)),
            "fallback_code": int(_json_value(bundle.canonical["fallback_code"], index)),
            "confidence_source": bundle.confidence_source,
        },
        "robot": {
            "qpos": bundle.robot["qpos"][index].tolist(),
            "qpos_joint_names": list(bundle.joint_names),
            "qpos_units": list(bundle.joint_units),
            "qvel": bundle.robot["qvel"][index].tolist(),
            "qvel_units": [f"{unit}/s" for unit in bundle.joint_units],
            "gripper_qpos": bundle.robot["gripper_qpos"][index].tolist(),
            "gripper_qpos_joint_names": list(bundle.gripper_joint_names),
            "gripper_qpos_units": list(bundle.gripper_joint_units),
            "T_B_TCP_target": _json_value(bundle.robot["T_B_TCP_target"], index),
            "T_B_TCP_fk": _json_value(bundle.robot["T_B_TCP_fk"], index),
            "position_error_m": float(_json_value(bundle.robot["position_error_m"], index)),
            "rotation_error_rad": float(_json_value(bundle.robot["rotation_error_rad"], index)),
            "ik_success": bool(_json_value(bundle.robot["ik_success"], index)),
            "collision_code": int(bundle.robot["collision_code"][index]),
        },
        "outgoing_transition": transition,
    }


def _write_jsonl(path: Path, bundle: EpisodeBundle) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index in range(bundle.num_frames):
            stream.write(_canonical_json(_frame_record(bundle, index)))
            stream.write("\n")


def _write_dataset(group: Any, name: str, values: NDArray[Any], **attrs: Any) -> Any:
    dataset = group.create_dataset(name, data=np.asarray(values), track_times=False)
    for key, value in attrs.items():
        dataset.attrs[key] = value
    return dataset


def _write_hdf5(path: Path, bundle: EpisodeBundle, fingerprint: str) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency message
        raise RuntimeError("HDF5 export needs h5py; install the project data extra") from exc

    with h5py.File(path, "w", libver="earliest", track_order=True) as root:
        root.attrs["schema_version"] = HDF5_SCHEMA
        root.attrs["format_claim"] = "project-local; not a LeRobot or robomimic dataset"
        root.attrs["episode_id"] = bundle.episode_id
        root.attrs["input_fingerprint"] = fingerprint

        episode = root.create_group("episode", track_order=True)
        episode.attrs["instruction"] = bundle.instruction
        episode.attrs["source_dataset"] = bundle.source_dataset
        episode.attrs["source_license"] = bundle.source_license
        episode.attrs["publishable"] = bundle.source_publishable
        episode.attrs["num_frames"] = bundle.num_frames
        episode.attrs["num_transitions"] = bundle.num_transitions

        timebase = root.create_group("timebase", track_order=True)
        timebase.attrs["source"] = bundle.timebase_source
        timebase.attrs["clock"] = "source video PTS in seconds"
        timebase.attrs["nominal_fps"] = bundle.nominal_fps
        _write_dataset(timebase, "frame_index", bundle.frame_index)
        _write_dataset(timebase, "timestamp_s", bundle.timestamp_s, unit="s")

        canonical = root.create_group("canonical", track_order=True)
        canonical.attrs["schema_version"] = "ego2robot.canonical_action.v1"
        canonical.attrs["hand_axis_size"] = 1
        _write_dataset(canonical, "T_C_TCP", bundle.canonical["T_C_TCP"], target_frame="C", source_frame="TCP", translation_unit="m")
        _write_dataset(canonical, "tcp_position_C_m", bundle.canonical["tcp_position_C_m"], frame="C", unit="m")
        _write_dataset(canonical, "tcp_quaternion_C_xyzw", bundle.canonical["tcp_quaternion_C_xyzw"], frame="C", order="xyzw")
        _write_dataset(canonical, "gripper_width_m", bundle.canonical["gripper_width_m"], unit="m", meaning="physical jaw opening width")
        for name in ("confidence", "valid", "fallback_code"):
            dataset = _write_dataset(canonical, name, bundle.canonical[name])
            if name == "confidence":
                dataset.attrs["source"] = bundle.confidence_source
                if bundle.confidence_source == "no_confidences_group":
                    dataset.attrs["meaning"] = "stored 1.0 is an annotation-presence placeholder, not measured detection confidence"
                else:
                    dataset.attrs["meaning"] = "see manifest hand_confidence_source"
        if "so3_correction_rad" in bundle.canonical:
            _write_dataset(canonical, "so3_correction_rad", bundle.canonical["so3_correction_rad"], unit="rad")

        transitions = root.create_group("transitions", track_order=True)
        transitions.attrs["num_transitions"] = bundle.num_transitions
        transitions.attrs["alignment"] = "row t maps source frame t to source frame t+1"
        transitions.attrs["position_delta_definition"] = "tcp_position_C_m[t+1] - tcp_position_C_m[t]"
        transitions.attrs["rotation_delta_definition"] = "Log_SO3(R_C_TCP[t+1] @ transpose(R_C_TCP[t]))"
        _write_dataset(transitions, "frame_index_from", bundle.frame_index[:-1])
        _write_dataset(transitions, "frame_index_to", bundle.frame_index[1:])
        _write_dataset(transitions, "timestamp_s_from", bundle.timestamp_s[:-1], unit="s")
        _write_dataset(transitions, "timestamp_s_to", bundle.timestamp_s[1:], unit="s")
        _write_dataset(transitions, "delta_position_C_m", bundle.delta["delta_position_C_m"], frame="C", unit="m")
        _write_dataset(transitions, "delta_rotvec_C_rad", bundle.delta["delta_rotvec_C_rad"], frame="C", unit="rad")
        gripper_dataset = _write_dataset(
            transitions,
            "gripper_channel_raw",
            bundle.delta["gripper_command"],
            unit="m" if bundle.gripper_command_mode == "width_m" else "unspecified",
            source_field="canonical/actions-delta.npz::gripper_command",
            alignment=f"source frame t; resolved gripper_command_mode={bundle.gripper_command_mode}",
            semantic_status="provisional; not normalized or a standardized control command",
        )
        gripper_dataset.attrs["explanation"] = _gripper_raw_semantics(bundle.gripper_command_mode)

        robot = root.create_group(f"robot/{bundle.robot_id}", track_order=True)
        robot.attrs["robot_id"] = bundle.robot_id
        robot.attrs["joint_names_json"] = _canonical_json(list(bundle.joint_names))
        robot.attrs["gripper_joint_names_json"] = _canonical_json(list(bundle.gripper_joint_names))
        robot.attrs["qpos_units_json"] = _canonical_json(list(bundle.joint_units))
        robot.attrs["gripper_qpos_units_json"] = _canonical_json(list(bundle.gripper_joint_units))
        robot.attrs["mjcf_sha256"] = str(bundle.robot_metadata.get("mjcf_sha256", ""))
        robot.attrs["pose_variable"] = "T_B_TCP"
        _write_dataset(robot, "T_C_B", bundle.T_C_B, target_frame="C", source_frame="B", translation_unit="m")
        _write_dataset(robot, "T_E_TCP", bundle.T_E_TCP, target_frame="E", source_frame="TCP", translation_unit="m")
        _write_dataset(robot, "frame_index", bundle.frame_index)
        _write_dataset(robot, "timestamp_s", bundle.timestamp_s, unit="s")
        for name, unit in (
            ("qpos", ",".join(bundle.joint_units)),
            ("qvel", ",".join(f"{unit}/s" for unit in bundle.joint_units)),
            ("gripper_qpos", ",".join(bundle.gripper_joint_units)),
            ("position_error_m", "m"),
            ("rotation_error_rad", "rad"),
            ("T_B_TCP_target", "m translation; rotation dimensionless"),
            ("T_B_TCP_fk", "m translation; rotation dimensionless"),
        ):
            _write_dataset(robot, name, bundle.robot[name], unit=unit)
        for name in ("ik_success", "collision_code", "joint_limit_hit", "ik_iterations", "solver_status"):
            _write_dataset(robot, name, bundle.robot[name])


def _load_font(size: int) -> Any:
    from PIL import ImageFont

    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _decode_representative_frame(bundle: EpisodeBundle, frame_index: int) -> NDArray[np.uint8]:
    import cv2

    video_path = bundle.input_paths["source_video"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video for preview: {video_path}")
    selected: NDArray[np.uint8] | None = None
    decoded = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if decoded == frame_index:
                selected = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            decoded += 1
    finally:
        capture.release()
    if decoded != bundle.num_frames:
        raise ValueError(f"video decoded {decoded} frames, expected {bundle.num_frames}")
    if selected is None:
        raise ValueError(f"representative frame {frame_index} was not decoded")
    return selected


def _project(K: NDArray[np.float64], p_C: NDArray[np.float64]) -> tuple[float, float] | None:
    if not np.all(np.isfinite(p_C)) or float(p_C[2]) <= 0.0:
        return None
    u = float(K[0, 0] * p_C[0] / p_C[2] + K[0, 2])
    v = float(K[1, 1] * p_C[1] / p_C[2] + K[1, 2])
    return u, v


def _write_overlay(path: Path, bundle: EpisodeBundle, frame_index: int) -> None:
    from PIL import Image, ImageDraw

    rgb = _decode_representative_frame(bundle, frame_index)
    image = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    width, height = image.size
    index = int(np.flatnonzero(bundle.frame_index == frame_index)[0])
    target_T_C_TCP = bundle.canonical["T_C_TCP"][index, 0]
    fk_T_C_TCP = bundle.T_C_B @ bundle.robot["T_B_TCP_fk"][index, 0]
    target_uv = _project(bundle.camera_K, target_T_C_TCP[:3, 3])
    fk_uv = _project(bundle.camera_K, fk_T_C_TCP[:3, 3])
    if target_uv is None or fk_uv is None:
        raise ValueError("representative target/FK TCP is behind the camera or non-finite")

    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(30)
    small = _load_font(24)
    tx, ty = target_uv
    fx, fy = fk_uv
    marker_radius = 15
    draw.line((tx, ty, fx, fy), fill=(248, 180, 35, 230), width=5)
    draw.line((tx - marker_radius, ty, tx + marker_radius, ty), fill=(255, 128, 0, 255), width=5)
    draw.line((tx, ty - marker_radius, tx, ty + marker_radius), fill=(255, 128, 0, 255), width=5)
    draw.ellipse((fx - marker_radius, fy - marker_radius, fx + marker_radius, fy + marker_radius), outline=(20, 220, 255, 255), width=5)

    header_height = 104
    draw.rectangle((0, 0, width, header_height), fill=(0, 0, 0, 180))
    draw.text((28, 16), f"Format trial · frame {frame_index}/{bundle.num_frames - 1} · PTS {bundle.timestamp_s[index]:.4f} s", font=font, fill=(255, 255, 255, 255))
    draw.text((28, 60), f"Orange + : canonical target TCP     Cyan circle: {bundle.robot_id} FK TCP", font=small, fill=(255, 255, 255, 255))

    footer_y = max(0, height - 58)
    draw.rectangle((0, footer_y, width, height), fill=(0, 0, 0, 165))
    draw.text((28, footer_y + 12), "Projection uses source K and stored T_C_B · preview evidence only", font=small, fill=(255, 255, 255, 255))
    image.convert("RGB").save(path, format="PNG", optimize=False, compress_level=6)


def _wrap_by_pixel_width(draw: Any, text: str, font: Any, max_width: int) -> str:
    lines: list[str] = []
    line = ""
    for character in text:
        candidate = line + character
        box = draw.textbbox((0, 0), candidate, font=font)
        if line and box[2] - box[0] > max_width:
            lines.append(line)
            line = character
        else:
            line = candidate
    if line:
        lines.append(line)
    return "\n".join(lines)


def _write_field_comparison(path: Path, bundle: EpisodeBundle) -> None:
    from PIL import Image, ImageDraw

    width, height = 1640, 1040
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(42)
    header_font = _load_font(27)
    body_font = _load_font(24)
    note_font = _load_font(21)

    draw.text((56, 34), f"{bundle.num_frames} 帧状态记录与 {bundle.num_transitions} 步相邻帧增量", font=title_font, fill=(24, 34, 52))
    draw.text((58, 92), f"{bundle.episode_id} · {bundle.robot_id} · PTS {bundle.timestamp_s[0]:.3f}–{bundle.timestamp_s[-1]:.3f} s", font=body_font, fill=(70, 83, 101))

    x0, x1, x2, x3 = 56, 450, 1035, 1584
    top = 152
    row_h = 97
    arm_units = bundle.joint_units[0] if len(set(bundle.joint_units)) == 1 else "mixed; see metadata"
    gripper_units = (
        bundle.gripper_joint_units[0]
        if bundle.gripper_joint_units and len(set(bundle.gripper_joint_units)) == 1
        else ("mixed; see metadata" if bundle.gripper_joint_units else "none")
    )
    hdf5_name = _output_names(bundle.robot_id)[0]
    headers = ((x0, x1, "字段含义"), (x1, x2, hdf5_name), (x2, x3, "frames.jsonl"))
    for left, right, text in headers:
        draw.rounded_rectangle((left, top, right, top + 58), radius=12, fill=(30, 73, 117))
        draw.text((left + 18, top + 12), text, font=header_font, fill=(255, 255, 255))

    rows = [
        ("状态样本", f"{bundle.num_frames} 帧；保留原 frame_index 与 source PTS", "每帧一行；frame_index / timestamp_s"),
        ("绝对 TCP", f"/canonical/T_C_TCP [{bundle.num_frames},1,4,4]；米", "canonical.T_C_TCP；4×4，C←TCP"),
        ("夹爪开度", f"/canonical/gripper_width_m [{bundle.num_frames},1]；物理米", "canonical.gripper_width_m；物理米"),
        (f"{bundle.robot_id} 状态", f"qpos [{bundle.num_frames},{len(bundle.joint_names)}] {arm_units}; gripper_qpos [{bundle.num_frames},{len(bundle.gripper_joint_names)}] {gripper_units}", "robot.qpos / gripper_qpos；关节名与单位随记录提供"),
        ("逐步位姿增量", f"/transitions/delta_* [{bundle.num_transitions},1,*]；t→t+1", f"前 {bundle.num_transitions} 行 outgoing_transition；末帧为 null"),
        ("夹爪通道原值", "/transitions/gripper_channel_raw；附单位与来源语义", "gripper_channel_raw；标出来源与 provisional 语义"),
    ]
    for row, values in enumerate(rows):
        y = top + 72 + row * row_h
        fill = (255, 255, 255) if row % 2 == 0 else (234, 239, 245)
        draw.rounded_rectangle((x0, y, x3, y + row_h - 8), radius=10, fill=fill)
        draw.multiline_text((x0 + 18, y + 16), _wrap_by_pixel_width(draw, values[0], header_font, x1 - x0 - 36), font=header_font, spacing=2, fill=(32, 47, 67))
        draw.multiline_text((x1 + 18, y + 13), _wrap_by_pixel_width(draw, values[1], body_font, x2 - x1 - 36), font=body_font, spacing=3, fill=(45, 58, 77))
        draw.multiline_text((x2 + 18, y + 13), _wrap_by_pixel_width(draw, values[2], body_font, x3 - x2 - 36), font=body_font, spacing=3, fill=(45, 58, 77))

    note_y = top + 72 + len(rows) * row_h + 3
    draw.rounded_rectangle((x0, note_y, x3, height - 40), radius=14, fill=(255, 244, 220), outline=(222, 170, 65), width=2)
    notes = [
        "gripper_command 的论文映射仍未决（ADR-0010）：导出保留原值和 resolved mode，不把字段名解释为控制命令。",
        "两种输出均是项目通用 schema；不宣称可由 LeRobot 或 robomimic 原生读取。视频不嵌入本次格式文件。",
    ]
    for i, note in enumerate(notes):
        draw.text((x0 + 24, note_y + 18 + i * 36), note, font=note_font, fill=(85, 63, 23))
    image.save(path, format="PNG", optimize=False, compress_level=6)


def _write_atomic(temp_path: Path, destination: Path, writer: Any) -> None:
    if temp_path.exists():
        temp_path.unlink()
    writer(temp_path)
    os.replace(temp_path, destination)


def _require_exportable_hand_count(episode_root: Path) -> None:
    """Refuse an H>1 canonical sequence before touching any robot artifact."""
    actions = episode_root / "canonical" / "actions.npz"
    if not actions.is_file():
        return
    with np.load(actions, allow_pickle=False) as data:
        if "valid" not in data.files:
            return
        require_hand_count_support("export", int(data["valid"].shape[1]))


def export_robot_formats(
    episode_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    force: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write custom HDF5 and JSONL views plus review images and a hash manifest."""

    source_root = Path(episode_root)
    resolved_repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    # Check the hand arity *before* loading the rest of the bundle: an H=2
    # canonical sequence would produce a 14-D action next to a 7-D robot record
    # (one robot group, hand_index 0 throughout), and the refusal must not depend
    # on unrelated robot artifacts being present.
    _require_exportable_hand_count(source_root)
    bundle = _load_bundle(source_root, resolved_repo)
    require_hand_count_support("export", int(np.asarray(bundle.canonical["valid"]).shape[1]))
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (bundle.episode_root / "previews" / "format-trial").resolve()
    )
    preview_root = (bundle.episode_root / "previews").resolve()
    if not destination.is_relative_to(preview_root):
        raise ValueError("format-trial output must remain under the episode previews directory")
    destination.mkdir(parents=True, exist_ok=True)

    input_hashes = _input_hashes(bundle)
    config = _config(bundle)
    output_names = _output_names(bundle.robot_id)
    config_hash = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    fingerprint = hashlib.sha256(
        _canonical_json({"inputs": input_hashes, "config_hash": config_hash}).encode("utf-8")
    ).hexdigest()
    if not force:
        cached = _resume_manifest(destination, fingerprint, output_names)
        if cached is not None:
            return {**cached, "resume_hit": True}

    representative_index = int(config["representative_frame_index"])
    frame_index = int(bundle.frame_index[representative_index])
    hdf5_path = destination / output_names[0]
    jsonl_path = destination / output_names[1]
    overlay_path = destination / output_names[2]
    comparison_path = destination / output_names[3]
    temp_paths = [path.with_name(path.name + ".tmp") for path in (hdf5_path, jsonl_path, overlay_path, comparison_path)]
    try:
        _write_atomic(temp_paths[0], hdf5_path, lambda path: _write_hdf5(path, bundle, fingerprint))
        _write_atomic(temp_paths[1], jsonl_path, lambda path: _write_jsonl(path, bundle))
        _write_atomic(temp_paths[2], overlay_path, lambda path: _write_overlay(path, bundle, frame_index))
        _write_atomic(temp_paths[3], comparison_path, lambda path: _write_field_comparison(path, bundle))
    except Exception:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()
        raise

    outputs = [
        {"path": path.name, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
        for path in (hdf5_path, jsonl_path, overlay_path, comparison_path)
    ]
    input_entries = []
    for name, path in sorted(bundle.input_paths.items()):
        try:
            stable_path = path.relative_to(bundle.repo_root).as_posix()
        except ValueError:
            try:
                stable_path = f"episode://{path.relative_to(bundle.episode_root).as_posix()}"
            except ValueError:
                stable_path = f"repo://{path.name}"
        input_entries.append({"name": name, "path": stable_path, "sha256": input_hashes[name]})

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "exporter": "ego2robot.export.robot_formats",
        "exporter_version": EXPORTER_VERSION,
        "status": "succeeded",
        "resume_hit": False,
        "episode_id": bundle.episode_id,
        "robot_id": bundle.robot_id,
        "input_fingerprint": fingerprint,
        "config_hash": config_hash,
        "config": config,
        "inputs": input_entries,
        "source": {
            "dataset": bundle.source_dataset,
            "license": bundle.source_license,
            "publishable": bundle.source_publishable,
        },
        "hand_confidence_source": {
            "source": bundle.confidence_source,
            "meaning": (
                "stored 1.0 values are placeholders because the source HDF5 has no /confidences group; "
                "they are not measured detection confidence"
                if bundle.confidence_source == "no_confidences_group"
                else "see hand/provenance.json for the provider-specific confidence semantics"
            ),
        },
        "timebase": {
            "source": bundle.timebase_source,
            "num_frames": bundle.num_frames,
            "num_transitions": bundle.num_transitions,
            "timestamp_start_s": float(bundle.timestamp_s[0]),
            "timestamp_end_s": float(bundle.timestamp_s[-1]),
            "frame_index_start": int(bundle.frame_index[0]),
            "frame_index_end": int(bundle.frame_index[-1]),
        },
        "gripper_channel": {
            "source_field": "canonical/actions-delta.npz::gripper_command",
            "mode": bundle.gripper_command_mode,
            "unit": "m" if bundle.gripper_command_mode == "width_m" else "unspecified",
            "meaning": _gripper_raw_semantics(bundle.gripper_command_mode),
            "paper_normalization_status": "unresolved; see ADR-0010",
        },
        "format_claims": {
            "hdf5": f"project-local canonical + {bundle.robot_id} trajectory schema; not robomimic-compatible",
            "jsonl": "project-local frame-record schema; not LeRobot-compatible",
        },
        "outputs": outputs,
    }
    manifest_tmp = _output_manifest_path(destination).with_name("manifest.json.tmp")
    manifest_tmp.write_text(_canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")
    os.replace(manifest_tmp, _output_manifest_path(destination))
    return manifest
