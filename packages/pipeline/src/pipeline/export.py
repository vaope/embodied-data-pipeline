"""Streaming custom HDF5/JSONL export for validated Stage 5 bundles."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .reader import BundleValidationError, _jsonable, open_bundle, sha256_file

EXPORT_SCHEMA = "learning.stage5.hdf5.v1"


def resolve_export_mode(bundle_path: str | Path, mode: str) -> str:
    """Auto only selects train when the declared bundle is already ready.

    The train reader subsequently verifies all evidence; a ready header alone
    cannot bypass validation. Auto never edits readiness, reviews, or QC.
    """
    if mode not in {"inspect", "train", "auto"}:
        raise ValueError("export mode must be inspect, train, or auto")
    if mode != "auto":
        return mode
    supplied = Path(bundle_path)
    path = supplied / "bundle.json" if supplied.is_dir() else supplied
    bundle = json.loads(path.read_text(encoding="utf-8-sig"))
    return "train" if bundle.get("training_readiness") == "training_ready" else "inspect"


def _json_string(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _write_arrays(group: h5py.Group, arrays: dict[str, np.ndarray] | None) -> None:
    if arrays is None:
        group.attrs["availability"] = "missing"
        return
    group.attrs["availability"] = "present"
    for name, array in arrays.items():
        if array.dtype.kind in {"U", "S"}:
            if array.ndim == 0:
                group.attrs[name] = str(array.item())
            else:
                group.create_dataset(name, data=array.astype(object), dtype=h5py.string_dtype("utf-8"))
        elif array.ndim and array.size:
            group.create_dataset(name, data=array, compression="gzip", shuffle=True)
        else:
            group.create_dataset(name, data=array)


def export_bundle_jsonl(bundle_path: str | Path, output_path: str | Path,
                        mode: str = "inspect", *, overwrite: bool = False) -> dict[str, Any]:
    """Export frame metadata; large RGB arrays stay in the source/HDF5 view."""
    requested_mode, mode = mode, resolve_export_mode(bundle_path, mode)
    output = Path(output_path).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing frame metadata: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    try:
        with open_bundle(bundle_path, mode=mode) as reader, temporary.open("x", encoding="utf-8", newline="\n") as stream:
            training_ready = mode == "train" and reader.training_readiness == "training_ready"
            for frame in range(reader.num_frames):
                metadata = reader.frame_metadata(frame)
                metadata.update({"schema_version": "learning.stage5.frame_export.v1",
                                 "episode_id": reader.episode_id, "source_key": reader.bundle["source_key"],
                                 "export_mode": mode, "training_ready": training_ready,
                                 "source_training_readiness": reader.training_readiness,
                                 "readiness_reasons": reader.readiness_reasons,
                                 "task_meta": reader.task_meta,
                                 "supervision_profile": reader.supervision_profile,
                                 "supervision": reader.supervision,
                                 "object_associations": reader.object_associations,
                                 "action_profile": reader.action_profile,
                                 "has_outgoing_action": frame < reader.num_frames - 1,
                                 "source_video_ref": reader.source_video_ref,
                                 "source_hdf5_ref": reader.source_hdf5_ref})
                if frame < reader.num_frames - 1:
                    action = reader.transitions["action7"][frame]
                    metadata["outgoing_transition"] = {
                        "source_frame_index": frame, "target_frame_index": frame+1,
                        "transition_valid": bool(reader.transitions["transition_valid"][frame]),
                        "reason_codes": reader.reason_codes(int(reader.transitions["reason_bits"][frame])),
                        "action7": action.tolist() if np.isfinite(action).all() else None,
                        "action_available": bool(np.isfinite(action).all()),
                        "action_missing_reason": None if np.isfinite(action).all() else "non_finite_or_missing_geometry",
                    }
                else:
                    metadata["outgoing_transition"] = None
                stream.write(_json_string(metadata) + "\n")
            count = reader.num_frames
        temporary.replace(output)
        return {"schema_version": "learning.stage5.jsonl_export_manifest.v1", "mode": mode,
                "requested_mode": requested_mode,
                "training_ready": training_ready, "num_frames": count, "path": str(output),
                "sha256": sha256_file(output), "bytes": output.stat().st_size}
    finally:
        if temporary.is_file():
            temporary.unlink()


def export_bundle_hdf5(bundle_path: str | Path, output_path: str | Path,
                       mode: str = "inspect", *, overwrite: bool = False) -> dict[str, Any]:
    """Export all N images and N-1 action slots without padding the last action.

    Invalid slots are retained in inspect exports.  Training exports include
    only a valid-transition selection array; original frame and action slots
    remain immutable and invalid slots cannot be silently consumed.
    """
    requested_mode, mode = mode, resolve_export_mode(bundle_path, mode)
    output = Path(output_path).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace an existing export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_jsonl = output.with_suffix(".frames.jsonl")
    manifest_path = output.with_suffix(".manifest.json")
    for companion in (frame_jsonl, manifest_path):
        if companion.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace existing export metadata: {companion}")
    temporary = output.with_name(output.name + f".partial-{os.getpid()}")
    temporary_jsonl = frame_jsonl.with_name(frame_jsonl.name + f".partial-{os.getpid()}")
    try:
        with open_bundle(bundle_path, mode=mode) as reader:
            first_image = reader.get_image(0)
            h, w = first_image.shape[:2]
            training_ready = mode == "train" and reader.training_readiness == "training_ready"
            with h5py.File(temporary, "x") as target, temporary_jsonl.open("x", encoding="utf-8", newline="\n") as frames:
                target.attrs["schema_version"] = EXPORT_SCHEMA
                target.attrs["mode"] = mode
                target.attrs["training_ready"] = training_ready
                target.attrs["source_training_readiness"] = reader.training_readiness
                target.attrs["readiness_reasons_json"] = _json_string(reader.readiness_reasons)
                target.attrs["episode_id"] = reader.episode_id
                target.attrs["run_id"] = reader.bundle["run_id"]
                target.attrs["source_key"] = reader.bundle["source_key"]
                target.attrs["action_profile"] = reader.action_profile or "unresolved"
                target.attrs["supervision_profile"] = reader.supervision_profile
                target.attrs["supervision_claims_json"] = _json_string(reader.supervision["claims"])
                target.attrs["observation_kind"] = reader.image_kind
                target.attrs["color_space"] = "RGB"
                target.attrs["action_shape_semantics"] = "N-1 adjacent transitions; final observation has no outgoing action"
                target.attrs["custom_format"] = "Not LeRobot/RLDS; no speed resampling or normalized gripper"
                obs = target.create_group("observations")
                images = obs.create_dataset("image_rgb", shape=(reader.num_frames, h, w, 3), dtype=np.uint8,
                                            chunks=(1, min(h, 128), min(w, 128), 3), compression="gzip", compression_opts=1)
                for frame in range(reader.num_frames):
                    image = first_image if frame == 0 else reader.get_image(frame)
                    if image.shape != (h, w, 3):
                        raise BundleValidationError("Cannot export mixed image resolutions in one observation view")
                    images[frame] = image
                    metadata = reader.frame_metadata(frame)
                    metadata.update({"schema_version": "learning.stage5.frame_export.v1",
                                     "episode_id": reader.episode_id, "source_key": reader.bundle["source_key"],
                                     "export_mode": mode, "training_ready": training_ready,
                                     "supervision_profile": reader.supervision_profile,
                                     "hdf5_image_dataset": "/observations/image_rgb", "hdf5_image_index": frame,
                                     "has_outgoing_action": frame < reader.num_frames-1,
                                     "source_video_ref": reader.source_video_ref,
                                     "source_hdf5_ref": reader.source_hdf5_ref})
                    frames.write(_json_string(metadata) + "\n")
                _write_arrays(target.create_group("timebase"), reader.timebase)
                _write_arrays(target.create_group("hand"), reader.hand)
                _write_arrays(target.create_group("canonical"), reader.canonical)
                _write_arrays(target.create_group("robot"), reader.robot)
                _write_arrays(target.create_group("quality"), reader.quality)
                _write_arrays(target.create_group("transitions"), reader.transitions)
                selected = np.flatnonzero(reader.transitions["transition_valid"]).astype(np.int64)
                target.create_dataset("training_transition_indices", data=selected if training_ready else np.empty(0, dtype=np.int64))
                annotations = target.create_group("annotations")
                string_dtype = h5py.string_dtype("utf-8")
                source_group = target.create_group("source")
                source_group.create_dataset("metadata_json", data=_json_string(reader.source_metadata), dtype=string_dtype)
                for name, value in (("stage3_v1_json", reader.stage3), ("step_frame_map_json", reader.step_map),
                                    ("object_associations_json", reader.object_associations),
                                    ("task_meta_json", reader.task_meta), ("supervision_json", reader.supervision)):
                    annotations.create_dataset(name, data=_json_string(value), dtype=string_dtype)
                provenance = target.create_group("provenance")
                provenance.create_dataset("bundle_json", data=_json_string(reader.bundle), dtype=string_dtype)
                provenance.attrs["bundle_sha256"] = sha256_file(reader.bundle_path)
                provenance.attrs["bundle_path"] = str(reader.bundle_path)
                provenance.create_dataset("quality_report_json", data=_json_string(reader.quality_report), dtype=string_dtype)
                result = {"schema_version": "learning.stage5.export_manifest.v1", "mode": mode,
                          "requested_mode": requested_mode, "supervision_profile": reader.supervision_profile,
                          "training_ready": training_ready, "source_training_readiness": reader.training_readiness,
                          "readiness_reasons": reader.readiness_reasons, "episode_id": reader.episode_id,
                          "source_key": reader.bundle["source_key"], "bundle_path": str(reader.bundle_path),
                          "bundle_sha256": sha256_file(reader.bundle_path), "num_frames": reader.num_frames,
                          "num_transition_slots": len(reader), "num_valid_transitions": int(len(selected)),
                          "num_training_transition_indices": int(len(selected)) if training_ready else 0,
                          "observation_kind": reader.image_kind, "action_profile": reader.action_profile,
                          "custom_format": "learning.stage5.hdf5.v1; not LeRobot/RLDS"}
            temporary.replace(output)
            temporary_jsonl.replace(frame_jsonl)
        result["hdf5"] = {"path": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size}
        result["frame_jsonl"] = {"path": str(frame_jsonl), "sha256": sha256_file(frame_jsonl), "bytes": frame_jsonl.stat().st_size}
        manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        result["manifest_path"] = str(manifest_path)
        return result
    finally:
        for partial in (temporary, temporary_jsonl):
            if partial.is_file():
                partial.unlink()
