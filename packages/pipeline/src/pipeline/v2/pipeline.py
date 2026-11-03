"""Pipeline V2 vertical slice: robot query -> Ego candidates -> selection -> export.

This module implements the parts of V2.1 that can run on this machine with data
already on disk. Anything requiring model weights we do not have (EgoANT / Flora),
CUDA extensions without nvcc (DROID-SLAM), a >12 GB GPU, or a training system is
reported as ``blocked`` with an explicit reason instead of being faked.

Run:
    python -m pipeline.v2 doctor  --out <dir>
    python -m pipeline.v2 catalog --droid-root <dir> --out <dir>
    python -m pipeline.v2 run     --droid-root <dir> --ego-root <dir> --query 0 --top-k 50 --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get('EDP_WORKSPACE', Path.cwd())).expanduser().resolve()
SCHEMA_VERSION = "learning.v2.run.v1"
ELIGIBILITY = (
    "captioning",
    "retrieval_text",
    "retrieval_motion",
    "video_only",
    "robot_action_supervision",
    "robotization_candidate",
)
REASON_REGISTRY = (
    "no_robot_action_gt",
    "missing_motion_feature",
    "missing_hdf5_pose",
    "missing_intrinsics",
    "insufficient_hand_visibility",
    "scale_ambiguous",
    "camera_pose_drift",
    "pose_jitter",
    "contact_unverified",
    "retarget_ik_fail",
    "weights_missing",
    "nvcc_missing",
    "vram_insufficient",
    "training_system_missing",
)


# --------------------------------------------------------------------------- utils
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_config() -> dict:
    import yaml  # PyYAML is a retrieval dependency

    import retrieval
    path = Path(os.environ['EDP_RETRIEVAL_CONFIG']).expanduser().resolve() if os.environ.get('EDP_RETRIEVAL_CONFIG') else Path(retrieval.__file__).parent / 'config.yaml'
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- semantic
def embed_texts(texts: list[str], config: dict) -> np.ndarray:
    """Same model/revision as the existing Ego index, so vectors are comparable."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from retrieval.features.semantic_encoder import SemanticEncoder

    encoder = SemanticEncoder(
        {
            "semantic_model": config["semantic_model"],
            "semantic_revision": config["semantic_revision"],
            "semantic_chunk_tokens": config["semantic_chunk_tokens"],
            "batch_size": config["batch_size"],
            "device": config["device"],
        }
    )
    return encoder.encode(texts).astype(np.float32)


# --------------------------------------------------------------------------- droid
def droid_episodes(root: Path) -> list[dict]:
    csv_path = root / "episodes.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def droid_query(root: Path, index: int) -> dict:
    episodes = droid_episodes(root)
    if not 0 <= index < len(episodes):
        raise SystemExit(f"query index {index} out of range 0..{len(episodes) - 1}")
    row = episodes[index]
    instruction = (row.get("language_instruction") or "").strip()
    if not instruction:
        raise SystemExit(f"DROID episode {index} has no language_instruction")
    steps_path = root / f"episode_{index:03d}" / "steps.csv"
    eef = {}
    if steps_path.is_file():
        with steps_path.open(encoding="utf-8-sig", newline="") as stream:
            steps = list(csv.DictReader(stream))
        pos = np.array([[float(s[f"action_cartesian_position_{i}"]) for i in range(3)] for s in steps], dtype=np.float64)
        grip = np.array([float(s["action_gripper_position"]) for s in steps], dtype=np.float64)
        delta = np.diff(pos, axis=0) if len(pos) > 1 else np.zeros((1, 3))
        eef = {
            "steps": len(steps),
            "duration_s": float(row.get("approx_seconds_at_15hz") or 0.0),
            "positions_action_cartesian": "6D pose columns action_cartesian_position_0..5 (commanded)",
            "path_length_m": float(np.linalg.norm(delta, axis=1).sum()),
            "max_step_m": float(np.linalg.norm(delta, axis=1).max()),
            "gripper_min": float(grip.min()),
            "gripper_max": float(grip.max()),
            "action_origin": "recorded_robot_command",
        }
    return {
        "source_dataset": "droid_100",
        "source_episode_id": f"droid_{index:03d}",
        "input_profile": "robot_native",
        "instruction": instruction,
        "extra_instructions": [row.get(k) for k in ("language_instruction_2", "language_instruction_3") if row.get(k)],
        "shard": row.get("shard"),
        "source_file_path": row.get("source_file_path"),
        "eef": eef,
        "capabilities": {
            "annotation_runner": "rental_pilot_pending: python -m pipeline.annotation",
            "has_rgb": None,
            "has_joint_state": steps_path.is_file(),
            "has_robot_command": steps_path.is_file(),
            "has_camera_intrinsics": None,
            "has_extrinsics": None,
            "has_hand_pose": False,
            "has_depth": None,
        },
        "evidence_kind": "real_source_annotation",
    }


# --------------------------------------------------------------------------- ego index
def ego_index(root: Path) -> dict:
    manifest = root / "manifest.jsonl"
    embeddings = root / "semantic_embeddings.npy"
    if not manifest.is_file() or not embeddings.is_file():
        raise SystemExit(f"Ego index incomplete under {root}")
    array = np.load(embeddings, mmap_mode="r")
    shape = tuple(int(v) for v in array.shape)
    return {
        "manifest": manifest,
        "embeddings": embeddings,
        "windows": shape[0],
        "dim": shape[1],
        "manifest_bytes": manifest.stat().st_size,
        "embeddings_sha256": sha256_file(embeddings),
        "manifest_sha256": sha256_file(manifest),
    }


def top_k(sims: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sims.size)
    part = np.argpartition(-sims, k - 1)[:k]
    return part[np.argsort(-sims[part])]


def collect_rows(manifest: Path, wanted: set[int]) -> dict[int, dict]:
    found: dict[int, dict] = {}
    with manifest.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("embedding_index")
            if index in wanted:
                found[index] = row
                if len(found) == len(wanted):
                    break
    return found


def ego_eligibility(row: dict) -> tuple[list[str], list[str]]:
    labels = ["video_only", "retrieval_text", "captioning"]
    reasons: list[str] = []
    hands = row.get("valid_hands") or {}
    if row.get("motion_path") and any(bool(v) for v in hands.values()):
        labels.append("retrieval_motion")
    else:
        reasons.append("missing_motion_feature")
    if row.get("hdf5_path"):
        labels.append("robotization_candidate")
    else:
        reasons.append("missing_hdf5_pose")
    reasons.append("no_robot_action_gt")
    return labels, sorted(set(reasons))


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def select(candidates: list[dict], max_total: int, max_per_task: int, max_per_video: int = 2,
           overlap: float = 0.5) -> tuple[list[dict], dict]:
    """Keep temporally distinct windows per source video, cap per video/task/total."""
    per_video: dict[str, list[tuple[float, float]]] = {}
    per_task: dict[str, int] = {}
    selected: list[dict] = []
    removed = {"overlap": 0, "video_cap": 0, "task_cap": 0, "max_total": 0}
    for candidate in candidates:
        if len(selected) >= max_total:
            removed["max_total"] += 1
            continue
        video_key = (candidate.get("source_hashes") or {}).get("video_path") or candidate["clip_id"]
        interval = (float(candidate["start_time"]), float(candidate["end_time"]))
        windows = per_video.setdefault(video_key, [])
        if any(_iou(interval, existing) > overlap for existing in windows):
            removed["overlap"] += 1
            continue
        if len(windows) >= max_per_video:
            removed["video_cap"] += 1
            continue
        task = candidate.get("task_text") or ""
        if per_task.get(task, 0) >= max_per_task:
            removed["task_cap"] += 1
            continue
        windows.append(interval)
        per_task[task] = per_task.get(task, 0) + 1
        selected.append(candidate)
    return selected, removed


# --------------------------------------------------------------------------- export
def to_sample(row: dict, query: dict, score: float, rank: int) -> dict:
    labels, reasons = ego_eligibility(row)
    return {
        "schema_version": "unified_sample_v1.draft",
        "sample_id": f"ego:{row['clip_id']}",
        "parent_sample_id": query["source_episode_id"],
        "granularity": "window",
        "source_type": "human",
        "source_dataset": "egodex",
        "source_episode_id": row["episode_id"],
        "segment": {"start_s": row["start_time"], "end_s": row["end_time"], "timebase": "pts"},
        "input_profile": "ego_structured",
        "geometry_level": "sensor_grounded" if row.get("hdf5_path") else "rgb_only",
        "action_origin": "missing",
        "modalities": ["rgb", "hand_pose"] + (["camera_pose"] if row.get("hdf5_path") else []),
        "labels": {"task": row.get("task_text"), "step": row.get("step_text"), "objects_visible": []},
        "motion": {
            "original_frame": "wrist_local",
            "reference_frame": "camera",
            "units": "m",
            "pose_representation": "quaternion_wxyz",
            "sampling_hz": row.get("fps"),
            "trajectory_ref": row.get("motion_path"),
        },
        "quality": {
            "source_video": "present",
            "reconstruction": {"status": "not_required", "reason": "source provides hand pose"},
            "geometric_scale": "known",
            "source_pose": "provided_annotation",
            "conversion_kinematics": "not_applicable",
            "conversion_semantics": "not_applicable",
            "contact_proxy": "not_applicable",
        },
        "usage_eligibility": labels,
        "reason_codes": reasons,
        "provenance": {
            "annotation_version": row.get("annotation_version"),
            "task_source": "official_instruction",
            "feature_version": row.get("feature_version"),
            "selection": {"query_episode": query["source_episode_id"], "rank": rank, "similarity": round(float(score), 6)},
        },
    }


def export_unified(samples: list[dict], out: Path) -> dict:
    unified = out / "unified"
    if unified.exists():
        shutil.rmtree(unified)
    unified.mkdir(parents=True)
    written = []
    for i, sample in enumerate(samples):
        path = unified / f"sample-{i:06d}.json"
        dump(path, sample)
        written.append({"path": str(path), "sha256": sha256_file(path)})
    dump(out / "unified" / "index.json", {"schema_version": "unified_sample_v1.draft", "samples": written})
    return {"count": len(written), "files": written}


def export_lerobot_draft(samples: list[dict], out: Path) -> dict:
    """LeRobot-shaped JSONL layout. Draft only: not the official LeRobot format."""
    root = out / "lerobot_draft"
    if root.exists():
        shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    frames = []
    for i, sample in enumerate(samples):
        episode = {
            "episode_index": i,
            "tasks": [sample["labels"].get("task")],
            "length": 1,
            "observation.state": None,
            "action": None,
            "source_sample_id": sample["sample_id"],
            "source_episode_id": sample["source_episode_id"],
            "start_s": sample["segment"]["start_s"],
            "end_s": sample["segment"]["end_s"],
            "eligibility": sample["usage_eligibility"],
        }
        frames.append(episode)
    dump_jsonl(data_dir / "episode-000000.jsonl", frames)
    dump(
        root / "meta" / "info.json",
        {
            "schema_version": "lerobot_draft.v0",
            "official_lerobot": False,
            "note": "Layout draft only. actions/observations are null because these Ego windows have no robot action GT.",
            "total_episodes": len(frames),
            "fps": None,
            "features": {"task": "string", "source_sample_id": "string"},
        },
    )
    return {"episodes": len(frames), "root": str(root)}


def verify_export(out: Path, selected_count: int) -> dict:
    unified = out / "unified" / "index.json"
    if not unified.is_file():
        raise SystemExit("unified index missing")
    index = json.loads(unified.read_text(encoding="utf-8"))
    missing, mismatched = [], []
    for row in index["samples"]:
        path = Path(row["path"])
        if not path.is_file():
            missing.append(str(path))
        elif sha256_file(path) != row["sha256"]:
            mismatched.append(str(path))
    chunk = out / "lerobot_draft" / "data" / "chunk-000" / "episode-000000.jsonl"
    loaded = sum(1 for line in chunk.read_text(encoding="utf-8").splitlines() if line.strip()) if chunk.is_file() else 0
    result = {
        "unified_samples": len(index["samples"]),
        "expected_selected": selected_count,
        "missing": missing,
        "hash_mismatch": mismatched,
        "lerobot_draft_episodes": loaded,
        "ok": not missing and not mismatched and len(index["samples"]) == selected_count and loaded == selected_count,
    }
    dump(out / "verify_export.json", result)
    return result


# --------------------------------------------------------------------------- doctor
def run_checked(command: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(command, capture_output=True, timeout=60)
        raw = (proc.stdout or b"") + (proc.stderr or b"")
        return proc.returncode == 0, raw.decode("utf-8", errors="replace").strip()[:500]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def doctor(out: Path, droid_root: Path, ego_root: Path) -> dict:
    has_nvcc, nvcc_text = run_checked(["bash", "-lc", "which nvcc"]) if platform.system() != "Windows" else run_checked(
        ["wsl", "-e", "bash", "-lc", "which nvcc"]
    )
    has_smi, smi_text = run_checked(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    vram_gib = None
    import re

    match = re.search(r"(\d+)\s*MiB", smi_text)
    if match:
        vram_gib = round(int(match.group(1)) / 1024, 2)
    model_root = Path(os.environ.get('EDP_MODEL_ROOT', WORKSPACE / 'models')).expanduser().resolve()
    weights = {
        "qwen3_6_27b": model_root / 'Qwen3.6-27B',
        "flora_captioner": model_root / 'Flora-Captioner',
        "flora_tempseg": model_root / 'Flora-TempSeg',
        "egoant_code": Path(__file__).parents[1] / 'annotation/vendor/egoant',
    }
    weight_state = {name: {"present": path.exists(), "path": str(path)} for name, path in weights.items()}
    try:
        import pyarrow  # noqa: F401

        pyarrow_ok = True
    except Exception:  # noqa: BLE001
        pyarrow_ok = False
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "platform": {"system": platform.system(), "python": sys.version.split()[0]},
        "gpu": {"nvidia_smi": has_smi, "detail": smi_text[:200], "vram_gib": vram_gib},
        "cuda_toolkit": {"nvcc": has_nvcc, "detail": nvcc_text[:200]},
        "weights": weight_state,
        "python_packages": {"pyarrow": pyarrow_ok},
        "datasets": {
            "droid_root": {"path": str(droid_root), "present": (droid_root / "episodes.csv").is_file()},
            "ego_index": {"path": str(ego_root), "present": (ego_root / "semantic_embeddings.npy").is_file()},
        },
        "capabilities": {
            "stage2b_recovery": "blocked" if (not has_nvcc or (vram_gib or 0) < 24) else "planned",
            "annotation_egoant": "planned" if weight_state["qwen3_6_27b"]["present"] else "blocked",
            "annotation_flora": "blocked" if not weight_state["flora_captioner"]["present"] else "planned",
            "lerobot_official": "blocked" if not pyarrow_ok else "planned",
        },
        "reason_codes": [code for code, cond in [
            ("nvcc_missing", not has_nvcc),
            ("vram_insufficient", (vram_gib or 0) < 24),
            ("weights_missing", not weight_state["qwen3_6_27b"]["present"]),
            ("training_system_missing", True),
        ] if cond],
    }
    dump(out / "doctor.json", report)
    return report


# --------------------------------------------------------------------------- run
def run(droid_root: Path, ego_root: Path, query_index: int, top_k_n: int, max_total: int, max_per_task: int,
        max_per_video: int, out: Path) -> dict:
    began = time.monotonic()
    config = load_config()
    index = ego_index(ego_root)
    query = droid_query(droid_root, query_index)
    vector = embed_texts([query["instruction"]], config)[0]
    embeddings = np.load(index["embeddings"], mmap_mode="r")
    sims = np.asarray(embeddings @ vector, dtype=np.float32)
    order = top_k(sims, top_k_n)
    rows = collect_rows(index["manifest"], set(int(i) for i in order))
    candidates = []
    for rank, embedding_index in enumerate(order.tolist()):
        row = rows.get(embedding_index)
        if row is None:
            continue
        candidates.append(
            {
                "clip_id": row["clip_id"],
                "episode_id": row["episode_id"],
                "task_text": row.get("task_text"),
                "step_text": row.get("step_text"),
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "fps": row.get("fps"),
                "valid_hands": row.get("valid_hands"),
                "motion_path": row.get("motion_path"),
                "hdf5_path": row.get("hdf5_path"),
                "video_path": row.get("video_path"),
                "source_hashes": row.get("source_hashes"),
                "annotation_version": row.get("annotation_version"),
                "feature_version": row.get("feature_version"),
                "similarity": float(sims[embedding_index]),
                "embedding_index": embedding_index,
                "rank": rank,
            }
        )
    selected, removed = select(candidates, max_total=max_total, max_per_task=max_per_task, max_per_video=max_per_video)
    samples = [to_sample(row, query, row["similarity"], row["rank"]) for row in selected]
    out.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out / "retrieved_candidates.jsonl", candidates)
    dump_jsonl(out / "selected_manifest.jsonl", selected)
    unified = export_unified(samples, out)
    lerobot = export_lerobot_draft(samples, out)
    verify = verify_export(out, len(samples))
    eligibility_counts: dict[str, int] = {}
    for sample in samples:
        for label in sample["usage_eligibility"]:
            eligibility_counts[label] = eligibility_counts.get(label, 0) + 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "method": "text-only cross-embodiment retrieval baseline (MiniLM, same model/revision as Ego index)",
        "query": query,
        "ego_index": {k: v for k, v in index.items() if k not in {"manifest", "embeddings"}},
        "counts": {
            "requested_top_k": top_k_n,
            "retrieved": len(candidates),
            "selected": len(samples),
            "unique_source_episodes": len({row["episode_id"] for row in selected}),
            "unique_source_videos": len({(row["source_hashes"] or {}).get("video_path") for row in selected}),
        },
        "selection": {"max_total": max_total, "max_per_task": max_per_task, "max_per_video": max_per_video, "removed": removed},
        "eligibility_counts": eligibility_counts,
        "exports": {"unified": unified["count"], "lerobot_draft": lerobot["episodes"]},
        "verify": verify,
        "elapsed_s": round(time.monotonic() - began, 2),
    }
    dump(out / "run_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.v2", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor")
    d.add_argument("--droid-root", type=Path, default=WORKSPACE / "data" / "droid")
    d.add_argument("--ego-root", type=Path, default=WORKSPACE / "artifacts" / "ego")
    d.add_argument("--out", type=Path, required=True)

    c = sub.add_parser("catalog")
    c.add_argument("--droid-root", type=Path, default=WORKSPACE / "data" / "droid")
    c.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("run")
    r.add_argument("--droid-root", type=Path, default=WORKSPACE / "data" / "droid")
    r.add_argument("--ego-root", type=Path, default=WORKSPACE / "artifacts" / "ego")
    r.add_argument("--query", type=int, default=0)
    r.add_argument("--top-k", type=int, default=50)
    r.add_argument("--max-total", type=int, default=20)
    r.add_argument("--max-per-task", type=int, default=3)
    r.add_argument("--max-per-video", type=int, default=2)
    r.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = doctor(args.out, args.droid_root, args.ego_root)
        print(json.dumps({"ok": True, "capabilities": report["capabilities"], "reason_codes": report["reason_codes"]}))
        return 0
    if args.command == "catalog":
        episodes = droid_episodes(args.droid_root)
        rows = []
        for i, row in enumerate(episodes):
            rows.append(
                {
                    "source_dataset": "droid_100",
                    "source_episode_id": f"droid_{i:03d}",
                    "input_profile": "robot_native",
                    "instruction": row.get("language_instruction"),
                    "steps": int(row.get("steps") or 0),
                    "approx_seconds": float(row.get("approx_seconds_at_15hz") or 0.0),
                    "capabilities": {"has_robot_command": True, "has_joint_state": True},
                }
            )
        args.out.mkdir(parents=True, exist_ok=True)
        dump_jsonl(args.out / "droid_catalog.jsonl", rows)
        dump(args.out / "droid_catalog_index.json", {"episodes": len(rows), "source": "droid_100 readable export"})
        print(json.dumps({"ok": True, "episodes": len(rows), "out": str(args.out)}))
        return 0
    report = run(args.droid_root, args.ego_root, args.query, args.top_k, args.max_total, args.max_per_task,
                 args.max_per_video, args.out)
    print(json.dumps({"ok": report["verify"]["ok"], "counts": report["counts"], "out": str(args.out)}))
    return 0 if report["verify"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
