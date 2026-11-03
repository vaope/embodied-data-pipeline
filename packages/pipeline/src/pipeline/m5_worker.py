"""Isolated, real ProPainter completion worker for the fixed 52-frame sample.

Only new completion outputs are writable.  Existing source frames/SAM3 masks,
official source, and weights remain read-only.  Native-resolution processing is
kept separate from review approval.  Run ``prepare`` on Windows, ``run`` under
the existing WSL m5 interpreter, and ``publish`` on Windows after success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True

# Direct-script invocation must not expose pipeline/statistics.py as the Python
# standard-library ``statistics`` imported internally by PyTorch.
if __package__ in (None, ""):
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]


def local(value: str | Path) -> Path:
    value = str(value).replace("\\", "/")
    if os.name != "nt" and len(value) > 2 and value[1] == ":":
        value = "/mnt/" + value[0].lower() + value[2:]
    return Path(value)


def windows(value: str | Path) -> str:
    value = str(Path(value).resolve()).replace("\\", "/")
    if value.startswith("/mnt/") and len(value) > 7:
        return value[5].upper() + ":" + value[6:]
    return value


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with local(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: str | Path) -> dict:
    path = local(path)
    return {"path": windows(path), "sha256": sha(path), "bytes": path.stat().st_size}


def read(path: str | Path) -> dict:
    return json.loads(local(path).read_text(encoding="utf-8-sig"))


def write(path: str | Path, value: dict) -> None:
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def checked_png(path: Path, digest: str, size: tuple[int, int]) -> None:
    from PIL import Image
    if sha(path) != digest:
        raise ValueError(f"SHA256 mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.size != size:
            raise ValueError(f"Unexpected image size: {path}: {image.size}")


def prepare(args) -> None:
    from PIL import Image
    episode = local(args.episode).resolve()
    out = local(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_path = episode / "segmentation/source-index.json"
    human_path = episode / "segmentation/bundle/index.json"
    source, human = read(source_path), read(human_path)
    old_bundle = read(args.bundle)
    source_sha = sha(episode / "source/0.mp4")
    if source_sha != source["source_video_sha256"] or source_sha != human["source_video_sha256"]:
        raise ValueError("Original video hash does not match M5 input indexes")
    if source_sha != old_bundle["refs"]["source_video"]["sha256"]:
        raise ValueError("Learning bundle/source video mismatch")
    if len(source["frames"]) != 52 or len(human["frames"]) != 52:
        raise ValueError("This worker is scoped to exactly the existing 000 52 frames")
    dimensions = (source["width"], source["height"])
    if dimensions != (1920, 1080):
        raise ValueError("Expected native 1920x1080 input")
    previous = -1.0
    for i, (rgb, mask) in enumerate(zip(source["frames"], human["frames"])):
        if rgb["frame_index"] != i or mask["frame_index"] != i:
            raise ValueError("Frame coverage is not contiguous")
        if abs(rgb["timestamp_s"] - mask["timestamp_s"]) > 1e-9 or rgb["timestamp_s"] <= previous:
            raise ValueError("PTS mismatch/nonmonotonic input")
        previous = rgb["timestamp_s"]
        for row, path_name, hash_name in ((rgb, "path", "sha256"), (mask, "raw_path", "raw_sha256"), (mask, "refined_path", "refined_sha256")):
            path = episode / row[path_name]
            checked_png(path, row[hash_name], dimensions)
            row[path_name] = windows(path)
        if not mask["valid"]:
            raise ValueError("Existing SAM3 refined mask is invalid")
    source["source_key"] = human["source_key"] = old_bundle["source_key"]
    source["provenance"]["original_index_ref"] = ref(source_path)
    human["provenance"]["original_index_ref"] = ref(human_path)
    write(out / "source-index.json", source)
    write(out / "human-index.json", human)
    repair_ref = None
    if args.repair_mask_index:
        import numpy as np
        repair_path = local(args.repair_mask_index)
        repair = read(repair_path)
        if (repair["source_video_sha256"] != source_sha or len(repair["frames"]) != 52 or
                (repair["width"], repair["height"]) != dimensions):
            raise ValueError("Repair mask override has different source/coverage/resolution")
        added = 0
        for i,(row,original) in enumerate(zip(repair["frames"],human["frames"])):
            if (row["frame_index"] != i or abs(row["timestamp_s"]-original["timestamp_s"])>1e-9 or not row["valid"]):
                raise ValueError("Repair mask override has different PTS/validity")
            for key,hash_key in (("raw_path","raw_sha256"),("refined_path","refined_sha256")):
                path = episode / row[key]
                checked_png(path,row[hash_key],dimensions)
                row[key] = windows(path)
            base = np.asarray(Image.open(local(original["refined_path"])).convert("L"))>0
            extra = np.asarray(Image.open(local(row["refined_path"])).convert("L"))>0
            if np.any(base & ~extra):
                raise ValueError("Repair override unexpectedly removes a human mask pixel")
            added += int(np.count_nonzero(extra & ~base))
        repair["source_key"] = old_bundle["source_key"]
        repair["provenance"].update(original_index_ref=ref(repair_path),
            evidence_kind="heuristic_shadow_repair_mask", role="repair_only_not_human_segmentation_truth",
            original_human_index_ref=ref(out / "human-index.json"), added_pixels_total=added)
        write(out / "repair-input-index.json",repair)
        repair_ref=ref(out / "repair-input-index.json")
    repo = local(args.repo).resolve()
    weight_refs = {name: ref(repo / "weights" / name) for name in ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth")}
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()
    config = {
        "schema_version": "learning.stage5.m5_completion_request.v1",
        "episode_id": old_bundle["episode_id"], "source_key": old_bundle["source_key"],
        "source_video_sha256": source_sha, "width": 1920, "height": 1080,
        "source_index_ref": ref(out / "source-index.json"), "human_index_ref": ref(out / "human-index.json"),
        "official_repository": windows(repo), "repository_revision": revision,
        "official_script_ref": ref(repo / "inference_propainter.py"), "weight_refs": weight_refs,
        "output_dir": windows(out), "seed": 0,
        "parameters": {"fp16": True, "neighbor_length": 2, "ref_stride": 10, "subvideo_length": 2, "mask_dilation": 4, "raft_iter": 20},
        "paper_parameters": {"fp16": True, "neighbor_length": 10, "ref_stride": 10, "subvideo_length": 80, "mask_dilation": 4, "raft_iter": 20},
        "variant": "official_cli_reduced_temporal_context_native_resolution",
        "review_status": "pending", "accepted": False,
    }
    if repair_ref:
        config["repair_index_ref"] = repair_ref
        config["variant"] += "_existing_heuristic_shadow_repair"
    if args.raft_pair_stream:
        config["raft_pair_stream"] = True
        config["variant"] += "_sequential_RAFT_pairs"
    write(out / "config.json", config)
    write(out / "input-audit.json", {"status": "verified", "frames": 52, "rgb_hashes_checked": 52,
        "raw_mask_hashes_checked": 52, "refined_mask_hashes_checked": 52, "timebase": "source_video_PTS",
        "dimensions": list(dimensions), "source_video_sha256": source_sha,
        "source_index_ref": config["source_index_ref"], "human_index_ref": config["human_index_ref"],
        "existing_results_modified": False, "models_executed": False})
    print(json.dumps({"config": windows(out / "config.json"), "status": "prepared", "frames": 52}))


def stage_frames(config: dict, root: Path, start: int, end: int, crop=None) -> tuple[Path, Path]:
    from PIL import Image
    source = read(config["source_index_ref"]["path"])
    human = read(config.get("repair_index_ref",config["human_index_ref"])["path"])
    frames, masks = root / "input/frames", root / "input/masks"
    frames.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(range(start, end)):
        for row, key, target in ((source["frames"][i], "path", frames), (human["frames"][i], "refined_path", masks)):
            incoming, outgoing = local(row[key]), target / f"{j:06d}.png"
            if crop is None:
                shutil.copyfile(incoming, outgoing)
            else:
                with Image.open(incoming) as image:
                    image.crop(crop).save(outgoing)
    return frames, masks


def execute_official(config: dict, frames: Path, masks: Path, output: Path, params: dict) -> None:
    repo = local(config["official_repository"])
    if sha(config["official_script_ref"]["path"]) != config["official_script_ref"]["sha256"]:
        raise ValueError("Pinned official inference script changed")
    for name, weight in config["weight_refs"].items():
        if sha(weight["path"]) != weight["sha256"]:
            raise ValueError(f"Pinned checkpoint changed: {name}")
    sys.path.insert(0, str(repo))
    import utils.download_util as download_util
    def offline_loader(url, model_dir=None, progress=True, file_name=None):
        from urllib.parse import urlparse
        name = file_name or Path(urlparse(url).path).name
        if name not in config["weight_refs"]:
            raise FileNotFoundError(f"Unapproved checkpoint requested: {name}")
        return str(local(config["weight_refs"][name]["path"]))
    download_util.load_file_from_url = offline_loader
    if config.get("raft_pair_stream"):
        import torch
        from model.modules.flow_comp_raft import RAFT_bi
        original = RAFT_bi.forward
        def sequential_pair_forward(self, clips, iters=20):
            # Exact official pair estimator, with adjacent pairs evaluated one
            # at a time rather than batched. No resize or checkpoint change.
            forward, backward = [], []
            for position in range(clips.shape[1]-1):
                f,b = original(self,clips[:,position:position+2],iters=iters)
                forward.append(f)
                backward.append(b)
                torch.cuda.empty_cache()
            return torch.cat(forward,dim=1),torch.cat(backward,dim=1)
        RAFT_bi.forward = sequential_pair_forward
    # The upstream __main__ is executed without modifying its source. It reads
    # staged PNG inputs and uses only the verified three local model weights.
    args = [str(repo / "inference_propainter.py"), "--video", str(frames), "--mask", str(masks),
        "--output", str(output), "--save_frames", "--save_fps", "30"]
    for key, value in params.items():
        if key == "fp16":
            if value:
                args.append("--fp16")
        else:
            args.extend(["--" + key, str(value)])
    prior_cwd, prior_argv = Path.cwd(), sys.argv
    try:
        os.chdir(repo)
        sys.argv = args
        runpy.run_path(str(repo / "inference_propainter.py"), run_name="__main__")
    finally:
        os.chdir(prior_cwd)
        sys.argv = prior_argv


def input_fingerprint(config: dict, config_path: str, params: dict, start: int, end: int, crop) -> str:
    """Recheck actual bytes, rather than trusting unchanged index text."""
    inventories = {}
    for key in ("source_index_ref", "human_index_ref", *(["repair_index_ref"] if "repair_index_ref" in config else [])):
        record = config[key]
        if sha(record["path"]) != record["sha256"]:
            raise ValueError(f"Input index hash changed: {key}")
        index = read(record["path"])
        records = []
        for row in index["frames"][start:end]:
            for name,hash_name in (("path","sha256"),("raw_path","raw_sha256"),("refined_path","refined_sha256")):
                if name in row:
                    checked_png(local(row[name]), row[hash_name], (config["width"],config["height"]))
                    records.append((row["frame_index"],name,row[hash_name]))
        inventories[key] = records
    for weight in config["weight_refs"].values():
        if sha(weight["path"]) != weight["sha256"]:
            raise ValueError("Checkpoint changed since preparation")
    script = config["official_script_ref"]
    if sha(script["path"]) != script["sha256"]:
        raise ValueError("Official entry script changed since preparation")
    repo = local(config["official_repository"])
    official_code = {path.relative_to(repo).as_posix():sha(path) for path in sorted(repo.rglob("*.py")) if ".git" not in path.parts}
    payload = {"config_sha256":sha(config_path),"parameters":params,"start":start,"end":end,"crop":crop,
        "actual_image_inventory":inventories,"official_code_inventory":official_code,
        "checkpoint_hashes":{name:row["sha256"] for name,row in config["weight_refs"].items()}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")).hexdigest()


def run(args) -> None:
    import random
    import numpy as np
    import torch
    config = read(args.config)
    for key in ("source_index_ref", "human_index_ref", *(["repair_index_ref"] if "repair_index_ref" in config else [])):
        if sha(config[key]["path"]) != config[key]["sha256"]:
            raise ValueError(f"Input index changed: {key}")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    root = local(config["output_dir"]) / "attempts" / args.name
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "run.json"
    params = dict(config["parameters"])
    if args.neighbor_length is not None:
        params["neighbor_length"] = args.neighbor_length
    if args.subvideo_length is not None:
        params["subvideo_length"] = args.subvideo_length
    fingerprint = input_fingerprint(config,args.config,params,args.start,args.end,args.crop)
    if record_path.exists() and read(record_path).get("status") == "succeeded":
        cached = read(record_path)
        if (cached["start"] != args.start or cached["end"] != args.end or
                cached["crop"] != args.crop or cached["parameters"] != params or
                cached.get("input_fingerprint") != fingerprint or
                cached["config_ref"]["sha256"] != sha(args.config)):
            raise ValueError("Existing successful attempt changed or uses different inputs; use a new attempt name")
        if [row["frame_index"] for row in cached["outputs"]] != list(range(args.start,args.end)):
            raise ValueError("Cached output coverage changed; use a new attempt name")
        for row in cached["outputs"]:
            checked_png(local(row["path"]),row["sha256"],tuple(cached["inference_resolution"]))
        print(json.dumps({"status": "cached", "path": windows(record_path)}))
        return
    start, end = args.start, args.end
    if not 0 <= start < end <= 52 or end - start < 2:
        raise ValueError("Run requires an explicit 2..52-frame subset of fixed 000")
    crop = tuple(args.crop) if args.crop else None
    frames, masks = stage_frames(config, root, start, end, crop)
    record = {"schema_version": "learning.stage5.m5_real_run.v1", "status": "running", "start": start, "end": end,
        "input_fingerprint":fingerprint,
        "crop": crop, "source_frame_indices": list(range(start, end)), "config_ref": ref(args.config), "parameters": params,
        "official_script_ref": config["official_script_ref"], "weight_refs": config["weight_refs"],
        "repository_revision": config["repository_revision"], "evidence_kind": "real_model", "review_status": "pending",
        "raft_pair_stream": config.get("raft_pair_stream",False),
        "runtime": {"python": sys.executable, "python_version": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "mem_free_total_before": list(torch.cuda.mem_get_info())},
        "inference_resolution": [crop[2]-crop[0], crop[3]-crop[1]] if crop else [1920, 1080],
        "source_resize": "identity", "output_frames_dir": windows(root / "model-output/frames/frames")}
    write(record_path, record)
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        execute_official(config, frames, masks, root / "model-output", params)
        from PIL import Image
        expected_size = tuple(record["inference_resolution"])
        results = []
        for j, i in enumerate(range(start, end)):
            result_path = root / f"model-output/frames/frames/{j:04d}.png"
            with Image.open(result_path) as image:
                if image.size != expected_size:
                    raise ValueError("Official result has unexpected resolution")
            results.append({"frame_index": i, **ref(result_path)})
        record.update(status="succeeded", outputs=results)
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record["elapsed_s"] = time.monotonic() - began
        record["runtime"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["runtime"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        # Every attempt, including OOM/import failure, retains a visible input
        # and mask panel; successful attempts also show actual model output.
        from PIL import Image, ImageDraw
        src = Image.open(frames / "000000.png").convert("RGB")
        mask = Image.open(masks / "000000.png").convert("L")
        overlay = Image.blend(src, Image.composite(Image.new("RGB", src.size, (255,40,40)), src, mask), .55)
        result_path = root / "model-output/frames/frames/0000.png"
        if record["status"] == "succeeded" and result_path.exists():
            result = Image.open(result_path).convert("RGB")
        else:
            result = Image.new("RGB", src.size, "#25252c")
        panel = Image.new("RGB", (1440,300), "#151a22")
        draw = ImageDraw.Draw(panel)
        for j,(image,label) in enumerate(((src,"Native source RGB"),(overlay,"Repair input mask / see provenance"),(result,"Real model result" if record["status"] == "succeeded" else "Model unavailable: " + record.get("error_type", "failure")))):
            panel.paste(image.resize((480,270)), (j*480,30))
            draw.text((j*480+8,9), label, fill="white")
        panel.save(root / "attempt-evidence.png")
        record["evidence_image_ref"] = ref(root / "attempt-evidence.png")
        write(record_path, record)


def publish(args) -> None:
    """Publish only actual model pixels inside the exact 4px repair mask."""
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy import ndimage
    config = read(args.config)
    out = local(config["output_dir"])
    source, human = read(config["source_index_ref"]["path"]), read(config["human_index_ref"]["path"])
    repair_input = read(config.get("repair_index_ref",config["human_index_ref"])["path"])
    records = [read(path) for path in args.runs]
    chosen = {}
    for path, run_record in zip(args.runs, records):
        if run_record["status"] != "succeeded" or run_record["crop"] is not None:
            raise ValueError("publish requires successful full-spatial-resolution runs")
        for result in run_record["outputs"]:
            i = result["frame_index"]
            edge_distance = min(i-run_record["start"], run_record["end"]-1-i)
            if i not in chosen or edge_distance > chosen[i][0]:
                chosen[i] = (edge_distance, result, path)
    if sorted(chosen) != list(range(52)):
        raise ValueError("Cannot publish an incomplete 52-frame background")
    bg_rows, masks = [], []
    output_frames = out / "background"
    output_frames.mkdir(exist_ok=True)
    (out / "repair-mask").mkdir(exist_ok=True)
    for i in range(52):
        src = np.asarray(Image.open(local(source["frames"][i]["path"])).convert("RGB"))
        mask = np.asarray(Image.open(local(repair_input["frames"][i]["refined_path"])).convert("L")) > 0
        repair = ndimage.binary_dilation(mask, iterations=4)
        result = chosen[i][1]
        if sha(result["path"]) != result["sha256"]:
            raise ValueError("Model frame hash changed before publication")
        prediction = np.asarray(Image.open(local(result["path"])).convert("RGB"))
        bg = src.copy()
        bg[repair] = prediction[repair]
        path = output_frames / f"frame-{i:06d}.png"
        Image.fromarray(bg).save(path)
        mask_path = out / f"repair-mask/frame-{i:06d}.png"
        Image.fromarray(repair.astype(np.uint8)*255).save(mask_path)
        if np.any(bg[~repair] != src[~repair]):
            raise ValueError("Background changed pixels outside the actual repair mask")
        bg_rows.append({"frame_index": i, "timestamp_s": source["frames"][i]["timestamp_s"], "valid": human["frames"][i]["valid"], **ref(path)})
        masks.append({"frame_index": i, "timestamp_s": source["frames"][i]["timestamp_s"], **ref(mask_path), "edited_pixels": int(repair.sum())})
    provenance = {"model_name": "official ProPainter inference_propainter.py", "backend": "WSL PyTorch CUDA",
        "repository_revision": config["repository_revision"], "checkpoint_sha256": {name: r["sha256"] for name,r in config["weight_refs"].items()},
        "weight_refs": config["weight_refs"], "official_script_ref": config["official_script_ref"],
        "inference_resolution": [1920,1080], "inference_width": 1920, "inference_height": 1080,
        "preview_only": False, "review_status": "pending", "evidence_kind": "real_model",
        "parameters": config["parameters"], "paper_parameters": config["paper_parameters"],
        "variant": config["variant"], "temporal_chunking": len(records)>1,
        "raft_pair_stream": config.get("raft_pair_stream",False),
        "temporal_join": "choose_prediction_with_largest_distance_to_clip_boundary; deterministic first on ties",
        "runs": [ref(p) for p in args.runs], "source_index_ref": config["source_index_ref"],
        "human_index_ref": config["human_index_ref"], "resize": "identity",
        "repair_input_index_ref": config.get("repair_index_ref",config["human_index_ref"]),
        "repair_mask_role": "human_union_heuristic_shadow" if "repair_index_ref" in config else "human_refined_mask",
        "repair_region": "scipy.ndimage.binary_dilation(indexed_repair_input_mask,iterations=4), connectivity1",
        "outside_repair_region_changed_pixels": 0}
    background = {"schema_version": "ego2robot.m5_background.v1", "source_key": config["source_key"],
        "source_video_sha256": config["source_video_sha256"], "width":1920,"height":1080,"frames":bg_rows,"provenance":provenance}
    write(out / "background-index.json", background)
    write(out / "repair-mask-index.json", {"schema_version":"learning.stage5.m5_repair_mask.v1","frames":masks})
    write(out / "visual-artifacts.json", {"schema_version":"learning.stage5.visual_artifacts.v1",
        "source_key":config["source_key"],"source_video_sha256":config["source_video_sha256"],
        "source_index_ref":config["source_index_ref"],"human_index_ref":config["human_index_ref"],
        "background_index_ref":ref(out / "background-index.json")})
    previews = out / "previews"
    previews.mkdir(exist_ok=True)
    for i in (0,7,15,22,29,36,44,51):
        src = Image.open(local(source["frames"][i]["path"])).convert("RGB")
        bg = Image.open(local(bg_rows[i]["path"])).convert("RGB")
        mask = Image.open(local(human["frames"][i]["refined_path"])).convert("L")
        overlay = Image.blend(src, Image.composite(Image.new("RGB", src.size, (255,40,40)), src, mask), .55)
        panels = [(src,"Source RGB"),(overlay,"Original SAM3 human mask")]
        if "repair_index_ref" in config:
            extra_mask = Image.open(local(repair_input["frames"][i]["refined_path"])).convert("L")
            extra_overlay = Image.blend(src, Image.composite(Image.new("RGB", src.size, (255,150,0)), src, extra_mask), .55)
            panels.append((extra_overlay,"Heuristic shadow repair mask"))
        panels.append((bg,"Real ProPainter 1080p / pending"))
        panel = Image.new("RGB", (480*len(panels),300), "#151a22")
        draw = ImageDraw.Draw(panel)
        for j,(image,label) in enumerate(panels):
            panel.paste(image.resize((480,270)), (j*480,30))
            draw.text((j*480+8,9), label, fill="white")
        panel.save(previews / f"review-{i:06d}.png")
    import cv2
    video = cv2.VideoWriter(str(previews / "background-review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280,360))
    if not video.isOpened():
        raise RuntimeError("Could not create review video")
    for i,row in enumerate(bg_rows):
        src = np.asarray(Image.open(local(source["frames"][i]["path"])).convert("RGB").resize((640,360)))
        bg = np.asarray(Image.open(local(row["path"])).convert("RGB").resize((640,360)))
        video.write(cv2.cvtColor(np.concatenate((src,bg),axis=1),cv2.COLOR_RGB2BGR))
    video.release()
    write(out / "manifest.json", {"status":"succeeded", "program_completed":True, "user_accepted":False,
        "review_status":"pending", "evidence_kind":"real_model", "frames":52,
        "inference_resolution":[1920,1080],"preview_only":False,
        "background_index_ref":ref(out / "background-index.json"),"visual_artifacts_ref":ref(out / "visual-artifacts.json"),
        "preview_video_ref":ref(previews / "background-review.mp4"), "limitations":[
            "Temporal context is reduced relative to paper; repair mask is original human SAM3 union an existing heuristic shadow candidate." if "repair_index_ref" in config else "Temporal context is reduced relative to paper; masks are reused without broadening to shadows.",
            "Native-resolution output is not a user acceptance or an independent quality claim."]})
    print(json.dumps({"status":"published","visual_artifacts":windows(out / "visual-artifacts.json"),"frames":52}))


def sequence(args) -> None:
    """Bounded resumable real-model windows; every child releases its GPU."""
    config=read(args.config)
    if not 2 <= args.window <= 52 or not 0 <= args.overlap < args.window:
        raise ValueError("Invalid temporal window/overlap")
    out=local(config["output_dir"])
    runs=[]
    sequence_path=out / f"sequence-window{args.window}-overlap{args.overlap}.json"
    state={"schema_version":"learning.stage5.m5_window_sequence.v1","status":"running",
        "config_ref":ref(args.config),"window":args.window,"overlap":args.overlap,
        "timeout_s_per_window":args.timeout,"runs":[],"review_status":"pending"}
    began=time.monotonic()
    try:
        for start in range(0,52,args.window-args.overlap):
            end=min(52,start+args.window)
            if end-start < 2:
                start=50
            name=f"window-{start:06d}-{end:06d}"
            log_path=out / f"{name}.log"
            run_path=out / "attempts" / name / "run.json"
            command=[sys.executable,"-B",str(Path(__file__).resolve()),"run","--config",str(local(args.config)),
                "--name",name,"--start",str(start),"--end",str(end)]
            environment=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
            print(json.dumps({"window":[start,end],"status":"starting"}),flush=True)
            with log_path.open("a",encoding="utf-8") as log:
                log.write(json.dumps({"event":"worker_invocation","command":command},ensure_ascii=False)+"\n")
                log.flush()
                try:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment,timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    from PIL import Image,ImageDraw
                    attempt=run_path.parent
                    timeout_record=read(run_path) if run_path.exists() else {}
                    timeout_record.update(status="failed",error_type="ResourceTimeout",
                        error=f"Window exceeded the explicit {args.timeout}s budget; child process killed and reaped.",
                        elapsed_s_lower_bound=args.timeout)
                    if "runtime" in timeout_record:
                        timeout_record["runtime"].update(peak_allocated_bytes=None,peak_reserved_bytes=None)
                    panel=Image.new("RGB",(960,300),"#151a22")
                    draw=ImageDraw.Draw(panel)
                    src=Image.open(attempt / "input/frames/000000.png").convert("RGB")
                    mask=Image.open(attempt / "input/masks/000000.png").convert("L")
                    overlay=Image.blend(src,Image.composite(Image.new("RGB",src.size,(255,40,40)),src,mask),.55)
                    panel.paste(src.resize((480,270)),(0,30));panel.paste(overlay.resize((480,270)),(480,30))
                    draw.text((8,9),"Resource timeout: native RGB input",fill="white")
                    draw.text((488,9),"Repair mask; no completed model result",fill="white")
                    panel.save(attempt / "timeout-evidence.png")
                    timeout_record["evidence_image_ref"]=ref(attempt / "timeout-evidence.png")
                    write(run_path,timeout_record)
                    raise
            if result.returncode != 0:
                raise RuntimeError(f"Real ProPainter window {start}:{end} failed; see {log_path}")
            run_record=read(run_path)
            if run_record["status"] != "succeeded":
                raise RuntimeError("Window process did not report success")
            runs.append(str(run_path))
            state["runs"].append({"start":start,"end":end,"run_ref":ref(run_path),"log_ref":ref(log_path)})
            write(sequence_path,state)
            print(json.dumps({"window":[start,end],"status":"succeeded","elapsed_s":run_record["elapsed_s"]}),flush=True)
            if end==52:
                break
        publish(argparse.Namespace(config=args.config,runs=runs))
        state["status"]="succeeded"
    except Exception as exc:
        state.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        state["elapsed_s"]=time.monotonic()-began
        write(sequence_path,state)


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--episode",required=True)
    prep.add_argument("--bundle",required=True)
    prep.add_argument("--repo",required=True)
    prep.add_argument("--output",required=True)
    prep.add_argument("--repair-mask-index",help="Optional existing, source-scoped repair-only shadow mask; original human truth remains separate")
    prep.add_argument("--raft-pair-stream",action="store_true",help="Run the unchanged official adjacent-pair RAFT estimator sequentially to bound GPU correlation memory")
    runner = sub.add_parser("run")
    runner.add_argument("--config",required=True)
    runner.add_argument("--name",required=True)
    runner.add_argument("--start",type=int,default=0)
    runner.add_argument("--end",type=int,default=52)
    runner.add_argument("--crop",type=int,nargs=4)
    runner.add_argument("--neighbor-length",type=int)
    runner.add_argument("--subvideo-length",type=int)
    pub = sub.add_parser("publish")
    pub.add_argument("--config",required=True)
    pub.add_argument("--runs",nargs="+",required=True)
    seq=sub.add_parser("sequence")
    seq.add_argument("--config",required=True)
    seq.add_argument("--window",type=int,default=8)
    seq.add_argument("--overlap",type=int,default=2)
    seq.add_argument("--timeout",type=float,default=240)
    args = parser.parse_args()
    {"prepare":prepare,"run":run,"publish":publish,"sequence":sequence}[args.command](args)


if __name__ == "__main__":
    main()
