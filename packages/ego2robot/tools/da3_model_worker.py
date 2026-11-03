"""Isolated Depth Anything 3 worker for the ``scene_depth`` stage.

Runs in the model environment (WSL conda/venv) so that torch and the official
``depth_anything_3`` package never enter the pipeline process.  The contract is a JSON
request in, one ``.npy`` per frame plus a JSON result out; the bridge
(:mod:`ego2robot.vision.da3_bridge`) validates that every returned path stays inside
the requested output directory.

Three things this worker deliberately refuses to do:

* **guess the pixel transform.**  The model returns depth on its own processing grid.
  If that grid is not exactly the working grid the request declared, the worker
  raises instead of cropping or letterboxing, because either would silently shift
  every depth sample relative to the RGB frame the compositor uses.
* **guess the units.**  Depth Anything 3 Metric-Large emits *focal-normalised* depth;
  metres are ``raw * focal_px / 300`` on the asserted grid.  The divisor is a request
  field, so a change is visible in the manifest rather than implicit.
* **fill holes.**  Invalid or non-positive samples become ``NaN``, never zero.

Usage::

    python tools/da3_model_worker.py --request req.json --result res.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    result_path = Path(args.result)
    result: dict = {"ok": False, "task": "scene_depth", "engine": "depth_anything_v3"}
    try:
        payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
        _run(payload, result)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-8000:]
        print(result["error"], file=sys.stderr)
        print(result["traceback"], file=sys.stderr)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.get("ok") else 1


def _run(payload: dict, result: dict) -> None:
    import numpy as np
    import torch
    from PIL import Image

    model_dir = Path(payload["model_dir"]).resolve()
    model_repo = Path(payload["model_repo"]).resolve()
    out_dir = Path(payload["out_dir"]).resolve()
    device = str(payload.get("device", "cuda"))
    seed = int(payload.get("seed", 0))
    mode = str(payload.get("mode", "metric"))
    divisor = float(payload.get("focal_scale_divisor", 300.0))
    work_w = int(payload["work_width"])
    work_h = int(payload["work_height"])
    width = int(payload["width"])
    height = int(payload["height"])
    K = np.asarray(payload["K"], dtype=np.float64)
    frames = payload["frames"]
    if not frames:
        raise ValueError("request contains no frames")
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    if mode not in {"metric", "relative"}:
        raise ValueError(f"unsupported mode {mode!r}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_name = str(config.get("model_name", "")).lower()
    if mode == "metric" and model_name != "da3metric-large":
        raise ValueError(
            f"metric mode needs the DA3METRIC-LARGE checkpoint, found model_name={model_name!r}"
        )

    sys.path.insert(0, str(model_repo / "src"))
    from depth_anything_3.api import DepthAnything3  # type: ignore

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = DepthAnything3.from_pretrained(str(model_dir)).to(device).eval()

    # Working intrinsics: the same anisotropic scale as the pixel resize, so the
    # principal point moves with the image instead of staying at the native centre.
    sx, sy = work_w / float(width), work_h / float(height)
    K_work = K.copy()
    K_work[0, 0] *= sx
    K_work[0, 2] *= sx
    K_work[1, 1] *= sy
    K_work[1, 2] *= sy
    focal_work = float((K_work[0, 0] + K_work[1, 1]) / 2.0)
    conversion = (focal_work / divisor) if mode == "metric" else 1.0

    depth_dir = out_dir / "depth"
    conf_dir = out_dir / "confidence"
    depth_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    metric_flags: list[bool] = []
    for row in frames:
        frame_index = int(row["frame_index"])
        image = Image.open(row["path"]).convert("RGB")
        if image.size != (width, height):
            raise ValueError(
                f"frame {frame_index}: image is {image.size}, request declared {(width, height)}"
            )
        work_image = np.asarray(image.resize((work_w, work_h), Image.Resampling.BILINEAR))
        torch.manual_seed(seed + frame_index)
        started = time.monotonic()
        with torch.inference_mode():
            prediction = model.inference(
                image=[work_image],
                intrinsics=K_work[None].astype(np.float32),
                process_res=max(work_w, work_h),
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
        raw = np.asarray(prediction.depth, dtype=np.float32)
        if raw.shape != (1, work_h, work_w):
            raise ValueError(
                f"frame {frame_index}: model depth grid is {raw.shape}, expected "
                f"{(1, work_h, work_w)}; refusing to guess a crop or K rescale"
            )
        metric_flags.append(bool(getattr(prediction, "is_metric", False)))
        converted = raw[0] * np.float32(conversion)
        depth = np.asarray(
            Image.fromarray(converted).resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).copy()
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        depth_path = depth_dir / f"frame-{frame_index:06d}.npy"
        np.save(depth_path, depth)
        entry = {
            "frame_index": frame_index,
            "path": str(depth_path),
            "valid_fraction": float(np.isfinite(depth).mean()),
            "inference_seconds": round(time.monotonic() - started, 3),
        }
        confidence = getattr(prediction, "conf", None)
        if confidence is not None:
            conf = np.asarray(confidence, dtype=np.float32)
            if conf.shape != raw.shape:
                raise ValueError(
                    f"frame {frame_index}: confidence {conf.shape} does not match depth {raw.shape}"
                )
            conf_native = np.asarray(
                Image.fromarray(conf[0]).resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ).copy()
            conf_dir.mkdir(parents=True, exist_ok=True)
            conf_path = conf_dir / f"frame-{frame_index:06d}.npy"
            np.save(conf_path, conf_native)
            entry["confidence_path"] = str(conf_path)
        rows.append(entry)
        print(
            json.dumps({"frame": frame_index, "valid": entry["valid_fraction"]}),
            flush=True,
        )

    revision: str | None = None
    try:
        import subprocess

        revision = subprocess.check_output(
            ["git", "-C", str(model_repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is optional
        revision = None

    result["frames"] = rows
    result["provenance"] = {
        "model": "Depth Anything 3",
        "model_name": model_name,
        "model_dir": str(model_dir),
        "model_repo": str(model_repo),
        "code_revision": revision,
        "mode": mode,
        "focal_work_px": round(focal_work, 6),
        "focal_scale_divisor": divisor,
        "raw_to_output_scale": round(float(conversion), 9),
        "conversion_rule": "focal_work_px * net_output / divisor" if mode == "metric" else "identity",
        "work_size": [work_w, work_h],
        "K_native": K.tolist(),
        "K_work": K_work.tolist(),
        "pixel_affine_native_to_work": [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        "pixel_transform_asserted": (
            "model depth grid equals the declared work grid exactly; bilinear inverse "
            "resize to native resolution"
        ),
        "confidence_semantics": "raw model score; no calibrated probability is claimed",
        "api_is_metric_flags": metric_flags,
        "native_resolution_inference": False,
        "temporal_postprocessing": "none (single view per frame)",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python": sys.version,
        "interpreter": sys.executable,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if device.startswith("cuda")
        else 0,
        "seed": seed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
