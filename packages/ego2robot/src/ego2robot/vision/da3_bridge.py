"""Process bridge for the isolated Depth Anything 3 worker.

Same shape as :mod:`ego2robot.vision.m5_bridge`: a validated JSON request goes to a
model interpreter (native or WSL), the child's log is captured, and every returned
path is checked to stay inside the requested output directory.  The path translation
helpers are the M5 ones on purpose — two stages mapping Windows and WSL paths
differently is exactly the kind of divergence that ends up in a wrong pixel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import m5_bridge
from .m5_contracts import RuntimeConfig

__all__ = ["Da3BridgeError", "run_da3"]


class Da3BridgeError(RuntimeError):
    """Raised when the DA3 worker fails or returns unusable artifacts."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tail(path: Path, limit: int = 12000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return "(worker log is unavailable)"


def _translate(payload: dict[str, Any], distribution: str, project_root: Path) -> dict[str, Any]:
    translated = {k: v for k, v in payload.items() if k != "frames"}
    for key in ("model_dir", "model_repo", "out_dir"):
        if payload.get(key):
            translated[key] = m5_bridge._wsl_path(str(payload[key]), distribution, project_root)
    translated["frames"] = [
        {**row, "path": m5_bridge._wsl_path(str(row["path"]), distribution, project_root)}
        for row in payload["frames"]
    ]
    return translated


def _host_paths(result: dict[str, Any], distribution: str) -> dict[str, Any]:
    """Translate the returned artifact paths back to host paths."""
    rows = []
    for row in result.get("frames", []):
        entry = dict(row)
        for key in ("path", "confidence_path"):
            if entry.get(key):
                entry[key] = m5_bridge._host_path(
                    str(entry[key]), distribution, _project_root()
                ).as_posix()
        rows.append(entry)
    result = dict(result)
    result["frames"] = rows
    return result


def run_da3(
    *,
    runtime: RuntimeConfig,
    model_dir: str,
    model_repo: str,
    frames: list[dict[str, Any]],
    camera_intrinsics: list[list[float]],
    width: int,
    height: int,
    work_width: int,
    work_height: int,
    mode: str,
    focal_scale_divisor: float,
    seed: int,
    out_dir: Path,
    request_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Run the DA3 worker and return its validated result payload."""
    if not str(model_dir).strip():
        raise Da3BridgeError("scene_depth.runtime.checkpoint_path (model_dir) is required")
    if not str(model_repo).strip():
        raise Da3BridgeError("scene_depth.runtime.repo_path is required")
    project_root = _project_root()
    worker = project_root / "tools" / "da3_model_worker.py"
    if not worker.is_file():
        raise Da3BridgeError(f"DA3 worker script is missing: {worker}")

    payload: dict[str, Any] = {
        "task": "scene_depth",
        "engine": "depth_anything_v3",
        "model_dir": str(model_dir),
        "model_repo": str(model_repo),
        "device": runtime.device,
        "seed": int(seed),
        "mode": mode,
        "focal_scale_divisor": float(focal_scale_divisor),
        "work_width": int(work_width),
        "work_height": int(work_height),
        "width": int(width),
        "height": int(height),
        "K": camera_intrinsics,
        "frames": frames,
        "out_dir": str(out_dir),
    }
    request_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()

    if runtime.execution == "wsl":
        distribution = runtime.wsl_distribution or ""
        if not distribution:
            raise Da3BridgeError("runtime.wsl_distribution is required when execution='wsl'")
        executable = runtime.python_executable or ""
        if not executable:
            raise Da3BridgeError("runtime.python_executable is required when execution='wsl'")
        if m5_bridge._is_windows_absolute(executable):
            raise Da3BridgeError(
                "runtime.python_executable must be a Linux path when execution='wsl'"
            )
        probe = m5_bridge._wsl_exec(distribution, "test", "-x", executable)
        if probe.returncode != 0:
            raise Da3BridgeError(
                f"configured WSL interpreter is missing or not executable: {executable} "
                f"(distribution {distribution})"
            )
        payload = _translate(payload, distribution, project_root)
        command = [
            "wsl.exe",
            "--distribution",
            distribution,
            "--exec",
            executable,
            m5_bridge._wsl_path(str(worker), distribution, project_root),
            "--request",
            m5_bridge._wsl_path(str(request_path), distribution, project_root),
            "--result",
            m5_bridge._wsl_path(str(result_path), distribution, project_root),
        ]
    else:
        executable = runtime.python_executable or sys.executable
        command = [
            executable,
            str(worker),
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]

    request_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    log_file = result_path.with_suffix(result_path.suffix + ".log")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        with log_file.open("w", encoding="utf-8", errors="replace") as log:
            completed = subprocess.run(
                command,
                cwd=str(project_root),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
            )
    except OSError as exc:
        raise Da3BridgeError(
            f"could not start the DA3 worker with {executable!r}; see {log_file}: {exc}"
        ) from exc

    if not result_path.is_file():
        raise Da3BridgeError(
            f"DA3 worker exited with status {completed.returncode} without writing a result; "
            f"see {log_file}\n{_tail(log_file)}"
        )
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("ok"):
        raise Da3BridgeError(
            f"DA3 worker reported failure: {result.get('error')}\n{_tail(log_file)}"
        )
    if completed.returncode != 0:
        raise Da3BridgeError(
            f"DA3 worker exited with status {completed.returncode} but wrote ok=true; "
            f"see {log_file}"
        )
    if runtime.execution == "wsl":
        result = _host_paths(result, runtime.wsl_distribution or "")

    rows = result.get("frames") or []
    if len(rows) != len(frames):
        raise Da3BridgeError(
            f"DA3 worker returned {len(rows)} frames for {len(frames)} requested"
        )
    root = out_dir.resolve()
    for row in rows:
        for key in ("path", "confidence_path"):
            value = row.get(key)
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                raise Da3BridgeError(f"worker returned a relative path: {value!r}")
            resolved = path.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise Da3BridgeError(
                    f"worker returned a path outside the output directory: {resolved}"
                ) from exc
            if not resolved.is_file():
                raise Da3BridgeError(f"worker declared a missing artifact: {resolved}")
    return result
