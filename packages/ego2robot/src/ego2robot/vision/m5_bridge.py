"""Process bridge for the isolated M5 model worker.

The bridge keeps PyTorch and model repositories out of the main process. It
passes a validated JSON request to either a native interpreter or WSL, captures
the child's log, and validates that returned artifacts belong to the requested
output directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .m5_contracts import ModelRequest, ModelResult, SourceFrameIndex

__all__ = ["run_model"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _wsl_exec(distribution: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["wsl.exe", "--distribution", distribution, "--exec", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _is_windows_absolute(value: str) -> bool:
    path = PureWindowsPath(value)
    return bool(path.drive) or value.startswith("\\\\")


def _is_posix_absolute(value: str) -> bool:
    return PurePosixPath(value).is_absolute()


def _host_path(value: str, distribution: str, project_root: Path) -> Path:
    """Resolve a request path on the host, converting WSL paths when needed."""
    if _is_windows_absolute(value):
        return Path(value)
    if _is_posix_absolute(value):
        converted = _wsl_exec(distribution, "wslpath", "-w", "-a", value)
        if converted.returncode != 0:
            raise RuntimeError(
                f"Could not map WSL path to the host: {value!r}: {converted.stderr.strip()}"
            )
        return Path(converted.stdout.strip())
    return (project_root / value).resolve()


def _wsl_path(value: str, distribution: str, project_root: Path) -> str:
    """Resolve a request path for WSL, translating Windows paths with wslpath."""
    if _is_posix_absolute(value):
        return value
    if _is_windows_absolute(value):
        host_path = Path(value)
    else:
        host_path = (project_root / value).resolve()
    converted = _wsl_exec(distribution, "wslpath", "-a", str(host_path))
    if converted.returncode != 0:
        raise RuntimeError(
            f"Could not map host path to WSL: {str(host_path)!r}: {converted.stderr.strip()}"
        )
    path = converted.stdout.strip()
    if not path.startswith("/"):
        raise RuntimeError(f"wslpath returned a non-absolute path for {host_path}: {path!r}")
    return path


def _host_json_path(request: ModelRequest, project_root: Path) -> Path:
    distribution = request.runtime.wsl_distribution or ""
    if request.runtime.execution == "wsl":
        return _host_path(request.source_index_path, distribution, project_root)
    value = request.source_index_path
    if _is_windows_absolute(value) or _is_posix_absolute(value):
        return Path(value)
    return (project_root / value).resolve()


def _translate_request_for_wsl(
    payload: dict[str, Any], distribution: str, project_root: Path
) -> dict[str, Any]:
    translated = dict(payload)
    for key in ("episode_root", "source_index_path", "output_dir", "mask_index_path"):
        value = translated.get(key)
        if value:
            translated[key] = _wsl_path(str(value), distribution, project_root)

    runtime = dict(translated["runtime"])
    for key in ("repo_path", "checkpoint_path"):
        value = runtime.get(key)
        if value:
            runtime[key] = _wsl_path(str(value), distribution, project_root)
    executable = runtime.get("python_executable")
    if not executable:
        raise ValueError("WSL model execution requires runtime.python_executable (a Linux path)")
    if _is_windows_absolute(str(executable)):
        raise ValueError(
            "runtime.python_executable must be a Linux path when runtime.execution='wsl'"
        )
    translated["runtime"] = runtime
    return translated


def _translate_frame_paths(paths: list[str], distribution: str) -> list[str]:
    """Translate only returned frame artifacts; preserve runtime provenance paths."""
    translated: list[str] = []
    for value in paths:
        if not _is_posix_absolute(value):
            raise ValueError(f"WSL worker returned a non-absolute frame path: {value!r}")
        converted = _wsl_exec(distribution, "wslpath", "-w", "-a", value)
        if converted.returncode != 0:
            raise RuntimeError(
                f"Could not map returned WSL frame path to the host: "
                f"{value!r}: {converted.stderr.strip()}"
            )
        translated.append(converted.stdout.strip())
    return translated


def _read_index_count(path: Path) -> int:
    try:
        index = SourceFrameIndex.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not validate source frame index {path}: {exc}") from exc
    return len(index.frames)


def _ensure_result_paths(result: ModelResult, output_dir: Path, expected_count: int) -> None:
    if len(result.frame_paths) != expected_count:
        raise ValueError(
            f"M5 worker returned {len(result.frame_paths)} frame paths; "
            f"source index contains {expected_count} frames"
        )
    root = output_dir.resolve()
    for index, raw_path in enumerate(result.frame_paths):
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError(f"M5 worker frame_paths[{index}] is not absolute: {raw_path!r}")
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"M5 worker frame_paths[{index}] escapes output_dir: {resolved}"
            ) from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"M5 worker frame_paths[{index}] does not exist: {resolved}")


def _tail(path: Path, limit: int = 12000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return "(worker log is unavailable)"


def run_model(
    request: ModelRequest,
    request_path: Path,
    result_path: Path,
) -> ModelResult:
    """Execute one isolated M5 model request and validate its returned PNGs.

    The request and result locations are host-side JSON paths. For WSL runs,
    Windows absolute paths are converted with wslpath while the configured
    Linux Python executable is passed through unchanged.
    """
    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a validated ModelRequest")
    request_file = Path(request_path).resolve()
    result_file = Path(result_path).resolve()
    if request_file == result_file:
        raise ValueError("request_path and result_path must be different files")

    project_root = _project_root()
    output_host = (
        _host_path(request.output_dir, request.runtime.wsl_distribution or "", project_root)
        if request.runtime.execution == "wsl"
        else (
            Path(request.output_dir)
            if _is_windows_absolute(request.output_dir) or _is_posix_absolute(request.output_dir)
            else (project_root / request.output_dir).resolve()
        )
    )
    source_index_host = _host_json_path(request, project_root)
    expected_count = _read_index_count(source_index_host)
    output_host.mkdir(parents=True, exist_ok=True)
    request_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    if not request.runtime.repo_path:
        raise ValueError("M5 runtime.repo_path is required")
    if not request.runtime.checkpoint_path:
        raise ValueError("M5 runtime.checkpoint_path is required")

    payload = request.model_dump(mode="json")
    if request.runtime.execution == "wsl":
        distribution = request.runtime.wsl_distribution or ""
        worker_script = _wsl_path(
            str(project_root / "tools" / "m5_model_worker.py"),
            distribution,
            project_root,
        )
        translated = _translate_request_for_wsl(payload, distribution, project_root)
        request_arg = _wsl_path(str(request_file), distribution, project_root)
        result_arg = _wsl_path(str(result_file), distribution, project_root)
        python_executable = str(translated["runtime"]["python_executable"])
        probe = _wsl_exec(distribution, "test", "-x", python_executable)
        if probe.returncode != 0:
            raise FileNotFoundError(
                f"Configured WSL Python interpreter is missing or not executable: "
                f"{python_executable} (distribution {distribution})"
            )
        command = [
            "wsl.exe",
            "--distribution",
            distribution,
            "--exec",
            python_executable,
            worker_script,
            "--request",
            request_arg,
            "--result",
            result_arg,
        ]
        payload = translated
    else:
        worker_script = project_root / "tools" / "m5_model_worker.py"
        python_executable = request.runtime.python_executable or sys.executable
        if not worker_script.is_file():
            raise FileNotFoundError(f"M5 model worker script is missing: {worker_script}")
        command = [
            python_executable,
            str(worker_script),
            "--request",
            str(request_file),
            "--result",
            str(result_file),
        ]

    request_file.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if result_file.exists():
        result_file.unlink()
    log_file = result_file.with_suffix(result_file.suffix + ".log")
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
        raise RuntimeError(
            f"Could not start M5 {request.task} worker with interpreter "
            f"{python_executable!r}; see {log_file}: {exc}"
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            f"M5 {request.task} worker exited with status {completed.returncode}; "
            f"see {log_file}\n{_tail(log_file)}"
        )
    if not result_file.is_file():
        raise RuntimeError(
            f"M5 {request.task} worker exited successfully without writing "
            f"its result JSON; see {log_file}"
        )

    raw_result: Any = json.loads(result_file.read_text(encoding="utf-8"))
    if request.runtime.execution == "wsl":
        if not isinstance(raw_result, dict) or not isinstance(raw_result.get("frame_paths"), list):
            raise ValueError("WSL worker result JSON must contain a frame_paths list")
        raw_result["frame_paths"] = _translate_frame_paths(
            raw_result["frame_paths"],
            request.runtime.wsl_distribution or "",
        )
    result = ModelResult.model_validate(raw_result)
    if result.task != request.task:
        raise ValueError(
            f"M5 worker task mismatch: requested {request.task!r}, returned {result.task!r}"
        )
    _ensure_result_paths(result, output_host, expected_count)
    return result
