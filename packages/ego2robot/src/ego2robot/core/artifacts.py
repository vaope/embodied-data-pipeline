"""Stage artifact layout, fingerprints, checksums, atomic publish and resume.

System design section 12 ("stage 契约") and section 13 ("canonical episode 存储布局").

Three guarantees implemented here:

1. **Determinism / resume** -- a stage records its input fingerprints and
   config hash; a rerun with identical values and a ``status: succeeded``
   manifest may be skipped.
2. **No half-products** -- every write goes to a sibling temporary directory and
   is published with an atomic rename only after it validates.  A failed stage
   writes a ``failed`` manifest and removes its partial output.
3. **Source immutability** -- nothing in ``source/`` is ever written by a stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from pydantic import BaseModel

from .schema import StageManifest

__all__ = [
    "EpisodeLayout",
    "atomic_write_bytes",
    "atomic_write_json",
    "config_hash",
    "fingerprint",
    "file_sha256",
    "find_repo_root",
    "hash_arrays",
    "hash_json",
    "read_json",
    "read_stage_manifest",
    "stable_json",
    "utc_now_iso",
    "write_stage_manifest",
]

# Directories inside an episode that a stage may never write to.
_READ_ONLY_DIRS = ("source",)


def utc_now_iso() -> str:
    """UTC timestamp with second resolution in ISO-8601 (``Z`` suffix)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk upwards looking for the project root (``pyproject.toml`` + ``docs``)."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "docs").is_dir():
            return candidate
    return here


# ---------------------------------------------------------------------------
# Hashing / fingerprinting
# ---------------------------------------------------------------------------


def stable_json(payload: Any) -> str:
    """Canonical JSON: sorted keys, no insignificant whitespace, ASCII-safe."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def hash_bytes(data: bytes, *, prefix: str = "sha256") -> str:
    digest = hashlib.sha256(data).hexdigest()
    return f"{prefix}:{digest}" if prefix else digest


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (never loads the whole file in memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_json(payload: Any) -> str:
    """SHA-256 over the canonical JSON encoding of ``payload``."""
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def hash_arrays(arrays: Mapping[str, Any]) -> str:
    """Content hash of a mapping of numpy arrays (shape + dtype + bytes)."""
    import numpy as np

    h = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def config_hash(resolved: Mapping[str, Any]) -> str:
    """Hash of a fully resolved configuration document."""
    return hash_json(resolved)


def fingerprint(
    *,
    stage: str,
    stage_version: str,
    inputs: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    code_revision: str = "",
) -> str:
    """Stage fingerprint: identity of (code, version, declared input hashes, config).

    Two runs with an identical fingerprint are interchangeable, which is what
    makes ``resume`` sound (system-design.md 12).
    """
    return hash_json(
        {
            "stage": stage,
            "stage_version": stage_version,
            "inputs": dict(sorted(inputs.items())),
            "config": resolved_config,
            "code_revision": code_revision,
        }
    )


# ---------------------------------------------------------------------------
# Atomic IO
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write bytes to ``path`` via a temporary file plus ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _json_default(value: Any) -> Any:
    """JSON fallback for numpy scalars/arrays and paths.

    ``json.dumps`` cannot serialise ``np.bool_`` / ``np.float64`` /
    ``np.ndarray``, and stage metrics routinely contain them.  Converting here
    keeps manifests writable without every stage having to remember to cast.
    """
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def json_safe(payload: Any) -> Any:
    """Recursively convert a payload into something ``json.dumps`` can encode.

    Public so report writers that build their own strings do not have to
    rediscover the numpy problem (see :func:`_json_default`).
    """
    if isinstance(payload, BaseModel):
        return json_safe(payload.model_dump(mode="json", by_alias=True))
    if isinstance(payload, Mapping):
        return {str(k): json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in payload]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    return _json_default(payload)


def atomic_write_json(path: str | Path, payload: BaseModel | Mapping[str, Any] | Any, *, indent: int = 2) -> Path:
    """Serialise ``payload`` to canonical JSON and publish it atomically."""
    if isinstance(payload, BaseModel):
        data = payload.model_dump(mode="json", by_alias=True)
    else:
        data = payload
    text = json.dumps(data, indent=indent, sort_keys=True, default=_json_default)
    return atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Episode layout
# ---------------------------------------------------------------------------


@dataclass
class EpisodeLayout:
    """Filesystem contract for one episode (system-design.md 13)."""

    root: Path
    robot_id: str = "franka_panda"

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- directories --------------------------------------------------------

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def hand_dir(self) -> Path:
        return self.root / "hand"

    @property
    def canonical_dir(self) -> Path:
        return self.root / "canonical"

    @property
    def robot_dir(self) -> Path:
        return self.root / "robot" / self.robot_id

    @property
    def rig_dir(self) -> Path:
        """Two-arm rig products (ADR-0022): one sub-directory per arm plus the rig record."""
        return self.root / "robot" / "rig"

    def rig_arm_dir(self, arm_id: str) -> Path:
        return self.rig_dir / arm_id

    @property
    def rig_json(self) -> Path:
        return self.rig_dir / "rig.json"

    @property
    def rig_contacts_npz(self) -> Path:
        return self.rig_dir / "contacts.npz"

    @property
    def visual_dir(self) -> Path:
        return self.root / "visual"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def depth_dir(self) -> Path:
        return self.root / "depth"

    @property
    def quality_dir(self) -> Path:
        return self.root / "quality"

    @property
    def previews_dir(self) -> Path:
        return self.root / "previews"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def stage_dir(self, stage: str) -> Path:
        """Directory that owns ``stage``'s outputs.

        Every stage must own a **distinct** directory, because a stage's
        ``stage-manifest.json`` lives there: ``robot`` and ``render`` both concern
        the robot, but if they shared ``robot/<robot_id>/`` one manifest would
        silently overwrite the other.  ``render`` therefore publishes into
        ``visual/``, which is also where system-design.md 13 puts the rendered
        video products.
        """
        return {
            "ingest": self.source_dir,
            "hand": self.hand_dir,
            "retarget": self.canonical_dir,
            "robot": self.robot_dir,
            "rig": self.rig_dir,
            "render": self.visual_dir,
            "scene_depth": self.root / "scene",
            "composite": self.root / "composite",
            "projection": self.previews_dir,
            "quality": self.quality_dir,
        }.get(stage, self.root / stage)

    # -- files --------------------------------------------------------------

    @property
    def episode_json(self) -> Path:
        return self.root / "episode.json"

    @property
    def camera_json(self) -> Path:
        return self.source_dir / "camera.json"

    @property
    def video_path(self) -> Path:
        return self.source_dir / "video.mp4"

    @property
    def hand_npz(self) -> Path:
        return self.hand_dir / "sequence.npz"

    @property
    def actions_npz(self) -> Path:
        return self.canonical_dir / "actions.npz"

    @property
    def raw_actions_npz(self) -> Path:
        return self.canonical_dir / "actions-raw.npz"

    @property
    def filtered_actions_npz(self) -> Path:
        return self.canonical_dir / "actions-filtered.npz"

    @property
    def delta_npz(self) -> Path:
        return self.canonical_dir / "actions-delta.npz"

    @property
    def base_json(self) -> Path:
        return self.robot_dir / "base.json"

    @property
    def trajectory_npz(self) -> Path:
        return self.robot_dir / "trajectory.npz"

    @property
    def render_npz(self) -> Path:
        return self.robot_dir / "render.npz"

    @property
    def quality_json(self) -> Path:
        return self.quality_dir / "report.json"

    @property
    def resolved_config_json(self) -> Path:
        return self.configs_dir / "resolved.json"

    def manifest_path(self, stage: str) -> Path:
        return self.stage_dir(stage) / "stage-manifest.json"

    def preview(self, name: str) -> Path:
        return self.previews_dir / name

    def ensure(self) -> EpisodeLayout:
        for d in (
            self.source_dir,
            self.hand_dir,
            self.canonical_dir,
            self.robot_dir,
            self.visual_dir,
            self.masks_dir,
            self.depth_dir,
            self.quality_dir,
            self.previews_dir,
            self.configs_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def is_source_path(self, path: str | Path) -> bool:
        """True when ``path`` lives under an immutable source directory."""
        try:
            rel = Path(path).resolve().relative_to(self.root.resolve())
        except ValueError:
            return False
        return bool(rel.parts) and rel.parts[0] in _READ_ONLY_DIRS

    def iter_artifacts(self) -> Iterator[Path]:
        """Yield every file in the episode (for checksum manifests)."""
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                yield path


# ---------------------------------------------------------------------------
# Stage manifest read/write
# ---------------------------------------------------------------------------


def read_stage_manifest(path: str | Path) -> StageManifest | None:
    """Load a stage manifest, returning ``None`` when absent or unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return StageManifest.model_validate(read_json(p))
    except Exception:
        return None


def write_stage_manifest(layout: EpisodeLayout, manifest: StageManifest) -> Path:
    """Atomically publish ``manifest`` next to its stage outputs."""
    manifest.schema_version = "ego2robot.stage_manifest.v1"
    return atomic_write_json(layout.manifest_path(manifest.stage), manifest)


@dataclass
class StageTimer:
    """Wall-clock timer used to fill ``StageManifest.duration_s``."""

    started_at: str = field(default_factory=utc_now_iso)
    _t0: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


@dataclass
class TempPublish:
    """Context manager that swaps a temp directory into place atomically.

    Products are written into a sibling ``.<name>.tmp-XXXX`` directory; on
    success the old directory is removed and the temp one renamed into place.
    On failure the temp directory is discarded, so a crash can never leave a
    directory that looks like a successful product (system-design.md 12).
    """

    target: Path
    prefix: str | None = None

    _tmp: Path | None = field(default=None, init=False)

    def __enter__(self) -> Path:
        self.target = Path(self.target)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        prefix = self.prefix or f".{self.target.name}.tmp-"
        self._tmp = Path(tempfile.mkdtemp(dir=str(self.target.parent), prefix=prefix))
        return self._tmp

    def publish(self) -> Path:
        if self._tmp is None:
            raise RuntimeError("TempPublish.publish() called outside of its context")
        backup: Path | None = None
        if self.target.exists():
            backup = self.target.with_name(f".{self.target.name}.old-{os.getpid()}")
            os.replace(self.target, backup)
        try:
            os.replace(self._tmp, self.target)
        except BaseException:
            if backup is not None and not self.target.exists():
                os.replace(backup, self.target)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        self._tmp = None
        return self.target

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
        return False


def cleanup_partials(directory: Path, *, keep: Iterable[str] = ()) -> list[Path]:
    """Remove leftover ``.tmp``/``.old`` entries from an interrupted publish."""
    removed: list[Path] = []
    keep_set = set(keep)
    if not directory.is_dir():
        return removed
    for entry in directory.iterdir():
        if entry.name in keep_set:
            continue
        if entry.name.startswith(".") and (".tmp-" in entry.name or ".old-" in entry.name):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            removed.append(entry)
    return removed
