"""Fetch the pinned MuJoCo Menagerie ViperX-300 model without updating its revision.

Usage: ``python scripts/fetch_viperx300.py``

The script uses a sparse Git checkout of one upstream model directory. Existing
asset contents are never deleted or overwritten: an exact existing copy is
verified, while a mismatch stops with a diagnostic so changes can be reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
UPSTREAM_PATH = "trossen_vx300s"
PINNED_COMMIT = "c96a32d28fb5da84da38c1da4d749e7a13212855"
LICENSE = "BSD-3-Clause"
MODEL_XML = "vx300s.xml"
TROSSEN_SPECS_URL = (
    "https://docs.trossenrobotics.com/interbotix_xsarms_docs/"
    "specifications/vx300s.html"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _fetch_source(temp_root: Path) -> Path:
    checkout = temp_root / "mujoco_menagerie"
    checkout.mkdir()
    _git("init", str(checkout))
    _git("-C", str(checkout), "remote", "add", "origin", REPOSITORY_URL)
    _git("-C", str(checkout), "fetch", "--depth=1", "--filter=blob:none", "origin", PINNED_COMMIT)
    _git("-C", str(checkout), "sparse-checkout", "init", "--cone")
    _git("-C", str(checkout), "sparse-checkout", "set", UPSTREAM_PATH)
    _git("-C", str(checkout), "checkout", "--detach", "FETCH_HEAD")
    actual_commit = _git("-C", str(checkout), "rev-parse", "HEAD")
    if actual_commit != PINNED_COMMIT:
        raise RuntimeError(f"checked out {actual_commit}, expected pinned commit {PINNED_COMMIT}")
    source = checkout / UPSTREAM_PATH
    if not (source / MODEL_XML).is_file() or not (source / "LICENSE").is_file():
        raise RuntimeError(f"upstream model directory is incomplete: {source}")
    return source


def _source_files(source: Path) -> list[Path]:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"no files found in upstream directory {source}")
    if any(path.is_symlink() for path in files):
        raise RuntimeError("symlinked upstream model files are not supported")
    return files


def _file_records(source: Path, files: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(source).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]


def _verify_existing(target: Path, expected: list[dict[str, Any]]) -> None:
    actual_files = sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and path.name != "PROVENANCE.json"
    )
    actual = {
        path.relative_to(target).as_posix(): _sha256(path)
        for path in actual_files
    }
    wanted = {record["path"]: record["sha256"] for record in expected}
    if actual != wanted:
        missing = sorted(set(wanted) - set(actual))
        extra = sorted(set(actual) - set(wanted))
        changed = sorted(name for name in set(actual) & set(wanted) if actual[name] != wanted[name])
        raise RuntimeError(
            "existing ViperX assets differ from the pinned source; "
            f"missing={missing}, extra={extra}, changed={changed}. "
            "Review and move the existing directory aside before fetching."
        )


def fetch(destination: Path) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e2r-viperx-fetch-") as temp:
        source = _fetch_source(Path(temp))
        files = _source_files(source)
        records = _file_records(source, files)
        if destination.exists():
            if not destination.is_dir():
                raise RuntimeError(f"destination exists and is not a directory: {destination}")
            _verify_existing(destination, records)
        else:
            destination.mkdir(parents=True)
            for src in files:
                relative = src.relative_to(source)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)

        provenance = {
            "schema_version": "ego2robot.robot_model_provenance.v1",
            "robot_id": "viperx300",
            "model_name": "Trossen ViperX-300 6DOF",
            "upstream_repository": REPOSITORY_URL,
            "upstream_path": UPSTREAM_PATH,
            "upstream_commit": PINNED_COMMIT,
            "upstream_commit_url": f"https://github.com/google-deepmind/mujoco_menagerie/commit/{PINNED_COMMIT}",
            "model_xml": MODEL_XML,
            "license": LICENSE,
            "manufacturer_specifications": {
                "url": TROSSEN_SPECS_URL,
                "published_reach_m": 0.75,
                "published_gripper_range_m": [0.042, 0.116],
                "measurement_note": (
                    "Manufacturer gripper range is retained as source evidence; the adapter uses "
                    "the MJCF collision-pad gap range for simulation mapping."
                ),
            },
            "files": records,
        }
        manifest_path = destination / "PROVENANCE.json"
        manifest_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return destination / "PROVENANCE.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "trossen_vx300s",
        help="destination directory (defaults to assets/mjcf/trossen_vx300s)",
    )
    args = parser.parse_args()
    manifest = fetch(args.destination)
    print(f"ViperX-300 model verified/fetched: {manifest.parent}")
    print(f"Provenance manifest: {manifest}")


if __name__ == "__main__":
    main()
