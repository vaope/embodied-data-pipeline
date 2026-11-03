"""Instantiate a robot adapter from the resolved robot profile.

This module keeps morphology selection outside perception and retargeting.
Each registered adapter owns its MJCF-derived kinematics and TCP contract.
"""

from __future__ import annotations

from importlib import import_module
import hashlib
from pathlib import Path
from typing import Any


_SPECS = {
    "franka_panda": ("ego2robot.robots.franka.adapter", "FrankaPanda", "build_franka_config", "franka_emika_panda/panda.xml"),
    "ur5e": ("ego2robot.robots.ur5e.adapter", "UR5e", "build_ur5e_config", "universal_robots_ur5e/ur5e.xml"),
    "xarm7": ("ego2robot.robots.xarm7.adapter", "XArm7", "build_xarm7_config", "ufactory_xarm7/xarm7.xml"),
    "viperx300": ("ego2robot.robots.viperx300.adapter", "ViperX300", "build_viperx_config", "trossen_vx300s/vx300s.xml"),
}


def resolve_mjcf(ctx: Any) -> Path:
    """Resolve the configured model, refusing an unknown robot ID."""

    cfg = ctx.config.robot_binding
    spec = _SPECS.get(cfg.robot_id)
    if spec is None:
        raise ValueError(f"unsupported robot_id {cfg.robot_id!r}; available: {', '.join(_SPECS)}")
    uri = ctx.config.case.get("mjcf_uri") or cfg.mjcf_uri or str(Path("assets/mjcf") / spec[3])
    path = Path(uri)
    if not path.is_absolute():
        path = Path(ctx.repo_root) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MJCF for {cfg.robot_id} is missing: {path}")
    return path


def model_asset_hashes(mjcf_uri: Path) -> dict[str, str]:
    """Fingerprint the MJCF and its colocated meshes/textures for resume."""

    suffixes = {".xml", ".obj", ".stl", ".msh", ".png", ".jpg", ".jpeg"}
    root = mjcf_uri.parent
    return {
        file.relative_to(root).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(root.rglob("*"))
        if file.is_file() and file.suffix.lower() in suffixes
    }


def adapter_source(robot_id: str) -> Path:
    spec = _SPECS.get(robot_id)
    if spec is None:
        raise ValueError(f"unsupported robot_id {robot_id!r}")
    module = import_module(spec[0])
    return Path(module.__file__).resolve()


def build_robot(ctx: Any, *, ik_settings: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """Return ``(RobotConfig, RobotModel)`` for the selected morphology."""

    cfg = ctx.config.robot_binding
    module_name, class_name, builder_name, _ = _SPECS[cfg.robot_id]
    mjcf_uri = resolve_mjcf(ctx)
    module = import_module(module_name)
    builder = getattr(module, builder_name)
    robot_class = getattr(module, class_name)
    robot_config = builder(mjcf_uri=mjcf_uri, profile=cfg)
    if robot_config.robot_id != cfg.robot_id:
        raise ValueError("robot adapter returned a different robot_id than the resolved profile")
    robot = robot_class(mjcf_uri=mjcf_uri, config=robot_config, ik_settings=ik_settings)
    return robot_config, robot


def build_robot_for(
    robot_id: str,
    *,
    repo_root: str | Path,
    ik_settings: dict[str, Any] | None = None,
    mjcf_uri: str | Path | None = None,
) -> tuple[Any, Any]:
    """Build a robot by explicit id, without a resolved-config context.

    The two-arm rig needs two adapters of possibly different morphologies while the
    configuration carries one ``robot_binding`` block; this is the seam that lets
    the rig stage build each arm independently without a fake context.
    """
    spec = _SPECS.get(robot_id)
    if spec is None:
        raise ValueError(f"unsupported robot_id {robot_id!r}; available: {', '.join(_SPECS)}")
    module_name, class_name, builder_name, default_rel = spec
    path = Path(mjcf_uri) if mjcf_uri is not None else Path(repo_root) / "assets" / "mjcf" / default_rel
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MJCF for {robot_id} is missing: {path}")
    module = import_module(module_name)
    builder = getattr(module, builder_name)
    robot_class = getattr(module, class_name)
    robot_config = builder(mjcf_uri=path, robot_id=robot_id)
    robot = robot_class(mjcf_uri=path, config=robot_config, ik_settings=ik_settings)
    return robot_config, robot
