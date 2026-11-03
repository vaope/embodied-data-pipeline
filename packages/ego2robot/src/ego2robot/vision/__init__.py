"""Vision layer: segmentation, inpainting, depth and rendering.

M5/M6 modules can be imported without initializing the MuJoCo renderer.  The
existing renderer exports remain available through lazy attribute loading.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .render import (
        MJ_CAM_TO_OPTICAL_BASIS,
        CameraSpec,
        MujocoRenderer,
        build_scene_xml,
        camera_spec_from_pinhole,
        mujoco_camera_basis,
        render_robot_scene,
        verify_projection,
    )

__all__ = [
    "CameraSpec",
    "MJ_CAM_TO_OPTICAL_BASIS",
    "MujocoRenderer",
    "build_scene_xml",
    "camera_spec_from_pinhole",
    "mujoco_camera_basis",
    "render_robot_scene",
    "verify_projection",
]

_RENDER_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Load existing render exports only when a caller requests one."""
    if name not in _RENDER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    render_module = import_module(".render", __name__)
    value = getattr(render_module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RENDER_EXPORTS)
