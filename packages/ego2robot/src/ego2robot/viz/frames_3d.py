"""3-D debug views: camera frustum, hand keypoints, TCP axes and the robot arm.

Uses matplotlib's 3-D axes with the unit cube forced to equal aspect so the
geometry is not visually distorted — a skewed aspect ratio is exactly the kind
of thing that makes a wrong axis convention look plausible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "draw_camera_frustum",
    "draw_frame_axes",
    "draw_robot_arm",
    "draw_trajectory_3d",
    "render_scene_figure",
]

#: canonical axis colours: x red, y green, z blue (matplotlib defaults)
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")


def draw_frame_axes(
    ax: Any,
    T_A_F: NDArray[np.floating[Any]],
    *,
    length: float = 0.05,
    label: str | None = None,
    alpha: float = 1.0,
    linewidth: float = 2.0,
) -> None:
    """Draw the three axes of frame ``F`` given ``T_A_F``."""
    T = np.asarray(T_A_F, dtype=np.float64)
    origin = T[:3, 3]
    for i, colour in enumerate(AXIS_COLORS):
        direction = T[:3, i] * length
        ax.quiver(
            *origin,
            *direction,
            color=colour,
            alpha=alpha,
            linewidth=linewidth,
            arrow_length_ratio=0.25,
        )
    if label:
        ax.text(*origin, f" {label}", color="black", fontsize=8)


def draw_camera_frustum(
    ax: Any,
    T_A_C: NDArray[np.floating[Any]],
    *,
    depth: float = 0.25,
    half_width: float = 0.12,
    half_height: float = 0.07,
    colour: str = "#444444",
    label: str = "camera C",
) -> None:
    """Draw a camera optical frustum given ``T_A_C`` (camera pose in A)."""
    T = np.asarray(T_A_C, dtype=np.float64)
    R, o = T[:3, :3], T[:3, 3]
    corners_C = np.array(
        [
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
        ]
    )
    corners = (R @ corners_C.T).T + o
    for c in corners:
        ax.plot([o[0], c[0]], [o[1], c[1]], [o[2], c[2]], color=colour, linewidth=0.8, alpha=0.8)
    loop = np.vstack([corners, corners[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=colour, linewidth=1.2)
    draw_frame_axes(ax, T, length=depth * 0.5, label=label)


def draw_robot_arm(
    ax: Any,
    body_positions: dict[str, NDArray[np.floating[Any]]],
    *,
    order: Sequence[str] = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"),
    colour: str = "#ff7f0e",
) -> None:
    """Draw the kinematic chain of the arm as a polyline plus joints."""
    pts = [np.asarray(body_positions[n], dtype=np.float64) for n in order if n in body_positions]
    if len(pts) < 2:
        return
    arr = np.vstack(pts)
    ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=colour, linewidth=2.5, marker="o", markersize=4)


def draw_trajectory_3d(
    ax: Any,
    points: NDArray[np.floating[Any]],
    *,
    colour: str = "#1f77b4",
    linewidth: float = 1.5,
    label: str | None = None,
    alpha: float = 1.0,
) -> None:
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
        return
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color=colour, linewidth=linewidth, label=label, alpha=alpha)


def render_scene_figure(
    path: str | Path,
    *,
    title: str,
    draw: Any,
    point_sets: Sequence[NDArray[np.floating[Any]]] = (),
    azimuth_deg: float = -60.0,
    elevation_deg: float = 18.0,
    legend: bool = True,
) -> Path:
    """Build a 3-D figure with equal aspect and save it.

    ``draw(ax)`` performs the actual drawing so callers control what appears.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    draw(ax)

    all_points = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in point_sets if np.size(p)]
    if all_points:
        stacked = np.vstack(all_points)
        finite = stacked[np.isfinite(stacked).all(axis=1)]
        if finite.size:
            centre = finite.mean(axis=0)
            radius = max(float(np.max(np.abs(finite - centre))) * 1.15, 0.05)
            ax.set_xlim(centre[0] - radius, centre[0] + radius)
            ax.set_ylim(centre[1] - radius, centre[1] + radius)
            ax.set_zlim(centre[2] - radius, centre[2] + radius)
            ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.view_init(elev=elevation_deg, azim=azimuth_deg)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper left", fontsize=8)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.stem}.tmp{p.suffix}")
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(p)
    return p
