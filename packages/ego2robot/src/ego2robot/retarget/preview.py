"""Retarget previews (M2 evidence artifacts).

The retarget stage turns hand geometry into canonical gripper poses.  Existing
previews show *per-frame* TCP axes on the video (``perception/hand/preview.py``);
what a reviewer cannot see there is the **relation between two hands over time**,
which is the whole point of the two-hand contract (ADR-0021).  This module adds:

``previews/gripper-width-per-hand.png``
    one opening curve per hand, with invalid frames marked;
``previews/tcp-relative-between-hands.png`` (H=2 only)
    distance and relative rotation between the two TCP frames, raw vs final, so
    an independent per-hand smoothing step that damaged the coordination is
    visible instead of being absorbed into a judgement about "smoothness";
``previews/tcp-raw-vs-final-per-hand.png``
    how far smoothing/filtering moved each hand;
``previews/tcp-3d.png``
    both TCP trajectories in frame C next to the camera frustum;
``previews/retarget-metrics.json``
    the numbers behind the plots (per-hand denominators included).

No new coupling optimiser is introduced here: the two-hand relation is
diagnostic only (PLAN-02 B1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.schema import CanonicalActionSequence
from ..core.transforms import geodesic_distance_rad
from ..viz.frames_3d import draw_camera_frustum, draw_trajectory_3d, render_scene_figure

__all__ = ["write_retarget_previews"]

_HAND_COLOURS = {"right": "#1f77b4", "left": "#ff7f0e"}


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _hand_labels(action: CanonicalActionSequence) -> list[str]:
    ids = action.hand_ids
    if len(ids) == action.num_hands:
        return list(ids)
    return [f"hand{h}" for h in range(action.num_hands)]


def write_retarget_previews(
    *,
    action: CanonicalActionSequence,
    raw_action: CanonicalActionSequence,
    previews_dir: Path,
    episode_id: str = "",
    dry_run: bool = False,
) -> list[Path]:
    """Write the per-hand / cross-hand retarget evidence figures."""
    outputs: list[Path] = []
    if dry_run:
        return outputs

    plt = _plt()
    hand_ids = _hand_labels(action)
    ts = np.asarray(action["timestamp_s"], dtype=np.float64)
    valid = np.asarray(action["valid"], dtype=bool)
    width = np.asarray(action["gripper_width_m"], dtype=np.float64)
    position = np.asarray(action["tcp_position_C_m"], dtype=np.float64)
    rotation = np.asarray(action["T_C_TCP"], dtype=np.float64)[:, :, :3, :3]
    raw_position = np.asarray(raw_action["tcp_position_C_m"], dtype=np.float64)
    raw_rotation = np.asarray(raw_action["T_C_TCP"], dtype=np.float64)[:, :, :3, :3]
    raw_valid = np.asarray(raw_action["valid"], dtype=bool)

    # ---- 1. gripper width per hand ----------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
    for h, hand_id in enumerate(hand_ids):
        colour = _HAND_COLOURS.get(hand_id, "#7f7f7f")
        w = width[:, h]
        ax.plot(ts, w * 1000.0, color=colour, linewidth=1.6, label=f"{hand_id} width")
        bad = ~valid[:, h]
        if bad.any():
            ax.scatter(
                ts[bad],
                np.nan_to_num(w[bad] * 1000.0, nan=0.0),
                color=colour,
                marker="x",
                s=18,
                alpha=0.65,
                label=f"{hand_id} invalid",
            )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("gripper opening [mm]")
    ax.set_title(f"canonical gripper width per hand ({episode_id})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = previews_dir / "gripper-width-per-hand.png"
    fig.savefig(p)
    plt.close(fig)
    outputs.append(p)

    # ---- 2. raw vs final displacement per hand ----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=110, sharex=True)
    for h, hand_id in enumerate(hand_ids):
        colour = _HAND_COLOURS.get(hand_id, "#7f7f7f")
        dp = np.linalg.norm(position[:, h] - raw_position[:, h], axis=1) * 1000.0
        dr = np.degrees(
            [
                geodesic_distance_rad(raw_rotation[t, h], rotation[t, h])
                if np.isfinite(raw_rotation[t, h]).all() and np.isfinite(rotation[t, h]).all()
                else np.nan
                for t in range(ts.shape[0])
            ]
        )
        axes[0].plot(ts, dp, color=colour, label=hand_id)
        axes[1].plot(ts, dr, color=colour, label=hand_id)
    axes[0].set_ylabel("|raw - final| [mm]")
    axes[1].set_ylabel("angle(raw, final) [deg]")
    axes[1].set_xlabel("time [s]")
    for a in axes:
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"effect of filtering/gaps/smoothing per hand ({episode_id})")
    fig.tight_layout()
    p = previews_dir / "tcp-raw-vs-final-per-hand.png"
    fig.savefig(p)
    plt.close(fig)
    outputs.append(p)

    # ---- 3. cross-hand relation (only meaningful with two hands) ---------
    metrics: dict[str, Any] = {
        "episode_id": episode_id,
        "hand_ids": hand_ids,
        "num_frames": int(ts.shape[0]),
        "per_hand": {},
        "notes": [
            "cross-hand distance/angle are diagnostics of the two canonical TCP frames, "
            "not evidence of physical contact or of a working two-arm controller",
            "raw and final curves are plotted together so independent per-hand smoothing "
            "cannot silently change the hand-to-hand relation",
        ],
    }
    for h, hand_id in enumerate(hand_ids):
        column = valid[:, h]
        tcp_steps = np.linalg.norm(np.diff(position[:, h], axis=0), axis=1)
        tcp_steps = tcp_steps[np.isfinite(tcp_steps)]
        metrics["per_hand"][hand_id] = {
            "valid_frames": int(np.count_nonzero(column)),
            "valid_ratio": round(float(np.mean(column)), 6) if column.size else 0.0,
            "width_min_m": round(float(np.nanmin(width[column, h])), 6) if column.any() else None,
            "width_max_m": round(float(np.nanmax(width[column, h])), 6) if column.any() else None,
            "fallback_frames": int(
                np.count_nonzero(np.asarray(action["fallback_code"])[:, h] != 0)
            ),
            # Path length of the canonical TCP.  A *static* hand still shows a
            # non-trivial value because the jaw centre follows finger noise, so
            # this is reported together with the per-frame step statistics and
            # must not be read as "how much this hand participated".
            "tcp_path_length_m": _travel(position[:, h]),
            "tcp_step_median_m": round(float(np.median(tcp_steps)), 6) if tcp_steps.size else 0.0,
            "tcp_step_p95_m": round(float(np.percentile(tcp_steps, 95)), 6) if tcp_steps.size else 0.0,
            "tcp_path_length_note": (
                "cumulative |dTCP|; includes per-frame annotation jitter, so a holding "
                "hand can have a large value -- not an activity or participation measure"
            ),
        }

    if len(hand_ids) == 2:
        both = valid[:, 0] & valid[:, 1]
        dist_final = np.linalg.norm(position[:, 0] - position[:, 1], axis=1)
        dist_raw = np.linalg.norm(raw_position[:, 0] - raw_position[:, 1], axis=1)
        angle_final = np.array(
            [
                np.degrees(geodesic_distance_rad(rotation[t, 0], rotation[t, 1]))
                if np.isfinite(rotation[t, 0]).all() and np.isfinite(rotation[t, 1]).all()
                else np.nan
                for t in range(ts.shape[0])
            ]
        )
        angle_raw = np.array(
            [
                np.degrees(geodesic_distance_rad(raw_rotation[t, 0], raw_rotation[t, 1]))
                if np.isfinite(raw_rotation[t, 0]).all() and np.isfinite(raw_rotation[t, 1]).all()
                else np.nan
                for t in range(ts.shape[0])
            ]
        )
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=110, sharex=True)
        axes[0].plot(ts, dist_final * 1000.0, color="#1f77b4", label="final")
        axes[0].plot(ts, dist_raw * 1000.0, color="#1f77b4", linestyle="--", alpha=0.6, label="raw")
        axes[0].set_ylabel("|TCP_R - TCP_L| [mm]")
        axes[1].plot(ts, angle_final, color="#d62728", label="final")
        axes[1].plot(ts, angle_raw, color="#d62728", linestyle="--", alpha=0.6, label="raw")
        axes[1].set_ylabel("relative rotation [deg]")
        axes[1].set_xlabel("time [s]")
        for a in axes:
            a.grid(True, alpha=0.3)
            a.legend(fontsize=8)
        fig.suptitle(f"hand-to-hand TCP relation, raw vs final ({episode_id})")
        fig.tight_layout()
        p = previews_dir / "tcp-relative-between-hands.png"
        fig.savefig(p)
        plt.close(fig)
        outputs.append(p)

        finite = both & np.isfinite(dist_final) & np.isfinite(angle_final)
        metrics["cross_hand"] = {
            "jointly_valid_frames": int(np.count_nonzero(both)),
            "jointly_valid_ratio": round(float(np.mean(both)), 6) if both.size else 0.0,
            "distance_min_m": round(float(np.min(dist_final[finite])), 6) if finite.any() else None,
            "distance_median_m": round(float(np.median(dist_final[finite])), 6) if finite.any() else None,
            "distance_max_m": round(float(np.max(dist_final[finite])), 6) if finite.any() else None,
            "relative_rotation_median_deg": (
                round(float(np.median(angle_final[finite])), 4) if finite.any() else None
            ),
            "raw_final_distance_median_delta_m": (
                round(float(np.median(np.abs(dist_final - dist_raw)[finite])), 9)
                if finite.any()
                else None
            ),
            "raw_final_rotation_median_delta_deg": (
                round(float(np.median(np.abs(angle_final - angle_raw)[finite])), 6)
                if finite.any()
                else None
            ),
            "statistics_scope": "frames where both hands are valid",
        }

    # ---- 4. 3-D trajectories --------------------------------------------
    def _draw(ax: Any) -> None:
        draw_camera_frustum(ax, np.eye(4), label="camera C")
        for h, hand_id in enumerate(hand_ids):
            colour = _HAND_COLOURS.get(hand_id, "#7f7f7f")
            draw_trajectory_3d(ax, position[:, h], colour=colour, label=f"{hand_id} TCP (final)")
            draw_trajectory_3d(
                ax,
                raw_position[:, h],
                colour=colour,
                linewidth=0.9,
                alpha=0.45,
                label=f"{hand_id} TCP (raw)",
            )

    p = render_scene_figure(
        previews_dir / "tcp-3d.png",
        title=f"canonical TCP trajectories in frame C ({episode_id})",
        draw=_draw,
        point_sets=[position.reshape(-1, 3), raw_position.reshape(-1, 3)],
    )
    outputs.append(p)

    payload_path = previews_dir / "retarget-metrics.json"
    from ..core import artifacts as A

    A.atomic_write_json(payload_path, metrics)
    outputs.append(payload_path)
    return outputs


def _travel(points: np.ndarray) -> float:
    """Cumulative finite path length of one hand's TCP, in metres."""
    P = np.asarray(points, dtype=np.float64)
    if P.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    steps = steps[np.isfinite(steps)]
    return round(float(np.sum(steps)), 4) if steps.size else 0.0
