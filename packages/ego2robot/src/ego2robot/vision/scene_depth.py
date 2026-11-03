"""Scene depth: the typed quantity the depth-aware compositor is allowed to consume.

ADR-0008 is the normative decision.  The short version:

* a depth field carries an explicit ``mode`` — ``metric``, ``aligned`` or
  ``relative`` — and the compositor **raises** on ``relative``;
* ``relative`` depth is affine-invariant, so it can never be compared with the
  metric depth MuJoCo renders, no matter how plausible the resulting frame looks;
* promoting ``relative`` to ``aligned`` goes through exactly one function,
  :func:`align_relative_depth`, which records the fit and its residual so a
  reviewer can see how much was invented.

Nothing in this module imports torch or a model library: inference lives behind
:mod:`ego2robot.vision.da3_bridge`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AnchorSamples",
    "DepthModeError",
    "DepthMode",
    "SceneDepth",
    "align_relative_depth",
    "anchors_from_hand",
    "anchors_from_tcp",
    "depth_preview",
    "validate_metric_scale",
]

DepthMode = Literal["metric", "aligned", "relative"]

_COMPOSABLE: frozenset[str] = frozenset({"metric", "aligned"})
_SCHEMA = "ego2robot.scene_depth.v1"


class DepthModeError(RuntimeError):
    """Raised when depth of the wrong metric semantics reaches a consumer."""


# ---------------------------------------------------------------------------
# anchor sampling
# ---------------------------------------------------------------------------


@dataclass
class AnchorSamples:
    """Points whose metric camera-frame depth is known independently.

    ``u``/``v`` are pixel coordinates and ``z_m`` is the trusted depth along the
    optical axis.  ``source`` names where the anchor came from so a residual can be
    attributed instead of averaged away.
    """

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    z_m: NDArray[np.float64]
    frame_index: NDArray[np.int64]
    source: str

    def __len__(self) -> int:
        return int(self.z_m.size)

    def subset(self, frame_index: NDArray[np.int64]) -> AnchorSamples:
        keep = np.isin(self.frame_index, np.asarray(frame_index, dtype=np.int64))
        return AnchorSamples(
            u=self.u[keep], v=self.v[keep], z_m=self.z_m[keep],
            frame_index=self.frame_index[keep], source=self.source,
        )


def _project(K: NDArray[np.float64], pts_C: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.where(np.abs(pts_C[..., 2]) < 1e-9, np.nan, pts_C[..., 2])
    return np.stack(
        [K[0, 0] * pts_C[..., 0] / z + K[0, 2], K[1, 1] * pts_C[..., 1] / z + K[1, 2]],
        axis=-1,
    )


def anchors_from_hand(
    hand_sequence: Any,
    K: NDArray[np.float64],
    frame_index: NDArray[np.int64],
) -> AnchorSamples:
    """Anchors from the annotated hand keypoints (Path A only).

    Only valid when the depth was estimated on frames that still show the hand: the
    inpainted background has the hand removed, so sampling a hand anchor there
    measures the surface *behind* the hand and would look like a huge error.
    """
    keypoints = np.asarray(hand_sequence["keypoints_C_m"], dtype=np.float64)
    valid = np.asarray(hand_sequence["valid"], dtype=bool)
    frames = np.asarray(hand_sequence["frame_index"], dtype=np.int64)
    wanted = np.asarray(frame_index, dtype=np.int64)
    rows_u: list[float] = []
    rows_v: list[float] = []
    rows_z: list[float] = []
    rows_f: list[int] = []
    for t, fi in enumerate(frames):
        if fi not in wanted:
            continue
        if valid.ndim == 2 and not bool(valid[t].any()):
            continue
        pts = keypoints[t, 0]
        finite = np.isfinite(pts).all(axis=-1) & (pts[:, 2] > 0)
        if not finite.any():
            continue
        uv = _project(K, pts[finite])
        for (u, v), p in zip(uv, pts[finite]):
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            rows_u.append(float(u))
            rows_v.append(float(v))
            rows_z.append(float(p[2]))
            rows_f.append(int(fi))
    return AnchorSamples(
        u=np.asarray(rows_u, dtype=np.float64),
        v=np.asarray(rows_v, dtype=np.float64),
        z_m=np.asarray(rows_z, dtype=np.float64),
        frame_index=np.asarray(rows_f, dtype=np.int64),
        source="hand_keypoints_C_m",
    )


def anchors_from_tcp(
    actions: Any,
    K: NDArray[np.float64],
    frame_index: NDArray[np.int64],
) -> AnchorSamples:
    """Anchors from the canonical TCP targets.

    The TCP sits at the human pinch point, i.e. on the manipulated object's
    surface.  After inpainting the object is still there while the hand is gone, so
    this anchor stays meaningful on the inpainted background — which is why it is
    the primary metric check rather than the hand keypoints.
    """
    # The canonical sequence exposes arrays by name only; iterating it would invoke
    # its sequence protocol, so probe the two known keys instead of listing them.
    position: np.ndarray | None = None
    for key in ("tcp_position_C_m", "position_C_m"):
        try:
            position = np.asarray(actions[key], dtype=np.float64)
        except (KeyError, IndexError):
            continue
        break
    if position is None:
        raise KeyError(
            "the canonical action sequence carries neither 'tcp_position_C_m' nor "
            "'position_C_m'"
        )
    if position.ndim == 3:
        # canonical layout is [T, H, 3]; the TCP anchors are per hand and this stage
        # is single-hand (system-design 7), so take the first hand
        position = position[:, 0]
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"expected TCP positions of shape [T, 3], got {position.shape}")
    valid = np.asarray(actions["valid"], dtype=bool).reshape(-1)
    frames = np.asarray(actions["frame_index"], dtype=np.int64)
    wanted = set(int(v) for v in np.asarray(frame_index, dtype=np.int64))
    rows_u: list[float] = []
    rows_v: list[float] = []
    rows_z: list[float] = []
    rows_f: list[int] = []
    for t, fi in enumerate(frames):
        if int(fi) not in wanted or not bool(valid[t]):
            continue
        p = position[t]
        if not np.isfinite(p).all() or p[2] <= 0:
            continue
        uv = _project(K, p)
        rows_u.append(float(uv[0]))
        rows_v.append(float(uv[1]))
        rows_z.append(float(p[2]))
        rows_f.append(int(fi))
    return AnchorSamples(
        u=np.asarray(rows_u, dtype=np.float64),
        v=np.asarray(rows_v, dtype=np.float64),
        z_m=np.asarray(rows_z, dtype=np.float64),
        frame_index=np.asarray(rows_f, dtype=np.int64),
        source="canonical_tcp_position_C_m",
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _sample_nearest(depth: NDArray[np.float32], u: float, v: float) -> float:
    h, w = depth.shape
    x = int(np.clip(round(u), 0, w - 1))
    y = int(np.clip(round(v), 0, h - 1))
    return float(depth[y, x])


def validate_metric_scale(
    seq: SceneDepth,
    anchors: AnchorSamples | list[AnchorSamples],
    cfg: Any,
    *,
    exclude_masks: dict[int, NDArray[np.bool_]] | None = None,
) -> dict[str, Any]:
    """Compare the depth field against independent metric anchors.

    Returns a report; it never rewrites the depth.  Callers decide whether a failing
    report blocks the run — the stage warns, and ``composite`` refuses ``relative``
    depth outright.

    ``exclude_masks`` drops anchors that fall inside a per-frame exclusion region.
    When the depth was estimated on an inpainted background, the hand region holds
    synthesised pixels rather than a measurement, so an anchor there would grade the
    inpainter rather than the depth model.  The compositor exempts the same region
    from occlusion for the same reason.
    """
    anchor_sets = anchors if isinstance(anchors, list) else [anchors]
    per_source: dict[str, Any] = {}
    pooled_residual: list[float] = []
    for anchor in anchor_sets:
        if len(anchor) == 0:
            per_source[anchor.source] = {"samples": 0, "note": "no anchors available"}
            continue
        idx = np.searchsorted(seq.frame_index, anchor.frame_index)
        residuals: list[float] = []
        used = 0
        skipped = 0
        for u, v, z, fi in zip(anchor.u, anchor.v, anchor.z_m, anchor.frame_index):
            if z < cfg.min_anchor_depth_m:
                skipped += 1
                continue
            pos = int(np.searchsorted(seq.frame_index, int(fi)))
            if pos >= seq.frame_index.size or int(seq.frame_index[pos]) != int(fi):
                skipped += 1
                continue
            mask = (exclude_masks or {}).get(int(fi))
            if mask is not None:
                xi = int(np.clip(round(float(u)), 0, mask.shape[1] - 1))
                yi = int(np.clip(round(float(v)), 0, mask.shape[0] - 1))
                if bool(mask[yi, xi]):
                    skipped += 1
                    continue
            measured = _sample_nearest(seq.depth_m[pos], float(u), float(v))
            if not np.isfinite(measured) or measured <= 0:
                skipped += 1
                continue
            residuals.append(abs(measured - float(z)))
            used += 1
        if residuals:
            arr = np.asarray(residuals, dtype=np.float64)
            pooled_residual.extend(residuals)
            per_source[anchor.source] = {
                "samples": used,
                "skipped": skipped,
                "median_abs_error_m": round(float(np.median(arr)), 6),
                "p90_abs_error_m": round(float(np.percentile(arr, 90)), 6),
                "max_abs_error_m": round(float(arr.max()), 6),
            }
        else:
            per_source[anchor.source] = {"samples": 0, "skipped": skipped}
    report: dict[str, Any] = {
        "mode": seq.mode,
        "anchor_depth_range_m": _depth_range(seq),
        "sources": per_source,
        "enabled": bool(getattr(cfg, "enabled", True)),
    }
    if pooled_residual:
        arr = np.asarray(pooled_residual, dtype=np.float64)
        report["pooled"] = {
            "samples": int(arr.size),
            "median_abs_error_m": round(float(np.median(arr)), 6),
            "p90_abs_error_m": round(float(np.percentile(arr, 90)), 6),
        }
    if not report["enabled"]:
        report["status"] = "disabled"
        return report
    pooled = report.get("pooled")
    if not pooled:
        report["status"] = "no_anchors"
        report["passes"] = False
        return report
    total = sum(int(v.get("samples", 0)) for v in per_source.values())
    possible = sum(int(v.get("samples", 0)) + int(v.get("skipped", 0)) for v in per_source.values())
    coverage = (total / possible) if possible else 0.0
    passes = (
        pooled["median_abs_error_m"] <= cfg.max_median_abs_error_m
        and pooled["p90_abs_error_m"] <= cfg.max_p90_abs_error_m
        and coverage >= cfg.min_anchor_coverage
    )
    report["anchor_coverage"] = round(float(coverage), 4)
    report["thresholds"] = {
        "max_median_abs_error_m": cfg.max_median_abs_error_m,
        "max_p90_abs_error_m": cfg.max_p90_abs_error_m,
        "min_anchor_coverage": cfg.min_anchor_coverage,
    }
    report["status"] = "passed" if passes else "failed"
    report["passes"] = bool(passes)
    return report


def _depth_range(seq: SceneDepth) -> dict[str, float] | None:
    finite = seq.depth_m[np.isfinite(seq.depth_m)]
    if finite.size == 0:
        return None
    return {
        "min": round(float(finite.min()), 4),
        "median": round(float(np.median(finite)), 4),
        "max": round(float(finite.max()), 4),
    }


# ---------------------------------------------------------------------------
# alignment (relative -> aligned)
# ---------------------------------------------------------------------------


def align_relative_depth(
    seq: SceneDepth,
    anchors: AnchorSamples | list[AnchorSamples],
    *,
    fit: Literal["scale", "scale_shift"] = "scale",
) -> tuple[SceneDepth, dict[str, Any]]:
    """Promote ``relative`` depth to ``aligned`` with a single recorded fit.

    This is the only sanctioned promotion path (ADR-0008 decision 2).  ``scale``
    solves ``z_metric ~ a * z_rel``; ``scale_shift`` solves
    ``z_metric ~ a * z_rel + b``.  The residual is returned with the fit and is the
    number a reviewer has to accept, because the fit is what turns a picture into a
    measurement.
    """
    if seq.mode != "relative":
        raise DepthModeError(
            f"align_relative_depth expects mode='relative', got {seq.mode!r}; "
            "refusing to re-fit an already metric or aligned field"
        )
    anchor_sets = anchors if isinstance(anchors, list) else [anchors]
    rel: list[float] = []
    met: list[float] = []
    for anchor in anchor_sets:
        for u, v, z, fi in zip(anchor.u, anchor.v, anchor.z_m, anchor.frame_index):
            pos = int(np.searchsorted(seq.frame_index, int(fi)))
            if pos >= seq.frame_index.size or int(seq.frame_index[pos]) != int(fi):
                continue
            measured = _sample_nearest(seq.depth_m[pos], float(u), float(v))
            if not (np.isfinite(measured) and measured > 0 and np.isfinite(z) and z > 0):
                continue
            rel.append(measured)
            met.append(float(z))
    if len(rel) < 8:
        raise DepthModeError(
            f"align_relative_depth needs at least 8 valid anchors, got {len(rel)}"
        )
    r = np.asarray(rel, dtype=np.float64)
    m = np.asarray(met, dtype=np.float64)
    if fit == "scale":
        a = float(np.dot(r, m) / np.dot(r, r))
        b = 0.0
    else:
        design = np.stack([r, np.ones_like(r)], axis=1)
        solution, *_ = np.linalg.lstsq(design, m, rcond=None)
        a, b = float(solution[0]), float(solution[1])
    predicted = a * r + b
    residual = np.abs(predicted - m)
    report = {
        "fit": fit,
        "scale": round(a, 6),
        "shift_m": round(b, 6),
        "anchors": int(r.size),
        "median_abs_error_m": round(float(np.median(residual)), 6),
        "p90_abs_error_m": round(float(np.percentile(residual, 90)), 6),
        "max_abs_error_m": round(float(residual.max()), 6),
        "anchor_sources": [a_.source for a_ in anchor_sets],
        "note": "aligned depth is a fitted quantity; the residual is its uncertainty",
    }
    aligned = seq.depth_m * np.float32(a)
    if fit == "scale_shift":
        aligned = aligned + np.float32(b)
    aligned[~(aligned > 0)] = np.nan
    out = SceneDepth(
        depth_m=aligned.astype(np.float32),
        frame_index=seq.frame_index.copy(),
        mode="aligned",
        confidence=seq.confidence,
        source=seq.source,
        provenance={**seq.provenance, "alignment": report},
    )
    return out, report


# ---------------------------------------------------------------------------
# the depth field itself
# ---------------------------------------------------------------------------


@dataclass
class SceneDepth:
    """Per-frame scene depth in the camera optical frame ``C``, metres.

    ``depth_m[t, v, u]`` is the metric z of the scene surface seen at pixel
    ``(u, v)`` of frame ``frame_index[t]``.  ``NaN`` means "no measurement"; it is
    never 0, because 0 is indistinguishable from a surface at the camera origin.
    """

    depth_m: NDArray[np.float32]
    frame_index: NDArray[np.int64]
    mode: DepthMode
    source: str = ""
    confidence: NDArray[np.float32] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
        self.frame_index = np.asarray(self.frame_index, dtype=np.int64)
        if self.depth_m.ndim != 3:
            raise ValueError(f"SceneDepth expects [T, H, W], got {self.depth_m.shape}")
        if self.frame_index.shape != (self.depth_m.shape[0],):
            raise ValueError(
                f"frame_index {self.frame_index.shape} does not match T={self.depth_m.shape[0]}"
            )
        if self.mode not in {"metric", "aligned", "relative"}:
            raise ValueError(f"unknown depth mode {self.mode!r}")
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)
            if self.confidence.shape != self.depth_m.shape:
                raise ValueError("confidence must have the same shape as depth_m")
        # a zero or negative depth is not a measurement
        self.depth_m = np.where(self.depth_m > 0, self.depth_m, np.float32("nan"))

    # -- contract ----------------------------------------------------------

    @property
    def height(self) -> int:
        return int(self.depth_m.shape[1])

    @property
    def width(self) -> int:
        return int(self.depth_m.shape[2])

    @property
    def num_frames(self) -> int:
        return int(self.depth_m.shape[0])

    @property
    def is_composable(self) -> bool:
        return self.mode in _COMPOSABLE

    def assert_composable(self, consumer: str) -> None:
        """Raise unless the mode may be compared with metric robot depth."""
        if not self.is_composable:
            raise DepthModeError(
                f"{consumer} received scene depth with mode={self.mode!r}. "
                "Relative depth is affine-invariant and cannot be compared with MuJoCo's "
                "metric robot depth (ADR-0008). Promote it with align_relative_depth() "
                "first, or use the debug_alpha compositor."
            )

    def frame(self, frame_index: int) -> NDArray[np.float32] | None:
        pos = int(np.searchsorted(self.frame_index, int(frame_index)))
        if pos >= self.frame_index.size or int(self.frame_index[pos]) != int(frame_index):
            return None
        return self.depth_m[pos]

    def statistics(self) -> dict[str, Any]:
        finite = self.depth_m[np.isfinite(self.depth_m)]
        if finite.size == 0:
            return {"num_frames": self.num_frames, "valid_fraction": 0.0}
        return {
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "valid_fraction": round(float(finite.size / self.depth_m.size), 6),
            "min_m": round(float(finite.min()), 4),
            "median_m": round(float(np.median(finite)), 4),
            "p95_m": round(float(np.percentile(finite, 95)), 4),
            "max_m": round(float(finite.max()), 4),
        }

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path, *, write_confidence: bool = True) -> Path:
        """Write one ``.npy`` per frame plus a manifest; returns the manifest path."""
        import numpy as _np

        out = Path(directory)
        depth_dir = out / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        frames: list[dict[str, Any]] = []
        conf_dir = out / "confidence"
        if self.confidence is not None and write_confidence:
            conf_dir.mkdir(parents=True, exist_ok=True)
        for pos, fi in enumerate(self.frame_index):
            name = f"frame-{int(fi):06d}.npy"
            path = depth_dir / name
            _np.save(path, self.depth_m[pos])
            row: dict[str, Any] = {
                "frame_index": int(fi),
                "path": f"depth/{name}",
                "valid_fraction": round(
                    float(np.isfinite(self.depth_m[pos]).mean()), 6
                ),
            }
            if self.confidence is not None and write_confidence:
                cpath = conf_dir / name
                _np.save(cpath, self.confidence[pos])
                row["confidence_path"] = f"confidence/{name}"
            frames.append(row)
        manifest = {
            "schema_version": _SCHEMA,
            "mode": self.mode,
            "quantity": "z_depth_m",
            "frame": "C",
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "frames": frames,
            "statistics": self.statistics(),
            "provenance": self.provenance,
        }
        path = out / "scene-depth.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, manifest_path: str | Path) -> SceneDepth:
        """Load a field written by :meth:`save`."""
        import numpy as _np

        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload.get("schema_version")
        if schema != _SCHEMA:
            raise ValueError(f"{path}: unsupported scene depth schema {schema!r}")
        root = path.parent
        depths: list[NDArray[np.float32]] = []
        confs: list[NDArray[np.float32] | None] = []
        frames: list[int] = []
        any_conf = False
        for row in payload["frames"]:
            depths.append(_np.load(root / row["path"], allow_pickle=False))
            frames.append(int(row["frame_index"]))
            if row.get("confidence_path"):
                confs.append(_np.load(root / row["confidence_path"], allow_pickle=False))
                any_conf = True
            else:
                confs.append(None)
        confidence = None
        if any_conf:
            shape = depths[0].shape
            confidence = np.stack(
                [
                    c if c is not None else np.full(shape, np.nan, dtype=np.float32)
                    for c in confs
                ]
            ).astype(np.float32)
        return cls(
            depth_m=np.stack(depths),
            frame_index=np.asarray(frames, dtype=np.int64),
            mode=payload["mode"],
            source=payload.get("source", ""),
            confidence=confidence,
            provenance=payload.get("provenance", {}),
        )


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


def _load_font(size: int) -> Any:
    """Best-effort labelled-preview font; falls back to PIL's built-in."""
    from PIL import ImageFont

    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def depth_preview(
    rgb: NDArray[np.uint8],
    depth_m: NDArray[np.float32],
    path: str | Path,
    *,
    vmin: float,
    vmax: float,
    title: str = "",
) -> Path:
    """Write a viewable RGB | depth pair at fixed metric colour limits.

    The limits are passed in rather than fitted per frame so that frames of one
    episode can be compared to each other; a per-frame auto-range hides exactly the
    scale jumps a reviewer is looking for.
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    h, w = depth_m.shape
    finite = np.isfinite(depth_m)
    t = np.clip((depth_m - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    t = np.where(finite, t, 1.0)
    # near = dark blue, far = warm yellow: a monotone, colour-blind safe ramp
    ramp = np.stack(
        [
            0.05 + 0.90 * t,
            0.10 + 0.55 * (1.0 - np.abs(0.5 - t) * 2.0),
            0.95 - 0.85 * t,
        ],
        axis=-1,
    )
    colour = (np.clip(ramp, 0, 1) * 255).astype(np.uint8)
    colour[~finite] = np.array([28, 30, 34], dtype=np.uint8)
    panel = Image.new("RGB", (w * 2 + 12, h + 44), (18, 20, 24))
    panel.paste(Image.fromarray(rgb.astype(np.uint8)), (0, 44))
    panel.paste(Image.fromarray(colour), (w + 12, 44))
    draw = ImageDraw.Draw(panel)
    font = _load_font(22)
    draw.text((8, 10), f"RGB · {title}", font=font, fill=(226, 232, 240))
    draw.text(
        (w + 20, 10),
        f"scene depth (m)  {vmin:.2f} – {vmax:.2f}   dark=near",
        font=font,
        fill=(226, 232, 240),
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
    imageio.imwrite(tmp, np.asarray(panel))
    tmp.replace(out)
    return out
