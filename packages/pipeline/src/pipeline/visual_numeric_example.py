"""Produce explicitly synthetic geometry evidence, without any model inference.

Run: python -m pipeline.visual_numeric_example --output <new-evidence-directory>
This is a numerical demonstration, not an automated test suite or user gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw

from .visual import DepthAnchor, align_depth, compose_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    h, w = 240, 480
    scene = np.full((h, w), 2.4, np.float64)
    scene[:, 170:340] = 0.8
    scene[:, 340:] = 1.6
    background = np.full((h, w, 3), [93, 116, 127], np.uint8)
    background[:, 170:340] = [213, 149, 75]
    background[:, 340:] = [141, 161, 142]
    robot_rgb = np.full((h, w, 3), [229, 72, 64], np.uint8)
    arm = np.zeros((h, w), bool)
    arm[100:145, 20:160] = True
    gripper = np.zeros((h, w), bool)
    gripper[105:125, 160:440] = True
    gripper[145:165, 160:440] = True
    robot_rgb[arm] = [75, 165, 225]
    robot_z = np.where(arm | gripper, 1.2, np.nan).astype(np.float32)
    # The foreground strip also crosses arm pixels: the arm remains visible.
    scene[100:145, 90:140] = 0.6
    background[100:145, 90:140] = [168, 118, 66]
    human = np.zeros((h, w), bool)
    human[95:135, 230:270] = True
    raw = (scene - 0.2) / 2.5
    inverse_raw = ((1.0 / scene) - 0.1) / 1.8
    anchors = [DepthAnchor(x=x, y=y, depth_m=float(scene[y, x]),
        evidence_ref="synthetic_scene_definition_in_this_script", evidence_kind="synthetic_fixture",
        surface_correspondence="exact synthetic scene pixel") for x, y in [(30, 20), (240, 20), (400, 20), (110, 120)]]
    aligned = align_depth(raw, mode="relative", quantity="relative_depth", anchors=anchors,
                         evidence_kind="synthetic_fixture", max_rmse_m=1e-5, max_residual_m=1e-5)
    inverse_aligned = align_depth(inverse_raw, mode="relative", quantity="relative_inverse_depth", anchors=anchors,
                         evidence_kind="synthetic_fixture", max_rmse_m=1e-5, max_residual_m=1e-5)
    composed = compose_frame(background, robot_rgb, arm, gripper, robot_z, aligned, human)
    unknown_scene = scene.copy()
    unknown_scene[145:165, 370:390] = np.nan
    metric_unknown = align_depth(unknown_scene, mode="metric", quantity="z_depth_m", evidence_kind="synthetic_fixture")
    unknown = compose_frame(background, robot_rgb, arm, gripper, robot_z, metric_unknown, human)
    palette = np.array([[93,116,127], [75,165,225], [229,72,64], [198,81,211], [40,40,40], [245,218,56]], np.uint8)
    panels = [(background, "SYNTHETIC scene: metric Z 0.6 / 0.8 / 1.6 / 2.4 m"),
              (composed.rgb, "SYNTHETIC composite: arm always visible; protected gripper"),
              (palette[composed.decision_map], "SYNTHETIC decisions: blue arm / red front / purple protected / black hidden"),
              (unknown.rgb, "SYNTHETIC missing depth: background retained; frame_valid=false")]
    canvas = Image.new("RGB", (w * 2, (h + 45) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (pixels, caption) in enumerate(panels):
        x, y = (index % 2) * w, (index // 2) * (h + 45)
        canvas.paste(Image.fromarray(pixels), (x, y + 45))
        draw.text((x + 6, y + 6), caption, fill="black")
        if index == 2:
            draw.text((x + 6, y + 23), "No EgoDex depth or real model was used. No visual gate accepted.", fill="black")
    image_path = args.output / "synthetic-depth-compositing.png"
    canvas.save(image_path)
    np.savez_compressed(args.output / "synthetic-numeric-arrays.npz", scene_depth_m=scene,
        relative_depth=raw, relative_inverse_depth=inverse_raw, aligned_depth_m=aligned.depth_m,
        robot_depth_m=robot_z, arm_mask=arm, gripper_mask=gripper, human_arm_mask=human,
        decision_map=composed.decision_map, unknown_decision_map=unknown.decision_map)
    report = {"evidence_kind": "synthetic_fixture", "model_execution": "none", "review_status": "pending",
        "code_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name("visual.py")]},
        "command": "python -m pipeline.visual_numeric_example --output " + str(args.output),
        "environment": {"numpy": np.__version__, "pillow": PIL.__version__},
        "training_ready": False, "scope": "four synthetic metric planes; known exact correspondences",
        "depth_alignment": {"scale": aligned.scale, "shift": aligned.shift,
            "rmse_m": aligned.residual_rmse_m, "max_residual_m": aligned.residual_max_m,
            "max_error_against_constructed_truth_m": float(np.max(np.abs(aligned.depth_m-scene)))},
        "inverse_depth_alignment": {"scale": inverse_aligned.scale, "shift": inverse_aligned.shift,
            "rmse_m": inverse_aligned.residual_rmse_m, "max_residual_m": inverse_aligned.residual_max_m},
        "composite": {"frame_numerically_valid": composed.frame_valid, **composed.diagnostics},
        "unknown_depth_composite": {"frame_numerically_valid": unknown.frame_valid, **unknown.diagnostics},
        "image": str(image_path.resolve())}
    (args.output / "numeric-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
