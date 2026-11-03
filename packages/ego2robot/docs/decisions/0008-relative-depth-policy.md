# ADR-0008: Metric-only depth compositing; relative depth is barred

- Status: accepted for the interface, implementation deferred to M6
- Date: 2026-09-20
- Tags: `[PAPER]` names Depth Anything V3, `[OPEN]` how it becomes metric

## Context

The paper composites with

```text
I_final(u,v) = I_robot(u,v)  if D_robot(u,v) < D_scene(u,v) and M_robot(u,v) = 1
               I_inpaint(u,v) otherwise
```

with `D_scene` "obtained from depth sensors or estimated via monocular depth"
(appendix A.4 names Depth Anything V3).

`docs/system-design.md` 9.5 flags the gap: it is never explained how a monocular
*relative* depth is put on the same metric scale as MuJoCo's rendered depth.
Comparing the two directly is a category error — Depth Anything outputs an
affine-invariant inverse depth (or a disparity-like quantity), not metres, and
its scale varies per frame. `docs/system-design.md` 14 lists "relative scene
depth and metric robot depth cannot be compared directly" as a merge-blocking
condition, and section 9.5 requires a recorded `metric | aligned | relative`
mode with confidence.

## Decision

The interface is fixed now, the alignment is not yet implemented:

1. `SceneDepth` carries an explicit `mode ∈ {metric, aligned, relative}` plus a
   confidence.
2. `DepthScaleAligner.align(relative_depth, anchors) -> MetricDepth` is the only
   sanctioned way to promote relative depth. Anchors are the quantities whose
   metric value is already known: the MuJoCo robot depth, the reconstructed hand
   depth, and — where the scene allows it — a known table plane.
3. The compositor **raises** on a `mode` mismatch rather than falling back. A
   relative depth reaching `composite()` is a programming error, not a runtime
   condition to tolerate.
4. Only `metric` and `aligned` depth may produce a formal synthetic video. A
   `relative` output is written to a differently named artifact and watermarked
   in the overlay, and it cannot be part of a `valid` episode.
5. `configs/pipeline/repro.yaml` records the intended mode (`aligned`) and
   `configs/pipeline/vision-m5-m6.yaml` keeps the model name and parameters.

## Consequences

* M6 cannot silently produce a plausible-looking but wrongly-occluded video:
  the mode is a required field, and the check is a raise.
* `arm body` and `gripper` are handled separately, as the paper does: the arm
  body is always composited (it is between the camera and the workspace and is
  never occluded by scene objects), and only the gripper undergoes per-pixel
  depth comparison, with the dilated hand mask protecting the interaction site.
  That is implemented in the design now
  (`docs/system-design.md` 9.6) so the M3 renderer already emits
  `robot_mask` (arm body) and `gripper_mask` separately, which it does.
* Choosing the anchor set and the fit (scale only, or scale + shift, or a
  per-frame affine fit with temporal smoothing) is the substantive M6 decision
  and remains open. It needs a fixture with known geometry — the
  `occlusion_tabletop` fixture named in implementation-plan.md section 15.

## Evidence

* `docs/system-design.md` 9.5, 9.6, 14
* `ego2robot/vision/render.py` emits `robot_mask` and `gripper_mask` separately
  and writes `NaN` for invalid depth, which is the precondition for the
  comparison
* `configs/pipeline/vision-m5-m6.yaml` (`scene_depth.mode: aligned`)

## Alternatives considered

* **Compare relative depth with metric depth and accept the error** — rejected:
  it is exactly the failure mode `docs/system-design.md` 14 forbids merging on,
  and it would be invisible in a video that "looks right".
* **Skip depth entirely and always draw the robot in front** — this is the
  documented `alpha-only debug compositor` path. Kept as a *debug* output, not
  as the formal one, because it makes the gripper never occluded, which is wrong
  for every grasp behind an object.
* **Use a stereo or ToF depth sensor if the source provides one** — preferred
  when available (`mode: metric`); no source in the current MVP scope provides
  one.
