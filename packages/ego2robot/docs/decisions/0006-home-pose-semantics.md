# ADR-0006: What "home configuration" means for large gaps

- Status: accepted
- Date: 2026-09-20
- Tags: `[PAPER]` names it, `[ENGINEERING]` defines it

## Context

Paper appendix A.1:

> Large gaps (> 10 frames) are filled with the robot's home configuration, with
> smooth blending transitions at the gap boundaries. The blend length is
> `n = max(5, min(90, ceil(0.6 n_pos + 0.4 n_rot)))` frames, where
> `n_pos = dp / 3.25 mm` and `n_rot = dtheta / 1.08 deg`.

The formula is published; **the home configuration itself never is**. Two
difficulties follow:

1. The gap handling runs in the `retarget` stage, at which point no robot has
   been chosen. The canonical action sequence is deliberately robot-independent
   (`docs/system-design.md` 6.3), so "the robot's home configuration" cannot be
   resolved there without breaking the layering.
2. Even on the robot side, "home" is ambiguous: a joint-space keyframe, a
   Cartesian home TCP pose, or a task-space park position.

## Decision

Gap filling operates on the **canonical** representation and therefore fills with
a *home TCP pose in the camera frame*, supplied as configuration:

* `gaps.home_gripper_width_m` (config key; the width assumed while parked)
* `home_position_C_m` / `home_rotation_C_TCP` arguments to
  `ego2robot.retarget.gaps.fill_gaps`

When the caller supplies no pose, `fill_gaps` derives one from the trajectory
itself — the mean position of the valid frames and the first valid orientation —
and **records that it did so** via `GapFillResult.home_was_derived`, which the
`retarget` stage surfaces as the `home_was_derived` metric. Nothing is silent.

What this deliberately does **not** do: it does not park the robot's joints. A
`retarget`-level decision cannot know the robot, and the plan's rule 2 (perception
is decoupled from the robot) would be violated. When M4/M7 add robot-aware gap
handling, the natural design is to re-fill gaps after base placement using
`FK(home_qpos)`, and to record which of the two was used.

## Consequences

* `retarget` never imports `robots`, so the layer boundary holds.
* A reviewer sees `home_was_derived: true` on any episode that used the fallback,
  which is the signal that the "home" pose is a heuristic for that episode.
* The blend length formula is implemented verbatim
  (`ego2robot.retarget.gaps.blend_length_frames`), including the `ceil` and the
  `[5, 90]` clamp.
* A gap at the very start or end of a clip has only one bracket, so it cannot be
  interpolated; it takes the large-gap path regardless of its length. The paper
  does not discuss this case.

## Evidence

* `ego2robot/retarget/gaps.py` (`GapConfigLike`, `fill_gaps`,
  `GapFillResult.home_was_derived`)
* `configs/pipeline/mvp.yaml` (`gaps.home_gripper_width_m`, `gaps.home_source`)
* `retarget` stage metrics on `case5_degenerate`, which contains a 14-frame
  dropout: `large_gaps: 1`, `gap_spans: [[50, 64, 14]]`, `blend_lengths: [5]`,
  `home_was_derived: true`

## Alternatives considered

* **Resolve home from the robot in `retarget`** — rejected: breaks the
  perception/robot decoupling and makes the canonical output robot-dependent.
* **Extend the canonical schema with a home pose field** — deferred: it would be
  the cleanest place to record the choice, but it changes a schema for a need
  that only materialises once base placement exists (M4). The metric plus the
  function argument are sufficient for M3.
* **Substitute zeros across the gap** — rejected outright: a parked gripper at
  the camera origin is not a pose, and implementation-plan.md Gate M1 bans
  zero-filling.
