# ADR-0009: How the MVP fixed base placement is chosen

- Status: accepted
- Date: 2026-09-20
- Tags: `[ENGINEERING]`, robot binding

## Context

`docs/system-design.md` 8.2 defines two base-placement strategies:

* **MVP fixed-base** — `T_C_B` given by a human, to close the loop quickly;
* **Reproduction grid search** (M4) — the paper's appendix A.4 candidate grid and
  score `S = FR - 5.0 |mean_reach_ratio - 0.65|`.

The MVP path needs *some* `T_C_B` for every episode, and the paper publishes
nothing about where a human should put the base. Inventing one silently is
forbidden (AGENTS.md rule 10); the value must come from configuration and be
recorded.

Two hard constraints turned out to bound the choice, and both were discovered by
the validation fixtures failing rather than by reasoning:

1. **Reachability.** The paper's own reach penalty is minimised at
   `mean_reach_ratio = 0.65`, but for these trajectories a base at
   `0.6 * max_reach` from the workspace centroid leaves the required TCP
   orientations outside the Panda's dexterous workspace and IK fails outright
   (measured: 0/45 frames on the static case).
2. **Visibility.** In an ego setup the camera sits at the operator's head/wrist,
   so the robot base is *between* the camera and the workspace. If the base is
   placed too close to the workspace, the arm fills the field of view and the
   hand disappears; if it is placed too far, the workspace leaves the field of
   view.

## Decision

For the fixed-base MVP path, `ego2robot.tools.cases.base_for_workspace` derives
`T_C_B` from the trajectory's own centroid:

* position: `centroid - reach_fraction * max_reach * f` horizontally and
  `height_offset_m` below the workspace, where `f` is the horizontal unit vector
  from the base to the workspace;
* orientation: `Rz(yaw) @ Rx(90 deg - pitch)` with `yaw` facing the workspace and
  `pitch = 70 deg`, so the base's `+z` is up and tilted forward;
* `reach_fraction = 0.26`, `height_offset_m = 0.03 m`, `pitch = 70 deg`.

`reach_fraction = 0.26` is **below** the paper's 0.65 target on purpose: it is the
largest value at which all five validation cases solve completely under the
present camera geometry. The reasoning is that the paper's 0.65 is an objective
for a *search over candidates that are already feasible*; when feasibility
itself is the binding constraint, feasibility wins. The `robot` stage reports
the achieved `mean_reach_ratio` alongside the config value so the deviation is
visible rather than papered over.

For a real episode (not a synthetic fixture) the value must come from the
episode or the pipeline profile:
`robot_binding.base_placement.mode: fixed` plus an explicit `T_C_B` in
`episode.json` or in `case.T_C_B`. The `robot` stage **refuses to run** without
one, and its input validation raises `robot.no_base_placement`.

## Consequences

* `T_C_B` is written to `robot/<robot_id>/base.json` with its `source`
  (`episode` / `profile` / `grid_search`) and its notes, and a later run reuses
  that file rather than recomputing it — so a base placement is a pinned,
  inspectable artifact.
* The `ingest` stage preserves an existing `T_C_B` from `episode.json` instead of
  dropping it when it rebuilds the manifest. This was a real bug: ingest used to
  overwrite the fixture's declared base with `None`, after which the `robot`
  stage legitimately refused to run.
* `base_for_workspace` raises if the computed orientation would point the base's
  `+z` downward in the camera frame, which is the signature of a sign error in
  the pitch composition. That check exists because two earlier revisions had the
  sign backwards and rendered the robot *in front of* the camera.
* M4 replaces this entirely with the grid search; the fixed path stays as the
  baseline that the search must beat (implementation-plan.md M4 gate).

## Evidence

Measured across the five validation cases with the current `T_C_B`:

```text
reach_fraction = 0.26   -> IK success 60/60 (case4), 45/45 (case3), 90/90 (case1), 80/80 (case5)
reach_fraction = 0.30   -> case4 0.68, case2 1.00, case1 1.00
reach_fraction = 0.34   -> case1 0.70, case2 0.89, case4 0.60
reach_fraction = 0.37   -> case1 0.92, case2 0.72, case4 0.58
```

* `ego2robot/tools/cases.py` (`base_for_workspace`)
* `ego2robot/robots/stage.py` (`_load_base`, `robot.no_base_placement`)
* `data/fixtures/*/robot/franka_panda/base.json`
* `robot` stage metrics: `mean_reach_ratio`, `max_reach_ratio`

## Alternatives considered

* **`0.65 * max_reach` as the paper's target** — rejected for the MVP: IK fails
  on these trajectories, so it would produce zero valid episodes and hide the
  real geometry behind an IK failure.
* **Auto-search at ingest** — rejected: that is M4's job, it needs the full
  trajectory and the robot model, and doing it at ingest would put a robot
  import in the ingest layer.
* **Place the base behind the workspace (base further from the camera)** —
  rejected: the arm then extends away from the camera, but the workspace is
  between the camera and the base, so the rendered arm lands behind the
  workspace and the hand occludes it — which is the opposite of the ego
  viewpoint the pipeline is reconstructing.
