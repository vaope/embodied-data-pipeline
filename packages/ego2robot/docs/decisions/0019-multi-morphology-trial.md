# ADR-0019: Early multi-morphology geometry trial

- Status: **engineering experiment; user visual review pending**
- Date: 2026-09-28
- Scope: the already reviewed 52-frame EgoDex source, three additional robot models; this does not mark M9 complete

## Context

The paper's appendix A.7 lists 15 robot morphologies. The repository originally
bound the canonical TCP sequence only to Franka Panda. The user requested a
comparison on several genuinely different robot models before the general M9
expansion. This is an explicit, narrow change to the original milestone order:
one reviewed source episode is reused for a geometry trial, while broad M9
dataset production and multi-arm scheduling remain outside this experiment.

The three trial profiles are `configs/robot/ur5e.yaml`,
`configs/robot/xarm7.yaml`, and `configs/robot/viperx300.yaml`. Their models are
vendored from a pinned MuJoCo Menagerie commit, with per-model `LICENSE`,
model XML, meshes and provenance (`SOURCES.md`, `PROVENANCE.md` or
`PROVENANCE.json`, depending on the model). The `robot_id` selects an adapter; the
perception, `HandSequence` and `CanonicalActionSequence` schemas are unchanged.

## Decision

1. Run the same source MP4/HDF5 through `ingest -> hand -> retarget` in three
   isolated episode directories under `data/morphology_trials/`. All canonical
   array fields must match before comparing robot results.
2. Add model-specific adapters and derive joint names, limits, `T_E_TCP`,
   gripper state mapping and base frame from each pinned MJCF. Persist model
   asset hashes in the robot stage fingerprint.
3. Choose `T_C_B` separately per morphology. The first pass uses
   `configs/trials/morphology-coarse-grid.yaml`: 162 candidates formed from a
   subset of the paper A.4 levels. This is an engineering speed choice, not
   the full paper grid. A wider xArm7 subset is recorded in
   `configs/trials/xarm7-expanded-grid.yaml` if the first pass misses the gate.
   In this episode, the xArm7 coarse pass produced 42.3% full-frame IK success;
   the 1,620-candidate expanded subset reached 52/52. Both placements are
   retained (`base-coarse-162.json` and `base.json`).
4. The Menagerie UR5e is a six-axis **arm-only** model with an attachment site
   and no finger joints. Its trial can measure TCP pose reachability and
   rendering, but cannot execute canonical gripper width or prove a grasp.
   `gripper_dof=0` and `unsupported_width_frames` must remain visible.
5. The ViperX profile uses the simulated collision-pad gap derived from its
   MJCF (15–87 mm). This differs from the manufacturer's physical gripper
   specification (42–116 mm). Out-of-range canonical widths are clamped to
   the simulated joint limits and counted in the robot manifest; no physical
   hardware claim follows from this mapping.
6. The xArm7 model has seven arm hinges and six gripper hinge coordinates
   driven by a coupled tendon actuator. The adapter maps one canonical jaw
   width to the six model coordinates, and the trajectory preserves all six.
7. Renderer state must use joint names and MuJoCo qpos addresses; camera pose
   must include the adapter's `T_W_B` when the chosen B frame is not MuJoCo
   world. Projection overlays must label arm-only output honestly.

## Consequences

- A successful solver result is a geometry result, not user acceptance. Each
  robot needs projection stills and render images for visual inspection.
- On the 52-frame trial, all three final robot trajectories reached 52/52
  pose IK success with no recorded joint jumps or collision frames. UR5e is
  still arm-only. The simulation width range clipped two xArm7 frames and
  four ViperX frames; the source widths remain unchanged in canonical output.
- Coarse base search can miss feasible placements. A low IK ratio remains a
  failed trial result until a stronger search or adapter correction is shown.
- Robot shapes, masks and TCP conventions can differ despite a common action
  target; visual position must be checked against the image, not inferred from
  an IK success flag.
- The trial does not provide a common physical gripper command, dynamic
  feasibility, or policy training data across robots.

## Evidence

- Paper model list: `docs/references/ego2robot-paper.html`, appendix A.7.
- Model source and license: `assets/mjcf/<model>/LICENSE` and the matching
  source/provenance file in that directory. All three models pin commit
  `c96a32d28fb5da84da38c1da4d749e7a13212855`.
- Resolved configuration and stage manifests: each directory under
  `data/morphology_trials/<robot_id>/egodex_vertical_pick_place_000_<robot_id>/`.
- User-facing images: each trial episode's `previews/projection-frame-*.png`
  and the model preview directory, where available.

## Alternatives

- Reusing Panda `qpos` with another robot's mesh would break joint semantics,
  FK and collision checks.
- Treating UR5e's attachment flange as a working gripper would hide its absent
  jaw actuation; this experiment keeps the limitation explicit.
