# ADR-0025: Joint two-arm rendering and the ownership map (B3, first half)

- Status: **implementation decision; robot-only render available, compositing not**
- Date: 2026-11-02
- Scope: PLAN-02 B3 first half — render both arms in one scene with an ownership map,
  per-arm masks and metric depth.  Human-arm removal, scene-depth alignment,
  ownership-aware compositing and export remain undone (B3 second half, B4).

## Context

ADR-0023 built the two-arm rig as a *collision-only* scene.  B3 needs pixels: the
arms rendered together, so the depth order between them comes from the geometry
rather than from draw order (ADR-0022 decision 9).  The existing
`vision/render.py::MujocoRenderer` renders one robot with one gripper mask; it cannot
say which arm a pixel belongs to.

## Decision

1. **One renderer, one scene.** `vision/render_rig.py::RigRenderer` renders the
   generated two-arm scene from a world-level camera.  The camera pose is written
   exactly as the single-arm renderer does it (`T_W_C = inv(T_C_B_right)`, times
   `MJ_CAM_TO_OPTICAL_BASIS`), so both renderers share one convention.
2. **The generated scene gains an optional camera.** `build_dual_scene(..., with_camera=True)`
   adds a placeholder `rig_cam`; its pose is overwritten at runtime, so the file stays
   deterministic and hash-named.
3. **Ownership from segmentation, not RGB.** Each geom's body name carries the arm
   prefix (`R`/`L`); the segmentation pass maps geom ids to those prefixes.  Output:
   `owner` (0 background, 1 right, 2 left), `arm_mask`, `gripper_mask` (geoms on
   `hand`/`left_finger`/`right_finger` of either arm), and `depth_m` (NaN where
   nothing was hit, restricted to robot pixels).
4. **Render a bounded sample, not the whole episode.** `rig.render_frames` (default 6)
   frames are rendered; the pick is deterministic (even spacing).  This is enough to
   judge geometry/ownership, and it keeps the stage under a minute.  A full-segment
   render is a separate run once the bases are final.
5. **Robot-only, and labelled as such.** No background, no human removal, no
   compositing.  The render is the *input* to compositing, not the final synthetic
   frame; `render.json` says so and lists the limits.

## Result

- Both arms render in one pass; owner maps are clean and per-arm masks non-empty on
  every sampled frame.
- Per-arm coverage of the frame: right 7.3–16.7 %, left 10.9–18.1 % (frame 109 shown
  in the learning-side page).  Robot depth at those frames is 0.17–0.22 m at the
  nearest arm pixel.
- This replaces the B2 projection proxy with rendered evidence, and it **confirms**
  the B2 finding: with the current bases the two arms sit between the camera and the
  workspace, so they occupy a large part of the ego view.  Whether that is acceptable
  is a visual design question for the user, not something the rig can fix silently.

## Consequences

- B3's remaining work needs the model runtime (SAM3 + ProPainter) and an
  ownership-aware composite; neither is started, and no synthetic video is claimed.
- The composite decision codes 0–4 stay as they are; adding ownership means a new
  per-arm aggregation in the composite manifest, not a redefinition of the codes.
- Frame selection is deterministic, so a later full render can be compared against
  these frames.

## Evidence

- `vision/render_rig.py`, `robots/rig.py::build_dual_scene(with_camera=...)`,
  `robots/rig_stage.py::_write_render`, `core/config.py` (`rig.write_render`,
  `rig.render_frames`).
- Episode `egodex_fold_unfold_paper_basic_000429`, run `93daba24534e`:
  `robot/rig/render/{rgb,owner,mask,depth}-frame-*.{png,npz}` + `render.json`.
- Learning-side page: `historical-local-evidence`.
