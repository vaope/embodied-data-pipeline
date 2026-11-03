# ADR-0002: Camera optical frame and the MuJoCo camera basis

- Status: accepted
- Date: 2026-09-20
- Tags: `[ENGINEERING]`, camera, rendering

## Context

Frame `C` is defined by `docs/system-design.md` 5.1 as the optical convention:
`+x` right, `+y` down, `+z` forward. MuJoCo camera bodies do not use that
convention: they look along their **local `-z`**, take local `+y` as image-up,
and their depth buffer is a positive distance along the viewing direction.

`docs/system-design.md` 11.2 explicitly warns about this and requires an
explicit basis transform plus a golden test, rather than tuning signs by eye.

Three revisions of this code were wrong before the empirical test settled it:

| basis | symptom |
|---|---|
| `diag(1, -1, 1)` | camera looks backwards; frames render empty, all segmentation labels `-1` |
| `diag(-1, -1, 1)` | same |
| `diag(1, 1, -1)` | camera axis right, image `v` flipped |
| **`diag(1, -1, -1)`** | analytic and rendered pixels agree to ~1 px |

## Decision

The renderer places MuJoCo camera bodies with

```text
R_B_MjCam = R_B_C @ diag(1, -1, -1)
```

which maps optical `+x -> camera +x`, optical `+y -> camera -y`,
optical `+z -> camera -z`. The matrix is a rotation (`det = +1`).

The MuJoCo depth buffer is used **unchanged**: MuJoCo reports positive distance
along its viewing direction, which is frame `C`'s `+z`, so it is already the
metric optical depth. Measured on the validation fixtures, the raw buffer over
robot pixels spans `0.26 - 0.83 m` for a workspace `~0.3 m` in front of the
camera, confirming the sign.

Camera intrinsics enter through the vertical field of view,
`fovy = 2 * atan(height / (2 * fy))`, so a camera-frame point projects
analytically to the same pixel the renderer produces.

## Consequences

* `egoz2robot.vision.render.MJ_CAM_TO_OPTICAL_BASIS` is the single definition.
* `verify_projection` renders emissive markers at known camera-frame points and
  compares rendered centroids with `K @ p_C`. It runs once per episode inside
  the `render` stage and writes `previews/projection-golden-markers.png` plus the
  residuals into the stage manifest.
* The renderer writes its generated scene XML *next to* the robot model, because
  MuJoCo resolves `meshdir` and every relative asset path against the
  **top-level** file's directory. A scene in `%TEMP%` cannot find `assets/` and
  fails with `Error opening file .../link0.stl`. The file is dot-prefixed and
  deleted on close.

## Evidence

* Determination method: 24 candidate signed permutation bases were rendered with
  markers at known camera-frame points and scored against the analytic
  projection. `diag(1, -1, -1)` is the only candidate that reproduced the
  analytic pixels; the mean residual was ~1 px for markers 0.5-0.6 m away, the
  rest of the residual being marker-centroid quantisation.
* `previews/projection-golden-markers.png` and
  `previews/projection-report.json` in every validation case.
* `ego2robot/vision/render.py` module docstring.

## Alternatives considered

* **Use `mj_cam` projections directly and skip `K`** — rejected: the episode has
  a calibrated `K` from its source, and the whole point of M3 is that the
  analytic projection and the render agree.
* **Flip the rendered image vertically instead of the camera basis** — rejected:
  it moves the discrepancy out of the geometry and into the pixels, where the
  depth buffer would still be inverted.
