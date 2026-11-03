# ADR-0005: `T_E_TCP` and the Panda end-effector frame

- Status: accepted (amended 2026-10-27: the rotation was wrong; see "Amendment")
- Date: 2026-09-20 (amended 2026-10-27)
- Tags: `[ENGINEERING]`, robot binding

## Context

`docs/system-design.md` 6.4 requires the robot configuration to store
`T_E_TCP`, the constant transform between the robot model's end-effector link
frame `E` and the canonical grasp frame `TCP`. The paper never publishes it.

Three facts had to be established from the model rather than assumed:

1. **Which body is `E`.** Upstream
   `mujoco_menagerie/franka_emika_panda/panda.xml` contains **no `<site>`
   element at all** — in particular no `attachment_site`, which is what `mink`'s
   own documentation and examples use. `mink` resolves only
   `body`/`geom`/`site` frames, so `frame_name="attachment_site"` cannot work
   against the upstream model. (`panda_nohand.xml` does define
   `attachment_site`, but it has no gripper.)
2. **Where the TCP is.** With both finger joints at zero the two finger bodies
   coincide; with them at 0.04 m they separate by exactly 0.08 m along the hand
   frame's `+y`. The jaw midpoint at zero opening is `hand + (0, 0, 0.0584)`.
3. **What the TCP axes are.** The paper's canonical frame (equation 3) is
   `x` approach, `y` jaw-plane normal, `z` grasp axis along the jaw line.

## Decision

`E` is the MJCF **`hand` body** and `frame_type="body"` in Mink. `T_E_TCP` is

```text
        [[0, 1, 0, 0     ],
         [0, 0, 1, 0     ],
         [1, 0, 0, 0.1031],
         [0, 0, 0, 1     ]]
```

i.e. the TCP axes `[x y z]` map to the hand-frame axes `[+z +x +y]`: `x` towards
the fingers (approach), `z` along the finger-travel direction (the jaw line), and
`y = z × x` the jaw-plane normal. This is a rotation (`det = +1`).

The **depth is the finger pad contact centre**, derived at config-build time from
the pad collision geoms (`pad_centre_depth_in_hand`), not the finger body origin:
the bodies sit at `hand + 0.0584 z` (where the fingers attach), while the pads span
`hand z ∈ [0.0979, 0.1084]`, centre `0.1031` m. Using the body origin put the
commanded grasp point 44.7 mm behind the real contact surface.

The gripper opening mapping is `finger_joint = width_m / 2` for both fingers, so
the published Panda range of 0-80 mm (paper appendix A.7) maps to
`finger_joint ∈ [0, 0.04]`, which is exactly the menagerie model's joint range.

`max_reach_m` is **0.855 m**, measured from the menagerie kinematic chain, not
the 1.272 m that paper appendix A.7 lists for Panda. The published figure is the
arm's reach; the value used here is the modelled chain including the hand, and it
is what reach ratios and workspace checks are normalised against.

## Consequences

* `ego2robot.robots.franka.build_franka_config` derives every kinematic value
  from the loaded model and, when a robot profile restates them, **raises** on a
  mismatch instead of warning.
* `configs/robot/franka_panda.yaml` restates the derived values, so a changed
  model shows up as a configuration diff.
* Because `max_reach_m` follows the model rather than the paper, a reach ratio
  computed here is not numerically comparable to one computed by the paper's
  own tooling. `base_search` reports the ratio it used alongside the value.

## Evidence

Measured on the vendored model (`mujoco.MjModel.from_xml_path`):

```text
nq = 9   (joint1..joint7, finger_joint1, finger_joint2)
nu = 8   (7 arm actuators + 1 gripper actuator)
bodies: world link0 link1 link2 link3 link4 link5 link6 link7 hand left_finger right_finger
finger_joint1/2 range [0, 0.04]
gripper 0 mm   -> both finger bodies at hand + (0, 0, 0.0584)
gripper 40 mm  -> separation 80.00 mm along the hand frame's +y
home keyframe qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]
```

* `ego2robot/robots/franka/adapter.py` module docstring and
  `PANDA_TCP_OFFSET_IN_HAND`
* `robot` stage metrics: `robot_kinematics.T_E_TCP`, `max_gripper_width_m`
* `docs/references/panda-mjcf-and-mink-research.md` (model licence: Apache-2.0)

## Alternatives considered

* **Use `panda_nohand.xml` + `attachment_site`** — rejected: no gripper, and the
  gripper is what the canonical frame describes.
* **Add a `<site>` to the vendored XML** — rejected: it modifies a vendored
  third-party asset and makes the model hash depend on this project.
* **Model the TCP at the finger pad centre instead of the jaw midpoint** —
  rejected: the paper's `p_tcp = (p_thumb + p_vf)/2` is the jaw midpoint.

## Amendment (2026-10-27): the rotation was 90° off

### What was wrong

The original matrix was the cyclic permutation `[[0,0,1],[1,0,0],[0,1,0]]`, which
maps TCP `x` onto hand `+y` — the finger-travel direction. The text above it
claimed the axes were `[+z +y +x]`, which is not the same permutation, and which
would have been a reflection (`det = -1`) anyway. Net effect: the canonical
`z = grasp axis` did **not** land on the jaw line; the `approach` axis did.

### Evidence

MuJoCo forward kinematics on the **executed** joint states of
`egodex_vertical_pick_place_000` (52 frames), measuring the vector between the
`left_finger` and `right_finger` bodies:

| comparison | angle |
|---|---|
| finger-to-finger line vs commanded `x` | 0.0° |
| finger-to-finger line vs commanded `z` | 90.0° |

Position and width were unaffected: the finger gap equalled the commanded width
(65.6 mm median), and the jaw midpoint sat 6.9-16.8 mm from the retargeted pinch
point (that residual is the position smoothing). Only the orientation was wrong.

Notably, none of the existing checks can catch this: `pos err` was 0.0027 mm,
`projected gap` 0.0091 px, and the M3 quality gate passed — IK had reached the
(mis-oriented) target exactly.

### Correction and verification

Two defects were fixed together.

**1. Rotation (90° off).** Replaced with `[[0,1,0],[0,0,1],[1,0,0]]`.

**2. Depth (44.7 mm short).** The translation used the finger **body origin**
(`hand + 0.0584 z`, where the fingers attach) instead of the **pad contact centre**
(`hand z ∈ [0.0979, 0.1084]`, centre `0.1031`). Consequence: the gripper's pads sat
44.7 mm *past* the commanded point, so the human fingertips visually lined up with
the gripper's root rather than its pads. The depth is now derived from the pad
collision geoms at build time.

Re-solving IK against the same stored `T_B_TCP` targets on all 52 frames, with the
corrected constant:

| metric | result |
|---|---|
| IK success | 52/52 |
| position error | median 0.0001 mm, max 0.0015 mm |
| rotation error | median 0.0000°, max 0.0001° |
| finger line vs commanded `z` | **0.0°** (was 90.0°) |
| finger line vs commanded `x` | 90.0° (was 0.0°) |
| **real pad centre ↔ human pinch point** | **12.1 mm median** (was 45.1 mm, max 65.0 → 26.7) |

The residual 12.1 mm equals the position-smoothing offset measured independently,
so no systematic depth bias remains.

### Migration

This is a coordinate-convention change, so existing artifacts are stale:

* every episode's `robot/`, `render/`, `projection/`, `quality/` and `visual/`
  outputs were produced under the old frame and must be regenerated
  (`e2r run <episode> --profile mvp --force-stage robot`, which re-runs robot and
  everything downstream);
* `fingerprint`s change, so caches invalidate automatically — do not copy old
  `robot/` or `projection/` artifacts into a new run;
* any previously recorded M3 review record binds the old projection video hashes
  and therefore no longer matches a regenerated episode;
* the exported `T_E_TCP` in `robot/trajectory.npz` metadata and in exported
  bundles (`robot_formats`) changes with the new constant;
* the other morphology adapters were re-checked with the same finger-line test on
  2026-10-27:
  * **xArm7** had the same class of defect (its `link_tcp` site axes are
    `-x/-y/+z` of `link7`, so the jaw line landed on canvas `y`). Its derivation now
    composes the site rotation with the canonicalising permutation; verified by
    driving both linkage drivers — the finger bodies separate along `z` (was `y`).
    Its `T_E_TCP` translation is still site-derived and unchanged; note that the
    site sits 0.172 m from `link7` while the jaw midpoint at zero driver angle is
    0.101 m, which the user may wish to review separately.
  * **ViperX-300** already puts the jaw line on `z`; unchanged.
  * **UR5e** has no gripper in the vendored model (`gripper_geom_names=[]`), so no
    jaw line exists to check; unchanged.
* trial artifacts for xArm7/ViperX-300/UR5e must be regenerated for the same
  reason as Panda before any of their results are quoted.

