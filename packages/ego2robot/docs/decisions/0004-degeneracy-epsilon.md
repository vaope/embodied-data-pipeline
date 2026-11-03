# ADR-0004: Numeric epsilon for the degeneracy tests

- Status: accepted
- Date: 2026-09-20
- Tags: `[PAPER]` says "approximately", `[ENGINEERING]` picks the number

## Context

Paper appendix A.3 freezes the grasp orientation in two situations:

1. `w < 0.01` m — the thumb tip and the virtual fingertip nearly coincide.
   The threshold is published.
2. `||z x d|| ~= 0` — the jaw line `z` and the wrist-to-fingertip direction `d`
   are nearly parallel, so the jaw plane has no unique normal, and
   `y = (z x d) / ||z x d||` is undefined.

Case 2's "approximately" has **no published epsilon**. Case 1's threshold is
published but the comparison is exact, so a width of exactly 0.01 m is a
boundary decision.

## Decision

Two configuration values in `configs/pipeline/mvp.yaml`, both under
`retarget.degeneracy`:

* `min_gripper_width_m: 0.01` — published; the comparison is strict (`w < 0.01`),
  matching the paper's wording, so `w == 0.01` is still valid.
* `parallel_dot_threshold: 0.999` — the degeneracy fires when
  `||z x d|| < sqrt(1 - 0.999^2)`, i.e. when `|z . d| > 0.999`, a mutual
  inclination of about **2.56 degrees**.

The test is expressed through the *dot* product rather than a raw cross-product
norm because the dot product is what has a geometrically interpretable
tolerance, and because `z` and `d` are both unit vectors by construction.

Both branches share the same fallback: freeze the orientation to the previous
valid estimate and set `FallbackCode.FROZEN_TO_PREVIOUS`. If no previous valid
orientation exists the frame is marked invalid with
`FallbackCode.NO_PREVIOUS_ORIENTATION` — an identity matrix is never substituted
(`docs/system-design.md` 7, `[ENGINEERING]`).

A third guard exists in code but not in the paper: an exact zero cross-product
norm (`< 1e-12`) is treated as the same degeneracy rather than dividing by zero.

## Consequences

* 2.56 degrees is generous enough to keep genuine pinch grasps valid (a real
  pinch has the jaw roughly perpendicular to the palm direction) and tight
  enough that the fallback is rare in practice. On the validation fixtures no
  frame trips it, which is the expected result: the fixtures keep the jaw plane
  well separated from the palm direction.
* A future data source with a different noise profile may need a different
  value; it is a config key, so no code change is required.
* `fallback_code` is persisted per frame, so the rate is measurable rather than
  hidden: the `retarget` stage reports `degenerate_frames` and
  `frozen_frames`.

## Evidence

* `configs/pipeline/mvp.yaml` (`retarget.degeneracy`)
* `ego2robot/retarget/hand_to_gripper.py`, `tcp_frame_from_keypoints`
* `retarget` stage metrics: `frozen_frames`, `degenerate_frames`,
  `fallback_frames`

## Alternatives considered

* **`||z x d|| < 1e-3`** (used by the third-party reproduction cited in
  `docs/system-design.md` 2.1) — rejected as a *paper* value: it is not from the
  paper. Numerically it is close to the choice here (1e-3 corresponds to about
  0.057 degrees), so it would fire far less often; the dot-product form is
  preferred because the tolerance then means something a reader can judge.
* **No epsilon; rely on `np.cross` + `np.linalg.norm` producing NaN** —
  rejected: `docs/system-design.md` 7 forbids letting an invalid frame produce
  NaN silently, and a NaN would propagate into the canonical action.
