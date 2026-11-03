# ADR-0010: The 1-D gripper channel in the 7-D action

- Status: **open** — required before M7 (export)
- Date: 2026-09-20
- Tags: `[OPEN]` in the paper, export

## Context

Paper appendix C.2:

> Each action step is 7-dim: 3D position delta `dp`, 3D rotation delta `domega`
> (rotation vector), 1D gripper.

The first six dimensions are fully specified (equation 10 fixes the frame
transform). The seventh is not: the paper never says whether the 1-D gripper
channel is a metric width in metres, a normalised opening in `[0, 1]`, a
normalised command in `[-1, 1]`, or a binary open/close. The only sentence in the
LaTeX source about normalisation — "Each dimension is Q99-normalized to
`[-1, 1]`" — is commented out and is absent from the published HTML and PDF.

This matters because a policy trained on one convention will not transfer to
another, and because implementation-plan.md section 14 forbids inventing values
the paper does not specify.

## Decision (provisional)

The canonical artifact stores the **physical** quantity and defers the command
mapping:

* `CanonicalActionSequence.gripper_width_m` is metres, jaw opening, persisted and
  authoritative.
* `delta_npz["gripper_command"]` currently carries the same metric width
  (`robot_binding.gripper_command_mode: width_m`), explicitly labelled as an
  export-profile decision rather than a paper value.
* The robot side maps width to joints as `finger_joint = width_m / 2`
  (ADR-0005), clamped to the model's joint range.

Normalisation is therefore an exporter concern, applied at M7 when there is a
training target to match, and it must be recorded in the export profile.

## What still has to be decided (M7)

1. Which convention the LeRobot/RLDS exporter emits: metric width, `[0,1]`, or
   `[-1,1]`.
2. If normalised: per-dataset or per-morphology statistics, and whether they are
   computed from L1-valid frames only (the paper's L2 already excludes invalid
   frames from its quantiles, which is the natural precedent).
3. Whether the "Q99-normalized to [-1,1]" sentence, were it ever published,
   applies per dimension to all seven or only to the gripper.
4. Whether a binary open/close variant is wanted for a discrete-action baseline.

None of these can be settled without knowing the consumer. Deciding now would
freeze a guess into a persisted schema.

## Consequences

* Nothing in M0-M3 depends on the choice: the canonical layer stores metres and
  the robot layer consumes metres.
* `gripper_command` in `actions-delta.npz` is documented as provisional in its
  array spec, so a consumer cannot mistake it for a paper-defined convention.
* M7's exporter must write the chosen convention into the export manifest, and
  the docs must record it as `[ENGINEERING]` or `[PAPER]` accordingly.

## Evidence

* `ego2robot/core/schema.py`, `ARRAY_SPECS["canonical_delta"]["gripper_command"]`
* `ego2robot/core/schema.py`, `CanonicalActionSequence.relative_action`
* `configs/pipeline/mvp.yaml`, `robot_binding.gripper_command_mode`
* `docs/references/ego2robot-paper.txt`, appendix C.2

## Alternatives considered

* **Normalise to `[0, 1]` by the morphology's published gripper range now** —
  rejected as premature: it is a guess about the consumer, and it destroys the
  physical meaning of the stored channel while looking authoritative.
* **Store both metric and normalised** — reasonable, but it doubles the schema
  surface for a decision no stage currently needs. Revisit at M7.
