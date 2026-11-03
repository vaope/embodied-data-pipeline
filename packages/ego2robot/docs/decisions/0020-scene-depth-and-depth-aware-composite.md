# ADR-0020: Scene depth source, scale evidence and the depth-aware composite

- Status: accepted for the interfaces and the first implementation; **visual
  acceptance is pending** (AGENTS.md rule 9)
- Date: 2026-10-28
- Tags: `[PAPER]` A.4 names Depth Anything V3 and the composite rule; `[ENGINEERING]`
  the backend, the anchor set and the NaN policy

## Context

ADR-0008 fixed the M6 *interface* — a `SceneDepth` carries an explicit
`mode ∈ {metric, aligned, relative}`, only `metric`/`aligned` may reach the
compositor, and `DepthScaleAligner` is the only sanctioned promotion path — but it
deliberately left three things open:

1. which model/inference path produces the depth, and on which pixels;
2. which anchors decide that a "metric" depth really is on the metric scale;
3. what the compositor does with missing depth and with the hand-mask exemption.

Those are now implemented in `ego2robot/vision/scene_depth.py`,
`ego2robot/vision/composite.py`, the two M6 stages and the isolated worker
`tools/da3_model_worker.py`.

## Decision

### 1. Backend: DA3 Metric-Large in an isolated interpreter, focal-normalised

Depth Anything 3 `DA3METRIC-LARGE` is run per frame through
`tools/da3_model_worker.py` in the model environment (WSL), never imported into the
pipeline process. Two properties are asserted rather than assumed:

- **the model grid must equal the declared working grid** (`756×420`). The worker
  raises if the returned array has any other shape, because cropping or letterboxing
  would silently move every depth sample relative to the RGB pixel the compositor
  compares it with;
- **metres come from the focal rule, not from the file name**:
  `depth_m = raw * focal_work_px / 300`. Metric-Large emits focal-normalised depth;
  the divisor is a request field so a change is visible in the manifest instead of
  hidden in a constant.

### 2. The depth is estimated on the pixels it will be compared with

`scene_depth.input: auto` prefers the **inpainted background** and falls back to the
decoded source frames, recording the choice. Compositing a robot onto the inpainted
background and depth-testing it against the depth of the *un-inpainted* frame would
compare two different scenes: the hand occupies exactly the pixels the composite has
to get right.

### 3. Scale evidence is a two-tier anchor test, and the tier matters

* **Primary screen — TCP anchors** (`canonical_tcp_position_C_m`), available on every
  episode and on the inpainted background. The TCP is the human pinch point, i.e. a
  point on the manipulated object, so its own camera-frame z is a metric reference.
  Anchor samples that fall inside the dilated hand region are dropped, because there
  the depth is inpainted content and the test would grade the inpainter.
* **Reference check — hand keypoints** (`validation.reference_frames`, default 6).
  The worker is run a second time on a handful of *original* frames, where the hand is
  still in the image, and the hand keypoints act as anchors with known metric z.

The TCP screen alone is **weak evidence and is documented as such**: the TCP sits in
the finger gap, so the ray through it frequently misses the object it supposedly
touches. Measured on `egodex_vertical_pick_place_000` (52 frames):

| anchor | samples | median \|error\| | p90 \|error\| |
|---|---|---|---|
| TCP (`tcp_position_C_m`) | 50 | **77.8 mm** | 141.2 mm |
| hand keypoints, 6 original frames | 102 | **40.6 mm** | 112.6 mm |

The TCP number is not a calibration error — a one-off signed comparison against the
hand keypoints on the same episode gave a **+6.7 mm median bias** (median \|error\|
39.5 mm, p90 119.1 mm), i.e. the scale is essentially unbiased while the *spread* is
tens of millimetres because a keypoint on a finger edge samples the background beyond
the hand. Conclusion that the code and this ADR both carry: **a "metric" DA3 depth on
this data is usable for occlusion at the decimetre scale, and must not be claimed to
be centimetre-accurate.**

A failing anchor test is a stage warning plus `metrics.validation.status`, not a hard
failure: the geometry gate (M3) remains the merge blocker, and the user decides.

### 4. Compositor semantics

`composite_frame` implements the paper's rule with the two asymmetries the paper
itself states:

* **arm body is always composited** (`arm_mask = robot ∧ ¬gripper`). The arm stands
  between the camera and the workspace; a scene-depth sample on an arm pixel belongs
  to whatever is behind the arm, so depth-testing it would delete the forearm wherever
  it crosses a table edge;
* **gripper is depth-tested** per pixel: hidden where
  `D_robot + bias ≥ D_scene` (paper: strict `<`; `occlusion_bias_m` defaults to 0 and
  exists only for a sensitivity sweep);
* **the dilated hand mask exempts the gripper** (5×5, `[PAPER]` A.4), because the hand
  region is what inpainting replaced — the depth there describes the surface behind the
  hand, and testing against it would hide the jaws at the grasp point;
* **unknown scene depth cannot hide the robot.** `NaN` is treated as "no evidence",
  not as "in front" and not as "behind". The alternative — dropping the gripper wherever
  the depth model has a hole — is invisible in a finished video and therefore worse.

Outputs per frame: composite RGB, visible mask, occluded mask, and an indexed
**decision map** (`0` background, `1` arm, `2` gripper in front, `3` gripper occluded,
`4` gripper exempt). The decision map is the artifact that makes an occlusion
explainable instead of plausible.

`mode: debug_alpha` draws the robot wherever it has a mask and never depth-tests. It
writes to different file names (`frames-alpha/`, `composite-alpha.mp4`) and sets
`valid_episode: false`, per ADR-0008 alternative B.

## Consequences

* M6 costs one DA3 pass per episode (52 frames: ≈ 0.23 s/frame inference plus ≈ 40 s
  model load; a run measured 95 s including the reference check) on the local RTX 5070.
* The stage depends on external assets (DA3 repository, `DA3METRIC-LARGE`, a model
  interpreter), configured in `configs/pipeline/m6.yaml` exactly like the M5 SAM 3 and
  ProPainter paths. Those are not vendored.
* **Only the gripper is occluded.** An arm that visually crosses a prop stays drawn,
  by the paper's own rule. The remaining "the arm goes through the box" impression is
  therefore *not* a compositing bug and cannot be fixed here; it needs scene geometry
  (object meshes) or a different arm-handling policy, which would be a new ADR.
* `scene/` (52 × 1920 × 1080 float32 ≈ 430 MB/episode) and `composite/` add real disk
  cost to an episode.
* Depth accuracy is quantified per episode, not assumed: `metrics.validation` and
  `metrics.reference_check` travel with the stage manifest.

## Evidence

* `src/ego2robot/vision/scene_depth.py`, `vision/composite.py`, `vision/m6_stages.py`,
  `vision/da3_bridge.py`, `tools/da3_model_worker.py`,
  `tools/composite_sanity_cases.py`
* `configs/pipeline/m6.yaml`
* `data/episodes/egodex_vertical_pick_place_000/scene/stage-manifest.json`
  (`metrics.validation`, `metrics.reference_check`)
* `data/episodes/egodex_vertical_pick_place_000/composite/stage-manifest.json`
* `previews/composite-review-frames.png`, `previews/composite-occlusion-cases.png`,
  `previews/scene-depth-frame-*.png`
* `docs/system-design.md` 9.5, 9.6, 14

## Alternatives considered

* **Compare relative depth directly with the metric robot depth.** Rejected by
  ADR-0008 and re-rejected here: it produces a video that looks plausible and is
  wrong, which is the failure mode this project's gates exist to prevent.
* **Estimate depth on the original frames and composite onto the inpainted ones.**
  Rejected: the two differ exactly where the hand was, i.e. at the grasp site.
* **Use the TCP anchor alone.** Rejected as *sufficient* evidence (77.8 mm median
  residual against a real-hand 40.6 mm) though kept as the always-available screen.
* **Rescale every depth so the anchor residual becomes zero.** Rejected: that turns a
  measurement into a fit and hides a model error behind a constant. The residual is
  reported instead.
* **Reconstruct object meshes and render the composite with real geometry.** The
  honest fix for arm-through-prop, and out of scope for M6: it needs per-episode object
  reconstruction, which the paper does not describe either.
