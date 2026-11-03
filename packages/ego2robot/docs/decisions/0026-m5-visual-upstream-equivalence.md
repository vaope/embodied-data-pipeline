# ADR-0026: Human-removal upstream equivalence and the two-arm rig review record

- Status: **implementation decision; applied to `segmentation` + `inpainting`**
- Date: 2026-11-02
- Scope: PLAN-02 B3 second half — let the human-removal stages (`segmentation`,
  `inpainting`) run on a **two-hand** episode whose reviewed visual upstream is the
  two-arm `rig` instead of the single-arm `projection` stage.
- Config keys: `segmentation.approval_record`, `segmentation.rig_approval_record`,
  `core/capabilities.py::STAGE_HAND_CAPABILITY`.

## Context

`vision/m5_stages.py::_m3_prerequisites` required three succeeded stages before SAM 3
could run:

```python
for stage in ("ingest", "hand", "projection"):
    _require_succeeded_manifest(ctx.layout, stage)
...
review_path, review = _approved_m3_record(ctx, episode, cfg)
```

`projection` is a single-arm stage: its contract is one base pose, one rendered
robot, `[:, 0]` on every hand axis, and `core/capabilities.py` refuses `H=2` for it.
A two-hand episode therefore has **no** succeeded `projection` and **no**
`quality/reviews/m3-video.json`; the M3 gate could never be satisfied, no matter how
good the two-arm geometry was.

At the same time `segmentation`/`inpainting` were declared `H <= 1`.  That refusal was
also wrong in a way worth naming exactly: the human-removal path **never reads a
hand**.  SAM 3 is prompted with text (`text_prompt: "person"`); the hand stream enters
only through `_validate_hand_timebase`, which uses it as an independent witness that
the decoded video and the canonical timebase agree.  The `H <= 1` entry was copied
from the single-arm assumption, not from an actual `[:, 0]` in this stage.

## Decision

1. **Upstream equivalence, explicit and ordered.** `_visual_upstream()` accepts
   `rig` **or** `projection`, preferring `rig`; both absent is an error that names
   both.  The error code stays `segmentation.invalid_input` — this is still "the
   inputs do not satisfy the contract", not a new failure class.
2. **The review record type is per upstream.** A `rig`-upstream episode must carry
   `quality/reviews/rig-geometry.json` (`schema_version: ego2robot.rig.review.v1`,
   `scope: episode_rig_geometry`).  A `projection`-upstream episode keeps
   `quality/reviews/m3-video.json` unchanged.  The single-arm branch is byte-for-byte
   the old code path.
3. **The rig record binds geometry, not narrative.** It must list, by episode-relative
   path and SHA-256, at least `robot/rig/rig.json` and one rendered frame under
   `robot/rig/render/`, and every listed file must exist and match.  A re-run that
   changes a base, a trajectory or a render therefore invalidates the approval
   instead of silently inheriting it.
4. **`reviewer` must be literally `user`.** The loader refuses any other value and
   says why.  An agent may not self-approve geometry (AGENTS.md rules 2 and 9).
5. **`segmentation` and `inpainting` accept `H=2`, with the reason recorded.**
   `_validate_hand_timebase` now asserts validity against the *actual* hand count
   (`expect_hands=hand.num_hands`) instead of hardcoding `1`.  `scene_depth` and
   `composite` stay at `H <= 1` — they do index a hand (`m6_stages.py` reads the
   hand mask per frame and picks a single arm), and flipping their table entry without
   changing that code is exactly the silent single-hand failure the table exists to
   prevent (ADR-0021).  They are B3 step 3 / B4 work.
6. **The single-arm fingerprint key set is preserved verbatim.** The `rig` branch adds
   `rig_manifest` and `rig_json`; the `projection` branch keeps
   `projection_manifest` and nothing else, so no one-hand episode picks up a spurious
   input.

## Consequences

- A two-hand episode now fails with **"two-arm rig review record is missing"** instead
  of **"projection manifest is missing"** — the correct new state, and the only thing
  standing between the episode and SAM 3.
- `pick_place_000` remains blocked, as before, by a *stale* M3 record (5 of its 6
  evidence hashes no longer match the artifacts on disk, a consequence of the
  2026-10-27 TCP fix).  This ADR neither causes nor hides it.
- **`input_fingerprints.self` changes for the four one-hand episodes** even though
  their inputs did not: `_m5_fingerprint` hashes `m5_stages.py` and `m5_contracts.py`
  into the fingerprint, and both files changed.  No one-hand stage is re-run here;
  the artifacts on disk keep their `succeeded` state.  Anyone who now runs
  `--profile m5` on a one-hand episode will re-execute `segmentation` — that is the
  intended meaning of "the implementation changed", not a regression.
- `e2r stages` is unchanged; the DAG is still linear and `rig` still precedes nothing
  that the M5 profile lists, so `m5` runs `segmentation -> inpainting` on a `rig`
  episode exactly as it did on a `projection` episode.
- A future third upstream (e.g. fused multi-camera) must add itself to
  `VISUAL_UPSTREAM_STAGES` and bring its own review-record schema; the mechanism is
  "one reviewed visual upstream + its own bound record", not "any manifest will do".

## Evidence

- Code: `src/ego2robot/vision/m5_stages.py` (`_visual_upstream`,
  `_approved_rig_record`, `_m3_prerequisites`, `_validate_hand_timebase`,
  `SegmentationStage.fingerprint`), `src/ego2robot/core/capabilities.py`,
  `src/ego2robot/vision/m5_contracts.py`.
- Records written: `data/episodes/egodex_fold_unfold_paper_basic_000429/quality/reviews/rig-geometry.json`
  and `.../egodex_screw_unscrew_bottle_cap_000391/quality/reviews/rig-geometry.json`
  (23 bound evidence items each; rig runs `93daba24534e` / `213373597b72`).
- Contract check across all six episodes on disk (verdicts before/after the records):
  `historical-local-evidence`,
  `scripts/check_erasure_contract.py`.
- Learning-side page: `experiments/bimanual_b4_20261102/review.html`.

## Alternatives

1. **Write a stub `projection` manifest for two-arm episodes.** Rejected: it would
   claim a single-arm stage succeeded on data that stage refuses, and would put a
   fabricated artifact in the resume path.
2. **Let a two-arm episode reuse `m3-video.json`.** Rejected: the record binds
   `previews/projection-overlay.mp4` and `robot/<id>/trajectory.npz`, neither of which
   exists or can exist for this episode; the loader would have to skip exactly the
   checks that make it meaningful.
3. **Implement real `H=2` in `projection`/`quality` first.** Rejected for ordering:
   `projection` exists to review a single rendered robot; the two-arm equivalent is
   `rig` + `render` (ADR-0025), which is already built and reviewed.
4. **Flip `scene_depth`/`composite` to `H=2` in the same commit.** Rejected: their code
   still selects one arm; that is step 3 of PLAN-02, and doing it here would trade a
   readable refusal for a plausible wrong result.
