# ADR-0021: Two-hand (bimanual) data contract on the existing `H` axis

- Status: **implementation decision; user review of the B1 evidence pending**
- Date: 2026-11-02
- Scope: `hand` + `retarget` only (PLAN-02 batches B0/B1). No robot, render,
  projection, quality or export stage is migrated; those refuse `H=2` by design.
- Type: [ENGINEERING]. The paper does not describe a two-hand scenario.

## Context

EgoDex HDF5 files carry both hands (`leftHand*` and `rightHand*` transforms), but
the Path A adapter read one hand, chosen by inspecting the file:

```python
hand = "right" if "rightHand" in probe.joint_names else "left"   # perception/hand/stage.py
```

`HandSequence` and `CanonicalActionSequence` already declared `[T, H, ...]`, and
`CanonicalActionSequence.relative_action` already looped per hand, but the config
rejected `hand_count != 1`, the adapter produced `H=1`, the retarget stage used
`hand_index=0`, and the previews/metrics indexed `[:, 0]`. Downstream stages
(`robot`, `render`, `projection`, `quality`, `export`, M5/M6) assume one arm, one
base pose and one rendered robot, and would have silently kept hand 0.

A two-hand episode read by the single-hand path is therefore **not wrong-looking
output, it is a plausible one-hand result**: that is the failure mode this ADR
addresses. Motion-length heuristics ("the left wrist barely moves") are not a
substitute: a static hand can be holding an object.

## Decision

1. **Keep one NPZ with the existing hand axis.** Both hands are stored along `H`
   in `hand/sequence.npz` and every `canonical/actions*.npz`, in the order given
   by the new `hand.sides` configuration (Path A default `[right, left]`). No
   `actions-hand0/hand1` split: a split would add a second, independent time axis
   and hand-identity bookkeeping without any consumer that needs it.
2. **Identity is explicit metadata, not array position.** The adapter and both
   stages persist `hand_ids` (npz `meta`, `hand/provenance.json`, stage metrics).
   `H > 1` without `hand_ids` is a validation error; `H == 1` without them stays
   readable as `hand.v1` and is reported as a warning (identity recovered from
   `handedness`). A hand id must agree with the `handedness` sign
   (`right=+1`, `left=-1`, [PAPER] appendix A.3); a conflict is an error.
3. **Stable slots: never swap hands between frames.** Both hands share the
   decoded frame range, PTS grid and `T_W_C`. A missing / low-confidence frame
   keeps its slot and is marked `valid=false`; a static hand may stay valid. No
   interpolation borrows from the other hand.
4. **Per-hand temporal processing.** Outlier rejection, gap filling, smoothing
   and SO(3) repair run independently per hand with the unchanged paper
   parameters, and the four canonical revisions (raw / filtered / final /
   delta) are stacked along `H`. The first implementation deliberately does not
   add a two-hand coupling optimiser; the cross-hand relation is *diagnosed*
   (distance and relative rotation curves, raw vs final) rather than optimised,
   so any change to it is visible instead of absorbed into "smoother output".
5. **Configuration is explicit and consistent.** `hand.selection` is
   `legacy | both | explicit`, with `hand.sides`; `hand.hand_count` must agree
   with the resolved sides in every case, and `selection: legacy` is single-hand
   by construction. `configs/pipeline/bimanual.yaml` is the first profile and
   stops at `retarget`.
6. **Un-migrated consumers refuse `H>1`.** `core/capabilities.py` declares the
   maximum hand count per stage; `hand`/`retarget` accept 2 and
   `robot/render/projection/quality/export/segmentation/scene_depth/composite`
   accept 1. Each of those entry points checks the loaded sequence and raises
   `HandCapabilityError` with the stage name, the observed `H` and the reason,
   instead of indexing `[:, 0]`.
7. **No schema version bump for the hand axis.** The axis already existed in
   `hand.v1` / `canonical_action.v1`; `H` moves from 1 to 2, which the declared
   shape already allows. `hand_ids` is additive metadata, and a multi-hand
   reader must validate it (rule 2). A new version becomes necessary if the
   per-joint skeleton semantics change (see PIT-034: the ARKit→MANO mapping still
   puts a MCP where a PIP belongs). That fix is **not** part of this change: the
   four retarget joints (wrist, thumb tip, index tip, middle tip) are measured,
   not derived, so the two-hand retarget is unaffected.

## Migration

- Existing `H=1` artifacts keep their values and paths. The B0 freeze
  (`experiments/bimanual_b0b1_20261102/frozen_h1_baseline/`) copies
  `hand/*.npz`, `canonical/*.npz` and the manifests of the four single-hand
  episodes; after the change, all four episodes were re-run through
  `hand`+`retarget` and every array is bit-identical to the frozen copy
  (`comparison.json`).
- Stage versions moved `hand` and `retarget` from `1` to `2`, and both
  fingerprints now include the selection/sides (hand) and the stage version
  (retarget), so an old cache entry can never be mistaken for an `H=2` run.
- New two-hand episodes are built in isolated directories
  (`data/episodes/egodex_<task>_<idx>/`) from the original MP4/HDF5; the
  `H=1` directories are not rewritten.
- Old readers of `hand/sequence.npz` (a) keep working for `H=1`, (b) must check
  `expect_hands` for `H=2`, and (c) get an explicit capability error from the
  in-repo consumers listed in decision 6.

## Consequences

- The perception/robot boundary keeps exactly one canonical representation; the
  two-hand extension does not leak robot-specific details into perception.
- A pipeline that asks for two hands can no longer silently produce one. The
  cost is that the bimanual profile cannot run past `retarget` until B2 (rig,
  base search, IK) and B3 (render, composite, ownership) exist.
- Per-hand filtering is more permissive than a shared threshold would be: each
  hand gets its own velocity median. On the screw/unscrew sample the final
  hand-to-hand relative rotation differs from the raw one by up to ~9° at some
  frames (see the cross-hand curve), so "smoothing preserved the coordination"
  is **not** claimed.
- `hand/validity-per-hand.json` adds a per-frame invalid list per hand; the
  existing single-hand `valid_ratio` metrics keep their meaning.

## Evidence

- `configs/pipeline/bimanual.yaml`, `src/ego2robot/core/capabilities.py`,
  `core/config.py::HandConfig`, `core/schema.py::ArrayRecord.validate_hand_identity`,
  `ingest/egodex.py::EgoDexHandAdapter`, `perception/hand/stage.py`,
  `retarget/stage.py`, `retarget/preview.py`, `perception/hand/preview.py`.
- Run records and images: learning workspace
  `experiments/bimanual_b0b1_20261102/` (`README.md`, `review.html`,
  `comparison.json`, `h2_guard.json`) and the episode directories
  `data/episodes/egodex_fold_unfold_paper_basic_000429/`,
  `data/episodes/egodex_screw_unscrew_bottle_cap_000391/`.
- The 2-D skeleton overlay of the EgoDex annotation does **not** line up tightly
  with the RGB hand (documented perspective mismatch; right hand
  `inside_image_ratio` 0.82 vs left 1.00 on the paper-fold sample). The two-hand
  overlay shows this honestly; it is not treated as a two-hand regression, and
  the M1 "overlay matches hand pixels" gate remains unmet for these episodes.

## Alternatives

- **Two files per hand** (`actions-hand0.npz`, `actions-hand1.npz`): convenient
  for the single-hand consumers, but adds timebase pairing, identity drift and a
  second "which file is this?" contract. Rejected for the canonical format; the
  robot side may still keep per-arm files in B2.
- **Automatic active-hand selection on motion length** (the old PLAN-02 stage 0):
  keeps a plausible-looking trajectory for the wrong reason and cannot tell a
  resting hand from a holding hand. Deferred; if resumed it must score
  translation, rotation, validity and the holding role, with a configured
  threshold and recorded evidence.
- **Bumping the schema version for `H=2`**: would force every `H=1` reader to
  migrate without changing any declared semantics, and would hide that the axis
  was always there. Rejected; identity metadata plus capability guards carry the
  change instead.
