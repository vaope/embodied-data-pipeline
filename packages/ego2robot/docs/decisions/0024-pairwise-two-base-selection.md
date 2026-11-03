# ADR-0024: Pairwise two-base selection and its negative result

- Status: **implemented; the joint-search result is negative and is reported as such**
- Date: 2026-11-02
- Scope: PLAN-02 B2 second half (the pairwise joint search).  Depends on ADR-0023.

## Context

ADR-0023 delivered two explicit bases chosen per arm.  The result was IK-perfect but
visually one-sided: the right arm's link chain is mostly outside the ego image
(visibility proxy 0.361 / 0.207 vs 1.000 / 0.690).  PLAN-02 asks for a joint choice:
per-arm top_k candidates → enumerate pairs → hard constraints → improve the worse arm
and the simultaneous-valid ratio → then compare error, continuity and occlusion.

## Decision

1. **Shortlist from the per-arm ranking** (`rig.search.top_k`, initial 5), then
   evaluate all pairs.  The per-arm ranking is kept intact so the joint step is a
   comparison of candidates, not a new search algorithm.
2. **Hard pair constraints**: base on its own side of the camera axis
   (`right x > 0 > left x`), base separation `>= 0.40 m`, keyframe IK pre-filter, and
   cross-arm contacts `<= max_cross_arm_contacts` sampled at joint keyframes.
3. **Joint quality order**: maximise the *worse* arm's visibility, then prefer more
   keyframes where both arms are visible, then smaller reach-ratio deviation, then
   candidate index.  A good arm cannot compensate for an invisible one.
4. **Joint keyframes** = union of each hand's TCP extremes on x/y/z, the closest and
   farthest TCP pair frames, and the min/max relative-rotation frames, thinned to 24.
   They are used for contact/visibility sampling only.
5. **Full-solve feasibility, not keyframe IK.**  The first implementation filtered
   pairs on isolated keyframe IK and rejected 21/25 pairs that the full solve (chained
   warm starts + 16 deterministic seeds) solves at 100%.  The pair evaluation therefore
   runs the same full per-arm solve the stage publishes, and keyframes only sample it.
6. **Joint jump is a preference here and a failing gate there.**  Every candidate pair
   still contains 1–3 frames above 1.5 rad (PIT-042).  Using it as a hard pair filter
   would reject all 25 pairs and hide the search behind an unrelated defect; the gate
   stays red and the number stays in `rig.json`, but it does not select the base.
   If any pair had been jump-free it would have been preferred.
7. **Reuse is keyed on the canonical action hash.**  A frozen `rig.json` is only reused
   when its `action_sha256` matches; otherwise the bases are re-searched with a warning.
   Silently reusing a placement for different targets is exactly the PIT-044 pattern.

## Result (negative, and that is the finding)

| episode | pairs | viable | rejected | winner min-visibility (keyframes) | simultaneously visible keyframes | chosen |
|---|---|---|---|---|---|---|
| fold_unfold_paper_000429 | 25 | 5 | 20 × `keyframe_ik` | 0.182 | 0 / 24 | R18 × L27 |
| screw_unscrew_bottle_cap_000391 | 25 | see `rig.json` | — | — | — | — |

The winner is **the same per-arm best pair** as ADR-0023: within the reduced 81-candidate
grid and the IK-based per-arm ranking, no pair makes both arms visible at the same time.
So the visibility defect is not a ranking bug — it is a property of the achievable
placement set (and, plausibly, of the ego viewpoint itself: two tabletop arms sit between
the camera and the workspace).

Next levers, in order of expected value:
1. visibility-aware candidate **generation** (sample positions/orientations with the
   joint visibility as a first-class term, instead of top-k of an IK-ranked grid);
2. a wider grid and larger `top_k`, budgeted as a background job;
3. accept the occlusion as a data-design fact and move to B3, where the arms are
   rendered and composited anyway — visibility then gets judged on rendered frames, not
   on a projection proxy.

## Consequences

- The rig remains a geometry product: both arms IK 100%, cross-arm contacts 0, joint
  FK self-check ≤1e-9 m, joint jump gate red.
- The base pair is recorded together with all 25 evaluated pairs and their reject
  reasons, so the negative result is auditable and re-runnable (`rig.json` →
  `joint_base_search`).
- No claim that "the two-base problem is solved" is made anywhere.

## Evidence

- `robots/joint_search.py`, `robots/rig_stage.py::_resolve_bases`, `configs/pipeline/bimanual-rig.yaml`
  (`rig.search.top_k`, `min_base_separation_m`).
- `data/episodes/egodex_fold_unfold_paper_basic_000429/robot/rig/rig.json` (`joint_base_search`,
  run `14b6dd6a5f25`), same block for `..._000391`.
- Learning-side record: `historical-local-evidence`.
