# ADR-0023: Two-arm rig assembly, per-arm artifacts and the joint-scene checks

- Status: **implementation decision; B2 geometry evidence awaits user review**
- Date: 2026-11-02
- Scope: PLAN-02 batch B2 first half — two explicit fixed bases, per-arm IK/FK, one
  MuJoCo scene for cross-arm contacts.  No vision (B3), no automatic two-base
  pairwise search (B2 second half), no export (B4).
- Depends on: ADR-0021 (hand contract), ADR-0022 (rig plan), ADR-0005 (TCP), ADR-0009
  (fixed base), PIT-044 (fixture base is stale, irrelevant here but the same class of
  mistake), PIT-047 (single-variable comparisons).

## Context

ADR-0021 made `hand`+`retarget` produce `[T, 2, ...]`.  Everything downstream still
assumed one arm, so `core/capabilities.py` refuses `H=2` in `robot`/`render`/etc.  B2
has to bind two arms without touching the reviewed single-arm path, and it has to make
the paper's `l1_cross_arm_contacts_max` threshold computable for the first time.

Two constraints shaped the design:

1. the single-arm `robot` stage is the reviewed M3 artifact; any refactor of its IK
   loop must be provably behaviour-preserving;
2. the two-arm scene must not become a second, silently different kinematic model.

## Decision

1. **One IK implementation, shared.**  The per-frame loop moved to
   `robots/ik.py::solve_trajectory`.  The single-arm stage and the rig stage call it.
   Proof it did not change: after the refactor, `egodex_pour_000` re-solved 190 frames
   and `trajectory.npz` is **bit-identical** to the pre-refactor file (all 14 arrays,
   IK 190/190), and the four `hand`/`canonical` files of the B0 freeze stay identical.
2. **Per-arm artifacts, not one 14-DoF trajectory.**  Each arm writes its own
   `robot/rig/<arm_id>/trajectory.npz` using the existing `RobotTrajectory` schema in
   **that arm's base frame**, with `arm_id`/`hand_id`/`T_C_B` in the meta.  A 14-DoF
   joint vector could not answer "which arm's FK is this", and the canonical `H` axis
   stays the only place where the two hands are indexed together.
3. **Joint scene = same model, twice, fully prefixed.**  `robots/rig.py` generates a
   collision-only XML next to the vendored model (`.<hash>.xml`, `meshdir="assets"`
   keeps resolving).  Every body/joint/tendon/equality/actuator/contact-exclude name is
   prefixed `R`/`L`, and the two arm subtrees are placed by their bases.  The scene
   world **is the right arm's base frame**, which is the frame the single-arm camera
   code already uses, so no new convention is introduced.
4. **The joint scene must reproduce the independent FK.**  Per frame, each arm's
   `FK(qpos)` inside the joint scene is compared with the independent single-arm
   `FK(qpos)` pushed through the relative placement.  Measured disagreement: right
   `0.0 m`, left `4.5e-9 m` (both episodes).  A failure here means the assembly is
   wrong and is a hard error, not a warning: it is the only cheap proof that prefixing,
   base placement and the relative transform agree.
5. **Bases: one per arm, explicit and frozen.**  `rig.base_mode: per_arm_search` runs
   the existing per-arm grid search (reduced 81-candidate set) with
   `lateral_sign=+1` for the right arm and `-1` for the left, then freezes both
   `T_C_B` in `robot/rig/rig.json`; later runs reuse them, and pasting them into the
   profile with `base_mode: fixed` makes the placement part of the config.  This is the
   "two explicit fixed bases" PLAN-02 asks for before any pairwise search.
6. **A new base-search constraint, default off.**  `BaseSearchConfig.min_forward_depth_m`
   rejects a base whose camera-frame `z` is too small.  The paper's 0.20 m constraint is
   a *distance*, which a base beside the camera satisfies while sitting almost in the
   camera plane; the first rig run did exactly that (right base at `z = 0.117 m`, arm
   projected across the frame).  The rig sets `0.30 m`; the single-arm default stays
   `0.0`, so its behaviour is unchanged.
7. **Contacts are reported in two units.**  Per frame: raw contact count and
   deduplicated body pairs.  The gate uses the paper's count-based threshold
   (`max_cross_arm_contacts`), while the pair list is the diagnostic.  Converting one
   into the other after the fact has already produced wrong conclusions in this project
   (PIT-047).
8. **Per-arm gates and a joint intersection.**  IK ratio, p95 errors, joint-limit
   frames, self/scene collision frames, joint-jump frames, reach ratio and a visibility
   proxy are computed per arm; the joint numbers are the intersection of valid frames,
   so one good arm cannot hide a bad one.  Joint jump is now an explicit gate (it was
   only a metric before), and it **fails** on both episodes (left 3 / 0 frames, right
   1 / 0 frames > 1.5 rad) — the PIT-042 continuity defect is still open.
9. **Visibility is a proxy, and labelled as one.**  The fraction of an arm's link
   origins that project inside the image and in front of the camera.  It is not a
   rendered occlusion test (that needs B3's render + depth), and it is reported as
   `not_evaluated` for anything beyond that: no claim about masks, occlusion or image
   quality is made from it.

## Consequences

- On both B2 episodes: both arms solve 100% of their valid frames with p95 position
  error `≈3e-6 m`, cross-arm contacts are `0` in every frame (max 0), the joint-scene
  FK agrees with the independent FK to `5e-9 m`, and 100% of frames are valid for both
  arms simultaneously.
- The joint-jump gate **fails**: 3 frames (left) on `fold_unfold_paper_000429` and 1
  frame (right) on `screw_unscrew_bottle_cap_000391` exceed 1.5 rad.  Solving that is a
  separate fix (branch-aware re-solve / short-window continuity), not hidden by the
  two-arm success.
- Visibility is asymmetric (`right` 0.36 / 0.21, `left` 1.00 / 0.69) because each base
  was chosen for IK feasibility alone.  The pairwise joint search (top_k × top_k with
  visibility, side, separation and contact constraints) is the next B2 step, and the
  current numbers are its baseline.
- 29 frames of `screw_unscrew_bottle_cap_000391` request a right-hand width above the
  Panda's 80 mm; they are counted, not clipped silently.  The clipping belongs to the
  robot binding/export decision (ADR-0010, B4).
- Nothing in `render`/`composite`/`quality`/`export` consumes the rig yet: they still
  refuse `H=2`, so a two-arm episode is a geometry product, not a visual one.
- The projected arms overlap the human arms in the RGB frames.  That is expected: B3
  removes the human arms; until then the overlay is a geometry check, not a preview of
  the final synthetic video.

## Evidence

- `configs/pipeline/bimanual-rig.yaml`, `robots/ik.py`, `robots/rig.py`,
  `robots/rig_stage.py`, `robots/base_search.py` (`min_forward_depth_m`),
  `core/config.py` (`RigConfig`), `core/artifacts.py` (`robot/rig/...` layout).
- Runs on `data/episodes/egodex_fold_unfold_paper_basic_000429` (`run 2a5baa2ba87a`,
  `a8832be7e904`) and `.../egodex_screw_unscrew_bottle_cap_000391` (`0751f0d89a17`).
- Learning-side record with images and numbers:
  `historical-local-evidence` (`README.md`,
  `review.html`).
- Single-arm preservation: `experiments/bimanual_b0b1_20261102/comparison.json`
  (four episodes, all arrays identical) plus the 190-frame `trajectory.npz` comparison
  recorded in that experiment's README.

## Alternatives

- **One 14-DoF IK problem for both arms.**  Would let the solver trade one arm's task
  error against the other's and mixes two task costs in one QP; also rewrites the
  reviewed M3 loop.  Rejected for the first delivery; the shared per-frame loop keeps
  the doors open for a joint solver later.
- **Two scenes composited by draw order.**  No cross-arm depth order or collision, which
  is the whole point of the joint scene.  Rejected (same reasoning as ADR-0022).
- **Mirroring one arm's model for the left side.**  A reflection has `det = -1`; the
  generated scene rejects such a placement explicitly.  Rejected.
- **Hand-tuning the two bases until the overlay "looks right".**  Not reproducible and
  hides the placement problem the search is supposed to expose.  The current bases are
  the recorded output of the per-arm search, warts included.
