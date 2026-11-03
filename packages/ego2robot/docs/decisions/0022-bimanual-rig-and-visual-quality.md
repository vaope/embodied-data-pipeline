# ADR-0022: Two-arm rig, joint scene render and ownership-aware composite

- Status: **proposed; not implemented** (PLAN-02 batches B2/B3). Do not read this
  as a decision taken: the table below is the plan the bimanual hand contract
  (ADR-0021) is being built towards.
- Date: 2026-11-02
- Type: [ENGINEERING] with [PAPER] anchors where the single-arm text applies.

## Context

ADR-0021 makes `hand`/`retarget` produce `H=2` canonical sequences. Everything
after that point is single-arm in the current code:

| Layer | Current assumption | Code |
|---|---|---|
| robot binding | one `robot_id`, one `T_C_B`, one `qpos`/`gripper_qpos` vector | `robots/stage.py`, `core/artifacts.py::robot_dir` |
| base placement | one grid search, one reach score `S = FR - 5|rho - 0.65|` | `robots/base_search.py` |
| IK/FK | `T_C_TCP[:, 0]`, `ik_success[:, 0]`, joint jump only recorded | `robots/stage.py`, `quality/l1.py` |
| render | one MuJoCo scene, one robot RGB/mask/depth | `vision/render_stage.py` |
| composite | one `robot_mask` + one `gripper_mask`, decision codes 0–4 | `vision/composite.py` |
| quality | `cross_arm_contacts_max` defined but never fed with contact data | `core/config.py`, `quality/l1.py` |
| export | one robot group, `hand_index=0` | `export/robot_formats.py` |

The failure mode is the same as in ADR-0021: a two-hand sequence silently bound to
one arm still produces a valid-looking episode. `core/capabilities.py` therefore
refuses `H=2` in all of the above today.

## Proposed decision (to be confirmed after B1 review)

1. **Rig model: two Panda arms in one MuJoCo scene**, keeping a binding interface
   so an ALOHA-class bimanual model can replace it later. Rationale: `T_E_TCP`,
   jaw axis and collision geometry are already measured and pinned for the Panda
   (ADR-0005, PIT-033, PIT-044); introducing a new morphology and a new two-arm
   layout at the same time would mix two variables (PIT-047).
2. **Left arm is a legal rigid placement, not a mirror.** `mirror()` would give
   `det(R) = -1`; the left base must be a proper rotation of the same model about
   the world/layout frame, and both `T_E_TCP` and the finger-pad centre must be
   re-derived per arm with `pad_centre_depth_in_hand`, not copied as constants.
3. **Name prefixing is total.** Every body/joint/geom/site/actuator/tendon/
   equality/default and referenced asset must be rewritten with an arm prefix;
   renaming only the root body produces a scene that loads and behaves wrongly.
4. **`hand_id` and `arm_id` are separate.** Default binding right→`right_arm`,
   left→`left_arm`; handedness sign is not a robot pose mirror.
5. **Two explicit fixed bases first.** Run the joint scene, IK/FK, projection and
   collision with two hand-placed bases before any automatic two-base search, so
   the assembly and the search algorithm are not validated at the same time.
6. **Joint base search later**: per-arm candidate ranking (top_k, initial value 5)
   → enumerate at most 25 pairs → hard constraints (rigid placement, side,
   inter-arm collision, reach per arm, joint visibility, per-arm continuity) →
   then maximise the *worse* arm's success and the simultaneous-valid ratio, then
   error/continuity/occlusion. No ad-hoc weighted sum before this ordering is
   measured. "Left of right" is a base-layout constraint, not a per-frame
   requirement that the two TCPs never cross.
7. **Per-arm quality, joint gates.** IK, FK error, joint limit, jump, gripper
   clipping/saturation and visibility are computed per arm *and* as the
   intersection of valid frames. Joint jump becomes an explicit gate with a
   deterministic re-solve/seed policy; smoothing `qpos` without re-checking FK is
   not a fix.
8. **Cross-arm contacts need real inputs.** The joint scene supplies per-frame
   contact counts plus a deduplicated body-pair diagnostic; the counting unit
   must be fixed before the existing threshold is reused. Missing metrics are
   `not_evaluated`, never 0.
9. **Joint render + ownership map.** Render both arms once, resolve inter-arm
   occlusion by metric depth, then compare with scene depth. Output joint
   RGB/depth/mask, an owner map (0 background, 1 right arm, 2 left arm) and
   per-arm gripper masks; composite keeps decision codes 0–4 and adds ownership
   statistics per arm. Hand-mask exclusion zones are bound to the hand that
   produced them, so the right hand's zone cannot keep the left gripper.
10. **Export gains per-arm fields**: hand→arm binding, per-arm joint names/order,
    commanded vs executed width, per-arm and joint validity masks, and the source
    time mapping. Two 7-D actions may form a derived 14-D view, but the canonical
    absolute pose keeps the `H` axis. Reader and learning-side Stage5 adapter are
    updated in the same batch.

## Consequences

- Until B2/B3 exist, a two-hand episode is a *data* product only: usable for
  reviewing hand geometry and canonical action, not for a robot episode.
- The two-arm work must not be used to mask the existing single-arm gaps (joint
  jump gate, cross-arm contact input, TCP pad-centre semantics, EgoDex 2D
  mismatch); B2's evidence must show both per-arm and joint metrics.
- Every new geometry result re-binds user review records by hash; old approvals
  do not carry over.

## Evidence

- Code and line references in the table above (2026-11-02 snapshot).
- Single-arm decisions this depends on: ADR-0005 (TCP offset), ADR-0006 (home
  semantics), ADR-0009 (fixed base), ADR-0020 (scene depth, depth-aware
  composite), PIT-033/PIT-044 (jaw axis and pad depth), PIT-042 (continuity),
  PIT-034 (ARKit→MANO mapping, only if a full 21-joint palm-plane method is
  introduced).
- Learning-side plan and drafts:
  `historical-local-evidence`,
  `ADR-B-双手数据契约-草稿.md`, `ADR-C-双臂Rig与视觉质量-草稿.md`.

## Alternatives

- **ALOHA/other true bimanual model first**: fewer naming complications, but a new
  gripper/TCP/joint mapping must be verified before any of the above can be
  trusted. Kept as the follow-up rig, not the first one.
- **Two independent single-arm scenes composited by layering**: no real inter-arm
  depth order, so occlusion between the arms would be decided by draw order.
  Rejected.
- **Mirroring one arm model**: reflection is not a rotation (`det = -1`) and
  produces a left arm with inverted chirality; rejected (ADR-0001 frame rules).
