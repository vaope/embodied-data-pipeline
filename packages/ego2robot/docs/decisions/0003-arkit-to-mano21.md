# ADR-0003: EgoDex ARKit skeleton -> MANO-21 arrays

- Status: accepted
- Date: 2026-09-20
- Tags: `[OPEN]` in the paper, `[ENGINEERING]` here, Path A, schema

## Context

The persisted hand contract is `HandSequence.keypoints_C_m` with shape
`[T, H, 21, 3]`, and `docs/system-design.md` 6.2 calls that a MANO-21 layout.
The paper describes Path B (WiLoR -> MANO parameters) in those terms, and
appendix A.1 states that the wrist-to-fingertip vector `d` is "taken directly
from the MANO keypoints".

EgoDex does **not** provide MANO parameters. Its HDF5 files store a *skeletal
transform tree* with 68 joints: 8 spine, 4 neck, 4 per arm, and **24 per hand**
(4 metacarpals + a 5-joint chain for each of index/little/middle/ring, and a
4-joint chain for the thumb). Apple's `utils/skeleton_tfs.py` is the
authoritative list and is vendored at `docs/references/egodex/`.

The paper never mentions ARKit and never specifies a remap. This is the largest
implementation gap on the Path A side.

The retargeting maths of equations 1-3 needs exactly four points: wrist, thumb
tip, index tip, middle tip. All four exist as real ARKit nodes.

## Decision

`ego2robot/ingest/egodex.py` defines an explicit, reviewable mapping
`MANO21_SOURCE_INDEX` from MANO-21 index to an ARKit joint suffix:

| MANO index | joint | ARKit source |
|---|---|---|
| 0 | wrist | `{hand}Hand` |
| 1, 2 | thumb cmc, mcp | `{hand}ThumbKnuckle` |
| 3 | thumb ip | `{hand}ThumbIntermediateTip` |
| 4 | thumb tip | `{hand}ThumbTip` |
| 6 | index mcp | `{hand}IndexFingerKnuckle` |
| 7 | index pip | `{hand}IndexFingerIntermediateTip` |
| 8 | index tip | `{hand}IndexFingerTip` |
| 10, 11, 12 | middle mcp, pip, tip | `{hand}MiddleFinger*` |
| 14, 15, 16 | ring | `{hand}RingFinger*` |
| 18, 19, 20 | little | `{hand}LittleFinger*` |
| 5, 9, 13, 17 | the four metacarpals | **no ARKit node** |

Rules that make this honest rather than convenient:

1. **Indices 5, 9, 13, 17 are written as `NaN`** with `keypoint_confidence = 0`
   and are listed in the persisted provenance under `nan_mano_indices`. They are
   not interpolated and not zero-filled; implementation-plan.md Gate M1 bans
   zero-filling a missing joint.
2. The adapter raises if any of the four retarget joints is absent, rather than
   silently degrading.
3. `HandSequence.validate` permits `NaN` in `keypoints_C_m` / `keypoints_uv_px`
   only; every other array must be finite.
4. The provenance records `source_joint_order: arkit24`,
   `target_joint_order: mano21` and the frame conversion, so a reader can always
   reconstruct how the numbers were produced.

Where ARKit has a joint but MANO-21 does not (the metacarpals) the information is
dropped. That is lossless for the current pipeline: nothing consumes them.

## Consequences

* Downstream code may only read the four retarget indices without checking a
  mask. Anything that wants all 21 joints must handle `NaN` and should treat the
  EgoDex source as a partial skeleton.
* The synthetic fixtures in `data/fixtures/` produce true MANO-21 skeletons, so
  the pipeline is exercised on a complete layout as well as the partial one.
* A Path B adapter (M8) will populate all 21 joints; the schema already allows
  both, and `keypoint_confidence` distinguishes them.

## Evidence

* `docs/references/egodex/utils__skeleton_tfs.py` (Apple, vendored)
* `docs/references/egodex/README.md` "Dataset Structure"
* `ego2robot/ingest/egodex.py` (`MANO21_SOURCE_INDEX`, `MANO21_JOINT_NAMES`)
* `provenance.json` written by the `hand` stage

## Alternatives considered

* **Interpolate the missing metacarpals** — rejected: it manufactures data the
  source does not contain, and a reviewer could not tell which joints were real.
* **Extend the schema to a 24-joint ARKit layout** — rejected for the MVP: it
  would force every consumer to know about a second joint vocabulary for four
  joints that nothing reads, and MANO-21 is what the paper's Path B produces.
* **Use `{hand}IndexFingerMetacarpal` as MANO index 5** — tempting but wrong:
  the ARKit metacarpal sits *behind* the knuckle along the bone, whereas MANO's
  index-5 is the knuckle-adjacent joint of a different topology; mixing them
  would silently change what "index mcp" means between Path A and Path B.
