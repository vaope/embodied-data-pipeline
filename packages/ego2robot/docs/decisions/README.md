# ADR index

Architecture decision records for choices the Ego2Robot paper does not publish.
AGENTS.md rule 10 requires every such choice to be recorded here and referenced
from the configuration that carries it.

Paper reference throughout: *Ego2Robot: Scalable Robot Data Synthesis from
Egocentric Human Data*, arXiv:2608.02580v1. A local copy of the extracted text
is in `../references/ego2robot-paper.txt`.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-transform-naming.md) | Direction of `T_A_B` and how a pose is written down | accepted |
| [0002](0002-camera-frame-convention.md) | Camera optical frame and the MuJoCo camera basis | accepted |
| [0003](0003-arkit-to-mano21.md) | EgoDex ARKit skeleton -> MANO-21 arrays | accepted |
| [0004](0004-degeneracy-epsilon.md) | Numeric epsilon for the "z parallel to d" degeneracy test | accepted |
| [0005](0005-franka-tcp-offset.md) | `T_E_TCP` and the Panda end-effector frame | accepted |
| [0006](0006-home-pose-semantics.md) | What "home configuration" means for large gaps | accepted |
| [0007](0007-ik-tolerance-semantics.md) | What the paper's IK "1e-5 / 100 iterations" controls | accepted |
| [0008](0008-relative-depth-policy.md) | Metric-only depth compositing; relative depth barred | accepted (M6 not implemented) |
| [0009](0009-fixed-base-placement.md) | How the MVP fixed base placement is chosen | accepted |
| [0010](0010-gripper-command-scaling.md) | The 1-D gripper channel in the 7-D action | open (M7) |
| [0011](0011-m5-runtime-and-mask-contract.md) | M5 model processes, frame indexes and mask refinement | implementation decision; visual review pending |
| [0012](0012-shadow-mask-experiment.md) | Episode 000 shadow-aware mask trial | experiment; visual review pending |
| [0013](0013-e2fgvi-comparison.md) | E2FGVI-HQ comparison runtime and evidence | experiment blocked by missing CUDA extension |
| [0014](0014-temporal-plate-experiment.md) | Episode 000 temporal clean-plate trial | experiment; visual review pending |
| [0015](0015-hybrid-inpainting-preview.md) | Combine sampled tabletop pixels with ProPainter fallback | experiment; visual review pending |
| [0016](0016-diffueraser-comparison.md) | DiffuEraser comparison on episode 000 | experiment run; visual review pending |
| [0017](0017-diffueraser-1080-memory-probe.md) | Native 1080p, tiled inference and INT8 memory probe | experiment run; visual review pending |
| [0018](0018-robot-format-trial.md) | Project-local HDF5 and JSONL robot trajectory export trial | experiment implemented; artifact review pending |
| [0019](0019-multi-morphology-trial.md) | Early geometry trial with three additional robot morphologies | experiment; visual review pending |
| [0020](0020-scene-depth-and-depth-aware-composite.md) | Scene depth backend, scale anchors and the depth-aware composite | implementation decision; visual review pending |
| [0021](0021-bimanual-hand-contract.md) | Two-hand data contract on the existing `H` axis (`hand`+`retarget` only) | implementation decision; B1 evidence review pending |
| [0022](0022-bimanual-rig-and-visual-quality.md) | Two-arm rig, joint scene render and ownership-aware composite | proposed; not implemented |
| [0023](0023-two-arm-rig-assembly.md) | Two-arm rig assembly: per-arm IK/FK, joint collision scene, per-arm artifacts | implementation decision; B2 geometry review pending |
| [0025](0025-joint-two-arm-render.md) | Joint two-arm render, ownership map, per-arm masks and metric depth | implementation decision; robot-only render, compositing pending |
| [0024](0024-pairwise-two-base-selection.md) | Pairwise two-base selection (top_k x top_k, worst-arm visibility) | implemented; result negative (no pair makes both arms visible in this grid) |
| [0026](0026-m5-visual-upstream-equivalence.md) | Human-removal upstream equivalence (`rig` or `projection`) and the two-arm rig review record | implementation decision applied to `segmentation`+`inpainting`; erase not yet run |

## How to add an ADR

Copy the structure of an existing record: **Context**, **Decision**,
**Consequences**, **Evidence**, **Alternatives**. State explicitly whether the
choice is `[PAPER]`, `[ENGINEERING]` or `[OPEN]`, and cite the config key that
carries it so a reviewer can find the value in a resolved config dump.
