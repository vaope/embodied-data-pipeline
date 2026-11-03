# Ego2Robot Reproduction Working Agreement

This repository implements the data-processing pipeline described in:

- [`docs/system-design.md`](docs/system-design.md): normative architecture, data contracts, coordinate conventions, module boundaries, and quality gates.
- [`docs/implementation-plan.md`](docs/implementation-plan.md): normative implementation order, acceptance criteria, and milestone checklist.

All code changes MUST follow these rules:

1. Read both documents before implementing a new module or changing a public schema.
2. Implement milestones in dependency order. Do not start a downstream visual stage before its upstream geometry gate passes.
3. Treat the official paper and appendix as the method source of truth. Treat third-party reproductions as references, not specifications.
4. Preserve the canonical, robot-independent representation between hand perception and robot-specific IK.
5. Use SI units, the transform naming convention `T_A_B` defined in the design, and quaternions in `xyzw` order at persisted boundaries.
6. Every pipeline stage must be deterministic for fixed inputs/configuration, emit a manifest, support resume, and expose typed inputs/outputs.
7. Do not write unit tests, integration tests, golden tests, or other automated test cases unless the user explicitly requests and provides them. Any automated-test tasks mentioned in the implementation plan are non-binding and are overridden by this rule.
8. Every verification run must produce user-viewable image artifacts showing the result and relevant overlays. Temporal stages may additionally produce videos, but representative result images are still required.
9. The user provides the test inputs and acceptance criteria and is the sole authority for pass/fail decisions. Do not mark a milestone or visual gate as accepted until the user has reviewed its artifacts and explicitly approved it.
10. Do not silently invent values that the paper does not specify. Add an ADR under `docs/decisions/`, record the evidence and trade-off, and reference it from the relevant configuration.
11. Schema or coordinate-convention changes require updating both normative documents and providing a migration path for existing artifacts.
12. Training, policy evaluation, and large-scale VLA work are outside the current project scope unless the project boundary is explicitly revised.
13. The initial validation source is the official public EgoDex test set. Start with one short single-hand episode and its paired MP4/HDF5 annotations; do not download or process the full training set unless the user explicitly requests it.

