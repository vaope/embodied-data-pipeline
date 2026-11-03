# Migration record

Snapshot date: 2026-11-03. Source copies are independent of the original research directories. SOURCE_SNAPSHOT.json records per-file source identity; no datasets, weights, run outputs or visual approval records were copied.

## Main package

The retrieval import name remains retrieval. The annotation, selection/baseline orchestration, bundle/readers, quality and isolated-worker adapters now live under pipeline in the same installable distribution. Existing tests were relocated to the package test tree; one fixture import was updated for that move.

Path changes: package schemas/configs/prompts use package locations; run outputs use the explicit/current workspace. EDP_WORKSPACE, EDP_MODEL_ROOT, EDP_RETRIEVAL_CONFIG and EDP_EGO2ROBOT_ROOT replace private-machine defaults. Stage 5 requires explicit admission reports/configs and audit inventory; its bounded output guard remains in force. Configuration and source identity changes intentionally invalidate historical cache identity; do not force old plans/approvals through.

The optional Hugging Face import in annotation asset preparation is delayed until after the remote-profile guard. Merely rejecting a nonlocal asset profile does not need that download dependency.

## Robotization

Actual sources, profiles, four robot model geometries and worker entry points were copied. Upstream model LICENSE/README files remain adjacent. The learned-model third_party directory, runtime caches, data, private approvals and hidden generated scene/rig XML files are excluded.

The current package requires an editable source installation so its repository-relative resources are available. Model interpreter/repository/checkpoint paths are explicit examples to configure locally. Original pyproject authorship/license declarations were not used to choose a new license for this project. No canonical geometry, motion, gates or IK logic changed. The original no-new-tests and image/user-review rules are preserved in the package AGENTS.md.

## Website

The old workspace-dependent export script is replaced with a builder consuming a frozen release manifest and static sources. It can build a no-media preview without other directories; --media-root attaches external selected media explicitly. Generated dist, private draft notes and prior screenshots are excluded from Git.

## Validation boundary

The main package is installed non-editably in a clean environment. Robotization uses its source-editable installation. Main-package existing checks, package-resource inspection, CLI help and the synthetic demo are checked separately; no new robotization model, IK or visual verification/acceptance is claimed. Public licensing and media review remain independent of a successful private source upload.
