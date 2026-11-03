# Embodied Data Pipeline

A research pipeline for video annotation, demonstration retrieval, data selection and traceable outputs. This repository includes an optional Ego2Robot reproduction and engineering implementation.

**Private preparation snapshot.** Source integration is in progress; this is not a production or fully accepted training pipeline. Original-code licensing and selected-media redistribution review are still pending. No model weights, raw datasets or private run outputs are included.

## What is included

| Module | Entry point | Current boundary |
|---|---|---|
| Video annotation | `edp-annotate` | AV-A/B/C wrappers, prompts, schemas, pinned upstream code, explicit execution/budgets and cache verification |
| Ego / Robot retrieval | `edp-retrieve`, `edp-sequence` | Ego pose/language recipe; Robot semantic retrieval, sequential matching and optional bounded effector-motion re-ranking |
| Selection and draft export | `edp-pipeline` | Text-baseline orchestration, overlap/source/task limits, eligibility reasons, unified draft export and readback |
| Artifact bundles | `edp-bundle` | Bounded historical Stage 5 integration; requires explicit inventories and source references |
| Optional robotization | `e2r` | Actual source, robot profiles, model geometry and explicit isolated workers; geometry/visual review rules retained |
| Static showcase | `website/` | Frozen annotation/query examples, clickable retrieval logic and sample JSON/CSV; media supplied separately |

Engineering checks, semantic correctness and user acceptance are recorded separately. A successful IK solve, export or demo does not establish real-world executability or training benefit. The draft export is not an official LeRobot compatibility claim.

## Install

Use Python 3.11+ and install FFmpeg / ffprobe on PATH. From the repository root:

```sh
python -m venv .venv
# Activate the environment using the command appropriate to your shell.
python -m pip install "./packages/pipeline[annotation,io]"
edp-demo --out ./outputs/synthetic-demo
```

Use a new demo output directory. This example generates three synthetic pose sequences and a gray video, builds nine retrieval windows, returns two candidates and reads features back. Semantic vectors are fixed synthetic values; it performs no language-model inference or download.

The optional semantic encoder is installed with `./packages/pipeline[semantic]`. Actual model/data setup is separate. Model worker environments can remain isolated.

Robotization is currently installed from its source checkout so model geometry, profiles and worker scripts remain available:

```sh
python -m pip install -e ./packages/ego2robot
# Additional video/data/robot extras are needed for the stages you choose.
e2r --help
```

See [installation and module usage](docs/USAGE.md), [third-party sources](docs/THIRD_PARTY.md) and [release status](docs/RELEASE_READINESS.md).

## Retrieval logic

- **Robot to Ego:** describe a step or step sequence, recall semantic candidates, optionally re-rank those candidates using observed effector speed, angular motion, stops and path ratios. Continuous queries additionally check order and step mappings. Re-ranking has no established aggregate quality benefit yet.
- **Ego to Ego:** combine language and initial hand-pose ranks for recall; compare four activity-weighted motion routes; fuse ranks and limit repeated source episodes.
- **Selection:** retain clip boundaries, annotations, available movement evidence and provenance. Unknown or unavailable information remains explicit.

[Continuous retrieval contract](docs/CONTINUOUS_SUBTASK_V1.md) · [Motion re-ranking](docs/MOTION_RERANKING_V2.md)

## Build the showcase

```sh
python website/build_showcase.py
python website/preview_server.py
```

Without external media, the page shows content, sample metadata and processing logic with explicit media placeholders. To include locally prepared selected media:

```sh
python website/build_showcase.py --media-root /path/to/selected-media
```

That directory must contain the `assets/showcase/` files named in `website/showcase.json`. Original research folders are not required to build the site. Do not add dataset videos to Git before checking their redistribution terms.

## Repository layout

```text
packages/pipeline/    annotation, retrieval, selection, artifact readers and exports
packages/ego2robot/   robotization source, profiles, geometry and isolated workers
website/             static source and frozen showcase manifest
examples/            small manifests and example-admission rules
scripts/             source/commit preparation checks
docs/                usage, contracts, provenance and current limitations
```

This is an independent reproduction/engineering project, not the official implementation of Ego2Robot, EgoANT, Flora or the underlying datasets. Each upstream component retains its own attribution and terms.
