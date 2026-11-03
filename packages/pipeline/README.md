# Pipeline package

Install from the repository root:

```sh
python -m pip install "./packages/pipeline[annotation,io,dev]"
```

Python 3.11+; FFmpeg and ffprobe must be on PATH. Semantic inference additionally needs the semantic extra and its explicitly prepared encoder. Annotation model execution needs separately configured API credentials or GPU workers; installation, CLI help, prepare and the synthetic demo do not call them.

## Entrypoints

- `edp-demo --out new-output`: synthetic retrieval/index/readback; no language model.
- `edp-retrieve --help`: indexed Ego retrieval. YAML paths resolve relative to the YAML file; dictionary paths resolve relative to the caller's working directory.
- `edp-sequence --help`: Robot step/sequence semantic retrieval and optional observed-effector motion re-ranking. Default encoder downloads are disabled unless explicitly enabled.
- `edp-annotate doctor`: offline pinned-vendor and environment checks.
- `edp-annotate prepare --manifest jobs.jsonl --out new-plan`: probe/hash input videos and prepare a plan. `run-chain --execute` is a separate, explicit model/API action.
- `edp-pipeline --help`: V2 text-baseline orchestration, selection, eligibility and draft export/readback. Its text-only baseline is separate from sequence motion re-ranking.
- `edp-bundle --help`: historical bounded Stage 5 integration, not an unrestricted production runner. Explicit samples, output and audit inventory are required.

## Data and runtime configuration

Example annotation JSONL (paths relative to the manifest unless `--data-root` is supplied):

```json
{"id":"episode001","video":"data/episode001.mp4","instruction":"Place the cup on the table.","instruction_source":"dataset_instruction","source_id":"dataset/episode001"}
```

A non-empty instruction must include its source. Human-entry duration/frame limits and pilot controls remain in force. Preserve source hashes and interval semantics; do not reuse caches across changed code/config/model identity.

`EDP_WORKSPACE` selects a run workspace (default: current working directory). `EDP_MODEL_ROOT` selects model locations for the legacy capability report. `EDP_RETRIEVAL_CONFIG` can override the bundled retrieval config. `EDP_EGO2ROBOT_ROOT` selects the optional robotization source checkout; default is `packages/ego2robot` under the workspace. No other drive or private checkout is assumed.

Bounded Stage 5 samples must explicitly carry `stage1_report` and `stage1_config`; the runner also requires `--audit-inventory`. New run outputs are confined to `outputs/stage5/runs/` or `outputs/stage5/completion/` under the selected workspace. Missing input evidence fails closed; no old approval or admission report is silently reused.

## Verification

From this package directory, after installing the dev extra:

```sh
python -m pytest tests -q -p no:cacheprovider
```

Existing annotation tests use local fixtures/mock backends, not paid endpoints. Test inputs requiring private historical data remain skipped. Test success does not establish caption accuracy, cross-embodiment quality or downstream training gains. Consult the root validation record for the exact tested environment and results.
