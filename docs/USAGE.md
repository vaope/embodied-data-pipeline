# Usage and data boundaries

Start with the synthetic retrieval demo in the root README. It is the only supplied no-data example; live semantic encoders, annotation APIs and robotization are separate operations.

## Annotation

```sh
edp-annotate doctor
edp-annotate prepare --manifest /path/to/jobs.jsonl --out ./outputs/annotation-plan
```

The default profile is the historical hosted AV-A profile. Its model ID, prices and concurrency describe a recorded engineering configuration, not a promise of current service availability or pricing. Set endpoint/model/budget configuration for your provider. Only `run-chain --execute` or an explicit stage execution invokes models. AV-B/C local GPU workers require separate environments and pinned upstream models.

## Retrieval and selection

```sh
edp-retrieve --help
edp-sequence --help
edp-pipeline run --droid-root ./data/droid --ego-root ./artifacts/ego \
  --query 0 --top-k 50 --max-total 20 --out ./outputs/selection
```

The last command is the historical V2 **text-only** baseline and needs a compatible prepared Ego index plus the pinned semantic encoder. It applies temporal-overlap and source/task caps, writes reasons/eligibility and verifies exported references. It is not the newer per-step motion route.

For sequential queries use `edp-sequence` with the explicit query/manifest contract and optional `--robot-motion` described in CONTINUOUS_SUBTASK_V1.md and MOTION_RERANKING_V2.md. Query and corpus granularity must be stated. Missing pose is unavailable evidence, not zero distance.

## Bundles and robotization

`edp-bundle run` requires `--samples`, `--output`, `--run-id` and `--audit-inventory`; samples also require explicit Stage 1 report/config references. See the installed command help for the bounded reuse/execute policy. It is an integration profile for audited artifacts, not a general automatic raw-video-to-training pipeline.

The optional source package's editable installation and data/environment requirements are documented in packages/ego2robot/README.md. No new geometry or visual approval is implied by installation or source migration. Existing review receipts remain bound to their exact artifacts and code/config identity.

## Showcase

`python website/build_showcase.py` uses only files inside this repository. Missing media produces an explicit content-preview state. `--media-root` accepts a separately prepared folder with the exact relative files listed in website/showcase.json; it copies them into the chosen output directory. `--out` can build into another directory without modifying research outputs.

The displayed selected results are frozen artifacts, not real-time search. Candidate ranks and annotation clocks retain their original meanings. The supplied metadata and stored sample hashes remain research evidence awaiting final public-media and semantic review.
