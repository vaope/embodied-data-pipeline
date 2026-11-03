# Portable retrieval preview

## Continuous subtask retrieval (V1)

`edp-sequence --manifest events.jsonl --query query.json --out new-output --cache embeddings.npz --render-clips 3`

Requires the `semantic` extra and the pinned local sentence-transformer. Downloads are disabled unless `--allow-download` is passed. The cache checks encoder configuration and implementation identity. Query JSON contains `query_id`, `key_step_id`, `steps` (each has `step_id`, `source_id`, `start_s`, `end_s`, `caption`, optional `video_path`). All times are half-open seconds; relative media paths resolve against the input file. Existing event manifests are adapted using annotation boundaries.

Single-key multi-location recall is followed by monotonic 1:1/1:2/2:1 alignment with bounded insertions, mandatory key, and explicit missing-step diagnostics. `results.json` includes mappings, thresholds, source hashes and single-key/concatenated-text baselines. `index.html` displays algorithmically complete candidates only; this is not a claim of human semantic correctness. FFmpeg is needed only for optional MP4 exports. The report preserves inserted actions within the continuous clip.

Serve exported reports with `python -m retrieval.sequence.serve --root new-output --port 8797`. This binds loopback only and supports MP4 byte ranges. Full development contract: `../../docs/CONTINUOUS_SUBTASK_V1.md`. Current defaults are uncalibrated; motion re-ranking is a later extension.

Retrieval is installable; annotation, selection orchestration and robotization migration remain pending. Project licensing/source clearance is pending before public release.

## Quick start

Python 3.11+ (verified Windows/Python 3.13.5); FFmpeg and ffprobe must be on PATH. From repository root:

```sh
python -m pip install ./packages/pipeline
edp-retrieve --help
edp-demo --out ./demo-output
```

Use a new output directory. The demo generates three synthetic pose sequences and a gray video, builds nine windows, retrieves two candidates and reads saved features back. Constant semantic vectors are explicitly synthetic: no model download, API call or language inference. See demo-output/demo_report.json. This does not establish real-data quality.

## Real data configuration

Install `./packages/pipeline[semantic]` to enable sentence-transformers. A real uncached index build loads the pinned model and may download it. Real inference was not rerun during migration.

```yaml
artifact_root: ../artifacts/my-index
packed_motion_path: null
schema_path: null
```

```sh
edp-retrieve --config configs/my-retrieval.yaml build-index --dataset-root /path/to/sources.json
edp-retrieve --config configs/my-retrieval.yaml retrieve --query CLIP_ID --top-k 10
```

YAML paths resolve relative to the configuration file; Python dictionary paths resolve relative to the current directory. Null schema selects bundled Schema v1. Source paths remain relative to sources.json. Motion/grasp/local-frame configurations and HTML template are bundled. Reports still need the local media server; Pages export is pending.

DROID routing requires explicit droid_root, ego_root and frozen candidates/candidate_ids. Automatic V2 text recall is not packaged and fails explicitly.

## Validation

Clean non-editable installation: 69 passed, 2 private-data tests skipped. Same synthetic fixture: original/packaged result IDs match, maximum motion-feature difference zero. Existing source files are unchanged. These checks do not establish semantic quality or large-index equivalence.

Moves/config changes alter fingerprints. Build into a new index directory; do not force-reuse original experiment caches. Install `[dev]` and run `python -m pytest tests -q` from this package directory for checks. Robotization rules remain separate.
