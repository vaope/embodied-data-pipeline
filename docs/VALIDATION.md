# First private snapshot validation

Date: 2026-11-03 (America/New_York).

- Clean environment: Windows, CPython 3.12.12. Main distribution installed non-editably; optional Ego2Robot installed editable from its source checkout so geometry/config/worker resources are available.
- Existing pipeline checks: **133 passed, 2 skipped**. Skipped checks require private historical/DROID inputs. The annotation API tests use local HTTP fixtures and fake model backends, not real service endpoints.
- Offline annotation doctor verifies the retained vendor manifest. CLI help for annotation, V2 orchestration, bundles and Ego2Robot succeeds; pip reports no broken requirements.
- Synthetic retrieval demo: three generated pose episodes, nine windows, two candidates, twelve motion fields read back; constant semantic vectors, no language inference.
- Site builds from repository-contained inputs into a fresh no-media directory; a separate external media root can attach precomputed selected clips.
- Source provenance, staged-file exclusions, finite secret/path patterns and maximum file size are checked before push.

Initial clean-env runs found a relocated annotation fixture import and undeclared requests dependency. They were corrected. Asset preparation also imported the optional download dependency before rejecting a remote-only profile; validation now runs before that import. The subsequent full existing suite passes.

No new robotization test cases, model inference, IK runs, image approvals, real-data quality claims or public deployment are part of this source migration. Fresh GPU execution and live browser/mobile visual verification remain separate. This environment's exact dependency freeze is retained in requirements-validation.txt; it is a validation record, not a universal GPU-runtime lock.
