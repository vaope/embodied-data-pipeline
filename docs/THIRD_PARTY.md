# Third-party code, assets and model sources

This private preparation snapshot has no selected license for the original project code yet. Do not infer that upstream code, weights, datasets and media share a single license. Public-release review remains open.

| Component | Included material / source | Terms and provenance |
|---|---|---|
| EgoANT | Pinned annotation helpers, revision `b27ea59f5fb85e55caeee9473afa7c221a9c67c4` | Original MIT license retained under `packages/pipeline/src/pipeline/annotation/vendor/egoant/LICENSE` |
| Flora | Pinned inference helpers, revision `ad744f46335aec671f5caac09ac97d817fbdb114` | Original LICENSE.pdf, MODEL_PROVENANCE, CODE_MANIFEST and THIRD_PARTY_NOTICES retained in the annotation vendor tree; prospective project licensing does not override those terms |
| Annotation vendor identity | File paths and SHA256 checks | `packages/pipeline/src/pipeline/annotation/vendor/SOURCES.json` |
| Franka Panda, ViperX, xArm7 and UR5e geometry | Selected MJCF/mesh assets, no learned weights | Each model retains LICENSE and available README/PROVENANCE/SOURCES beside its files under `packages/ego2robot/assets/mjcf/` |
| SAM3 / ProPainter / depth workers | Adapter/wrapper code and explicit configuration | External upstream repositories and weights are not bundled; users prepare them separately under their own terms |
| EgoDex / DROID / showcase clips | Source identifiers, selected descriptions, result metadata, and derived figures/GIF for the private preview | Raw datasets and full demonstration videos are excluded from Git. Public redistribution of selected visuals is still under review |

`SOURCE_SNAPSHOT.json` records the source-byte hashes of this migration snapshot, with portable origin labels. Changed path/configuration wrappers are described in MIGRATION.md; vendored annotation bytes are checked against their retained upstream manifest.

Original model inference, upstream implementation and this project's engineering adaptations must be credited separately. This file records provenance and pending review; it does not grant permissions on behalf of any upstream owner.
