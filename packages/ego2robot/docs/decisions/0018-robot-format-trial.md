# ADR-0018: Project-local robot format export trial

- Status: **experiment implemented; artifact review pending**
- Date: 2026-09-28
- Scope: one existing, M3-reviewed EgoDex episode; no change to the canonical core schema or runtime stage graph

## Context

The repository's M7 plan calls for an exporter skeleton after the canonical representation is stable. This trial compares two small, inspectable ways to carry the existing canonical TCP and Franka trajectory together:

1. a typed HDF5 file with one group per semantic section;
2. JSON Lines with one record per source video frame and an optional outgoing transition.

The trial uses `egodex_vertical_pick_place_000`, which already has a reviewed projection preview. Its canonical and Franka records each contain 52 frames, while `actions-delta.npz` contains 51 transitions. The source PTS run from 0.0 to 1.7 seconds.

## Decision

Implement these explicitly project-local schemas:

- `ego2robot.export.canonical_hdf5.v1`: `timebase`, absolute `canonical` values, `transitions`, and `robot/<robot_id>` state and FK/IK fields.
- `ego2robot.export.frame_jsonl.v1`: one record per frame. Frames 0–50 carry their outgoing `t -> t+1` delta. The final frame has `outgoing_transition: null`.

Both files retain the source frame indices and PTS. Absolute `T_C_TCP` and physical `gripper_width_m` remain frame-aligned state. Relative position is `p[t+1] - p[t]`; the camera-frame rotation delta is `Log_SO3(R[t+1] @ transpose(R[t]))`, matching `system-design.md` section 6.3.

The raw `gripper_command` channel is preserved as `gripper_channel_raw`, with its source field, resolved mode, unit and provisional meaning. In this sample the mode is `width_m`; that value is a metric width passthrough aligned to the source frame. It is not described as a normalized or standardized control command. The paper mapping remains open under [ADR-0010](0010-gripper-command-scaling.md).

`confidence_source` is carried into the manifest, HDF5 metadata and JSONL records. For this episode the source HDF5 has no `/confidences` group, so the stored `1.0` values are annotation-presence placeholders, not measured detector confidence.

## Format compatibility

These files are not claimed to be native LeRobot or robomimic datasets.

- [LeRobotDataset v3 documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) describes low-dimensional signals in Parquet, videos in MP4 shards, and JSON/Parquet metadata. This episode-only HDF5/JSONL layout does not implement that storage contract.
- [robomimic dataset documentation](https://robomimic.github.io/docs/datasets/overview.html) describes a single HDF5 organized under `data/demo_*`, including environment metadata and state/action/reward/done/observation fields. This trial does not fabricate environment state, rewards, termination labels, or observations to imitate that contract.

The local environment has `h5py`, but does not have `lerobot`, `robomimic` or `pyarrow`. No official-format loader or compatibility check was run.

## Consequences and limitations

- HDF5 is convenient for grouped arrays and metadata. JSONL is easy to inspect and stream line by line. Both are readable with installed general-purpose libraries.
- These exports include robot geometry/state fields but do not embed source video or rendered images. The source video and camera are identified by manifest hashes; an image overlay is saved separately for visual review.
- qpos order comes from the robot profile, and units are checked against the MJCF joint types. Panda arm hinge coordinates are radians; the two finger slide coordinates are metres.
- Deterministic input/config fingerprints and output checksums support resume. A changed input, exporter source, CLI source or ADR invalidates the cache.
- The source license is `CC-BY-NC-ND` and the episode marks itself `publishable: false`; generated files remain local trial artifacts and must not be treated as approved for redistribution.
- This experiment does not complete M7, validate policy-training behavior, or constitute user acceptance of the visual artifacts.

## Evidence

The exporter writes its machine-readable run record and review images under:

`data/episodes/egodex_vertical_pick_place_000/previews/format-trial/`

The manifest records source/config/code hashes, frame and transition counts, PTS bounds, output hashes, the gripper mode and confidence provenance.

