# ADR-0014: Episode 000 temporal clean-plate experiment

- Status: experiment generated; visual review pending
- Date: 2026-09-27
- Scope: one isolated comparison artifact for egodex_vertical_pick_place_000
- Configuration: data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/temporal-plate/configuration.json
- Implementation: tools/m5_temporal_plate.py

## Context

Episode 000 already has 52 source PNGs with a canonical PTS index and 52 refined SAM 3 person masks. A mask-occupancy diagnostic reports 467,045 pixels that were masked in at least one frame and 45,974 pixels masked in every frame; the remaining temporal visibility suggests that some real background can be sampled from other frames. This is only an upper-bound visibility signal: shadows, moving objects, segmentation misses, and registration errors can still contaminate a sample.

The paper does not specify a clean-plate algorithm for this episode. This experiment is a bounded comparison candidate and does not change the formal M5 stage.

## Decision

1. Register each source frame to frame 0 with OpenCV phase correlation. The existing refined person masks and protected tray/package ROIs are excluded when estimating translation. The frame's estimated displacement is recorded; no response threshold is treated as an acceptance criterion.
2. Build a per-pixel channel median from aligned source observations that lie outside the 48-pixel dilated person mask and outside the protected ROIs. Use a deterministic nearest-to-median contributing frame as a representative source attribution; ties use the lower frame index. The median can combine values from several frames, so the attribution map names a representative contributor rather than claiming that one frame alone produced every pixel value.
3. Define the repair target as the refined person mask dilated by 48 pixels. This fixed image-space halo is an engineering approximation intended to include nearby arm shadows and mask-edge contamination. It is not a shadow detector, and shadows outside the halo may remain or contaminate the plate.
4. Protect the manipulated package with manually placed, padded image-space bounding boxes interpolated between the saved keyframes. Protect the whole transparent tray with a manually drawn conservative polygon. These are conservative review ROIs, not object segmentation. Hand pixels intersecting these ROIs remain in the output and are counted as retained residuals.
5. Fill only target pixels with at least one eligible sample from a different source frame. Where no eligible sample exists, or the target overlaps a protected ROI, retain the original source pixel. Save separate sample-valid, unrecovered, repair-target, protected, and source-attribution PNG masks for every frame.
6. Preserve the original PTS source index byte-for-byte and repeat each frame_index and timestamp_s in the result and coverage indices. The fixed-rate MP4 is a review artifact and does not replace canonical timestamps.
7. Define no quality or acceptance threshold. The report measures available coverage and residuals for review; it does not determine whether the candidate is acceptable.

## Results recorded from the generated candidate

- 52 source/result frames at 1920×1080; source PTS runs from 0.0 s through 1.7 s.
- 8 representative source/mask/result images and one 52-frame review video were generated.
- The refined person masks cover 10,711,501 pixels summed across frames. Of those, 1,741,397 overlap a protected package/tray ROI and are deliberately retained.
- Of 8,970,104 person-mask pixels outside protected ROIs, 4,476,329 were filled from other frames (49.90%).
- Including the 48-pixel repair halo, 7,459,217 of 14,288,985 eligible target pixels were filled (52.20%). The remainder without an eligible sample is retained and marked.
- The report also records 3,940,743 repair-target pixels inside the protected ROIs. Protected pixels are excluded from the eligible coverage denominator.

These values describe this run only. They are not acceptance thresholds and do not imply that the sampled color is semantically correct.

## Limitations and review focus

- The 48-pixel halo is not guaranteed to include all arm shadows. Inspect retained shadows and check sampled table pixels for shadow contamination.
- Transparent tray walls refract the background, and a clean plate cannot reproduce their optics. The entire tray is protected to avoid painting over its contents; this intentionally leaves hand/tray overlap visible.
- The package ROIs are hand-authored and broad. Inspect the package's table-to-tray movement boundaries for over-protection or missed pixels.
- A median may blur temporal changes or hide a moving object if the masks/ROIs miss it. Inspect all representative frames and the review video.
- The result is only a comparison candidate. It has not been accepted by the user or integrated into formal M5 outputs.
