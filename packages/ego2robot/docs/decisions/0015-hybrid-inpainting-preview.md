# ADR-0015: Episode 000 hybrid preview

- Status: experiment only; pending user review
- Date: 2026-09-27
- Implementation and configuration: `tools/m5_hybrid_preview.py`
- Evidence: `data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/hybrid-preview/`

## Context

The episode 000 temporal clean-plate trial can retrieve a background sample from
another frame for only 49.90% of eligible person-mask pixels. The other pixels
retain the person in its standalone result. The shadow-expanded ProPainter
result removes more of the person but retains low-detail areas and residual
shadow. This preview checks whether retrieved source pixels can sharpen the
tabletop while ProPainter supplies the remaining pixels.

The paper does not specify this hybrid, the table polygon, or feather width.
They are engineering trial values, not paper settings.

## Decision

For each of the same 52 PTS-aligned frames, start with the shadow-mask
ProPainter result. Use the temporal plate's `sample-valid` mask only inside
the normalized tabletop polygon
`[(0.365,0.495), (1,0.565), (1,1), (0.05,0.905)]`. Blend the plate image into
the ProPainter result with an alpha derived from distance to the valid-region
edge, saturating at 24 source pixels. Protected and unrecovered pixels keep
the ProPainter value. Every input PNG is checked against its indexed SHA-256.
The output keeps frame index and source PTS, with an index, manifest, 52-frame
video and eight representative comparison images.

## Evidence

The experiment manifest records 5,994,435 source-sampled tabletop pixels
across the clip. In the first, seventh and 29th frames, the combined result
looks close to the shadow-mask ProPainter result at the review scale. It does
not visibly resolve the broad gray shadow or prove sharper tabletop detail.
The standalone temporal plate shows obvious retained hands and hard patch
boundaries; those are avoided where the ProPainter fallback is used.

No ground-truth empty scene exists for a numerical fidelity comparison. The
stored `succeeded` status means artifact generation completed, not that M5 or
this candidate passed visual review.

## Consequences and alternatives

- Sample provenance shows that another frame supplied a pixel, but does not
  prove that registration, lighting or object visibility are correct. Errors
  can create seams or ghosting despite edge feathering.
- The manual table polygon only applies to this camera and episode. A moving
  camera or another scene needs new registration and object protection.
- The fallback retains the 640×360 ProPainter inference detail limit and any
  residual shadow. This trial does not address those remaining defects.
- The formal M5 segmentation and inpainting bundles are unchanged. The user
  must review visual evidence before any candidate is promoted.

Alternatives considered: use the temporal plate alone (large unrecovered
regions), use ProPainter alone (current shadow/blur issue), or use a second
video model (E2FGVI-HQ inference blocked in the current runtime; ADR-0013).
