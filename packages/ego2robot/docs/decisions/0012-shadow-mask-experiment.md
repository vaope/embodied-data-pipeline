# ADR-0012: Episode-Specific Shadow-Aware Mask Candidate

- Status: experiment-only; pending visual and inpainting review
- Date: 2026-09-27
- Scope: episode 000 only; does not change the formal M5 segmentation stage
- Implementation: `tools/m5_shadow_mask.py`
- Evidence directory: `data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/shadow-mask/`

## Context

The episode 000 M5 delivery report records that the current SAM 3 masks remove
the visible arms, while arm shadows and local blur remain in the ProPainter
preview. The existing mask sequence has 52 frames and matches the source PTS
timeline from 0.000 s through 1.700 s. This experiment asks whether a bounded
shadow extension to the inpainting mask improves that result while preserving
the handled carton and the clear destination box.

The paper and its appendix do not specify shadow detection, tabletop polygons,
photometric thresholds, or object-protection regions. Every numeric threshold
and polygon below is an episode-specific engineering candidate. None is
presented as a paper parameter or as a generally validated setting.

## Decision

Add a standalone shadow-mask candidate tool. It reads the indexed source RGB
frames and the existing raw/refined SAM 3 mask index, then writes a separate
`MaskSequenceIndex` at
`previews/m5-alternatives/shadow-mask/index.json`. Each candidate refined mask
is the union of the existing refined SAM 3 mask and new shadow pixels. The raw
mask path and hash, frame index, timestamp, validity flag, source-video hash,
and original dimensions are preserved. Formal segmentation artifacts are not
modified.

The candidate detector uses a temporal per-pixel RGB reference from source
observations outside an 8 px dilation of the SAM 3 mask. It selects the 85th
percentile from at least six observations and calls a reference stable when its
relative temporal luminance median absolute deviation is at most 0.18. Each
frame receives median-ratio global exposure compensation estimated on stable,
unprotected tabletop pixels.

An added pixel must be on the configured tabletop, within 240 px of a SAM 3
person mask in that frame, have table-like chromaticity (RGB chromaticity L1
difference at most 0.16), and be at least 12% darker than either the temporal
reference or a local tabletop upper envelope with 60 px radius. The local
reference is included to catch stationary arm shadows that do not disappear
from the temporal sequence. A 14 px dilated protection margin prevents new
shadow pixels from entering the manually outlined clear destination box or
the handled carton's initial footprint.

The 1920×1080 normalized image-space table polygon is
`[(0.365,0.495), (1,0.565), (1,1), (0.05,0.905)]`. Protection polygons are
stored in the script configuration and repeated in the index provenance:

- clear destination box: `[(0.472,0.418), (0.718,0.418), (0.778,0.505), (0.757,0.665), (0.502,0.665)]`
- handled carton initial footprint: `[(0.565,0.715), (0.665,0.715), (0.665,0.865), (0.565,0.865)]`

## Evidence and reproducibility

For fixed source/index bytes and configuration, the tool computes a SHA-256
fingerprint, writes every candidate mask and preview atomically, records output
hashes in `manifest.json`, and reuses the output only when the fingerprint
and every recorded output hash still match. It emits an index, diagnostic
JSON, a per-frame four-panel review video, and eight representative images.
The evidence is an experiment result, not an acceptance record.

The generated episode 000 candidate records 52 frames on the same PTS timeline
as the source index. Its mask index schema is
`ego2robot.m5_masks.v1`. Its generated fingerprint and per-output checksums are
in the experiment manifest; the report records per-frame additions and
protection suppression counts.

The parent task then ran the existing ProPainter worker against this candidate
mask and compared its output with the original M5 result. The comparison covers
52 frames; representative evidence is in
`data/episodes/egodex_vertical_pick_place_000/previews/m5-alternatives/shadow-propainter/previews/comparison.mp4`
and `compare-frame-000000.png`, `compare-frame-000029.png`, and
`compare-frame-000051.png`. In those reviewed stills, the right-arm projected
shadow is visibly reduced relative to the original ProPainter output, while a
gray tabletop residue remains and the left-hand area remains blurred. This is
evidence that the candidate input mask improves the observed right-arm shadow
removal for this run; it does not establish generally suitable thresholds,
complete shadow removal, or acceptable temporal behavior.

## Limitations and review conditions

- A dark tabletop pixel is not proof of a cast shadow. Wood texture, reflections,
  changing illumination, or camera auto-exposure can trigger false additions.
- The temporal reference cannot recover a clean value where an arm shadow is
  present throughout the clip. The local upper envelope may fill this gap, but
  it can also mark ordinary local contrast around the arm.
- Table and object polygons are manual image-space regions for this fixed
  camera. They must not be reused for another episode, resolution, camera, or
  crop without review.
- The protection polygons suppress only newly proposed shadow pixels. The
  existing SAM 3 mask remains unchanged, including any pre-existing leakage
  over the carton or box. A held carton may also leave its protected initial
  footprint before entering the protected destination region.
- The candidate mask and candidate ProPainter output have not been accepted by
  the user. The observed right-arm improvement is partial: gray tabletop
  residue remains, and blur remains around the left hand. Mask coverage, object
  preservation, contact edges, and temporal appearance require user review.
  The program's success status does not pass M5.

## Consequences

The experiment can be passed to the existing inpainting worker through its
versioned mask-index contract without replacing the formal segmentation
artifacts. If the user finds the result useful, the thresholds, protection
regions, and any migration into a supported stage require a separate design
review and new evidence.
