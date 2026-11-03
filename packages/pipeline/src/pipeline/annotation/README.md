# Annotation wrappers

The AV-A/B/C annotation implementation is included in the pipeline package. Use the current root README and docs/USAGE.md for installation and commands; original local machine setup and historical budget/provider details are not generic installation requirements.

- AV-A: pinned EgoANT segmentation/refinement and a configured caption API/model.
- AV-B: A's same segmentation cache and the pinned Flora captioner.
- AV-C: Flora native segmentation and captioner.

Default config.json is a recorded hosted AV-A profile. Local GPU profiles under configs/ are separate. Model credentials are read from environment variables. prepare and doctor are offline metadata operations; actual model execution remains explicit, budgeted and identity-bound.

Upstream source files and original terms are retained under vendor/. SOURCES.json checks their bytes. Model weights are not bundled; the model downloader additionally requires its optional Hugging Face dependency and an explicitly pinned local profile. Source migration and future package changes invalidate old plan identities; re-prepare into a new directory instead of bypassing checks.
