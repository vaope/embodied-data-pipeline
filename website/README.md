# Static showcase

Build from the frozen, path-free showcase.json and static/ sources:

```sh
python website/build_showcase.py
python website/preview_server.py
```

The default output is website/dist. No original research workspace is needed. The page includes three annotation videos' metadata (AV-A/B/C), Robot/Ego retrieval examples, clickable route explanations and selected-result JSON/CSV.

Videos and posters are external inputs. Without --media-root, the page deliberately shows media placeholders while metadata, route diagrams and downloads remain usable. To attach your separately prepared assets:

```sh
python website/build_showcase.py --media-root /path/to/selected-media --out /path/to/site-output
```

That directory must contain the exact assets/showcase paths listed in showcase.json. This command performs no inference, encoder download or API calls. It does not fetch raw datasets. The manifest retains original query/result clocks and frozen ranks; a matched window and its display context remain distinct.

## Figures and offline visual HTML

The [English README](../README.md) and [Chinese README](../README.zh-CN.md) lead with real-source frames, timelines, retrieval candidates and a short animation. `docs/showcase.html` (English) and `docs/showcase.zh-CN.html` (Chinese) each embed the visual gallery as a single file that can be downloaded and opened without a server. Each edition works independently; switching language requires the other file alongside it. These are lightweight visual exports; the full review interface retains the video/timeline controls described above and its existing Chinese UI.

To regenerate figures from your external media root, install matplotlib and Pillow in a separate visualization environment, then run:

```sh
python website/export_readme_figures.py --media-root /path/to/selected-media
python website/export_readme_figures.py --media-root /path/to/selected-media --language zh-CN --font /path/to/cjk-font.ttf
```

Render English first, then Chinese. On Windows, `--font` can be omitted to use the installed Microsoft YaHei font; no font file is redistributed. Chinese figure captions are display translations only; frozen English annotations remain unchanged. The exporter reads the same showcase.json, retains original timestamps/ranks, and records source/output hashes in docs/visuals/SOURCES.json. These are data figures, not browser screenshots. No model is invoked.

Robotization stays as a process placeholder; source code is in packages/ego2robot, but no new visual acceptance is implied by this page.

Current status: private preparation. Full demonstration videos are not committed; selected stills and a derived GIF are included for this private review. Public-media rights and the final project license remain pending. Computer Use was unavailable during the most recent edit, so data/media/figure checks are separate from live browser and mobile visual review.
