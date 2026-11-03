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

Robotization stays as a process placeholder; source code is in packages/ego2robot, but no new visual acceptance is implied by this page.

Current status: private preparation. Original demonstration media is not committed. Public-media rights and the final project license remain pending. Computer Use was unavailable during the most recent edit, so DOM/data/media checks are separate from live browser and mobile visual review.
