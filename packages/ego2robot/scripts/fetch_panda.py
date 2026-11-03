"""Download the MuJoCo Menagerie Franka Emika Panda model (XML + assets) via GitHub raw URLs.

Uses the GitHub contents API to discover the real file names, then downloads
every file (recursively into assets/) with a small thread pool.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_DIR = "franka_emika_panda"
API = f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/{REPO_DIR}"
RAW = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/{REPO_DIR}"
DEST = Path(__file__).resolve().parents[1] / 'assets/mjcf/franka_emika_panda'
HEADERS = {"User-Agent": "dsh-agent", "Accept": "application/vnd.github+json"}


def api_listing(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(rel: str) -> tuple[str, str, int]:
    out = DEST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW}/{rel}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dsh-agent"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        out.write_bytes(data)
        return ("OK", rel, len(data))
    except Exception as exc:  # noqa: BLE001 - report every failure, never skip
        return ("FAIL", f"{rel} :: {exc}", 0)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)

    top = api_listing(API)
    print(f"TOP_LEVEL_ENTRIES\t{len(top)}")
    targets: list[str] = []
    for entry in top:
        if entry["type"] == "file":
            targets.append(entry["name"])
        elif entry["type"] == "dir":
            sub = api_listing(entry["url"])
            print(f"DIR\t{entry['name']}\t{len(sub)} files")
            for s in sub:
                if s["type"] == "file":
                    targets.append(f"{entry['name']}/{s['name']}")

    print(f"TOTAL_TARGETS\t{len(targets)}")
    ok = fail = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download, rel): rel for rel in targets}
        for fut in as_completed(futures):
            status, info, size = fut.result()
            if status == "OK":
                ok += 1
                print(f"OK\t{info}\t{size}")
            else:
                fail += 1
                failures.append(info)
                print(f"FAIL\t{info}")

    print(f"SUMMARY\tok={ok}\tfail={fail}")
    for f in failures:
        print(f"FAILED_ENTRY\t{f}")
    print("DOWNLOAD_DONE")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
