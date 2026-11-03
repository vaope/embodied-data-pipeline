"""Human review: run several Robot queries, then serve a click-to-label page.

    python -m pipeline.v2.review run   --queries 0,1,2 --top-k 10 --out experiments/v2_eval/review_20261023
    python -m pipeline.v2.review serve --report experiments/v2_eval/review_20261023 --port 8780

The page streams EgoDex playback videos with HTTP Range, shows the query text and the
candidate window, and POSTs labels to labels.jsonl. It is a review tool, not a benchmark.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import pipeline as core

_OLD_HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>检索复查</title><style>
body{margin:0;background:#0f141b;color:#dce6f2;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}
header{position:sticky;top:0;background:#17202b;padding:10px 16px;z-index:2;border-bottom:1px solid #2a3646}
main{display:grid;grid-template-columns:1.15fr 1fr;gap:16px;padding:16px}
#player{position:sticky;top:60px;align-self:start}
video{width:100%;background:#000;border-radius:8px}
video.robot{width:100%;max-height:220px;background:#000;border-radius:8px;margin:6px 0}
.card{background:#17202b;border:1px solid #2a3646;border-radius:10px;padding:10px 12px;margin:8px 0;cursor:pointer}
.card.active{border-color:#4f8cff}
.card .row{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.rank{color:#8fb3ff;font-weight:700}.score{color:#9fb0c6}
.task{color:#cfe0f5} .meta{color:#7f93aa;font-size:12px}
button{background:#22314a;color:#dce6f2;border:1px solid #35507a;border-radius:6px;padding:4px 10px;cursor:pointer;margin-right:6px}
button.ok{background:#1d4d33;border-color:#2f7a52}button.no{background:#4d1f28;border-color:#7a3140}
select{background:#22314a;color:#dce6f2;border:1px solid #35507a;border-radius:6px;padding:3px}
.done{opacity:.55}
#stat{padding:0 16px;color:#9fb0c6}
a{color:#8fb3ff}
</style>
<header><b>检索复查</b> <span id="stat">loading…</span> · <a href="/labels.csv">导出 labels.csv</a></header>
<main><div id="player"><video id="v" controls preload="metadata"></video>
<p class="meta" id="now">点击右侧候选加载对应片段</p></div><div id="list"></div></main>
<script>
let data=null, labels={}, current=null;
const CATS=["","same_verb_wrong_object","same_object_wrong_action","motion_similar_no_contact","different_task","unclear"];
async function boot(){
  data=await (await fetch('/candidates.json')).json();
  labels=await (await fetch('/labels.json')).json();
  const list=document.getElementById('list');
  let total=0, done=0;
  for(const q of data.queries){
    const h=document.createElement('h3'); h.textContent=q.query_id+" · "+q.instruction; list.appendChild(h);
    const meta=document.createElement('div'); meta.className='meta';
    meta.textContent=`${q.eef.steps} steps / ${q.eef.duration_s}s / path ${q.eef.path_length_m.toFixed(3)}m`;
    list.appendChild(meta);
    if(q.robot_video){
      const rv=document.createElement('video'); rv.className='robot'; rv.controls=true; rv.loop=true; rv.muted=true;
      rv.preload='metadata'; rv.src='/robot/'+encodeURIComponent(q.query_id);
      rv.title='DROID robot demonstration (three cameras)';
      list.appendChild(rv);
    } else {
      const note=document.createElement('div'); note.className='meta';
      note.textContent='（该查询没有导出机器人视频）'; list.appendChild(note);
    }
    for(const c of q.candidates){
      total++;
      const id=q.query_id+"|"+c.clip_id;
      const lab=labels[id];
      if(lab!==undefined) done++;
      const card=document.createElement('div'); card.className='card'+(lab!==undefined?' done':''); card.dataset.id=id;
      card.innerHTML=`<div class="row"><span class="rank">#${c.rank}</span><span class="score">sim ${c.similarity.toFixed(4)}</span>
        <span class="task">${(c.task_text||'').slice(0,110)}</span></div>
        <div class="meta">${c.episode_id} · ${c.start_time.toFixed(1)}–${c.end_time.toFixed(1)}s · ${c.clip_id}</div>
        <div style="margin-top:6px"><button class="ok">相关</button><button class="no">不相关</button>
        <select>${CATS.map(x=>`<option value="${x}">${x||'错误类别(可选)'}</option>`).join('')}</select></div>`;
      card.onclick=(ev)=>{ if(ev.target.tagName==='BUTTON'||ev.target.tagName==='SELECT') return; play(c,card); };
      card.querySelector('.ok').onclick=()=>mark(id,1,card);
      card.querySelector('.no').onclick=()=>mark(id,0,card);
      card.querySelector('select').onchange=(e)=>{ card.dataset.cat=e.target.value; };
      list.appendChild(card);
    }
  }
  updateStat();
  const first=document.querySelector('.card');
  if(data.queries.length && data.queries[0].candidates.length && first){
    load(data.queries[0].candidates[0], first, false);
  }
}
function load(c,card,autoplay){
  const v=document.getElementById('v');
  v.onerror=()=>{ document.getElementById('now').textContent='视频加载失败 error='+(v.error&&v.error.code)+' clip='+c.clip_id; };
  v.src='/media/'+encodeURIComponent(c.clip_id);
  v.onloadedmetadata=()=>{ v.currentTime=Math.max(0,c.start_time); if(autoplay) v.play().catch(()=>{}); };
  document.getElementById('now').textContent=`${c.clip_id} · ${c.start_time.toFixed(2)}–${c.end_time.toFixed(2)}s`;
  document.querySelectorAll('.card').forEach(x=>x.classList.remove('active'));
  if(card) card.classList.add('active');
}
function play(c,card){ load(c,card,true); }
async function mark(id,label,card){
  const cat=card.querySelector('select').value;
  await fetch('/label',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id,label,category:cat})});
  labels[id]=label; card.classList.add('done'); updateStat();
}
function updateStat(){
  const total=Object.keys(labels).length? document.querySelectorAll('.card').length : document.querySelectorAll('.card').length;
  const done=document.querySelectorAll('.card.done').length;
  document.getElementById('stat').textContent=`已标 ${done} / ${total}`;
}
boot();
</script></html>"""

from .review_page import HTML  # noqa: E402  (page markup kept in its own module)


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _diversify(order, rows: dict, top_k: int, max_per_episode: int, max_per_family: int,
               used_videos: set[str], overlap: float = 0.5) -> list[int]:
    """Ties are expected: all windows of an episode share one task-only embedding.

    Also dedupes by video file (across queries when ``used_videos`` is shared) and caps
    the number of near-identical scenes from one task family.
    """
    chosen: list[int] = []
    per_episode: dict[str, list[tuple[float, float]]] = {}
    per_family: dict[str, int] = {}
    for idx in order.tolist():
        row = rows.get(int(idx))
        if row is None:
            continue
        video = row.get("playback_video_path") or row.get("video_path") or row["clip_id"]
        if video in used_videos:
            continue
        episode = row["episode_id"]
        intervals = per_episode.setdefault(episode, [])
        if len(intervals) >= max_per_episode:
            continue
        interval = (float(row["start_time"]), float(row["end_time"]))
        if any(_iou(interval, existing) > overlap for existing in intervals):
            continue
        family = episode.rsplit("_", 1)[0]
        if per_family.get(family, 0) >= max_per_family:
            continue
        intervals.append(interval)
        per_family[family] = per_family.get(family, 0) + 1
        used_videos.add(video)
        chosen.append(int(idx))
        if len(chosen) >= top_k:
            break
    return chosen


def _collect(droid_root: Path, ego_root: Path, queries: list[int], top_k: int, max_per_episode: int = 2,
             max_per_family: int = 3, dedupe_across_queries: bool = True) -> dict:
    config = core.load_config()
    index = core.ego_index(ego_root)
    qspecs, vectors, skipped = [], [], []
    for q in queries:
        try:
            spec = core.droid_query(droid_root, q)
        except SystemExit as exc:
            skipped.append({"query": q, "reason": str(exc)})
            continue
        qspecs.append(spec)
        vectors.append(core.embed_texts([spec["instruction"]], config)[0])
        robot = droid_root / f"episode_{q:03d}" / "three_cameras.mp4"
        spec["robot_video"] = str(robot) if robot.is_file() else None
    import numpy as np

    embeddings = np.load(index["embeddings"], mmap_mode="r")
    sims = np.asarray(embeddings @ np.asarray(vectors).T, dtype=np.float32)  # windows x queries
    wanted: set[int] = set()
    orders = []
    for j in range(len(qspecs)):
        order = core.top_k(sims[:, j], min(5000, sims.shape[0]))
        orders.append(order)
        wanted.update(int(i) for i in order)
    rows = core.collect_rows(index["manifest"], wanted)
    out = {"queries": [], "skipped": skipped, "index": {k: v for k, v in index.items() if k not in {"manifest", "embeddings"}}}
    used_videos: set[str] = set()
    for j, spec in enumerate(qspecs):
        candidates = []
        chosen = _diversify(orders[j], rows, top_k, max_per_episode, max_per_family,
                            used_videos if dedupe_across_queries else set())
        for rank, idx in enumerate(chosen, 1):
            row = rows[idx]
            candidates.append(
                {
                    "rank": rank,
                    "clip_id": row["clip_id"],
                    "episode_id": row["episode_id"],
                    "task_text": row.get("task_text"),
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "similarity": float(sims[idx, j]),
                    "video_path": row.get("video_path"),
                    "playback_video_path": row.get("playback_video_path"),
                }
            )
        out["queries"].append({"query_id": spec["source_episode_id"], "instruction": spec["instruction"], "eef": spec["eef"],
                               "robot_video": spec.get("robot_video"), "candidates": candidates})
    return out


def _media_map(payload: dict) -> dict:
    mapping = {}
    for query in payload["queries"]:
        for row in query["candidates"]:
            path = row.get("playback_video_path") or row.get("video_path")
            if path:
                mapping[row["clip_id"]] = path
    return mapping


def run(droid_root: Path, ego_root: Path, queries: list[int], top_k: int, out: Path, max_per_episode: int = 1,
        max_per_family: int = 3, dedupe_across_queries: bool = True) -> dict:
    payload = _collect(droid_root, ego_root, queries, top_k, max_per_episode, max_per_family, dedupe_across_queries)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "media_map.json").write_text(json.dumps(_media_map(payload), ensure_ascii=False), encoding="utf-8")
    robot_map = {q["query_id"]: q["robot_video"] for q in payload["queries"] if q.get("robot_video")}
    (out / "robot_media.json").write_text(json.dumps(robot_map, ensure_ascii=False), encoding="utf-8")
    (out / "index.html").write_text(HTML, encoding="utf-8")
    labels = out / "labels.jsonl"
    labels.touch(exist_ok=True)
    summary = {"queries": len(payload["queries"]), "skipped": len(payload.get("skipped", [])),
               "candidates": sum(len(q["candidates"]) for q in payload["queries"]),
               "distinct_episodes_per_query": [len({c["episode_id"] for c in q["candidates"]}) for q in payload["queries"]],
               "top_k": top_k, "max_per_episode": max_per_episode}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _labels(path: Path) -> dict:
    latest = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["id"]] = row["label"]
    return latest


def make_server(report: Path, port: int):
    media = json.loads((report / "media_map.json").read_text(encoding="utf-8"))
    labels = report / "labels.jsonl"
    snapshot = report / "labels.snapshot.json"
    if labels.is_file() and labels.stat().st_size > 0:
        backup = report / f"labels-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        if not backup.exists():
            backup.write_bytes(labels.read_bytes())

    def _latest() -> dict:
        latest: dict[str, int] = {}
        if labels.is_file():
            for line in labels.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    latest[row["id"]] = row["label"]
        return latest

    def _save_snapshot() -> None:
        snapshot.write_text(json.dumps(_latest(), ensure_ascii=False, indent=2), encoding="utf-8")

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            if not getattr(self, "_is_media", False):
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _json(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == "/candidates.json":
                return self._json(json.loads((report / "candidates.json").read_text(encoding="utf-8")))
            if route == "/labels.json":
                return self._json(_latest())
            if route == "/labels.csv":
                rows = ["id,label"]
                for key, value in _latest().items():
                    rows.append(f"{key},{value}")
                body = ("\n".join(rows) + "\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if route.startswith("/media/"):
                return self._media(route[len("/media/"):])
            if route.startswith("/robot/"):
                return self._robot(route[len("/robot/"):])
            return super().do_GET()

        def do_POST(self):
            if unquote(urlsplit(self.path).path) != "/label":
                return self.send_error(404)
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            payload.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
            with labels.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            _save_snapshot()
            return self._json({"ok": True, "saved": len(_latest())})

        def _media(self, clip_id):
            # Reload the map on every request: candidates.json may be regenerated
            # while the server is running, and a stale map returns 404 for new clips.
            try:
                media_now = json.loads((report / "media_map.json").read_text(encoding="utf-8"))
            except Exception:
                media_now = media
            return self._stream(Path(media_now.get(clip_id, "")))

        def _robot(self, query_id):
            try:
                robot_now = json.loads((report / "robot_media.json").read_text(encoding="utf-8"))
            except Exception:
                robot_now = {}
            return self._stream(Path(robot_now.get(query_id, "")))

        def _stream(self, path):
            if not path.is_file():
                self.log_message("media 404 path=%s", path)
                return self.send_error(404)
            size = path.stat().st_size
            value = self.headers.get("Range")
            start, end, status = 0, size - 1, 200
            if value:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
                if not match or not any(match.groups()):
                    return self.send_error(416)
                if not match[1]:
                    start = max(0, size - int(match[2]))
                else:
                    start = int(match[1])
                    if match[2]:
                        end = min(end, int(match[2]))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return None
                status = 206
            handle = path.open("rb")
            handle.seek(start)
            self.remaining = end - start + 1
            self._is_media = True
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(self.remaining))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                while True:
                    block = handle.read(min(1024 * 1024, self.remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    self.remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                handle.close()

    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(report)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.v2.review")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--queries", default="0")
    r.add_argument("--top-k", type=int, default=10)
    r.add_argument("--max-per-episode", type=int, default=1)
    r.add_argument("--max-per-family", type=int, default=3)
    r.add_argument("--allow-cross-query-reuse", action="store_false", dest="dedupe_across_queries")
    r.add_argument("--droid-root", type=Path, default=core.WORKSPACE / "data" / "droid")
    r.add_argument("--ego-root", type=Path, default=core.WORKSPACE / "artifacts" / "ego")
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("serve")
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--port", type=int, default=8780)
    args = parser.parse_args(argv)
    if args.command == "run":
        queries = [int(x) for x in args.queries.split(",") if x.strip()]
        summary = run(args.droid_root, args.ego_root, queries, args.top_k, args.out,
                      args.max_per_episode, args.max_per_family, args.dedupe_across_queries)
        print(json.dumps({"ok": True, **summary, "out": str(args.out)}))
        return 0
    server = make_server(args.report, args.port)
    print(f"serving http://127.0.0.1:{args.port}/  (report={args.report})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
