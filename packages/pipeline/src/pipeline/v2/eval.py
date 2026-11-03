"""Annotation template + retrieval metrics for the cross-embodiment eval set.

No public benchmark exists for "Robot Demo -> Human Ego candidate" relevance, so this
module makes the smallest honest evaluation: a CSV of candidates to label, and a scorer
for Recall@K / precision@K / mAP over the labelled queries.

Usage:
    python -m pipeline.v2.eval make-template --candidates <run>/retrieved_candidates.jsonl --out eval/
    python -m pipeline.v2.eval score         --annotations eval/annotation.csv --out eval/metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = {"1", "0", ""}
ERROR_CATEGORIES = (
    "same_verb_wrong_object",
    "same_object_wrong_action",
    "motion_similar_no_contact",
    "different_task",
    "unclear",
)


def make_template(candidates: list[dict], query_id: str, query_text: str, out: Path, top_n: int = 10) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "annotation.csv"
    exists = path.is_file()
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["query_id", "query_text", "rank", "clip_id", "task_text", "start_s", "end_s",
                        "label", "error_category", "note"],
        )
        if not exists:
            writer.writeheader()
        for rank, row in enumerate(candidates[:top_n], 1):
            writer.writerow(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "rank": rank,
                    "clip_id": row["clip_id"],
                    "task_text": (row.get("task_text") or "")[:160],
                    "start_s": row["start_time"],
                    "end_s": row["end_time"],
                    "label": "",
                    "error_category": "",
                    "note": "",
                }
            )
    return path


def _labels(rows: list[dict]) -> dict[str, list[int]]:
    by_query: dict[str, list[int]] = {}
    for row in rows:
        text = (row.get("label") or "").strip()
        if text not in {"0", "1"}:
            continue
        by_query.setdefault(row["query_id"], []).append((int(row["rank"]), int(text)))
    return {q: sorted(pairs) for q, pairs in by_query.items()}


def metrics(rows: list[dict], ks=(1, 5, 10)) -> dict:
    by_query = _labels(rows)
    if not by_query:
        return {"queries": 0, "note": "no labelled rows (label must be 1 or 0)"}
    recalls = {f"Recall@{k}": 0.0 for k in ks}
    precisions = {f"Precision@{k}": 0.0 for k in ks}
    ap_sum = 0.0
    for pairs in by_query.values():
        ranked = [label for _, label in pairs]
        positives = sum(ranked)
        for k in ks:
            top = ranked[:k]
            hit = sum(top)
            recalls[f"Recall@{k}"] += 1.0 if hit > 0 else 0.0
            precisions[f"Precision@{k}"] += hit / k
        if positives:
            hits, seen = 0, 0
            for position, label in enumerate(ranked, 1):
                if label:
                    hits += 1
                    seen += hits / position
            ap_sum += seen / positives
    n = len(by_query)
    result = {"queries": n, "labelled_rows": sum(len(v) for v in by_query.values())}
    result.update({k: round(v / n, 4) for k, v in recalls.items()})
    result.update({k: round(v / n, 4) for k, v in precisions.items()})
    result["mAP"] = round(ap_sum / n, 4)
    return result


def score(path: Path, out: Path, ks=(1, 5, 10)) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = metrics(rows, ks=ks)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def from_review(review: Path, out: Path) -> Path:
    """Convert review labels.jsonl + candidates.json into an annotation.csv."""
    payload = json.loads((review / "candidates.json").read_text(encoding="utf-8"))
    latest: dict[str, int] = {}
    labels = review / "labels.jsonl"
    if labels.is_file():
        for line in labels.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["id"]] = int(row["label"])
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for query in payload["queries"]:
        for candidate in query["candidates"]:
            key = f"{query['query_id']}|{candidate['clip_id']}"
            if key not in latest:
                continue
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query_text": query["instruction"],
                    "rank": candidate["rank"],
                    "clip_id": candidate["clip_id"],
                    "task_text": candidate["task_text"],
                    "start_s": candidate["start_time"],
                    "end_s": candidate["end_time"],
                    "label": latest[key],
                    "error_category": "",
                    "note": "",
                }
            )
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "query_text", "rank", "clip_id", "task_text",
                                                    "start_s", "end_s", "label", "error_category", "note"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.v2.eval")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("make-template")
    t.add_argument("--candidates", type=Path, required=True)
    t.add_argument("--query-id", default="droid_000")
    t.add_argument("--query-text", default="")
    t.add_argument("--top-n", type=int, default=10)
    t.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--annotations", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("from-review")
    f.add_argument("--review", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "from-review":
        path = from_review(args.review, args.out)
        print(json.dumps({"ok": True, "path": str(path)}))
        return 0
    if args.command == "make-template":
        rows = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
        text = args.query_text
        if not text:
            report = args.candidates.parent / "run_report.json"
            if report.is_file():
                text = json.loads(report.read_text(encoding="utf-8")).get("query", {}).get("instruction", "")
        text = text or (rows[0].get("task_text") if rows else "")
        path = make_template(rows, args.query_id, text, args.out, args.top_n)
        print(json.dumps({"ok": True, "path": str(path), "rows": min(len(rows), args.top_n)}))
        return 0
    result = score(args.annotations, args.out)
    print(json.dumps({"ok": True, "metrics": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
