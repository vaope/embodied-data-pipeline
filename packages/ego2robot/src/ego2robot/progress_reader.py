"""Bounded, read-only views of the repository's progress evidence.

This module never runs the pipeline or treats a successful stage as user acceptance.
Only named documentation and selected summary fields from local artifacts are exposed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOCUMENTS = {
    "readme": "README.md",
    "system_design": "docs/system-design.md",
    "implementation_plan": "docs/implementation-plan.md",
    "m5_m8_workplan": "docs/m5-m8-workplan.md",
    "modification_todo": "docs/modification-todo.md",
    "morphology_review": "docs/morphology-trial-review.md",
    "long_episode_review": "docs/long-episode-review.md",
    "m5_alternatives_review": "docs/m5-alternatives-review.md",
    "mcp_setup": "docs/mcp-progress.md",
}

STAGE_LOCATIONS = {
    "ingest": "source/stage-manifest.json",
    "hand": "hand/stage-manifest.json",
    "retarget": "canonical/stage-manifest.json",
    "render": "visual/stage-manifest.json",
    "projection": "previews/stage-manifest.json",
    "quality": "quality/stage-manifest.json",
    "segmentation": "segmentation/stage-manifest.json",
    "inpainting": "inpainting/stage-manifest.json",
}

EVIDENCE_FILES = (
    "previews/projection-frame-000029.png",
    "previews/projection-overlay.mp4",
    "previews/projection-report.json",
    "segmentation/previews/segmentation-frame-000029.png",
    "segmentation/previews/mask-review.mp4",
    "inpainting/previews/inpainting-frame-000029.png",
    "inpainting/previews/background-review.mp4",
    "previews/m5-delivery.png",
    "previews/m5-delivery-report.json",
)

_EPISODE_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
_ISSUE = re.compile(r"^### \[([ x~])\] (MOD-\d+)[：:]\s*(.+)$")


class ProgressReader:
    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("E2R_PROJECT_ROOT")
        selected_root = root if root is not None else Path(configured) if configured else Path(__file__).resolve().parents[2]
        self.root = selected_root.resolve()
        if not (self.root / "docs" / "system-design.md").is_file():
            raise ValueError(f"Not an Ego2Robot project root: {self.root}")

    def _json(self, relative: str) -> dict[str, Any] | None:
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root) or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def read_document(self, document: str, offset: int = 0, max_chars: int = 12000) -> dict[str, Any]:
        """Read a named project document, with paging and no arbitrary path input."""
        if document not in DOCUMENTS:
            raise ValueError(f"Unknown document. Choose one of: {', '.join(DOCUMENTS)}")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        if not 1 <= max_chars <= 20000:
            raise ValueError("max_chars must be between 1 and 20000")
        relative = DOCUMENTS[document]
        content = (self.root / relative).read_text(encoding="utf-8")
        end = min(len(content), offset + max_chars)
        return {
            "document": document,
            "source": relative,
            "content": content[offset:end],
            "total_chars": len(content),
            "next_offset": end if end < len(content) else None,
        }

    def open_issues(self) -> list[dict[str, str]]:
        """Return the explicit MOD issue checklist without inferring that code fixes are accepted."""
        content = (self.root / DOCUMENTS["modification_todo"]).read_text(encoding="utf-8")
        priority = "unknown"
        issues: list[dict[str, str]] = []
        for line in content.splitlines():
            if line.startswith("## P0"):
                priority = "P0"
            elif line.startswith("## P1"):
                priority = "P1"
            elif line.startswith("## P2"):
                priority = "P2"
            match = _ISSUE.match(line)
            if match:
                mark, issue_id, title = match.groups()
                issues.append({
                    "id": issue_id,
                    "priority": priority,
                    "title": title.strip(),
                    "code_change_state": "checked" if mark == "x" else "in_progress" if mark == "~" else "open",
                    "source": DOCUMENTS["modification_todo"],
                })
        return issues

    def _episode_root(self, episode_id: str) -> Path:
        if not _EPISODE_ID.fullmatch(episode_id):
            raise ValueError("Invalid episode_id")
        episode = self.root / "data" / "episodes" / episode_id
        if not episode.is_dir() or not episode.resolve().is_relative_to((self.root / "data" / "episodes").resolve()):
            raise ValueError(f"Unknown episode_id: {episode_id}")
        return episode

    def list_episodes(self) -> list[str]:
        base = self.root / "data" / "episodes"
        if not base.is_dir():
            return []
        return sorted(path.name for path in base.iterdir() if path.is_dir() and _EPISODE_ID.fullmatch(path.name))

    def episode_progress(self, episode_id: str) -> dict[str, Any]:
        episode = self._episode_root(episode_id)
        relative = episode.relative_to(self.root).as_posix()
        stages: list[dict[str, Any]] = []
        locations = dict(STAGE_LOCATIONS)
        robot_dir = episode / "robot"
        if robot_dir.is_dir():
            for robot in sorted(robot_dir.iterdir()):
                if robot.is_dir() and _EPISODE_ID.fullmatch(robot.name):
                    locations[f"robot/{robot.name}"] = f"robot/{robot.name}/stage-manifest.json"
        for name, path in locations.items():
            manifest = self._json(f"{relative}/{path}")
            if manifest is None:
                continue
            metrics = manifest.get("metrics")
            stages.append({
                "stage": name,
                "run_status": manifest.get("status", "unknown"),
                "started_at": manifest.get("started_at"),
                "duration_s": manifest.get("duration_s"),
                "manifest": f"{relative}/{path}",
                "metrics": self._safe_metrics(metrics),
            })
        reviews = []
        review_dir = episode / "quality" / "reviews"
        if review_dir.is_dir():
            for path in sorted(review_dir.glob("*.json")):
                review = self._json(path.relative_to(self.root).as_posix())
                if review and review.get("reviewer") == "user" and isinstance(review.get("accepted"), bool):
                    evidence = review.get("evidence")
                    evidence_matches = isinstance(evidence, list) and bool(evidence)
                    if evidence_matches:
                        for item in evidence:
                            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                                evidence_matches = False
                                break
                            artifact = (episode / item["path"]).resolve()
                            if not artifact.is_relative_to(episode.resolve()) or not artifact.is_file():
                                evidence_matches = False
                                break
                            try:
                                with artifact.open("rb") as stream:
                                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                            except OSError:
                                evidence_matches = False
                                break
                            if digest != item["sha256"]:
                                evidence_matches = False
                                break
                    reviews.append({
                        "scope": review.get("scope"),
                        "accepted": review["accepted"],
                        "evidence_matches_current_files": evidence_matches,
                        "current_artifacts_accepted": review["accepted"] and evidence_matches,
                        "reviewed_on": review.get("reviewed_on"),
                        "record": path.relative_to(self.root).as_posix(),
                    })
        evidence = [f"{relative}/{item}" for item in EVIDENCE_FILES if (episode / item).is_file()]
        for folder, pattern in (
            ("previews", "projection-frame-*.png"),
            ("segmentation/previews", "segmentation-frame-*.png"),
            ("inpainting/previews", "inpainting-frame-*.png"),
        ):
            images = sorted((episode / folder).glob(pattern)) if (episode / folder).is_dir() else []
            if images:
                evidence.append(images[len(images) // 2].relative_to(self.root).as_posix())
        quality = self._json(f"{relative}/quality/report.json")
        gate = quality.get("gate_verdict") if quality else None
        return {
            "episode_id": episode_id,
            "stages": stages,
            "user_reviews": reviews,
            "evidence_paths_local_only": evidence,
            "machine_gate": {
                "gate": gate.get("gate"),
                "pass": gate.get("pass"),
                "failing_checks": gate.get("failing_checks", []),
                "source": f"{relative}/quality/report.json",
            } if isinstance(gate, dict) else None,
            "acceptance_rule": "A succeeded run or machine gate never implies user acceptance; only scoped user review records do.",
        }

    @staticmethod
    def _safe_metrics(value: Any) -> dict[str, int | float | bool]:
        if not isinstance(value, dict):
            return {}
        return {
            key: metric for key, metric in value.items()
            if isinstance(key, str) and isinstance(metric, (int, float, bool))
            and (not isinstance(metric, float) or math.isfinite(metric))
        }

    def morphology_summary(self) -> dict[str, Any]:
        relative = "data/morphology_trials/comparison-manifest.json"
        report = self._json(relative)
        if report is None:
            return {"available": False, "source": relative}
        models = {}
        for model_id, data in report.get("models", {}).items():
            if not isinstance(data, dict):
                continue
            binding = data.get("gripper_binding") or {}
            models[model_id] = {
                "ik_success_ratio": data.get("ik_success_ratio"),
                "p95_fk_position_error_m": data.get("p95_fk_position_error_m"),
                "p95_projected_gap_px": data.get("p95_projected_gap_px"),
                "joint_jump_frames": data.get("joint_jump_frames"),
                "gripper_actuated": binding.get("actuated") if isinstance(binding, dict) else None,
                "unsupported_width_frames": binding.get("unsupported_width_frames") if isinstance(binding, dict) else None,
            }
        return {
            "available": True,
            "source": relative,
            "recorded_acceptance": report.get("acceptance"),
            "frame_count": report.get("frame_count"),
            "models": models,
            "comparison_image_local_only": report.get("comparison_image"),
        }

    def project_progress(self) -> dict[str, Any]:
        """Summarize authored status plus current local manifests and review records."""
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        match = re.search(r"(?ms)^## Status\s*\n(.*?)(?=^## |\Z)", readme)
        status_section = match.group(1).strip() if match else "Status section unavailable"
        episodes = self.list_episodes()
        issues = self.open_issues()
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "authored_status": status_section,
            "authored_status_source": "README.md#status",
            "episode_ids": episodes,
            "episode_summaries": [self.episode_progress(episode_id) for episode_id in episodes],
            "morphology_trial": self.morphology_summary(),
            "open_issue_counts": {
                priority: sum(1 for issue in issues if issue["priority"] == priority and issue["code_change_state"] != "checked")
                for priority in ("P0", "P1", "P2")
            },
            "acceptance_rule": "Read user_reviews.scope and accepted. Do not infer milestone acceptance from stage status, README implementation state, or machine gate verdict.",
        }
