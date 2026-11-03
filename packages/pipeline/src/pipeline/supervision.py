"""Versioned language supervision without generating or changing Stage3 labels.

The source-task profile conditions one right-arm action stream on the original
EgoDex instruction. It makes no Step, bilateral-task, outcome, or semantic-QA
claim. Source provenance is checked independently by the reader.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

STRICT_STEPS = "stage3_steps_strict.v1"
SOURCE_TASK = "egodex_source_task.v1"
SUPERVISION_PROFILES = (STRICT_STEPS, SOURCE_TASK)
SUPERVISION_SCHEMA = "learning.stage5.supervision.v1"


def plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def source_task(attributes: dict, source_hdf5_ref: dict) -> dict:
    """Use EgoDex's selector, never silently choose another description."""
    attributes = plain(attributes)
    selector_present = "which_llm_description" in attributes
    selector = str(attributes.get("which_llm_description", "1"))
    field = {"1": "llm_description", "2": "llm_description2"}.get(selector)
    raw = attributes.get(field) if field else None
    available = isinstance(raw, str) and raw.strip().lower() not in {"", "none", "null", "nan"}
    reasons = []
    if field is None:
        reasons.append("S5_SOURCE_TASK_SELECTOR_INVALID")
    elif not available:
        reasons.append("S5_SOURCE_TASK_MISSING")
    return {"availability": "available" if available else "missing",
            "instruction": raw if available else None,
            "source": "egodex_hdf5_attribute", "field": field,
            "selector": selector, "selector_present": selector_present,
            "selection_rule": "which_llm_description" if selector_present else "legacy_default_description_1",
            "raw_attributes": {name: attributes.get(name) for name in
                               ("llm_description", "llm_description2", "which_llm_description", "task", "llm_objects", "object")},
            "source_hdf5_ref": source_hdf5_ref, "outcome": "unknown",
            "semantic_review": "not_evaluated", "reason_codes": reasons}


def annotation_diagnostics(attributes: dict, annotation: dict | None) -> tuple[list[str], list[dict]]:
    """Retain source inventory differences as unverified evidence, not corrections.

    Literal content-word overlap is only a conservative screening rule. It is
    not object identity, semantic equivalence, or a judgment of which text wins.
    No rule depends on an episode id.
    """
    if annotation is None:
        return ["S5_STAGE3_MISSING"], []
    reasons: list[str] = []
    if any(step.get("acting_hand") not in {"right", "none"} for step in annotation.get("steps", [])):
        reasons.append("S5_HAND_COVERAGE_PARTIAL")
    source_objects = plain(attributes).get("llm_objects", [])
    if isinstance(source_objects, str):
        source_objects = [source_objects]
    annotated_objects = [str(obj.get("name", "")) for obj in annotation.get("objects", [])]
    ignored = {"a", "an", "the", "of", "with", "and", "on", "in", "colored", "colour", "color"}
    def tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower())) - ignored
    differences = []
    for original in source_objects:
        if not any(tokens(str(original)) & tokens(other) for other in annotated_objects):
            differences.append({"status": "unverified_inventory_difference", "source_object": str(original),
                                "stage3_objects": annotated_objects,
                                "rule": "no_literal_content_token_overlap.v1",
                                "interpretation": "Potential language/source conflict; neither source is corrected or verified."})
    if differences:
        reasons.append("S5_LANGUAGE_SOURCE_CONFLICT")
    return reasons, differences


def build_supervision(profile: str, attributes: dict, source_hdf5_ref: dict,
                      annotation: dict | None) -> dict:
    if profile not in SUPERVISION_PROFILES:
        raise ValueError(f"Unsupported supervision profile: {profile}")
    task = source_task(attributes, source_hdf5_ref)
    diagnostics, evidence = annotation_diagnostics(attributes, annotation)
    if profile == SOURCE_TASK:
        blocking = task["reason_codes"]
        selected_task = task
    else:
        blocking = list(diagnostics)
        if annotation is not None and not annotation.get("steps"):
            blocking.append("S5_STAGE3_STEPS_MISSING")
        if annotation is not None and annotation.get("steps"):
            covered = set()
            for step in annotation["steps"]:
                covered.update(range(step["start_frame"], step["end_frame_exclusive"]))
            if covered != set(range(annotation["frame_count"])):
                blocking.append("S5_STAGE3_COVERAGE_INCOMPLETE")
        selected_task = annotation.get("task") if annotation else None
        if selected_task is not None and not any(isinstance(selected_task.get(k), str) and selected_task[k].strip()
                                                 for k in ("goal", "description", "instruction")):
            blocking.append("S5_STAGE3_TASK_MISSING")
    return {"schema_version": SUPERVISION_SCHEMA, "profile": profile,
            "availability": "ready" if not blocking else "missing_or_partial",
            "selected_task": selected_task, "source_task": task,
            "stage3_availability": "available" if annotation else "missing",
            "step_supervision": "selected" if profile == STRICT_STEPS else "not_selected",
            "step_sidecar_status": "available" if annotation else "missing",
            "diagnostic_reason_codes": diagnostics, "blocking_reason_codes": sorted(set(blocking)),
            "inventory_difference_evidence": evidence,
            "claims": {"task_conditioning": True, "step_conditioning": profile == STRICT_STEPS,
                       "hand_stream": "right", "bilateral_task_conversion": False,
                       "task_success_verified": False, "source_text_semantically_verified": False}}
