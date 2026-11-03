"""Per-stage hand-arity capabilities (ADR-0021).

The hand/canonical schemas carry a hand axis ``H`` from the start, but the
*consumers* of those arrays are not equally ready.  This module is the single
place that declares which stage can handle how many hands, so a two-hand
sequence cannot be silently squeezed to one hand by a ``[:, 0]`` somewhere deep
in a stage.

Why a table rather than a ``try/except`` on shapes: the failure mode this guards
against is not a crash but a *plausible* result.  A stage written for one hand
that receives ``[T, 2, ...]`` and indexes ``[:, 0]`` produces a perfectly valid
trajectory for the right hand while silently discarding the left one.  The
capability table turns that into an explicit, readable error that names the
stage and the hand count.

Reference: ``docs/decisions/0021-bimanual-hand-contract.md`` (B1 scope) and the
learning-side plan ``docs/handoff/PLAN-02-双臂支持-20261102.md``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "HandCapabilityError",
    "STAGE_HAND_CAPABILITY",
    "describe_capabilities",
    "require_hand_count_support",
    "require_sequence_support",
]


class HandCapabilityError(ValueError):
    """Raised when a stage is asked to consume a hand count it does not support."""

    def __init__(self, stage: str, hand_count: int, supported: int) -> None:
        self.stage = stage
        self.hand_count = int(hand_count)
        self.supported = int(supported)
        super().__init__(
            f"stage {stage!r} does not support H={self.hand_count} hands "
            f"(declared capability: H <= {self.supported}). "
            "The bimanual path is scoped to hand+retarget until that stage is "
            "migrated (ADR-0021); refusing rather than silently using hand 0."
        )


#: Maximum hand count each consumer accepts.
#:
#: ``hand``/``retarget`` were migrated to the existing ``[T, H, ...]`` axis in
#: ADR-0021.  ``rig`` consumes both hands (ADR-0022).  ``segmentation`` and
#: ``inpainting`` were migrated in ADR-0026: the human-removal path never indexes
#: a hand — it segments the text prompt and only checks the hand stream for its
#: timebase — so accepting ``H=2`` means "check both hands' timebase", not "use
#: hand 0 and hope".  Everything else still assumes a single arm, one base pose
#: and one rendered robot; those stages are B3/B4 work and must fail loudly today.
STAGE_HAND_CAPABILITY: dict[str, int] = {
    "hand": 2,
    "retarget": 2,
    "rig": 2,
    "robot": 1,
    "render": 1,
    "projection": 1,
    "quality": 1,
    "export": 1,
    "segmentation": 2,
    "inpainting": 2,
    "scene_depth": 1,
    "composite": 1,
}


def _supported(stage: str) -> int:
    try:
        return STAGE_HAND_CAPABILITY[stage]
    except KeyError:  # pragma: no cover - a new stage must opt in explicitly
        raise KeyError(
            f"stage {stage!r} has no declared hand capability; add it to "
            f"core/capabilities.py::STAGE_HAND_CAPABILITY. Known: "
            f"{sorted(STAGE_HAND_CAPABILITY)}"
        ) from None


def require_hand_count_support(stage: str, hand_count: int) -> None:
    """Raise :class:`HandCapabilityError` unless ``stage`` handles ``hand_count``."""
    supported = _supported(stage)
    if int(hand_count) > supported:
        raise HandCapabilityError(stage, int(hand_count), supported)


def require_sequence_support(stage: str, sequence: Any, *, hand_count: int | None = None) -> None:
    """Check a loaded ``HandSequence``/``CanonicalActionSequence`` against ``stage``.

    ``hand_count`` is accepted explicitly for callers that know the configured
    arity before loading the array; otherwise it is read from the record's
    ``num_hands`` property.
    """
    count = hand_count
    if count is None:
        value = getattr(sequence, "num_hands", None)
        if value is None:
            raise ValueError(
                f"{type(sequence).__name__} exposes no num_hands; cannot check the "
                f"hand capability of stage {stage!r}"
            )
        count = int(value)
    require_hand_count_support(stage, int(count))


def describe_capabilities() -> dict[str, int]:
    """Copy of the capability table, for reports and ``e2r stages``-style output."""
    return dict(STAGE_HAND_CAPABILITY)
