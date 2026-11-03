"""Stage registry and discovery.

Stages register themselves with :func:`register_stage`; :func:`default_registry`
imports every in-tree stage module so the CLI can build the DAG without the
caller enumerating modules by hand.
"""

from __future__ import annotations

import importlib
import sys
from typing import Callable, Iterable, Iterator, Mapping

from .stage import Stage

__all__ = [
    "STAGE_MODULES",
    "StageRegistry",
    "default_registry",
    "register_stage",
]

#: Import order matters only for reproducibility of the printed DAG; the linear
#: order actually executed comes from the ``stages`` list in the pipeline profile.
STAGE_MODULES: tuple[str, ...] = (
    "ego2robot.ingest.stage",
    "ego2robot.perception.hand.stage",
    "ego2robot.retarget.stage",
    "ego2robot.robots.stage",
    "ego2robot.robots.rig_stage",
    "ego2robot.vision.render_stage",
    "ego2robot.viz.projection_debug",
    "ego2robot.quality.stage",
    "ego2robot.vision.m5_stages",
    "ego2robot.vision.m6_stages",
)


class StageRegistry:
    """Ordered collection of stages keyed by ``Stage.name``."""

    def __init__(self) -> None:
        self._stages: dict[str, Stage] = {}
        self._sources: dict[str, str] = {}

    def add(self, stage: Stage, *, source: str = "") -> None:
        name = getattr(stage, "name", None)
        if not name or not isinstance(name, str):
            raise TypeError(f"{stage!r} has no usable 'name' attribute")
        if name in self._stages and self._stages[name] is not stage:
            raise KeyError(
                f"duplicate stage name {name!r} (already registered from "
                f"{self._sources.get(name, '?')}); stage names must be unique"
            )
        self._stages[name] = stage
        self._sources[name] = source or type(stage).__module__

    def extend(self, stages: Iterable[Stage], *, source: str = "") -> None:
        for s in stages:
            self.add(s, source=source)

    def __contains__(self, name: object) -> bool:
        return name in self._stages

    def __getitem__(self, name: str) -> Stage:
        try:
            return self._stages[name]
        except KeyError as exc:
            raise KeyError(f"stage {name!r} is not registered; known: {sorted(self._stages)}") from exc

    def __iter__(self) -> Iterator[Stage]:
        return iter(self._stages.values())

    def __len__(self) -> int:
        return len(self._stages)

    def names(self) -> list[str]:
        return list(self._stages)

    def as_mapping(self) -> dict[str, Stage]:
        return dict(self._stages)

    def subset(self, names: Iterable[str]) -> dict[str, Stage]:
        return {n: self[n] for n in names}

    def describe(self) -> str:
        lines = [f"{len(self._stages)} stage(s) registered:"]
        for name, stage in self._stages.items():
            lines.append(f"  {name} v{stage.version}  <- {self._sources.get(name, '?')}")
        return "\n".join(lines)


#: module-level registry used by the decorator and by discovery
_GLOBAL_REGISTRY = StageRegistry()


def register_stage(
    cls: type | None = None, *, instance: Stage | None = None
) -> Callable[[type], type] | Stage:
    """Register a stage class (instantiated once) or an instance.

    Used as a class decorator::

        @register_stage
        class IngestStage:
            name = "ingest"
            ...
    """
    if instance is not None:
        _GLOBAL_REGISTRY.add(instance)
        return instance

    def _wrap(target: type) -> type:
        _GLOBAL_REGISTRY.add(target())  # type: ignore[call-arg]
        return target

    if cls is None:
        return _wrap
    return _wrap(cls)


def default_registry(
    *, modules: Iterable[str] | None = None, refresh: bool = False
) -> StageRegistry:
    """Import every in-tree stage module and return the populated registry.

    ``refresh=True`` drops the registry and ``reload``\\ s the stage modules so
    their ``@register_stage`` decorators run again.  A plain ``import_module`` is
    not enough for the second call: the modules are already in ``sys.modules``,
    so their decorators would not re-execute and the registry would come back
    empty.
    """
    module_names = list(modules or STAGE_MODULES)
    if refresh:
        _GLOBAL_REGISTRY._stages.clear()
        _GLOBAL_REGISTRY._sources.clear()
        for name in module_names:
            if name in sys.modules:
                importlib.reload(sys.modules[name])

    failures: list[str] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:  # optional heavy dependency (mujoco/mink)
            failures.append(f"{module_name}: {exc}")
    _GLOBAL_REGISTRY.add_import_warnings(failures)
    return _GLOBAL_REGISTRY


def _add_import_warnings(self: StageRegistry, messages: list[str]) -> None:
    """Record which stage modules could not be imported and why."""
    self.import_warnings = list(messages)  # type: ignore[attr-defined]


StageRegistry.add_import_warnings = _add_import_warnings  # type: ignore[attr-defined]


def registry_mapping() -> Mapping[str, Stage]:
    """Convenience accessor for the global registry contents."""
    return _GLOBAL_REGISTRY.as_mapping()
