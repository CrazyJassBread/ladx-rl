"""Generic, JSON-safe semantic state differencing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ValueChange:
    previous: Any
    current: Any

    def as_dict(self) -> dict[str, Any]:
        return {"previous": self.previous, "current": self.current}


@dataclass(frozen=True)
class StateDelta:
    changes: dict[str, ValueChange]

    def changed(self, path: str) -> bool:
        return path in self.changes

    def get(self, path: str) -> ValueChange | None:
        return self.changes.get(path)

    def increased(self, path: str) -> bool:
        change = self.get(path)
        return bool(change and _numeric(change.current) and _numeric(change.previous) and change.current > change.previous)

    def decreased(self, path: str) -> bool:
        change = self.get(path)
        return bool(change and _numeric(change.current) and _numeric(change.previous) and change.current < change.previous)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {path: change.as_dict() for path, change in self.changes.items()}


def diff_states(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    ignored_prefixes: Iterable[str] = ("meta.frame", "raw"),
) -> StateDelta:
    """Return leaf changes between two semantic states."""

    before = _flatten(previous)
    after = _flatten(current)
    ignored = tuple(ignored_prefixes)
    changes: dict[str, ValueChange] = {}
    for path in sorted(before.keys() | after.keys()):
        if any(path == prefix or path.startswith(prefix + ".") for prefix in ignored):
            continue
        old = before.get(path)
        new = after.get(path)
        if old != new:
            changes[path] = ValueChange(old, new)
    return StateDelta(changes)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, path))
        return flattened
    return {prefix: value}


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
