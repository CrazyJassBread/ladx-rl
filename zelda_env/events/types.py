"""JSON-safe game-event records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GameEvent:
    type: str
    frame: int
    data: dict[str, Any] = field(default_factory=dict)
    source_paths: tuple[str, ...] = ()
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "frame": self.frame,
            "data": self.data,
            "source_paths": list(self.source_paths),
            "confidence": self.confidence,
        }
