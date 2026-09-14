"""Event detector protocol."""

from __future__ import annotations

from typing import Any, Protocol

from zelda_env.events.types import GameEvent
from zelda_env.memory.delta import StateDelta


class EventDetector(Protocol):
    def reset(self, state: dict[str, Any]) -> None: ...

    def detect(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
        delta: StateDelta,
        *,
        frame: int,
    ) -> tuple[GameEvent, ...]: ...
