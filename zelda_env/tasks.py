"""Task definitions and success/failure evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zelda_env.events.types import GameEvent


@dataclass(frozen=True)
class TaskSpec:
    id: str = "free_play"
    max_steps: int | None = None
    success_events: frozenset[str] = field(default_factory=frozenset)
    failure_events: frozenset[str] = field(default_factory=lambda: frozenset({"player_died"}))
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, events: tuple[GameEvent, ...]) -> tuple[bool, bool, str | None]:
        event_types = {event.type for event in events}
        success = bool(self.success_events & event_types)
        failure = bool(self.failure_events & event_types)
        reason = None
        if success:
            reason = sorted(self.success_events & event_types)[0]
        elif failure:
            reason = sorted(self.failure_events & event_types)[0]
        return success, failure, reason
