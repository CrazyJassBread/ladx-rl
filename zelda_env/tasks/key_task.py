"""Small-key collection objectives unlocked by defeating enemies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from zelda_env.tasks.base import EventList, GameState, TaskStep
from zelda_env.tasks.entity_task import DefeatEntitiesTask


class KillAndCollectTask(DefeatEntitiesTask):
    """Defeat reset-time targets, then collect the resulting small key."""

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        drop_type: int = 0x30,
        **kwargs: Any,
    ) -> None:
        super().__init__(target_types, **kwargs)
        self.drop_type = int(drop_type)
        self._initial_keys = 0
        self._drop_seen = False

    def reset(self, state: GameState) -> dict[str, Any]:
        self._initial_keys = state["progress"]["small_keys"]
        self._drop_seen = False
        return super().reset(state)

    def step(
        self,
        previous: GameState,
        current: GameState,
        events: EventList,
    ) -> TaskStep:
        drop_visible = any(entity["type"] == self.drop_type for entity in current["entities"])
        first_drop = drop_visible and not self._drop_seen
        self._drop_seen |= drop_visible

        result = super().step(previous, current, events)
        if result.info["failure"] is not None:
            return result
        collected = current["progress"]["small_keys"] > self._initial_keys
        was_collected = previous["progress"]["small_keys"] > self._initial_keys
        success = self._cleared and collected
        signals = dict(result.info["reward_signals"])
        if first_drop:
            signals["key_drop_seen"] = 1.0
        if collected and not was_collected:
            signals["key_collected"] = 1.0

        return self._result(current, signals, success=success, failure=None)

    def _success_condition(self, state: GameState) -> bool:
        return state["progress"]["small_keys"] > self._initial_keys

    def _phase(self, state: GameState) -> str:
        if state["progress"]["small_keys"] > self._initial_keys:
            return "complete"
        if self._cleared:
            return "collect_key"
        return "defeat_targets"

    def _extra_info(self, state: GameState) -> dict[str, Any]:
        current_keys = state["progress"]["small_keys"]
        return {
            "drop_type": self.drop_type,
            "drop_seen": self._drop_seen,
            "initial_small_keys": self._initial_keys,
            "current_small_keys": current_keys,
            "key_collected": current_keys > self._initial_keys,
        }
