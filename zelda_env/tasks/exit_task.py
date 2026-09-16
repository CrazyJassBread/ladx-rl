"""Navigation objectives that become available after clearing a room."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from zelda_env.tasks.base import GameState
from zelda_env.tasks.entity_task import DefeatEntitiesTask


class DefeatAndExitTask(DefeatEntitiesTask):
    """Defeat all reset-time targets, then enter one explicit destination room."""

    def __init__(
        self,
        target_types: Iterable[int],
        *,
        target_room: Iterable[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(target_types, **kwargs)
        self.target_room = tuple(int(value) for value in target_room)
        if len(self.target_room) != 3:
            raise ValueError("target_room must be (is_indoor, map_id, room_id)")

    def reset(self, state: GameState) -> dict[str, Any]:
        if _room_key(state) == self.target_room:
            raise ValueError("target_room must differ from the task start room")
        return super().reset(state)

    def _success_condition(self, state: GameState) -> bool:
        return False

    def _room_exit_success(self, state: GameState) -> bool:
        return self._cleared and _room_key(state) == self.target_room

    def _phase(self, state: GameState) -> str:
        if self._room_exit_success(state):
            return "complete"
        return "exit_room" if self._cleared else "defeat_targets"

    def _extra_info(self, state: GameState) -> dict[str, Any]:
        return {
            "target_room": list(self.target_room),
            "destination_reached": self._room_exit_success(state),
        }


def _room_key(state: GameState) -> tuple[int, int, int]:
    room = state["room"]
    return room["is_indoor"], room["map_id"], room["id"]
